
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import os
import os.path as osp
from evaluator import Evaluator
from dataset_audio import WavImageDataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

class TrainClassifier:
    def __init__(self, data_dir, model, epochs, batch_size, lr, lr_decay_factor, lr_decay_step_size,
                 weight_decay, log_dir=None, checkpoint_dir=None, checkpoint_name="checkpoint.pt", device_no="cuda",
                 isMultiLabel=False):

        self.device = torch.device(device_no if torch.cuda.is_available() else 'cpu')



        self.data_dir = data_dir
        self.transforms = transforms.Compose([transforms.Resize((128, 128)),transforms.ToTensor()])

        self.model = model
        self.evaluator = Evaluator()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_step_size = lr_decay_step_size
        self.weight_decay = weight_decay
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        print("checkpoint_dir", checkpoint_dir)
        self.checkpoint_name = checkpoint_name
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.best_val_acc = 0


        self.init_datasets()

        self.reg_criterion = torch.nn.CrossEntropyLoss()




    def init_datasets(self):

        train_transforms = transforms.Compose([
           # transforms.RandomCrop(28, padding=2),
            transforms.Pad(1),
            transforms.Resize((299, 299)),
            transforms.ToTensor()
        ])



        self.train_data_set = WavImageDataset(data_dir=self.data_dir, csv_file="data/test/train.csv", transforms=train_transforms)
        self.test_data_set = WavImageDataset(data_dir=self.data_dir, csv_file="data/test/test.csv", transforms=train_transforms)
        self.val_data_set = WavImageDataset(data_dir=self.data_dir, csv_file="data/test/val.csv", transforms=train_transforms)

        self.train_loader = DataLoader(self.train_data_set, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data_set, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_data_set, batch_size=self.batch_size, shuffle=False)

        print(f'Number of training examples: {len( self.train_data_set)}')
        print(f'Number of validation examples: {len(self.test_data_set)}')
        print(f'Number of testing examples: {len(self.val_data_set)}')



    def logger(self, info):
        epoch = info['epoch']
        train_loss, val_loss, val, train, test = info['train_loss'], info['val_loss'], info['val'], info["train"], info[
            "test"]
        print('{} - {:.4f} : {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f}'.format(
            epoch, train_loss, train.values, val.values, test.values))

    def run(self):
       # self.model.reset_parameters()

        self.model.to(self.device)
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=self.lr_decay_step_size, gamma=self.lr_decay_factor)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)
        if self.checkpoint_dir is not None :
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.log_dir is not '':
            os.makedirs(self.log_dir, exist_ok=True)
            writer_train = SummaryWriter(log_dir=os.path.join(self.log_dir, "train"))
            writer_val = SummaryWriter(log_dir=os.path.join(self.log_dir, "val"))
            writer_test = SummaryWriter(log_dir= os.path.join(self.log_dir, "test"))


        best_info = None
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train(self.model, optimizer, self.train_loader)
            _, train_dict = self.eval(self.model, self.train_loader, self.evaluator)
            val_loss, val_dict = self.eval(self.model, self.val_loader, self.evaluator)
            _, test_dict = self.eval(self.model, self.test_loader, self.evaluator)
            eval_info = {

                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val': val_dict,
                'test': test_dict,
                'train': train_dict
            }

            if self.log_dir is not '':
                writer_val.add_scalar('loss', val_loss, epoch)
                writer_train.add_scalar('loss', train_loss, epoch)

                writer_train.add_scalar('acc', train_dict["acc"], epoch)
                writer_val.add_scalar('acc', val_dict["acc"], epoch)
                writer_test.add_scalar('acc', test_dict["acc"], epoch)

                writer_train.add_scalar('mcc', train_dict["mcc"], epoch)
                writer_val.add_scalar('mcc', val_dict["mcc"], epoch)
                writer_test.add_scalar('mcc', test_dict["mcc"], epoch)

                writer_train.add_scalar('f1', train_dict["f1"], epoch)
                writer_val.add_scalar('f1', val_dict["f1"], epoch)
                writer_test.add_scalar('f1', test_dict["f1"], epoch)

            if self.logger is not None:
                print(epoch,
                      '{:.4f} - {:.4f} | {:.3f} {:.3f} | {:.3f} {:.3f}'.format(train_loss, val_loss, train_dict["acc"],
                                                                               val_dict["acc"], train_dict["mcc"],
                                                                               val_dict["mcc"]))

            acc = val_dict["acc"]
            if acc > self.best_val_acc:
                self.best_val_acc = acc
                best_info = eval_info
                if self.checkpoint_dir is not None :
                    print('Saving checkpoint...', acc)
                    checkpoint = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'scheduler_state_dict': scheduler.state_dict(), 'best_test_acc': self.best_val_acc,
                                  'num_params': self.num_params}
                    torch.save(checkpoint, osp.join(self.checkpoint_dir, self.checkpoint_name))

            scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize(device=self.device)

        if self.log_dir is not '':
            writer_val.close()
            writer_test.close()
            writer_train.close()


        if best_info is None:  # no best detected in fold
            best_info = eval_info  # get mlast
        return best_info

    def train(self, model, optimizer, loader):
        model.train()
        total_loss = 0
        for img, label in loader:
            optimizer.zero_grad()
            data = img.to(self.device)
            y = label.to(self.device)

            out = model(data)

            loss = self.reg_criterion(out, y)  # .view(-1)
            loss.backward()
            optimizer.step()
            print(loss.item())
            total_loss += loss.item()



        return total_loss / len(loader.dataset)

    def eval(self, model, loader, evaluator):
        model.eval()
        y_true = []
        y_pred = []
        loss_all = 0
        for img, label in loader:

            data = img.to(self.device)
            y = label.to(self.device)

            with torch.no_grad():
                pred = model(data)

            loss = self.reg_criterion(pred, y)  #
            loss_all += loss.item()

            y_true.append(y.view(-1).detach().cpu())
            y_pred.append(pred.max(1)[1].detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return loss_all / len(loader.dataset), evaluator.eval(input_dict)
