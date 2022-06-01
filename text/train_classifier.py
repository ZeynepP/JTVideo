import datasets
import torch
from torch.nn import DataParallel

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

import os
import os.path as osp

from tqdm import tqdm

from evaluator import Evaluator
from torch.utils.data import Dataset, DataLoader

import numpy as np

class PandasDataset(Dataset):

    # convert a df to tensor to be used in pytorch
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.length = 320000


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        df = self.dataframe.iloc[index]
        values = df["input_values"]
        return torch.from_numpy(np.array(df["input_ids"])),torch.from_numpy(np.array(df["attention_mask"])),torch.from_numpy(np.array(values)),torch.tensor(df["label"])

class TrainClassifier:
    def __init__(self, dataset, model, epochs, batch_size, lr, lr_decay_factor, lr_decay_step_size,
                 weight_decay, log_dir=None, checkpoint_dir=None, checkpoint_name="checkpoint.pt", device_no="cuda",
                 isMultiLabel=False):

        self.device = torch.device(device_no if torch.cuda.is_available() else 'cpu')


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

        self.best_val_acc = 0


        self.train_loader = DataLoader(PandasDataset(dataset["train"]), batch_size=self.batch_size, shuffle=True)

        self.val_loader = DataLoader(PandasDataset(dataset["val"]), batch_size=self.batch_size, shuffle=False)

        self.reg_criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([2.16034755,3.01211454 ,0.43774008 ,7.90462428 ,0.83844267,2.15864246,
 0.41320441, 1.38902996, 2.30607083 ,1.4025641,  1.06089992, 3.34352078,
 0.6442874  ,0.48596304]).to(self.device))

        self.model = model.to(self.device) #MultipleModel(num_classes=14, num_layers=2, num_features=768, drop_ratio=0.5, hidden=128)
        self.model = DataParallel(model, device_ids=[0, 1])
        self.num_params = sum(p.numel() for p in self.model.parameters())


    def logger(self, info):
        epoch = info['epoch']
        train_loss, val_loss, val, train = info['train_loss'], info['val_loss'], info['val'], info["train"]
        print('{} - {:.4f} : {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} '.format(
            epoch, train_loss, val_loss, train.values, val.values))

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
            writer_train = SummaryWriter(log_dir=os.path.join(self.log_dir, "train"))
            writer_val = SummaryWriter(log_dir=os.path.join(self.log_dir, "val"))
         #   writer_test = SummaryWriter(log_dir= os.path.join(self.log_dir, "test"))


        best_info = None
        for epoch in range(1, self.epochs + 1):

            train_loss = self.train(self.model, optimizer, self.train_loader)
            _, train_dict = self.eval(self.model, self.train_loader, self.evaluator)
            val_loss, val_dict = self.eval(self.model, self.val_loader, self.evaluator)
          #  _, test_dict = self.eval(self.model, self.test_loader, self.evaluator)


            eval_info = {

                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val': val_dict,
           #     'test': test_dict,
                'train': train_dict
            }

            if self.log_dir is not '':
                writer_val.add_scalar('loss', val_loss, epoch)
                writer_train.add_scalar('loss', train_loss, epoch)

                writer_train.add_scalar('acc', train_dict["acc"], epoch)
                writer_val.add_scalar('acc', val_dict["acc"], epoch)
            #    writer_test.add_scalar('acc', test_dict["acc"], epoch)


                writer_train.add_scalar('f1', train_dict["f1"], epoch)
                writer_val.add_scalar('f1', val_dict["f1"], epoch)
              #  writer_test.add_scalar('f1', test_dict["f1"], epoch)

            #if self.logger is not None:
            print(eval_info)

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
          #  writer_test.close()
            writer_train.close()


        if best_info is None:  # no best detected in fold
            best_info = eval_info  # get mlast
        return best_info

    def train(self, model, optimizer, loader):
        model.train()
        total_loss = 0.0
        for  input_id, mask, input_values, label in tqdm(loader):

            label = label.to(self.device)
            mask = mask.to(self.device)
            input_id = input_id.to(self.device)
            input_values = input_values.to(self.device)

            output = model(input_id, mask, input_values)

            loss = self.reg_criterion(output, label)
            total_loss += loss.item()



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values

        return total_loss / len(loader)

    def eval(self, model, loader, evaluator):
        model.eval()
        y_true = []
        y_pred = []
        loss_all = 0
        for input_id, mask, input_values, label in tqdm(loader):
            label = label.to(self.device)
            mask = mask.to(self.device)
            input_id = input_id.to(self.device)
            input_values = input_values.to(self.device)

            with torch.no_grad():
                output = model(input_id, mask, input_values)

            loss = self.reg_criterion(output, label)
            loss_all += loss.item()

            y_true.append(label.view(-1).detach().cpu())
            y_pred.append(output.max(1)[1].detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()
        y_pred = torch.cat(y_pred, dim=0).numpy()
        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return loss_all / len(loader), evaluator.eval(input_dict)
