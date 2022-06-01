import json
from itertools import product

import argparse
import torchvision
from torch.nn import DataParallel

import train_classifier


parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.1)
parser.add_argument('--lr_decay_step_size', type=int, default=10)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--drop_ratio', type=int, default=0.5)
parser.add_argument('--log_dir', type=str, default="/usr/src/images/logs/")
parser.add_argument('--data_dir', type=str, default="/usr/src/samsung/")
parser.add_argument('--checkpoint_dir', type=str, default="")

args = parser.parse_args()

#TODO: add all to args and config file & copy config file to results folder to track the config
nets = [ConvNet,LeNet]

with open(args.log_dir + "read.me","w+") as r:
    json.dump(args.__dict__, r, indent=2)

results = []


model = torchvision.models.inception_v3()
model.aux_logits = False

name = 'inception'


classifier = train_classifier.TrainClassifier(
    args.data_dir,
    model,
    epochs=args.epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    lr_decay_factor=args.lr_decay_factor,
    lr_decay_step_size=args.lr_decay_step_size,
    weight_decay=args.weight_decay,
    log_dir=args.log_dir +  "/" + name
   # checkpoint_dir=args.checkpoint_dir ,
   # checkpoint_name=name + "checkpoint.pt"

)

best_info = classifier.run()
best_info["model"] = name
print('{} - {}: {}'.format(args.dataset, model, best_info))
results.append(best_info)

print("##################")
for r in results:
    print(r)

