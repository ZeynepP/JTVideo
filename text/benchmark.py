import json
import argparse

import datasets
from torch.nn import DataParallel

import train_classifier

import os
import pandas as pd

from models import multiple
import pyarrow as pa
def process(batch, args):
    for a in args:
        print(a)
  #  flags = pa.compute.match_substring(batch["audio"], pattern="201912031")



    return batch


def init_datasets(data_dir):
    print("[INFO] Getting dataset audio...")
    if os.path.exists("/usr/src/audio/data/val"):
        val_audio = datasets.load_from_disk("/usr/src/audio/data/val")
        val_audio = val_audio.remove_columns(["labels","label"]).data

    print(f'Number of audio validation examples: {len(val_audio)}')

    print("[INFO] Getting dataset val text...")
    if os.path.exists(data_dir + "val"):
        val_text = datasets.load_from_disk(data_dir + "val")
        val_text = val_text.remove_columns(["text"]).data
    print(f'Number of text validation examples: {len(val_text)}')
    val_data = None
    for a in val_audio.to_batches():
        val_df = a.to_pandas()
        val_df["id"] = val_df["audio"].apply(lambda ex: ex.split("/")[-1].replace(".wav", ""))
        val_df["id"] = val_df["id"].astype("int32")
        flags = pa.compute.is_in(val_text["id"], value_set=pa.array(val_df["id"]))
        t = val_text.filter(flags)
        if val_data is None:
            val_data = pd.merge(val_df, t.to_pandas(), how="inner", on="id")
        else:
            val_data = val_data.append(pd.merge(val_df, t.to_pandas(), how="inner", on="id"))

    print(f'Number of  val examples: {len(val_data)}')


    if os.path.exists("../audio/data/train"):
        train_audio = datasets.load_from_disk("/usr/src/audio/data/train")
        train_audio = train_audio.remove_columns(["labels","label"]).data
    print(f'Number of audio training examples: {len(train_audio)}')


    if os.path.exists(data_dir + "train"):
        train_text = datasets.load_from_disk(data_dir + "train")
        train_text = train_text.remove_columns(["text"]).data

    print(f'Number of text training examples: {len(train_text)}')

    train_data = None
    for a in train_audio.to_batches():
        train_df = a.to_pandas()
        train_df["id"] = train_df["audio"].apply(lambda ex: ex.split("/")[-1].replace(".wav", ""))
        train_df["id"] = train_df["id"].astype("int32")
        flags = pa.compute.is_in(train_text["id"], value_set=pa.array(train_df["id"]))
        t = train_text.filter(flags)
        if train_data is None:
            train_data = pd.merge(train_df, t.to_pandas(), how="inner", on="id")
        else:
            train_data = train_data.append(pd.merge(train_df, t.to_pandas(), how="inner", on="id"))

    print(f'Number of  train examples: {len(train_data)}')

    return {"train": train_data, "val":val_data}


def init_datasets_pandas(data_dir):


    print("[INFO] Getting dataset audio...")
    if os.path.exists("../audio/data/val"):
        val_audio = datasets.load_from_disk("../audio/data/val")
        val_audio = val_audio.remove_columns(["labels","label"])
        val_audio_df = val_audio.to_pandas()
        del val_audio
        #val_audio_df["id"] = val_audio_df.apply(lambda example: example['audio'].split("/")[-1].replace(".wav", ""), axis=1)
    print(f'Number of audio validation examples: {len(val_audio_df)}')

    if os.path.exists("../audio/data/train"):
        train_audio = datasets.load_from_disk("../audio/data/train")
        train_audio = train_audio.remove_columns(["labels","label"])
        print("tn to pandas ")
        train_audio_df = train_audio.to_pandas()
        del train_audio

    print(f'Number of audio training examples: {len(train_audio_df)}')

    print("[INFO] Getting dataset text...")
    if os.path.exists(data_dir + "val"):
        val_text = datasets.load_from_disk(data_dir + "val")
        val_df = val_text.to_pandas()
        val_df["id"] = val_df["id"].astype("string")
    print(f'Number of text validation examples: {len(val_text)}')

    if os.path.exists(data_dir + "train"):
        train_text = datasets.load_from_disk(data_dir + "train")
        train_df = val_text.to_pandas()
        train_df["id"] = train_df["id"].astype("string")

    print(f'Number of text training examples: {len(train_text)}')


    train = pd.merge(train_audio_df, train_df, how="inner", on="id")
    print(f'Number of training examples: {len(train)}')

    val = pd.merge(val_audio_df, val_df,how="inner", on="id")
    print(f'Number of validation examples: {len(val)}')



    return {"train": train, "val":val}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lr_decay_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_step_size', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--drop_ratio', type=int, default=0.5)
    parser.add_argument('--log_dir', type=str, default="/usr/src/temp/logs/")
    parser.add_argument('--data_dir', type=str, default="/usr/src/temp/data/")
    parser.add_argument('--checkpoint_dir', type=str, default="")

    args = parser.parse_args()
    if args.log_dir is not '':
        os.makedirs(args.log_dir, exist_ok=True)

    with open(args.log_dir + "read.me","w+") as r:
        json.dump(args.__dict__, r, indent=2)

    results = []

    dataloaders = init_datasets(args.data_dir)


    model = multiple.MultipleModel(num_classes=14,num_layers=1, hidden=128, drop_ratio=0.1)

    name = 'multi'


    classifier = train_classifier.TrainClassifier(
        dataloaders,
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

