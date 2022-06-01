import json

import datasets
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import class_weight, compute_class_weight
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm
from transformers import TrainingArguments, FlaubertModel, FlaubertForSequenceClassification, FlaubertTokenizer, \
    SchedulerType, \
    Trainer, AutoConfig, BertModel
import torch
from dataset_text import JTTextDataset, NewsGroupsDataset
import numpy as np
import torch.nn as nn
import pandas as pd
from evaluator import Evaluator
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')

        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(
            [2.16071429, 3.01357213, 0.43773754, 7.88937729, 0.83838069,
             2.15768383, 0.41322282, 1.38910029, 2.3047619, 1.40239615,
             1.06098522, 3.34440994, 0.64436799, 0.4859986]).to("cuda"))

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")
    print({"accuracy": accuracy,  "f1": f1})
    return {"accuracy": accuracy,  "f1": f1}


from datetime import datetime
import time


def convert_time(doctime):
    doctime = doctime.split("T")[1]
    if "." in doctime:
        doctime = doctime.split(".")[0]  # 20:20:28
    else:
        doctime = doctime.split("+")[0]
    return datetime.strptime(doctime, "%H:%M:%S")

def json_to_csv(file_path):

    out_path = file_path.replace(".json",".csv")
    num = sum(1 for line in open(file_path))
    classes_labels = {'santé': 0, 'culture-loisirs': 1, 'société': 2, 'sciences_et_techniques': 3, 'economie': 4,
                      'environnement': 5,
                      'politique_france': 6, 'sport': 7, 'histoire-hommages': 8, 'justice': 9, 'faits_divers': 10,
                      'education': 11,
                      'catastrophes': 12, 'international': 13}
    media_labels = {"tv/fr2" :0, "tv/fr3":1, "tv/m6_":2,"tv/tf1":3}

    start = datetime.strptime("19:00:00", "%H:%M:%S")

    with open(out_path, mode="a+", encoding='utf-8') as out:
        with open(file_path, encoding='utf-8') as file:

            for line in tqdm(file, total=num):
                try:
                    json_obj = json.loads(line)
                    # tokenizing text
                    text_raw = json_obj["document"]
                    label_str = json_obj["topics"][0]["label"]
                    #      "2017-06-07T20:20:28.2+02:00"

                    dtime = convert_time(json_obj["docTime"])
                    etime = convert_time(json_obj["endTime"])
                    duree = (etime - dtime).total_seconds()
                    stime = (dtime - start).total_seconds()

                    out.write(",".join([str(json_obj["id"]), str(media_labels[str(json_obj["media"])]), str(duree), str(stime),  str(classes_labels[label_str.replace(" ", "_").lower()]), text_raw]))
                    out.write("\n")

                except Exception as ex:
                    print(ex, line)




import os

# set constants
WORKING_DIR = "/usr/src/temp/"
#WORKING_DIR = ""
DATA_FOLDER = WORKING_DIR + "data/"
MODEL_NAME = "flaubert/flaubert_base_uncased"

TRAIN = os.path.join(DATA_FOLDER ,"train.json.csv")
VALIDATION = os.path.join(DATA_FOLDER, "val.json.csv")
TEST = os.path.join(DATA_FOLDER, "test.json.csv")
ALL = os.path.join(DATA_FOLDER, "all.csv")

LABEL_COL = "label"
INPUT_COL = "text"

def preprocess(batch):
    processor = FlaubertTokenizer.from_pretrained(MODEL_NAME)
    out = processor(batch[INPUT_COL],  padding='max_length', max_length=512, truncation=True)
    out["label"] = list(batch[LABEL_COL])
    out["id"] = list(batch["id"])
    out["media"] = list(batch["media"])
    out["duration"] = list(batch["duration"])
    out["stime"] = list(batch["stime"])
    return out

def train_kfold(output, n=10):
    data_df = pd.read_csv(DATA_FOLDER + "all.csv", delimiter=",", header=None, names=["id","media","duration","stime","label","text"])
    rs = StratifiedShuffleSplit(n_splits=n, test_size=.1, random_state=123456)

    counter = 0
    for train_index, rest_index in rs.split(data_df[INPUT_COL], data_df[LABEL_COL]):
        print(train_index, rest_index)
        train_ds = get_dataset("train",data_df.iloc[train_index], fold=counter )
        val_ds = get_dataset("val", data_df.iloc[rest_index], fold=counter)

        print(f'Number of training examples: {len(train_ds)}')
        print(f'Number of validation examples: {len(val_ds)}')
        train(output + str(counter), train_ds, val_ds)
        counter += 1


def train(output, train_ds, val_ds):

    model = FlaubertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=14)
    training_args = TrainingArguments(
        output_dir=output,
        num_train_epochs=4,
        per_device_train_batch_size=8,  # batch size per device during training
        learning_rate=0.0001,  # strength of weight decay
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        log_level="debug",
        fp16=True,
        eval_steps=10,
        logging_steps=10,
        save_total_limit=5,
        gradient_accumulation_steps=32,
    )

    trainer = CustomTrainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics
    )
    trainer.train()
    metrics = trainer.evaluate()
    print( metrics)
    trainer.save_model(output)





def get_dataset(type, df, fold):
    path = os.path.join(DATA_FOLDER ,str(fold) , type)
    if os.path.exists(path ):
        ds = datasets.load_from_disk(path)
    else:
        ds =  Dataset.from_pandas(df)
        print(fold, type, " Creating emb...")
        ds = ds.map(preprocess, batched=True)
        print(fold, type, "Saving emb...")
        ds.save_to_disk(path)

    return ds





def evaluate(model_path):
    evaluator = Evaluator()
    if not os.path.exists(TEST):
        get_dataset(TEST)
    test = load_dataset("csv", data_files=TEST, delimiter=",")
    if os.path.exists(DATA_FOLDER + "test"):
        test = datasets.load_from_disk(DATA_FOLDER + "test")
    else:
        print("Creating test emb...")
        test = test.map(preprocess, batched=True)
        print("Saving test emb...")
        test.save_to_disk(DATA_FOLDER + "test")


    model = FlaubertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    y_true = []
    y_pred = []
    for i in range(len(test["train"])) : # hg adds train
        print(i)
        temp = test["train"][i]
        # Then you do test data pre-processing and apply the model on test data
        with torch.no_grad():
            output = model(  input_ids= torch.tensor(temp["input_ids"]).unsqueeze(0),
                attention_mask=torch.tensor(temp["attention_mask"]).unsqueeze(0),
                token_type_ids=torch.tensor(temp["token_type_ids"]).unsqueeze(0))


        y_true.append(torch.tensor([temp["label"]]))
        y_pred.append(output.logits.max(1)[1].detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)



if __name__ == '__main__':
    print("starting...")
    #json_to_csv("./data/subjects_v3_2017-2019.json")
    train_kfold("/usr/src/temp/t3_trainer/",n=5)
    #train("/usr/src/temp/text_trainer/")
   # eval_result = evaluate("/usr/src/temp/text_trainer/")
   # print(eval_result)
   #  dataset = load_dataset("csv", data_files=ALL, delimiter=",")
   #  if os.path.exists(DATA_FOLDER + "all"):
   #      dataset = datasets.load_from_disk(DATA_FOLDER + "all")
   #  else:
   #      print("Creating all emb...")
   #      dataset = dataset.map(preprocess, batched=True)
   #      print("Saving all emb...")
   #      dataset.save_to_disk(DATA_FOLDER + "all")