import datasets
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import compute_class_weight
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification

from models import multiple
from evaluator import Evaluator


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass

        logits = model(**inputs)
      #  logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor([2.16034755, 3.01211454, 0.43774008, 7.90462428, 0.83844267, 2.15864246,
                                 0.41320441, 1.38902996, 2.30607083, 1.4025641, 1.06089992, 3.34352078,
                                 0.6442874, 0.48596304]).to("cuda"))
        loss = loss_fct(logits.view(-1, self.model.num_labels), labels.view(-1))
        return (loss, (loss, logits)) if return_outputs else loss


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    print(pred)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")
    print({"accuracy": accuracy, "f1": f1})
    return {"accuracy": accuracy, "f1": f1}


# set constants
WORKING_DIR = "/usr/src/temp/"
#WORKING_DIR = "./"
DATA_FOLDER = "/usr/src/temp/data/"
#DATA_FOLDER = "./data/"
MODEL_NAME = "flaubert/flaubert_base_uncased"

LABEL_COL = "label"
INPUT_COL = "text"


def get_dataset(type, fold):#precalculated embeddings
    ds = None
    path = os.path.join(DATA_FOLDER ,str(fold) , type)
    if os.path.exists(path ):
        ds = datasets.load_from_disk(path)
    return ds

def train_kfold(output, n=10):
    for counter in range(n):

        val_ds = get_dataset("val",  fold=counter)
        train_ds = get_dataset("train", fold=counter)

        print(f'Number of training examples: {len(train_ds)}')
        print(f'Number of validation examples: {len(val_ds)}')

        train(output + str(counter), train_ds, val_ds)



def train(output, train, val):

    training_args = TrainingArguments(
        output_dir=output,
        num_train_epochs=10,
        per_device_train_batch_size=8,  # batch size per device during training
        learning_rate=0.003,  # strength of weight decay
        load_best_model_at_end=True,
        weight_decay=0.3,
        evaluation_strategy="steps",
        fp16=True,
        eval_steps=100,
        logging_steps=100,
        save_total_limit=5,
        gradient_accumulation_steps=32,

    )

    model = multiple.MultipleModel(num_classes=14, num_layers=10, hidden=300, drop_ratio=0.5)

    trainer = CustomTrainer(
        model=model, args=training_args, train_dataset=train, eval_dataset=val,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output)

    metrics = trainer.evaluate()
    print(metrics)



def evaluate(model_path, test):
    evaluator = Evaluator()
    model = multiple.MultipleModel(num_classes=14, num_layers=2, hidden=2560, drop_ratio=0.5)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    y_true = []
    y_pred = []
    for i in range(len(test)) : # hg adds train
        print(i)
        temp = test[i]
        # Then you do test data pre-processing and apply the model on test data
        with torch.no_grad():
            output = model(  input_ids= torch.tensor(temp["input_ids"]).unsqueeze(0),
                attention_mask=torch.tensor(temp["attention_mask"]).unsqueeze(0),
                stime=torch.tensor(temp["stime"]).unsqueeze(0),
                             media=torch.tensor(temp["media"]).unsqueeze(0),duration=torch.tensor(temp["duration"]).unsqueeze(0))


        y_true.append(torch.tensor([temp["label"]]))
        y_pred.append(output.max(1)[1].detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)




if __name__ == '__main__':
    train_kfold("./multi2_trainer")
   #  val_ds = get_dataset("val",  fold=0)
   #  results = evaluate("/usr/src/temp/multi_trainer/pytorch_model.bin", val_ds)
   #  print(results)
