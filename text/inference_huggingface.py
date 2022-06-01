
import datasets
from datasets import load_dataset

from transformers import  FlaubertForSequenceClassification
import torch
from evaluator import Evaluator
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import os


# set constants
WORKING_DIR = "/usr/src/temp/"
WORKING_DIR = ""
DATA_FOLDER = WORKING_DIR + "data/"
MODEL_NAME = "flaubert/flaubert_base_uncased"

TRAIN = os.path.join(DATA_FOLDER ,"train.json.csv")
VALIDATION = os.path.join(DATA_FOLDER, "val.json.csv")
TEST = os.path.join(DATA_FOLDER, "test.json.csv")
ALL = os.path.join(DATA_FOLDER, "all.csv")

LABEL_COL = "label"
INPUT_COL = "text"

labels = {'santé': 0, 'culture-loisirs': 1, 'société': 2, 'sciences_et_techniques': 3, 'economie': 4,
                  'environnement': 5,
                  'politique_france': 6, 'sport': 7, 'histoire-hommages': 8, 'justice': 9, 'faits_divers': 10,
                  'education': 11,
                  'catastrophes': 12, 'international': 13}
def evaluate(model_path):
    evaluator = Evaluator()
    test = load_dataset("csv", data_files=TEST, delimiter=",")
    if os.path.exists(DATA_FOLDER + "test"):
        test = datasets.load_from_disk(DATA_FOLDER + "test")

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

    cm = confusion_matrix(y_true, y_pred)

    return evaluator.eval(input_dict),cm




#eval_result,cm = evaluate("/usr/src/temp/text_trainer/")
#print(eval_result)
#print(cm)

#{'mcc': 0.7822066274545914, 'acc': 0.8050971380823062, 'f1': 0.8055146844254812}
