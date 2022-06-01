
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from transformers import AutoProcessor, AutoModel, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor, \
    Trainer, TrainingArguments, Wav2Vec2ForSequenceClassification
import torch

from evaluator import Evaluator
from dataset_audio import  HGAudioDataset
import numpy as np
import torch.nn as nn
import datasets
from sklearn.metrics import confusion_matrix, precision_score, recall_score


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([2.16034755,3.01211454 ,0.43774008 ,7.90462428 ,0.83844267,2.15864246,
 0.41320441, 1.38902996, 2.30607083 ,1.4025641,  1.06089992, 3.34352078,
 0.6442874  ,0.48596304]).to("cuda"))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    print(pred)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")
    print({"accuracy": accuracy,  "f1": f1})
    return {"accuracy": accuracy,  "f1": f1}



model_checkpoint = "LeBenchmark/wav2vec2-FR-7K-base"
batch_size = 32
max_duration = 10 #seconds
WORKING_DIR = "/usr/src/temp/"
#WORKING_DIR = "./"
def evaluate(model_path):
    evaluator = Evaluator()
    data_dir = WORKING_DIR + "data/"
    print(data_dir)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True,
                                                 max_length=int(16000 * max_duration))
    if os.path.exists(data_dir +"val"):
        val_data = datasets.load_from_disk(data_dir +"val")
    else:
        val_data = HGAudioDataset(data_dir + "/val.json", AUDIO_DIR, feature_extractor,max_duration)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_path,
        num_labels=14
    )
    model.eval()
    y_true = []
    y_pred = []
    print(val_data)
    for i in range(val_data.num_rows) : # hg adds train
        print(i)
        temp = val_data[i]
        # Then you do test data pre-processing and apply the model on test data
        with torch.no_grad():
            output = model(
                input_values= torch.tensor(temp["input_values"]).unsqueeze(0),
                labels=torch.tensor(temp["labels"]).unsqueeze(0))


        y_true.append(torch.tensor([temp["label"]]))
        y_pred.append(output.logits.max(1)[1].detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}

    cm = confusion_matrix(y_true, y_pred)

    return evaluator.eval(input_dict),cm

AUDIO_DIR = WORKING_DIR + './audio'
def train():

    data_dir = WORKING_DIR + "data/"
    print(data_dir)
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                 do_normalize=True, return_attention_mask=True,
                                                 max_length=int(16000 * max_duration))
    if os.path.exists(data_dir +"val"):
        val_data = datasets.load_from_disk(data_dir +"val")
    else:
        val_data = HGAudioDataset(data_dir + "/val.json", AUDIO_DIR, feature_extractor,max_duration)

    if os.path.exists(data_dir +"train"):
        train_data = datasets.load_from_disk(data_dir +"train")
    else:
        train_data = HGAudioDataset(data_dir + "train.json", AUDIO_DIR, feature_extractor, max_duration)
    print(train_data[0])

    training_args = TrainingArguments(
        output_dir='./test_trainer/',
        num_train_epochs=4,
        per_device_train_batch_size=8,  # batch size per device during training
        learning_rate=0.01,# strength of weight decay
        load_best_model_at_end=False,
        log_level="debug",
        save_strategy="no",
        evaluation_strategy="epoch",
    )

    # instantiate a data collator that takes care of correctly padding the input dat

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=14
    )


    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(val_data)}')

    trainer = CustomTrainer(
        model=model, args=training_args,  train_dataset=train_data, eval_dataset=val_data, compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(WORKING_DIR + "./test_trainer/")

import os
if __name__ == '__main__':
    #train()
    results, cm = evaluate(WORKING_DIR + "./test_trainer/")
    print(results)
    print(cm)

