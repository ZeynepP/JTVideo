import json

import torchaudio
from datasets import DatasetDict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils import class_weight, compute_class_weight
from tqdm import tqdm
from transformers import TrainingArguments, Trainer, AutoModelForAudioClassification, Wav2Vec2Processor, \
    Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
from dataset_audio import  HGAudioDataset
import numpy as np
import torch.nn as nn
import datasets

from dataclasses import dataclass
from typing import Union, Optional, Dict, List


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



model_checkpoint = "./models/wav2vec2-base"
batch_size = 32
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_checkpoint)
max_duration = 10 #seconds
# feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True, max_length=int(16000 * max_duration))
# tokenizer = Wav2Vec2CTCTokenizer("./models/wav2vec2-base/vocab.json",max_length=int(16000 * max_duration), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)



print(feature_extractor.sampling_rate)
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
    )
    return inputs

def train():


    training_args = TrainingArguments(
        output_dir='./test_trainer/',
        num_train_epochs=4,
        per_device_train_batch_size=4,  # batch size per device during training
        learning_rate=0.00001,# strength of weight decay
        load_best_model_at_end=False,
        log_level="debug",
        save_strategy="no",
        evaluation_strategy="epoch",
    )

    # instantiate a data collator that takes care of correctly padding the input data


    data_dir = "./data"
    AUDIO_DIR = "./data/test/"
    device = "cuda"
    classes_labels = ['santé', 'culture-loisirs', 'société', 'sciences_et_techniques', 'economie',
                           'environnement',
                           'politique_france', 'sport', 'histoire-hommages', 'justice', 'faits_divers',
                           'education',
                           'catastrophes', 'international']



    train_data = HGAudioDataset(data_dir + "/train_short.json",  AUDIO_DIR, feature_extractor, max_duration)



    print(train_data[0])

    val_data = HGAudioDataset(data_dir + "/val_short.json", AUDIO_DIR,feature_extractor, max_duration)

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
    trainer.save_model("./test_trainer/")

import os
if __name__ == '__main__':
    # if os.path.exists("./data/train"):
    #     train_audio = datasets.load_from_disk("./data/train")
    # print(train_audio.features)
    #
    # train_text = torch.load("./data/train.json.emb")
    # print(train_audio[0])
    train()