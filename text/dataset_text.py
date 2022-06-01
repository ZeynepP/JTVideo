
import torch

from tqdm import tqdm
from transformers import  FlaubertTokenizer
import numpy as np
import json
import os
os.environ['http_proxy'] = "http://firewall.ina.fr:81"
os.environ['HTTP_PROXY'] = "http://firewall.ina.fr:81"
os.environ['https_proxy'] = "http://firewall.ina.fr:81"
os.environ['HTTPS_PROXY'] = "http://firewall.ina.fr:81"

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
       item = {k: v[idx] for k, v in self.encodings.items()}
       item["labels"] = self.labels[idx]
       return item



        #mask = self.encodings["attention_mask"][idx]
        #return {"input_ids": ids,"attention_mask": mask}, self.labels[idx

    def __len__(self):
        return len(self.labels)

class JTTextDataset(torch.utils.data.Dataset):

    def __init__(self, data_file='./data', truncate_type = "middle"):
        self.modelname = 'flaubert/flaubert_base_uncased'
        self.file_path = data_file
        self.classes_labels = {'santé': 0, 'culture-loisirs': 1, 'société': 2, 'sciences_et_techniques': 3, 'economie': 4, 'environnement': 5,
         'politique_france': 6, 'sport': 7, 'histoire-hommages': 8, 'justice': 9, 'faits_divers': 10, 'education': 11,
         'catastrophes': 12, 'international': 13}

        self.texts = []
        self.labels = []
        self.truncate_type = truncate_type
        self.process()

        print(data_file + ".emb", os.path.exists( data_file + ".emb"))
        if os.path.exists( data_file + ".emb"):
            self.encodings = torch.load(data_file + ".emb")
        else:
            tokenizer = FlaubertTokenizer.from_pretrained(self.modelname)
            self.encodings = [tokenizer(text,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in self.texts]
            torch.save(self.encodings, data_file + ".emb")

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs

        return self.encodings[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

    def process(self):

      #  self.tokenizer = FlaubertTokenizer.from_pretrained(self.modelname)
        counter = 0
        num = sum(1 for line in open(self.file_path))
        data_texts=[]
        with open(self.file_path, encoding='utf-8') as file:
            for line in tqdm(file, total=num):

                    json_obj = json.loads(line)
                    # tokenizing text
                    text_raw = self.truncate_text(json_obj["document"])
                   # self.text.append()
                    self.texts.append(text_raw)

                    # getting labels and converting to int
                    label_str = json_obj["topics"][0]["label"]
                    self.labels.append(self.classes_labels[label_str.replace(" ","_").lower()])
                    counter+=1


       # self.embeddings = self.tokenizer(text_raw, padding='max_length', max_length=512, verbose=True, truncation=True)

    def truncate_text(self, text, length=512):
        content = text.split(" ")
        l = len(content)
        if l <= length:
            return text
        if self.truncate_type == "start":
            return ' '.join(content[:length])
        elif self.truncate_type == "end":
            return ' '.join(content[-length:])
        else:
            m = int(l / 2)
            return ' '.join(content[m - int(length / 2):m + int(length / 2) + 1])



    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = JTTextDataset(data_file="./data/test.json", truncate_type="middle") #/usr/src/temp/data/ina/
    print(dataset)
    print(dataset[5])
    print(dataset[5].y)