import json
import os

import numpy as np
import torch
import datasets

import pandas as pd
import torchaudio
from datasets import Features, Audio
from tqdm import tqdm


from transformers import AutoFeatureExtractor, Wav2Vec2Processor


class HGAudioDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_file,
                 audio_dir,processor,max_duration):
        self.data_file = data_file
       # self.annotations = pd.read_csv(annotations_file, header=None, sep="\t")
        self.audio_dir = audio_dir
     #   f = datasets.Dataset.from_json(data_file)
        self.transforms = None
        self.max_duration = max_duration
        self._data =[]
        self._format_type = None
        self.truncate_type = "middle"
        self.classes_labels = {'santé': 0, 'culture-loisirs': 1, 'société': 2, 'sciences_et_techniques': 3, 'economie': 4,
                           'environnement': 5,
                           'politique_france': 6, 'sport': 7, 'histoire-hommages': 8, 'justice': 9, 'faits_divers': 10,
                           'education': 11,
                           'catastrophes': 12, 'international': 13}

        self.target_sampling_rate = processor.sampling_rate
        self.processor = processor

        print(data_file + ".emb", os.path.exists( data_file + ".emb"))
        if os.path.exists( data_file + ".emb"):
            self._data = torch.load(data_file + ".emb")
        else:
            self.process()
            torch.save(self._data, data_file + ".emb")



    def __len__(self):
        return len(self._data)


    def process(self):

        num = sum(1 for line in open(self.data_file))

        with open(self.data_file, encoding='utf-8') as file:
            for line in tqdm(file, total=num):
                try:
                    json_obj = json.loads(line)
                # getting labels and converting to int
                    label_str = json_obj["topics"][0]["label"]
                    label = self.classes_labels[label_str.replace(" ", "_").lower()]

                    id = str(json_obj["id"])
                    audio_sample_path = self.audio_dir + str(label) + "/" + id + ".wav"
                    signal, sr = torchaudio.load(audio_sample_path)
                    signal = self.cut_audio(signal, self.truncate_type, sr * 20)
                    signal = self.resample_if_necessary(signal = signal, sr = sr, target_sample_rate= self.target_sampling_rate)
                    signal = self.mix_down_if_necessary(signal)
                    signal = self.right_pad_if_necessary(signal,self.target_sampling_rate*20)
                    tmp ={}
                    tmp["label"] = label
                    print(signal.shape)
                    input = self.processor(
                        signal,
                        sampling_rate=self.target_sampling_rate,
                        max_length=int(self.target_sampling_rate * 20),
                        truncation=True
                    )
                    tmp["input_values"] = input.input_values

                    self._data.append(tmp)

                except Exception as e:
                    print(e)


    def resample_if_necessary(self, signal, sr,target_sample_rate):
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            signal = resampler(signal).squeeze(1)
        return signal


    def __getitem__(self, idx):
       return self._data[idx]


    def cut_audio(self, signal, truncate_type, length):
        l = signal.shape[1]
        if l <= length:
            return signal
        if truncate_type == "start":
            return signal[:, :length]
        elif truncate_type == "end":
            return signal[:, -length:]
        else:
            m = int(l / 2)

            return signal[:, m - int(length / 2):m + int(length / 2) + 1]

    def right_pad_if_necessary(self, signal,num_samples):
        length_signal = signal.shape[1]
        if length_signal < num_samples:
            num_missing_samples = num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


    def mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
           signal = torch.mean(signal, dim=0, keepdim=True)
        return signal


#
#
# class JTAudioDataset(torch.utils.data Dataset):
#
#     def __init__(self,
#                  data_file,
#                  audio_dir,
#                  transformation,
#                  target_sample_rate,
#                  num_samples,
#                  device):
#         self.data_file = data_file
#        # self.annotations = pd.read_csv(annotations_file, header=None, sep="\t")
#         self.audio_dir = audio_dir
#         self.device = device
#         self.transformation = transformation
#         self.target_sample_rate = target_sample_rate
#         self.num_samples = num_samples
#         self.truncate_type = "middle"
#         self.labels = []
#         self.embeddings = []
#
#         self.classes_labels = {'santé': 0, 'culture-loisirs': 1, 'société': 2, 'sciences_et_techniques': 3, 'economie': 4,
#                            'environnement': 5,
#                            'politique_france': 6, 'sport': 7, 'histoire-hommages': 8, 'justice': 9, 'faits_divers': 10,
#                            'education': 11,
#                            'catastrophes': 12, 'international': 13}
#         print(data_file + "data.emb", os.path.exists(data_file + "data.emb"))
#         if os.path.exists(data_file + "data.emb"):
#             self.embeddings = torch.load(data_file + "data.emb")
#             self.labels =  torch.load(data_file + "label.emb")
#         else:
#             self.process()
#             torch.save(self.embeddings, data_file + "data.emb")
#             torch.save(self.labels, data_file + "label.emb")
#
#
#
#
#     def __len__(self):
#         return len(self.labels)
#
#     def process(self):
#
#         num = sum(1 for line in open(self.data_file))
#
#         with open(self.data_file, encoding='utf-8') as file:
#             for line in tqdm(file, total=num):
#                 try:
#                     json_obj = json.loads(line)
#                 # getting labels and converting to int
#                     label_str = json_obj["topics"][0]["label"]
#                     label = self.classes_labels[label_str.replace(" ", "_").lower()]
#                     self.labels.append(label)
#
#                 # traiting audio
#
#                     id = str(json_obj["id"])
#                     self.embeddings.append(self.process_signal(self.audio_dir + str(label) + "/" + id + ".wav"))
#
#                 except Exception as ex:
#                     print(ex)
#                     print(id)
#
#     def process_signal(self, audio_sample_path):
#         signal, sr = torchaudio.load(audio_sample_path)
#         signal = self.cut_audio(signal, self.truncate_type, self.num_samples)
#         signal = self._resample_if_necessary(signal, sr)
#         signal = self._mix_down_if_necessary(signal)
#         signal = self._right_pad_if_necessary(signal)
#         return self.transformation(signal)
#
#     def __getitem__(self, idx):
#
#        return {"input_values" : self.embeddings[idx], "label" : self.labels[idx]}
#
#
#
#
#     def cut_audio(self,signal, truncate_type, length):
#         l = signal.shape[1]
#         if l <= length:
#             return signal
#         if truncate_type == "start":
#             return signal[:, :length]
#         elif truncate_type == "end":
#             return signal[:, -length:]
#         else:
#             m = int(l / 2)
#
#             return signal[:, m - int(length / 2):m + int(length / 2) + 1]
#
#     def _right_pad_if_necessary(self, signal):
#         length_signal = signal.shape[1]
#         if length_signal < self.num_samples:
#             num_missing_samples = self.num_samples - length_signal
#             last_dim_padding = (0, num_missing_samples)
#             signal = torch.nn.functional.pad(signal, last_dim_padding)
#         return signal
#
#     def _resample_if_necessary(self, signal, sr):
#         if sr != self.target_sample_rate:
#             resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
#             signal = resampler(signal)
#         return signal
#
#     def _mix_down_if_necessary(self, signal):
#         if signal.shape[0] > 1:
#             signal = torch.mean(signal, dim=0, keepdim=True)
#         return signal
#
#
#
#
# from IPython.display import Audio, display
#
#
#
# if __name__ == "__main__":
#     ANNOTATIONS_FILE = "./data/val_short.json"
#     AUDIO_DIR = "./data/test/"
#     SAMPLE_RATE = 22050 * 2
#     NUM_SAMPLES = 22050 * 20
# #10 seconds
#     if torch.cuda.is_available():
#         device = "cuda"
#     else:
#         device = "cpu"
#     print(f"Using device {device}")
#
#     mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#         sample_rate=SAMPLE_RATE,
#         n_fft=1024,
#         hop_length=512,
#         n_mels=64
#     )
#
#     usd = JTAudioDataset(ANNOTATIONS_FILE,
#                          AUDIO_DIR,
#                          mel_spectrogram,
#                          SAMPLE_RATE,
#                          NUM_SAMPLES,
#                          device)
#     print(f"There are {len(usd)} samples in the dataset.")
#     for i in range(0,2):
#         signal = usd[i]
#         print(signal["input_values"].shape,  signal["label"])
#   #  play_audio(signal, SAMPLE_RATE)
#
