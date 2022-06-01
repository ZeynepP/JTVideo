import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
import pandas as pd
from torchvision.datasets import VisionDataset


class WavImageDataset(VisionDataset):

    def __init__(self, data_dir, csv_file, transforms=None):
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file, sep="\t",  names=["img_path", "label"])
        self.transforms = transforms
        self.length = len(self.data)
        self.num_classes =self.data['label'].nunique()


    def __getitem__(self, index):
        img_path, label = list(self.data.iloc[index].values)
        image = Image.open(self.data_dir + img_path.strip())
        image = image.convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        return image,  label

    def __len__(self):
        return self.length
