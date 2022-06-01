import json
import os.path as osp
from collections import Counter

import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
import torch_geometric.transforms as T

from JTDataset import JTDataset
from JTDatasetJson import JTDatasetJson
import numpy as np

import os

from images.dataset import WavImageDataset

os.environ['http_proxy'] = "http://firewall.ina.fr:81"
os.environ['HTTP_PROXY'] = "http://firewall.ina.fr:81"
os.environ['https_proxy'] = "http://firewall.ina.fr:81"
os.environ['HTTPS_PROXY'] = "http://firewall.ina.fr:81"

def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)

def calculate_multilabel_weights(y):
    print(y.shape)
    print(y)
    all_sum = torch.sum(y, 1)
    print(all_sum, )
    return all_sum / y.shape[0]


def get_dataloaders(data_dir, transforms, batch_size):
    train_data_set = WavImageDataset(data_dir=data_dir, csv_file="data/test/train.csv", transforms=transforms)
    test_data_set = WavImageDataset(data_dir=data_dir, csv_file="data/test/test.csv", transforms=transforms)
    val_data_set = WavImageDataset(data_dir=data_dir, csv_file="data/test/val.csv", transforms=transforms)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    val_laoder = DataLoader(val_data_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_laoder, test_loader


# def get_dataloaders(dataset, split_idx, batch_size):
#
#     train_dataset = dataset[split_idx["train"]]
#     test_dataset = dataset[split_idx["test"]]
#     val_dataset = dataset[split_idx["valid"]]
#
#     train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
#
#     return train_loader, val_loader, test_loader


def get_split_idx_from_file(path):
    splits =[]
    for p in path:
        with open(p) as json_file:
         splits.append(json.load(json_file))
    return splits

def get_split_idx_kfold(dataset, folds):

    splits = []
    # this part for test
    ids = [i for i in range(len(dataset))]
    y = dataset.data.y.cpu().detach().numpy()

    rs = StratifiedShuffleSplit(n_splits=folds, test_size=.2, random_state=123456)  # replace with StratifiedShuffleSplit
    val_rs = StratifiedShuffleSplit(n_splits=1, test_size=.5)

    for train_index, test_index in rs.split(ids, y):
        split_idx = {}
        split_idx["train"], split_idx["test"] = train_index, test_index
        test_to_split = split_idx["test"]
        for valid_index, test_index in val_rs.split(test_to_split, y[test_to_split]):
            split_idx["valid"], split_idx["test"] = test_to_split[valid_index], test_to_split[test_index]

        print(y[split_idx["train"]])

        c = Counter(y[split_idx["train"]])
        print( "train", c)
        c = Counter(y[split_idx["valid"]])
        print("valid", c)
        c = Counter(y[split_idx["test"]])
        print("test", c)

        split_idx["train"] = torch.from_numpy(np.array(split_idx["train"], dtype=np.int64))
        split_idx["test"] = torch.from_numpy(np.array(split_idx["test"], dtype=np.int64))
        split_idx["valid"] = torch.from_numpy(np.array(split_idx["valid"], dtype=np.int64))
        print("train: ", split_idx["train"].size(), " val: ", split_idx["valid"].size(), " test:", split_idx["test"].size() )
        splits.append(split_idx)

    return splits

def dict2file(dictionary, embeddings, meta_out_file):
    fw = open(meta_out_file, 'w', encoding='utf8')
    for i in range(embeddings.shape[0]):
        fw.write(dictionary[i] + "\n")
    fw.close()


def embedding2file(embeddings, embeddings_out_file):
    print("Embedding:", embeddings.shape)
    fw = open(embeddings_out_file, 'w', encoding='utf8')
    for i in range(embeddings.shape[0]):
        line = ''
        for j in range(embeddings.shape[1]):
            line = line + str(embeddings[i, j]) + '\t'
        fw.write(line.strip() + '\n')
    fw.close()



class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data

def logger(info):
    epoch =  info['epoch']
    train_loss, val_loss, test_acc = info['train_loss'], info['val_loss'], info['test_acc']
    print('{:03d}:Train Loss: {:.4f}, Val Loss: {:.4f}, Test Accuracy: {:.3f}'.format(
         epoch, train_loss, val_loss, test_acc))




def get_dataset_local(name, task=None, sparse=True, cleaned=False):
    if name == "JT":
        dataset = JTDataset(root='/usr/src/temp/data/ina-clean/')
    elif name == "JT-json":
        dataset = JTDatasetJson(root='/usr/src/temp/data/ina-json/')

    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    # dataset.data.edge_attr = None
    print(dataset.data.x.dtype,dataset.data.edge_index.dtype)
    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset.copy(torch.tensor(indices))

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset

