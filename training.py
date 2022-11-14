import os
import math
import omegaconf
from tqdm import tqdm
from omegaconf import OmegaConf

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold

import torch
import torch.nn as nn
from torch import dropout, optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from pytorch_metric_learning import losses as L
from pytorch_metric_learning import miners

import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


class XRDDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.materials_ids_list = data['id'].unique()

    def __getitem__(self, idx):
        material_id = self.materials_ids_list[idx]
        sample = self.data[['intensity']][self.data['id'] == material_id].values
        # label = self.data[['first_material', 'second_material']][self.data['id'] == material_id].values
        label = self.data['material'][self.data['id'] == material_id].values[0]

        # print(material_id)
        # print(self.data['material'][self.data['id'] == material_id].values[0])
        # print(sample)

        sample = torch.tensor(sample, dtype=torch.float32).transpose(1, 0)
        #label = torch.tensor(label, dtype=torch.float32).transpose(1, 0)
        label = torch.tensor(label, dtype=torch.long)

        return {
            'sample': sample,
            'label': label
        }

    def __len__(self):
        return len(self.materials_ids_list)


class DataPreprocesser():
    def __init__(self, config):
        self.config = config
        self.data = pd.read_csv(config.paths.path_to_csv)

    def __call__(self):
        self.data = self.preprocess_data(self.data)
        train_data, train_targets, val_data, val_targets = self.generate_data(self.data)

        return train_data, train_targets, val_data, val_targets

    def generate_data(self, data):
        kfold = GroupKFold(n_splits=5)
        current_fold = 0
        target_column = '2theta'

        for fold, (train_idx, val_idx) in enumerate(kfold.split(data, data[target_column], groups=data['material'])):
            if fold == current_fold:
                train_data = data.iloc[train_idx].reset_index(drop=True)
                train_targets = data[target_column].iloc[train_idx].values
                val_data= data.iloc[val_idx].reset_index(drop=True)
                val_targets = data[target_column].iloc[val_idx].values

                break

        return train_data, train_targets, val_data, val_targets

    def preprocess_data(self, data):
        # Drop single-element phases
        data = data[data['material'].str.contains('_')]

        # Encode phases
        encodings = dict(list(enumerate(data['material'].unique())))
        data['material'] = data['material'].apply(lambda x: list(encodings.keys())[list(encodings.values()).index(x)])
        # data['first_material'] = data['material'].apply(lambda x: x.split('_')[0])
        # data['second_material'] = data['material'].apply(lambda x: x.split('_')[1])
        # materials = np.unique([*data['first_material'].unique(), *data['second_material'].unique()])

        # encodings = dict(list(enumerate(materials)))

        # data['first_material'] = data['first_material'].apply(lambda x: list(encodings.keys())[list(encodings.values()).index(x)])
        # data['second_material'] = data['second_material'].apply(lambda x: list(encodings.keys())[list(encodings.values()).index(x)])

        return data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)

        return x


# class XRDModel(nn.Module):
#     def __init__(self):
#         super(XRDModel, self).__init__()

#         # self.transformer = nn.Transformer(d_model=2250, nhead=10, batch_first=True)
#         self.embedding = nn.Embedding(1, 2250)
#         self.pos_encoding = PositionalEncoding(d_model=2250, dropout=0.5)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=2250, nhead=10, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

#     def forward(self, x, y):
#         # x = self.transformer(x, y)
#         # x = self.embedding(x)
#         x = self.pos_encoding(x)
#         x = self.transformer_encoder(x)

#         x = x.squeeze(1)

#         return x


class XRDModel(nn.Module):
    def __init__(self):
        super(XRDModel, self).__init__()

        # self.transformer = nn.Transformer(d_model=2250, nhead=10, batch_first=True)
        self.embedding = nn.Embedding(1, 2250)
        self.lstm = nn.LSTM(input_size=2250, hidden_size=256, num_layers=1, dropout=0, bidirectional=False, batch_first=True)

    def forward(self, x, y):
        # x = self.transformer(x, y)
        # x = self.embedding(x)
        # x = self.pos_encoding(x)
        # x = self.transformer_encoder(x)
        x, h = self.lstm(x)

        x = x.squeeze(1)

        return x


class XRDLoss():
    def __init__(self):
        # self.contrastive_loss = L.ContrastiveLoss(pos_margin=0, neg_margin=1)
        self.contrastive_loss = L.TripletMarginLoss()
        #self.contrastive_loss = L.CosFaceLoss(margin=0.3, num_classes=15, embedding_size=2250, scale=64)

        # self.miner = BatchEasyHardMiner(pos_strategy='easy', neg_strategy='easy')
        self.miner = miners.PairMarginMiner(pos_margin=0.5, neg_margin=0.5)
        ### Metric learning loss for embeddings ###

    def __call__(self, outputs, labels):
        print(outputs.shape)
        print(labels)
        # loss = self.cross_entropy(outputs, labels)
        #pairs = self.miner(outputs, labels)
        #print(pairs)
        loss = self.contrastive_loss(outputs, labels)

        return loss


def train(config, dataloader, loss_function, model, optimizer, scheduler, scaler):
    total_loss = 0.0

    model.train()

    for step, batch in enumerate(tqdm(dataloader)):
        inputs, labels = batch['sample'].to(config.training.device), batch['label'].to(config.training.device)

        if config.training.mixed_precision:
            with amp.autocast():
                outputs = model(inputs, labels)

                loss = loss_function(outputs, labels)
        else:
            outputs = model(inputs, labels)
            # print(outputs.shape, labels.shape)
            # sample = outputs[0].cpu().detach().numpy()
            # plt.plot(range(256), sample)
            # plt.show()

            loss = loss_function(outputs, labels)
        
        print(loss.item())

        total_loss += loss.item()

        if config.training.mixed_precision:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
    print('Learning rate:', optimizer.param_groups[0]['lr'])

    return total_loss / len(dataloader)


def validate():
    pass


def run(config):
    data_preprocessor = DataPreprocesser(config)
    train_data, _, val_data, _ = data_preprocessor()
    print(train_data)
    print(train_data.shape, val_data.shape)

    train_dataset = XRDDataset(train_data)
    val_dataset = XRDDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=config.data.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.data.val_batch_size, shuffle=False)

    model = XRDModel().to(config.training.device)
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    loss_function = XRDLoss()
    if config.training.mixed_precision:
        scaler = amp.GradScaler()
    else:
        scaler = None

    train_loss = train(config, train_loader, loss_function, model, optimizer, scheduler, scaler)
    print(train_loss)



if __name__ == '__main__':
    config = {
        'general': {
            'seed': 121
        },
        'paths': {
            'path_to_csv': 'data.csv'
        },
        'data': {
            'train_batch_size': 8,
            'val_batch_size': 8,
        },
        'training': {
            'num_epochs': 10,
            'lr': 0.001,
            'device': 'cuda',

            'mixed_precision': False
        }
    }

    config = OmegaConf.create(config)

    run(config)
    # data = pd.read_csv('data.csv')
    # print(data[data['id'] == '4105680_1534878_2'])



