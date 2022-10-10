import os
import omegaconf
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from omegaconf import OmegaConf
from xgboost import train


class XRDDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        sample = self.data[['2theta', 'intensity']].iloc[idx]
        label = self.data['material'].iloc[idx] ### ENCODE TARGET

        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return {
            'sample': sample,
            'label': label
        }

    def __len__(self):
        return len(self.data)


def data_generator(config):
    data = pd.read_csv(config.paths.path_to_csv)

    kfold = KFold(n_splits=5, shuffle=True, random_state=config.general.seed)
    current_fold = 0
    target_column = 'material'

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data, data[target_column])):
        if fold == current_fold:
            train_data = data.iloc[train_idx]
            train_targets = data[target_column].iloc[train_idx].values
            val_data= data.iloc[val_idx]
            val_targets = data[target_column].iloc[val_idx].values

            break

    return train_data, train_targets, val_data, val_targets


class XRDModel(nn.Module):
    def __init__(self):
        super(XRDModel, self).__init__()

        self.transformer = nn.Transformer()

    def forward(self, x):
        x = self.transformer(x)

        return x


class XRDLoss():
    def __init__(self):
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, outputs, labels):
        loss = self.cross_entropy(outputs, labels)

        return loss


def train(config, dataloader, loss_function, model, optimizer, scheduler, scaler):
    total_loss = 0.0

    model.train()

    for step, batch in enumerate(tqdm(dataloader)):
        inputs, labels = batch['sample'].to(config.training.device), batch['label'].to(config.training.device)

        if config.training.mixed_precision:
            with amp.autocast():
                outputs = model(inputs)

                loss = loss_function(outputs, labels)


def validate():
    pass


def run(config):
    train_data, train_targets, val_data, val_targets = data_generator(config)
    print(train_data.shape, val_data.shape)

    train_dataset = XRDDataset(train_data)
    val_dataset = XRDDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = XRDModel().to(config.training.device)
    optimizer = optim.Adam(model.parameters(), lr=config.training.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    loss_function = XRDLoss()
    if config.training.mixed_precision:
        scaler = amp.GradScaler()
    else:
        scaler = None

    train(config, train_loader, loss_function, model, optimizer, scheduler, scaler)



if __name__ == '__main__':
    config = {
        'general': {
            'seed': 121
        },
        'paths': {
            'path_to_csv': 'data.csv'
        },
        'training': {
            'num_epochs': 10,
            'lr': 0.001,
            'device': 'cuda',

            'mixed_precision': True
        }
    }

    config = OmegaConf.create(config)

    run(config)



