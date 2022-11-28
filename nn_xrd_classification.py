import os
import math
import json
import random
from matplotlib.streamplot import OutOfBounds
from omegaconf import OmegaConf
from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy import signal
from sklearn import model_selection, preprocessing, metrics

import torch
import torch.nn as nn
from torch.cuda import amp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
import matplotlib.pyplot as plt

from draft import intensities_to_list

import warnings
warnings.filterwarnings('ignore')


class XRDModel(nn.Module):
    def __init__(self, input_size=1550, hidden_size=128, num_layers=1, dropout=0, bidirectional=False):
        super(XRDModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0, bidirectional=bidirectional, batch_first=True)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.linear_1 = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, hidden_size // 2)
        self.output_layer = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 105) 

        # nn.init.kaiming_normal_(self.output_layer.weight.data)

    def forward(self, x):
        features, hidden = self.lstm(x)
        outputs = self.dropout(self.activation(features))
        logits = None

        logits = self.output_layer(outputs)

        outputs = nn.functional.softmax(logits, dim=-1)

        return outputs, logits


# class XRDDataset(Dataset):
#     def __init__(self, data):
#         self.data  = data 
#         self.id_values = self.data['id'].unique()

#     def __getitem__(self, idx):
#         intensity = self.data['intensity'][self.data['id'] == self.id_values[idx]].values
#         label = self.data['material'][self.data['id'] == self.id_values[idx]].values[0]

#         intensity = torch.tensor(intensity, dtype=torch.float32)
#         label = torch.tensor(label, dtype=torch.long)

#         sample = {
#             'data': intensity,
#             'label': label
#         }

#         return sample

#     def __len__(self):
#         return len(self.id_values)
    
class XRDDataset(Dataset):
    def __init__(self, data):
        self.data  = data 

    def __getitem__(self, idx):
        intensity = np.array(self.data['intensity'].iloc[idx])
        label = self.data['material'].iloc[idx]

        intensity = torch.tensor(intensity, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        sample = {
            'data': intensity,
            'label': label
        }

        return sample

    def __len__(self):
        return len(self.data)


class DataHandler():
    def __init__(self, config, path_to_data):
        super(DataHandler, self).__init__()

        self.config = config
        self.data = self.read_data(path_to_data)
        
        self.label_encoder = preprocessing.LabelEncoder()
        self.data = self.preprocess_data(self.data)

    def __call__(self, current_fold=0):
        train_data, val_data = self.split_data(self.data, current_fold)
        train_loader, val_loader = self.get_dataloaders(train_data, val_data)

        return train_loader, val_loader

    def read_data(self, path):
        data = pd.read_csv(path)

        return data

    # def create_spectr(self, mz, intensity):
    #     ''' Scale intensity values in a common range '''

    #     spec = []
    
    #     for x in range(200, 1350): # 200 1350
    #         if x in mz:
    #             spec.append(intensity[mz.index(x)])
    #         else:
    #             spec.append(0)
            
    #     return spec

    def preprocess_data(self, data):
        print('Preprocessing data...')

        # Normalize pattern in range [0, 1]
        for x in range(0, len(data) - 2250, 2250):
            intensity = data['intensity'].iloc[x:x + 2250].values
            intensity = (intensity - min(intensity)) / (max(intensity) - min(intensity))

            data['intensity'].iloc[x:x + 2250] = intensity

        # Remote single-element patterns
        data = data[data['id'].str.contains('_')].reset_index(drop=True)

        # Encode material names
        data['material'] = self.label_encoder.fit_transform(data['material'])
        np.save('material_encodings.npy', self.label_encoder.classes_)

        data = intensities_to_list(data)

        return data

    # def split_data(self, data, current_fold):
    #     kfold = model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.config.general.seed)

    #     for fold, (train_idx, val_idx) in enumerate(kfold.split(data['intensity'], data['material'], groups=data['id'])):
    #         if fold == current_fold:
    #             train_data = data.iloc[train_idx].reset_index(drop=True)
    #             val_data = data.iloc[val_idx].reset_index(drop=True)

    #             return train_data, val_data

    def split_data(self, data, current_fold):
        kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.general.seed)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(data['intensity'], data['material'])):
            if fold == current_fold:
                train_data = data.iloc[train_idx].reset_index(drop=True)
                val_data = data.iloc[val_idx].reset_index(drop=True)

                return train_data, val_data

    def get_dataloaders(self, train_data, val_data):
        train_dataset = XRDDataset(train_data)
        val_dataset = XRDDataset(val_data)

        train_loader = DataLoader(train_dataset, batch_size=self.config.data.train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.data.val_batch_size, shuffle=False)

        return train_loader, val_loader


def set_seed(seed):
    '''Set a random seed for complete reproducibility.'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def train(config, dataloader, model, loss_function, optimizer, scheduler, scaler):
    total_loss = 0.0

    model.train()

    for step, batch in enumerate(tqdm(dataloader)):
        inputs, labels = batch['data'].to(config.training.device), batch['label'].to(config.training.device)

        if config.training.mixed_precision:
            with amp.autocast():
                outputs, _ = model(inputs)
                outputs = outputs.squeeze(1)

                loss = loss_function(outputs, labels)
        else:
            outputs, _ = model(inputs)
            outputs = outputs.squeeze(1)
            # print(outputs.shape)

            loss = loss_function(outputs, labels)
        
        total_loss += loss.item()

        if config.training.mixed_precision:
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    
    scheduler.step()

    # print('Learning rate:', optimizer.param_groups[0]['lr'])

    return total_loss / len(dataloader)


def validation(config, dataloader, model, loss_function):

    total_loss = 0.0
    total_score = 0.0

    model.eval()

    for step, batch in enumerate(tqdm(dataloader)):
        inputs, labels = batch['data'].to(config.training.device), batch['label'].to(config.training.device)

        with torch.no_grad():
            outputs, logits = model(inputs)   
            outputs = outputs.squeeze(1)   

        loss = loss_function(outputs, labels)

        #print(torch.sort(outputs, descending=True))
        outputs = torch.argmax(outputs, axis=-1)
        score = metrics.f1_score(labels.cpu().detach().numpy(), outputs.cpu().detach().numpy(), average='macro')

        total_loss += loss.item()
        total_score += score

    return total_loss / len(dataloader), total_score / len(dataloader)


if __name__ == '__main__':
    config = {
        'general': {
            'seed': 121
        },
        'training': {
            'lr': 0.001, 
            'num_epochs': 150, 

            'device': 'cuda',
            'mixed_precision': False
        },
        'data': {
            'train_batch_size': 32, 
            'val_batch_size': 32
        }
    }
    config = OmegaConf.create(config)

    set_seed(config.general.seed)

    cv_scores = []

    for fold in range(5):
        data_handler = DataHandler(config=config, path_to_data='data/data.csv')
        train_loader, val_loader = data_handler(current_fold=fold)

        model = XRDModel(input_size=2250, hidden_size=128, num_layers=1, dropout=0, bidirectional=False).to(config.training.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.num_epochs // 5) 
        loss_function = nn.CrossEntropyLoss(reduction='mean')
        if config.training.mixed_precision:
            scaler = amp.GradScaler()
        else:
            scaler = None
    
        best_score = 0.0

        for epoch in range(config.training.num_epochs):
            train_loss = train(config, train_loader, model, loss_function, optimizer, scheduler, scaler)
            val_loss, val_score = validation(config, val_loader, model, loss_function)

            if best_score < val_score:
                best_score = val_score

                #torch.save(model.state_dict(), f'nn_models/fold_{fold + 1}_epoch_{epoch + 1}_{round(best_score, 4)}.pt')
                torch.save(model.state_dict(), f'model_weights/fold_{fold + 1}.pt')

            if epoch % 10 == 0:
                print(f'Epoch {epoch + 1}')
                print(f'Val loss: {val_loss}, Val F1: {round(best_score, 4)}')
                #print(f'T loss: {t_loss}')


        cv_scores.append(best_score)

    for i, score in enumerate(cv_scores):
        print(f'Fold {i + 1} F1: {score}')
    print(f'5-fold CV F1: {sum(cv_scores) / len(cv_scores)}')

    # data = next(iter(train_loader))
    # print(data['data'].shape, data['label'].shape)
    # print(data['data'][0].numpy())
    # print(data['label'])

    # plt.plot(range(1550), data['data'][0].numpy())
    # plt.show()


