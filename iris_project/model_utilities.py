"""
IRIS
"""

import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F


class IrisModel(torch.nn.Module):
    def __init__(self, input_dim, n_neurons=128, n_linear_layers=2):
        super(IrisModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, n_neurons)

        layers = []
        for i in range(n_linear_layers - 1):
            layers.extend([nn.Linear(n_neurons, n_neurons), nn.ReLU()])
        self.linear_layers = nn.Sequential(*layers)

        self.final_layer = nn.Linear(n_neurons, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.linear_layers(x)
        x = F.softmax(self.final_layer(x), dim=1)
        return x


class IrisDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(x.values).float()
        self.y = torch.Tensor(y.values).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
