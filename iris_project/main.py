import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from iris_utilities import *
from model_utilities import *
import torch.utils.data as data_utils
import time
import wandb
import torch
from torch import nn
import torch.utils.data as utils
import torch.utils.data as td


def main(n_neurons=128, n_linear_layers=2):
    start = time.perf_counter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iris = load_iris()
    X = pd.DataFrame(iris.data)
    y = pd.DataFrame(iris.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Dataset
    # Create a dataset and loader for the training data and labels
    train_dataset = IrisDataset(X_train, y_train)

    # Create a dataset and loader for the test data and labels
    test_dataset = IrisDataset(X_test, y_test)

    # Dataloaders
    batch_size = 10
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model3
    learning_rate = 1e-3
    weight_decay = 1e-7
    model = IrisModel(X_train.shape[1], n_neurons, n_linear_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.NLLLoss()
    epochs = 6

    # Weights and biases logging
    config = {'n_neuros': n_neurons, 'n_linear_layers': n_linear_layers, 'weight_decay': weight_decay,
              'batch_size': batch_size, 'learning_rate': learning_rate, 'optimizer': 'Adam'}
    name = f'layers:{n_linear_layers},neurons:{n_neurons},a:{weight_decay},lr:{learning_rate},batch:{batch_size}'
    wandb.init(project='IRIS', entity="vassia_zrk", name=name, config=config, notes='', group='',reinit=True)

    for epoch in range(1, epochs + 1):
        iris_train_wandb(epoch, train_loader, model, loss_fn, optimizer, device)
        iris_evaluate_wandb(epoch, test_loader, model, loss_fn, device)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")


if __name__ == '__main__':
    n_neuron_list = [5, 25, 128, 256]
    n_layers_list = [1, 2, 3, 4, 5]
    for n_neurons, n_layers in zip(n_neuron_list, n_layers_list):
        main(n_neurons, n_layers)
