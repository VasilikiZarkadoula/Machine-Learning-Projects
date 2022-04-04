"""
IRIS
"""

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import wandb



def iris_train(epoch, dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    y_pred = torch.Tensor()
    y_true = torch.Tensor()

    for batch_num, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y.view(-1,))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save y and y_pred
        y_pred_batch = torch.argmax(output.cpu(), dim=1)
        y_pred = torch.cat((y_pred, y_pred_batch))
        y_true = torch.cat((y_true, y.cpu()))

        # if batch_num % 100 == 0:
        #     loss, current = loss.item(), batch_num * len(x)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_acc = accuracy_score(y_true, y_pred)
    print(f'Epoch {epoch} Training accuracy: {train_acc}')


def iris_evaluate(epoch, dataloader, model, loss_fn, device):
    model.eval()
    y_pred = torch.Tensor()
    y_true = torch.Tensor()

    with torch.no_grad():
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y.view(-1,)).item()

            # Save y and y_pred
            y_pred_batch = torch.argmax(output.cpu(), dim=1)
            y_pred = torch.cat((y_pred, y_pred_batch))
            y_true = torch.cat((y_true, y.cpu()))

    test_accuracy = accuracy_score(y_true, y_pred)
    print(f'Epoch {epoch} Test accuracy: {test_accuracy}')


def iris_train_wandb(epoch, dataloader, model, loss_fn, optimizer, device):
    model.train()
    y_pred = torch.Tensor()
    y_true = torch.Tensor()

    for batch_num, (X, y) in enumerate(tqdm(dataloader)):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        output = model(X)
        loss = loss_fn(output, y.view(-1,))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save y and y_pred
        y_pred_batch = torch.argmax(output.cpu(), dim=1)
        y_pred = torch.cat((y_pred, y_pred_batch))
        y_true = torch.cat((y_true, y.cpu()))

        wandb.log({'epoch': epoch, 'train_loss': loss.item()})

    train_acc = accuracy_score(y_true, y_pred)
    wandb.log({'epoch': epoch, 'train_accuracy': train_acc})
    print(f'Epoch {epoch} Training accuracy: {train_acc}')


def iris_evaluate_wandb(epoch, dataloader, model, loss_fn, device):
    model.eval()
    y_pred = torch.Tensor()
    y_true = torch.Tensor()

    with torch.no_grad():
        for (X, y) in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y.view(-1,))

            wandb.log({'epoch': epoch, 'test_loss': loss.item()})

            # Save y and y_pred
            y_pred_batch = torch.argmax(output.cpu(), dim=1)
            y_pred = torch.cat((y_pred, y_pred_batch))
            y_true = torch.cat((y_true, y.cpu()))

    test_accuracy = accuracy_score(y_true, y_pred)
    wandb.log({'epoch': epoch, 'test_accuracy': test_accuracy})