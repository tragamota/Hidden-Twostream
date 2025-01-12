import argparse
import os
from os import path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from Resnet import ResNet, BasicBlock
from TemporalNet import TemporalNet
from datasets.UFC101 import UFC101

def build_model(model_name):

    if model_name == "SpatialNet":
        return ResNet(BasicBlock, input_channels=33, layers=[2, 2, 2, 2], num_classes=101, use_classifier=True)
    if model_name == "TemporalNet":
        return TemporalNet('./checkpoints/motionNet/last.pt', BasicBlock, layers=[2, 2, 2, 2], num_classes=101)
    if model_name == "TwoStreamNet":
        return None

def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    return device


def train(dataloader, model, loss_fn, optim, device):
    running_loss = 0.0
    running_accuracy = 0.0

    model.train()

    for X, y in tqdm(dataloader):
        X = X.to(device, non_blocking=True).float()
        X = X / 255
        y = y.to(device, non_blocking=True)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optim.step()
        optim.zero_grad()

        prob = nn.Softmax(dim=1)(pred)

        running_accuracy += (prob.argmax(1) == y).sum().item()
        running_loss += loss.item()

    return running_loss / len(dataloader), running_accuracy / len(dataloader.dataset)


def validate(dataloader, model, loss_fn, device):
    running_loss = 0.0
    running_accuracy = 0.0

    model.eval()

    with torch.no_grad():
        for X, y in tqdm(dataloader):
            X = X.to(device, non_blocking=True).float()
            X = X / 255
            y = y.to(device, non_blocking=True)

            pred = model(X)

            prob = nn.Softmax(dim=1)(pred)

            running_accuracy += (prob.argmax(1) == y).sum().item()
            running_loss += loss_fn(pred, y).item()

    return running_loss / len(dataloader), running_accuracy / len(dataloader.dataset)


def main(args):
    UFC101_train_transform = A.Compose([
        A.Resize(224, 224),
        A.SafeRotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.2),
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.3),
        ToTensorV2()
    ])

    UFC101_val_transform = A.Compose([
        A.Resize(224, 224),
        ToTensorV2()
    ])

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    device = get_device()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(path.join(args.dir, 'train.csv'))

    train_df, validation_df = train_test_split(df, test_size=0.2, shuffle=True)

    train_data = UFC101(train_df, args.dir, transform=UFC101_train_transform)
    validation_data = UFC101(validation_df, args.dir, transform=UFC101_val_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation_data, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_model(args.modal)
    model.to(device)

    summary(model, input_size=(33, 224, 224))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    train_losses = np.zeros(args.epochs, dtype=float)
    train_accuracies = np.zeros(args.epochs, dtype=float)

    eval_losses = np.zeros(args.epochs, dtype=float)
    eval_accuracies = np.zeros(args.epochs, dtype=float)

    best_val_acc = -np.inf

    for epoch in range(args.epochs):
        train_results = train(train_loader, model, criterion, optimizer, device)
        eval_results = validate(validation_loader, model, criterion, device)

        train_losses[epoch] = train_results[0]
        train_accuracies[epoch] = train_results[1]

        eval_losses[epoch] = eval_results[0]
        eval_accuracies[epoch] = eval_results[1]

        print('---' * 40)
        print(f"Epoch: {epoch + 1}/{args.epochs}\n\t\t Train Loss: {train_losses[epoch]}, "
              f"Train Accuracy: {train_accuracies[epoch]}, "
              )

        print(f"\n\t\t Validation Loss : {eval_losses[epoch]}, Validation Accuracy: {eval_accuracies[epoch]}")
        print('---' * 40 + "\n")

        if eval_results[1] > best_val_acc:
            best_val_acc = eval_results[1]

            torch.save(model.state_dict(), os.path.join(args.output, "best.pt"))

    torch.save(model.state_dict(), os.path.join(args.output, "last.pt"))
    print("Finished Training")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Hidden two-stream model')

    args.add_argument('--dir', help='Directory of data', required=True, type=str, default='data')
    args.add_argument('--epochs', help='Number of epochs', default=50, type=int)
    args.add_argument('--seed', help='Seed for random number generator', default=42, type=int)
    args.add_argument('--batch', help='Batch size', default=16, type=int)
    args.add_argument('--lr', help='Learning rate', default=1e-4, type=float)
    args.add_argument('--workers', help='Number of data loading workers', default=4, type=int)
    args.add_argument('--momentum', help='Momentum', default=0.9, type=float)
    args.add_argument('--decay', help='Decay', default=0.999, type=float)
    args.add_argument('--output', help='Output directory of model', required=True, type=str)
    args.add_argument('--modal', help='Model to train', choices=['SpatialNet', 'TwoStreamNet'])

    main(args.parse_args())
