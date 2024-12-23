import argparse
import os


import numpy as np
import pandas as pd
import albumentations as A
import torch

from os import path

from albumentations.pytorch import ToTensorV2
from torch.distributions.constraints import nonnegative
from torchvision.transforms.v2 import ToTensor
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.model_selection import train_test_split

from Loss import MotionNetLoss
from MotionNet import MotionNet
from SpatialTemporalNet import ResNet, BasicBlock
from datasets.UFC101 import UFC101


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

        optim.zero_grad()

        optical_flow = model(X)

        loss_2 = loss_fn(X, optical_flow[0])
        loss_3 = loss_fn(X, optical_flow[1])
        loss_4 = loss_fn(X, optical_flow[2])
        loss_5 = loss_fn(X, optical_flow[3])
        loss_6 = loss_fn(X, optical_flow[4])

        loss = 0.02 * loss_2 + 0.04 * loss_3 + loss_4 + 0.08 * loss_5 + 0.16 * loss_6

        loss.backward()
        optim.step()

        running_loss += loss.item()

    return running_loss / len(dataloader), running_accuracy / len(dataloader.dataset)


def main(args):
    UFC101_train_transform = A.Compose([
        A.Resize(224, 224),
        A.SafeRotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.2),
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.3),
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

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    model = MotionNet().to(device)

    summary(model, (33, 224, 224))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = MotionNetLoss().to(device)

    train_losses = np.zeros(args.epochs, dtype=float)
    train_accuracies = np.zeros(args.epochs, dtype=float)

    for epoch in range(args.epochs):
        train_results = train(train_loader, model, criterion, optimizer, device)

        train_losses[epoch] = train_results[0]

        print('---' * 40)
        print(f"Epoch: {epoch + 1}/{args.epochs}\n\t\t Train Loss: {train_losses[epoch]}, "
              f"Train Accuracy: {train_accuracies[epoch]}"
              )
        print('---' * 40 + "\n")

    torch.save(model.state_dict(), os.path.join(args.output, "last.pt"))

    print("Finished Training")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Hidden two-stream model')

    args.add_argument('--dir', help='Directory of data', required=True, type=str, default='data')
    args.add_argument('--epochs', help='Number of epochs', default=50, type=int)
    args.add_argument('--seed', help='Seed for random number generator', default=42, type=int)
    args.add_argument('--batch', help='Batch size', default=16, type=int)
    args.add_argument('--lr', help='Learning rate', default=1e-4, type=float)
    args.add_argument('--workers', help='Number of data loading workers', default=6, type=int)
    args.add_argument('--output', help='Output directory of model', required=True, type=str)

    main(args.parse_args())
