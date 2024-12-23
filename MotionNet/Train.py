import argparse
import os
from os import path

import albumentations as A
import numpy as np
import pandas as pd
import torch

from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from Loss import MotionNetLoss
from MotionNet import MotionNet
from datasets.UFC101 import UFC101


def get_device():
    return (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def train_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()
    running_loss = 0.0

    with tqdm(dataloader, desc="Training", leave=True) as pbar:
        for X, _ in pbar:
            X = X.to(device, non_blocking=True).float() / 255.0

            optimizer.zero_grad()

            optical_flow = model(X)

            loss = [w * loss_fn(X, of) for w, of in zip([0.01, 0.02, 0.04, 0.08, 0.16], optical_flow)]
            loss = sum(loss)

            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

            running_loss += loss.item()

    return running_loss / len(dataloader)


def prepare_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.SafeRotate(limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.2),
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.3),
        ToTensorV2()
    ])


def main(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = get_device()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    df = pd.read_csv(path.join(args.dir, 'train.csv'))
    train_df, _ = train_test_split(df, test_size=0.2, shuffle=True, random_state=args.seed)

    train_data = UFC101(train_df, args.dir, transform=prepare_transforms())
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    model = MotionNet().to(device)
    summary(model, (33, 224, 224))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = MotionNetLoss().to(device)

    train_losses = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(train_loader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)

        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), path.join(args.output, "last.pt"))
    print("Training complete. Model saved to", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MotionNet for unsupervised learning')
    parser.add_argument('--dir', help='Directory of data', required=True, type=str)
    parser.add_argument('--epochs', help='Number of epochs', default=50, type=int)
    parser.add_argument('--seed', help='Seed for random number generator', default=42, type=int)
    parser.add_argument('--batch', help='Batch size', default=16, type=int)
    parser.add_argument('--lr', help='Learning rate', default=1e-4, type=float)
    parser.add_argument('--workers', help='Number of data loading workers', default=6, type=int)
    parser.add_argument('--output', help='Output directory of model', required=True, type=str)

    main(parser.parse_args())
