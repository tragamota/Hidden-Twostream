import argparse
import os
import sys
from os import path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

sys.path.append(os.path.abspath(os.getcwd()))

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


def train_epoch(epoch_info, dataloader, model, loss_fn, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0

    desc = f"Training Epoch {epoch_info[0]}/{epoch_info[1]}"

    pbar = tqdm(dataloader, desc=desc, leave=True, dynamic_ncols=True)

    for X, _ in pbar:
        X = X.to(device, non_blocking=True).float() / 255.0

        optimizer.zero_grad()

        optical_flow = model(X)

        loss = [loss_fn(X, of, w) for w, of in zip([0.01, 0.02, 0.04, 0.08, 0.16], optical_flow)]
        loss = sum(loss)

        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        running_loss += loss.item()

    # pbar.set_postfix({"Loss": running_loss / len(dataloader)})
    # pbar.update()

    return running_loss / len(dataloader)


def prepare_transforms():
    return A.Compose([
        A.Resize(224, 224),
        # A.SafeRotate(limit=20, p=0.3),
        # A.HorizontalFlip(p=0.3),
        # A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.2),
        # A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.3),
        ToTensorV2()
    ])


def main(args):
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    device = get_device()

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    df = pd.read_csv(path.join(args.dir, 'train.csv'))

    train_data = UFC101(df, args.dir, transform=prepare_transforms())
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    model = MotionNet().to(device)
    summary(model, (33, 224, 224))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    loss_fn = MotionNetLoss().to(device)

    train_losses = []

    for epoch in range(args.epochs):
        train_loss = train_epoch((epoch + 1, args.epochs), train_loader, model, loss_fn, optimizer, None, device)
        train_losses.append(train_loss)

        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}", flush=True)

    torch.save(model.state_dict(), path.join(args.output, "last.pt"))

    print("Training complete. Model saved to", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MotionNet for unsupervised learning')
    parser.add_argument('--dir', help='Directory of data', required=True, type=str)
    parser.add_argument('--epochs', help='Number of epochs', default=50, type=int)
    parser.add_argument('--seed', help='Seed for random number generator', default=42, type=int)
    parser.add_argument('--batch', help='Batch size', default=16, type=int)
    parser.add_argument('--lr', help='Learning rate', default=3.2e-5, type=float)
    parser.add_argument('--workers', help='Number of data loading workers', default=6, type=int)
    parser.add_argument('--output', help='Output directory of model', required=True, type=str)

    main(parser.parse_args())
