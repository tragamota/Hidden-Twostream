import argparse
import os
import sys
from os import path

import albumentations as A
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import GradScaler, autocast

from datasets.UFC101 import UFC101
from MotionNet.Loss import MotionNetLoss
from MotionNet.MotionNet import TinyMotionNet, MotionNet

from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm


def get_device():
    return (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def train_epoch(epoch_info, dataloader, model, loss_fn, optimizer, scaler, loss_params, device):
    model.train()
    running_loss = 0.0
    desc = f"Training Epoch {epoch_info[0]}/{epoch_info[1]}"

    with tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=True) as pbar:
        for X, _ in pbar:
            X = X.to(device, non_blocking=True).float() / 255.0
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device, dtype=torch.float16, enabled=True):
                optical_flow = model(X)
                loss = sum(
                    loss_fn(X, of, weight, flow_scale, border_mask, flow_mask)
                    for weight, flow_scale, border_mask, flow_mask, of in zip(
                        loss_params["ssim"],
                        loss_params["flow"],
                        loss_params["border_masks"],
                        loss_params["flow_masks"],
                        optical_flow
                    )
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

            running_loss += loss.item()

    return running_loss / len(dataloader)


def create_flow_border_mask(shape):
    mask = torch.ones(shape)

    for i in range(shape[1] // 2):
        mask[:, i * 2, :, -1] = 0
        mask[:, i * 2 + 1, -1, :] = 0

    return mask


def create_border_mask(shape, border_ratio=0.1):
    mask = torch.ones(shape)

    border_width = round(shape[-1] * border_ratio)

    mask[:, :, :border_width, :] = 0  # Top border
    mask[:, :, -border_width:, :] = 0  # Bottom border
    mask[:, :, :, :border_width] = 0  # Left border
    mask[:, :, :, -border_width:] = 0  # Right border

    return mask


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

    os.makedirs(args.output, exist_ok=True)
    device = get_device()

    df = pd.read_csv(path.join(args.dir, 'train.csv'))

    train_df, validation_df = train_test_split(df, test_size=0.2, shuffle=True)

    train_data = UFC101(train_df, args.dir, transform=UFC101_train_transform)
    validation_data = UFC101(validation_df, args.dir, transform=UFC101_val_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation_data, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    if args.tiny:
        model = TinyMotionNet()

        ssim_scales = [0.01, 0.02, 0.04]
        flow_scales = [10, 5, 2.5]
        border_masks = [create_border_mask((1, 1, size, size)).to(device)
                        for size in [112, 56, 28]]
        flow_border_mask = [
            create_flow_border_mask((1, 20, size, size)).to(device)
            for size in [112, 56, 28]
        ]

    else:
        model = MotionNet()

        ssim_scales = [0.01, 0.02, 0.04, 0.08, 0.16]
        flow_scales = [10, 5, 2.5, 1.25, 0.625]
        border_masks = [
            create_border_mask((1, 1, size, size)).to(device)
            for size in [112, 56, 28, 14, 7]
        ]
        flow_border_mask = [
            create_flow_border_mask((1, 20, size, size)).to(device)
            for size in [112, 56, 28, 14, 7]
        ]

    loss_params = {
        "ssim": ssim_scales,
        "flow": flow_scales,
        "border_masks": border_masks,
        "flow_masks": flow_border_mask
    }

    model = model.to(device)
    summary(model, (33, 224, 224))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=True)
    loss_fn = MotionNetLoss().to(device)

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            (epoch + 1, args.epochs), train_loader, model, loss_fn, optimizer, scaler, loss_params, device
        )
        print(f"Epoch {epoch + 1}/{args.epochs} - Train Loss: {train_loss:.4f}", flush=True)

        # Save model state every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = path.join(args.output, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    torch.save(model.state_dict(), path.join(args.output, "last.pt"))
    print(f"Training complete. Model saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MotionNet for unsupervised learning')
    parser.add_argument('--dir', help='Directory of data', required=True, type=str)
    parser.add_argument('--epochs', help='Number of epochs', default=50, type=int)
    parser.add_argument('--seed', help='Seed for random number generator', default=42, type=int)
    parser.add_argument('--batch', help='Batch size', default=16, type=int)
    parser.add_argument('--lr', help='Learning rate', default=3.2e-5, type=float)
    parser.add_argument('--workers', help='Number of data loading workers', default=6, type=int)
    parser.add_argument("--tiny", action='store_true', help="Use tiny motionet", default=False)
    parser.add_argument("--weights", required=True, help="Weights to pretrained motionnet")
    parser.add_argument('--output', help='Output directory of model', required=True, type=str)

    main(parser.parse_args())
