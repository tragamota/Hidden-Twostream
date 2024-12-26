import argparse
import os
import sys
from os import path

import albumentations as A
import pandas as pd
import torch
from torch import GradScaler, autocast

sys.path.append(os.path.abspath(os.getcwd()))

from albumentations.pytorch import ToTensorV2
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


def train_epoch(epoch_info, dataloader, model, loss_fn, optimizer, scaler, border_masks, device):
    model.train()
    running_loss = 0.0
    desc = f"Training Epoch {epoch_info[0]}/{epoch_info[1]}"

    with tqdm(dataloader, desc=desc, dynamic_ncols=True, leave=True) as pbar:
        for X, _ in pbar:
            X = X.to(device, non_blocking=True).float() / 255.0
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device, dtype=torch.bfloat16, enabled=True):
                optical_flow = model(X)
                loss = sum(
                    loss_fn(X, of, weight, flow_scale, border_mask)
                    for weight, flow_scale, border_mask, of in zip(
                        [0.01, 0.02, 0.04, 0.08, 0.16],
                        [0.625, 1.25, 2.5, 5, 10],
                        border_masks,
                        optical_flow
                    )
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

            running_loss += loss.item()

    return running_loss / len(dataloader)


def prepare_transforms():
    return A.Compose([
        A.Resize(224, 224),
        ToTensorV2()
    ])


def create_border_mask(shape):
    mask = torch.ones(shape)

    mask[0, :] = 0  # Top border
    mask[-1, :] = 0  # Bottom border
    mask[:, 0] = 0  # Left border
    mask[:, -1] = 0  # Right border

    return mask


def main(args):
    os.makedirs(args.output, exist_ok=True)
    device = get_device()

    df = pd.read_csv(path.join(args.dir, 'train.csv'))
    train_data = UFC101(df, args.dir, transform=prepare_transforms())
    train_loader = DataLoader(
        train_data, batch_size=args.batch, shuffle=True, pin_memory=True,
        pin_memory_device=device, num_workers=args.workers
    )

    border_masks = [
                       create_border_mask((1, 1, size, size)).to(device)
                       for size in [7, 14, 28, 56, 112]
                   ][::-1]

    model = MotionNet().to(device)
    summary(model, (33, 224, 224))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=True)
    loss_fn = MotionNetLoss().to(device)

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            (epoch + 1, args.epochs), train_loader, model, loss_fn, optimizer, scaler, border_masks, device
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
    parser.add_argument('--output', help='Output directory of model', required=True, type=str)

    main(parser.parse_args())
