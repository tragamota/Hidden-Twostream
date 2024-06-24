import os

import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.io import read_video

def _pick_frame_even(video, transform):
    T, C, H, W = video.size()

    frames = torch.zeros(33, 224, 224, dtype=torch.float)
    frame_numbers = np.linspace(0, T - 1, 11, dtype=int)

    for idx, frame_number in enumerate(frame_numbers):
        frames[idx*3:idx*3+3, :, :] = transform(video[frame_number, :, :, :])

    return frames


class UFC101(Dataset):

    def __init__(self, annotations, root_dir, transform=None):
        self.items = annotations
        self.dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        video_path = os.path.join(self.dir, self.items.iloc[idx, 0])
        label = self.items.iloc[idx, 1] - 1

        video, _, _ = read_video(video_path, output_format="TCHW", pts_unit='sec')
        frames = _pick_frame_even(video, self.transform)

        return frames, label
