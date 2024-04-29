import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video

def _pick_frame_even(video):
    T, C, H, W = video.size()

    frames = torch.zeros(33, H, W, dtype=torch.float)
    frame_numbers = np.linspace(0, T, 11)

    for idx, frame_number in enumerate(frame_numbers):
        frames[idx*3:idx*3+3, :, :, :] = video[frame_number, :, :, :]

    return frames


class UFC101(Dataset):

    def __init__(self, annotations, root_dir, transform=None):
        self.items = pd.read_csv(annotations)
        self.dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        video_path = os.path.join(self.dir, self.items.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 2]

        video = read_video(video_path, output_format="TCHW")
        frames = _pick_frame_even(video)

        for i in range(frames.size()[0] // 3):
            if self.transform:
                frames[i * 3:i * 3 + 3, :, :] = self.transform(frames[i * 3: i * 3 + 3, :, :])

        return frames, label
