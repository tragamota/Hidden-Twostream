import os
import random

import torch

from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder


class EvenVideoSampler:
    def __init__(self, num_samples: int):
        self.num_samples = num_samples

    def __call__(self, total_frames: int) -> torch.Tensor:
        indices = torch.linspace(0, total_frames - 1, steps=self.num_samples)

        return indices


class ConsecutiveVideoSampler:
    def __init__(self, num_samples: int, min_stride: int = 1, max_stride: int = 1):
        self.num_samples = num_samples
        self.min_stride = min_stride
        self.max_stride = max_stride

    def __call__(self, total_frames: int) -> torch.Tensor:
        stride = random.randint(self.min_stride, self.max_stride)

        max_start = total_frames - (self.num_samples * stride)

        if max_start <= 0:
            return torch.linspace(0, total_frames - 1, steps=self.num_samples).long()

        start = random.randint(0, max_start)

        indices = torch.tensor([start + i * stride for i in range(self.num_samples)], dtype=torch.long)

        return indices


class UFC101(Dataset):

    def __init__(self, samples, root_dir, sampler, transform=None):
        self.samples = samples
        self.dir = root_dir
        self.sampler = sampler
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path = os.path.join(self.dir, self.samples.iloc[idx, 0])
        label = self.samples.iloc[idx, 1] - 1

        decoder = VideoDecoder(video_path, num_ffmpeg_threads=4, seek_mode="approximate")

        frame_sample_indices = self.sampler(decoder.metadata.num_frames)

        video = decoder.get_frames_at(indices=frame_sample_indices)
        video = video.data

        if self.transform:
            video = self.transform(video)

        T, C, H, W = video.shape

        video = video.reshape(T * C, H, W)

        return video, label
