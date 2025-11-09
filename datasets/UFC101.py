import os

from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from torchvision.io import read_video

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

        # frame_sample_indices = self.sampler(decoder.metadata.num_frames)

        frame_sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        video = decoder.get_frames_at(indices=frame_sample_indices)
        video = video.data

        if self.transform:
            video = self.transform(video)

        T, C, H, W = video.shape

        video = video.reshape(T * C, H, W)

        return video, label
