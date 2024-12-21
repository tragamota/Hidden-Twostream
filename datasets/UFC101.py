import os

from torch.utils.data import Dataset
from torchvision.io import read_video


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

        video, _, _ = read_video(video_path, output_format="THWC", pts_unit='sec')
        video = video.numpy()

        if self.transform is not None:
            video = self.transform(images=video)["images"]

        T, C, H, W = video.shape

        video = video.reshape(T * C, H, W)


        return video, label
