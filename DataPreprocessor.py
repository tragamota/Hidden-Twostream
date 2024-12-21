import argparse
import os
from os import path

import pandas as pd
import albumentations as A
import numpy as np
import torch
import cv2


codec = cv2.VideoWriter_fourcc(*'mp4v')

def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_channels = 3

    frames = np.empty((total_frames, frame_height, frame_width, frame_channels), dtype=np.uint8)

    for i in range(total_frames):
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {i} / {total_frames} from {video_path}")
            break

        frames[i] = frame

    frames = frames[:i]

    cap.release()

    return frames

def save_frame_as_video(video, name):
    T, H, W, C = video.shape

    out = cv2.VideoWriter(name, codec, 11, (W, H), True)

    for frame in video:
        out.write(frame)

    out.release()


def extract_evenly_distributed_frames(video_tensor, num_frames=11):
    T, C, H, W = video_tensor.shape

    indices = torch.linspace(0, T - 1, num_frames).long()

    return video_tensor[indices, :, :, :]


def main(args):
    data_df = pd.read_csv(os.path.join(args.split_dir, args.split_name), sep=' ', header=None)
    output_path = args.output_dir

    os.makedirs(output_path, exist_ok=True)

    output_data_pair = []

    for index, row in data_df.iterrows():
        video_path = row[0]
        video_label = row[1]

        video_file = os.path.basename(video_path)
        video_name = os.path.splitext(video_file)[0]

        video = load_video_frames(os.path.join(args.split_dir, video_path))
        video = extract_evenly_distributed_frames(video, args.num_frames)

        save_frame_as_video(video, os.path.join(args.output_dir, f'{video_name}.mp4'))

        output_data_pair.append((f'{video_name}.mp4', video_label))

    output_pd = pd.DataFrame(output_data_pair, columns=['filename', 'label'])
    output_pd.to_csv(os.path.join(args.output_dir, args.output_name), header=False, index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--split_name', required=True, help="Data split file path")
    args.add_argument('--split_dir', required=True, help="Output directory")
    args.add_argument('--output_dir', required=True, help="Output path")
    args.add_argument("--output_name", required=True, help="Output file name")
    args.add_argument('--num_frames', type=int, default=11, help="Number of frames")

    main(args.parse_args())