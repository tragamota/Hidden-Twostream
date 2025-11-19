import argparse
import json
import os

import cv2
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.model_selection import train_test_split
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

import numpy as np
import torch
import cv2


codec = cv2.VideoWriter_fourcc(*'mp4v')

def load_video_frames(video_path):


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


def convert_video(src_video_path, output_video_path):
    cap = cv2.VideoCapture(src_video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_channels = 3

    frames = np.empty((total_frames, frame_height, frame_width, frame_channels), dtype=np.uint8)

    for i in range(total_frames):
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {i} / {total_frames} from {src_video_path}")
            break

        frames[i] = frame

    frames = frames[:i]

    cap.release()

    h, w, _ = frames[0].shape
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(output_video_path, fourcc, 30, (w, h))
        for f in frames:
            out.write(f)
        out.release()
    except Exception as e:
        return None, f"Encode error in {output_video_path}: {e}"

    return output_video_path, None


def process_split(split_name, split_df, base_dir, output_dir, max_workers):
    manifest_entries = []
    futures = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _, row in split_df.iterrows():
            rel_path, label = row["path"], int(row["label"])
            src_video_path = os.path.join(base_dir, rel_path)
            video_name = os.path.splitext(os.path.basename(rel_path))[0]
            output_video_path = os.path.join(output_dir, f"{video_name}.avi")

            if os.path.exists(output_video_path):
                manifest_entries.append({
                    "path": os.path.relpath(output_video_path, output_dir),
                    "label": label
                })
                continue

            futures[executor.submit(convert_video, src_video_path, output_video_path)] = label

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name}"):
            result, err = future.result()
            label = futures[future]
            if result is not None:
                manifest_entries.append({
                    "path": os.path.relpath(result, output_dir),
                    "label": label
                })
            else:
                print(f"⚠️ {err}")

    return manifest_entries


def main(args):
    df = pd.read_csv(args.data, sep=" ", header=None, names=["path", "label"])
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

    os.makedirs(args.output, exist_ok=True)
    base_dir = os.path.dirname(args.data)

    manifest = {
        "train": process_split("train", train_df, base_dir, args.output, args.workers),
        "val": process_split("val", val_df, base_dir, args.output, args.workers),
        "test": process_split("test", test_df, base_dir, args.output, args.workers)
    }

    manifest_path = os.path.join(args.output, "manifest.json")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Done! MJPEG videos saved in: {args.output}")
    print(f"📄 Manifest written to: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Text file with 'path label' per line")
    parser.add_argument("--output", required=True, help="Output directory for MJPEG .avi files")
    parser.add_argument("--workers", type=int, default=os.cpu_count() // 2,
                        help="Number of parallel processes")
    main(parser.parse_args())
