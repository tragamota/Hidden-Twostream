import argparse
import json
import os

import cv2
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.model_selection import train_test_split
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm


def convert_video(src_video_path, output_video_path):
    try:
        decoder = VideoDecoder(src_video_path, num_ffmpeg_threads=2, seek_mode="approximate")
        frames = [frame.permute(1, 2, 0).numpy()[:, :, ::-1] for frame in decoder]  # to HWC, BGR
    except Exception as e:
        return None, f"Decode error in {src_video_path}: {e}"

    if not frames:
        return None, f"No frames in {src_video_path}"

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
