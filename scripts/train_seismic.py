#!/usr/bin/env python3
"""Prepare 512x256 grayscale dataset and launch training."""
import argparse
import glob
import os
import subprocess
from PIL import Image

def prepare_data(src_dir: str, out_root: str = "data/seismic_velocity"):
    """
    Converts source images to 512x256 grayscale and saves them.
    The actual 256x256 cropping will be handled randomly during training.
    """
    train_dir = os.path.join(out_root, "train", "class")
    test_dir = os.path.join(out_root, "test", "class")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    if not img_paths:
        raise ValueError(f"No PNG images found in '{src_dir}'. Please provide the correct directory.")

    # Split images into training and testing sets
    if len(img_paths) <= 1:
        train_paths = img_paths
        test_paths = []
    else:
        split_index = max(1, int(0.9 * len(img_paths)))
        train_paths = img_paths[:split_index]
        test_paths = img_paths[split_index:]

    def convert_and_save(paths, out_dir):
        """Loads, converts to grayscale, resizes to 512x256, and saves."""
        for path in paths:
            with Image.open(path) as img:
                # Resize to the full rectangular shape
                processed_img = img.convert("L").resize((512, 256), Image.LANCZOS)
                processed_img.save(os.path.join(out_dir, os.path.basename(path)))

    print(f"Processing {len(train_paths)} images for training and {len(test_paths)} for testing...")
    convert_and_save(train_paths, train_dir)
    convert_and_save(test_paths, test_dir)
    return len(train_paths), len(test_paths)


def main():
    p = argparse.ArgumentParser(description="Prepare dataset and launch training")
    p.add_argument("--dataset_dir", default="processed_images",
                   help="Folder containing source PNG images")
    p.add_argument("--config", default="seismic_rect_safe")
    p.add_argument("--name", default="seismic_exp_safe")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    p.add_argument("--generate", type=int, default=16)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()

    n_train, n_test = prepare_data(args.dataset_dir)
    print(f"✔ Prepared {n_train} training images and {n_test} test images.")

    cmd = ["python", "scripts/train_gen.py",
           "--config", f"{args.config}.yml", # Pass the full config filename
           "--name", args.name,
           "--epochs", str(args.epochs),
           "--batch_size", str(args.batch_size),
           "--generate", str(args.generate),
           "--steps", str(args.steps)]
    if args.progress:
        cmd.append("--progress")

    print("\n🟢 Launching training script...")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
