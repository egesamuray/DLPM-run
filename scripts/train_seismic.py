#!/usr/bin/env python3
"""Prepare a 512x256 grayscale dataset and train DLPM.

This script replicates the Colab workflow in a single command-line tool.
The dataset directory can be overridden with --dataset_dir.
"""
import argparse, glob, os, subprocess
from PIL import Image


def prepare_data(src_dir: str, out_root: str = "data/seismic_velocity"):
    train_dir = os.path.join(out_root, "train", "class")
    test_dir = os.path.join(out_root, "test", "class")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(src_dir, "*.png")))
    if len(img_paths) <= 1:
        train_paths = img_paths
        test_paths = []
    else:
        split = max(1, int(0.9 * len(img_paths)))
        train_paths = img_paths[:split]
        test_paths = img_paths[split:]

    def convert_and_save(paths, out_dir):
        for path in paths:
            with Image.open(path) as img:
                g = img.convert("L").resize((512, 256), Image.LANCZOS)
                g.save(os.path.join(out_dir, os.path.basename(path)))

    convert_and_save(train_paths, train_dir)
    convert_and_save(test_paths, test_dir)
    return len(train_paths), len(test_paths)


def main():
    p = argparse.ArgumentParser(description="Prepare dataset and launch training")
    p.add_argument("--dataset_dir", default="processed_images",
                   help="Folder containing source PNG images")
    p.add_argument("--config", default="seismic_rect_safe.yml")
    p.add_argument("--name", default="seismic_exp_safe")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--generate", type=int, default=16)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--progress", action="store_true")
    args = p.parse_args()

    n_train, n_test = prepare_data(args.dataset_dir)
    print(f"Prepared {n_train} training images and {n_test} test images.")

    cmd = ["python", "train_gen.py",
           "--config", args.config,
           "--name", args.name,
           "--epochs", str(args.epochs),
           "--batch", str(args.batch),
           "--generate", str(args.generate),
           "--steps", str(args.steps)]
    if args.progress:
        cmd.append("--progress")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
