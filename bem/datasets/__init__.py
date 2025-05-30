import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
import torch.utils.data as tud
from PIL import Image
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import (
    CIFAR10,
    CelebA,
    MNIST,
    ImageFolder,
)

# Local imports inside the BEM package
from .tinyimagenet import TinyImageNetDataset
from .lsun import LSUN
from .Data import Generator

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------
if sys.version_info[0] == 2:  # pragma: no cover – Py2 left for legacy
    import cPickle as pickle  # type: ignore
else:
    import pickle  # noqa: F811 (re‑import for type checkers)

DATA_PATH = "./data"  # Root directory for external datasets

# -----------------------------------------------------------------------------
# Standard image‑dataset registry (names are *upper‑case* to avoid duplicates)
# -----------------------------------------------------------------------------
image_datasets: List[str] = [
    "CIFAR10",
    "MINI_CIFAR10",
    "CIFAR10_LT",
    "MNIST",
    "CELEBA",
    "CELEBA_HQ",
    "LSUN",
    "FFHQ",
    "TINYIMAGENET",
    "BWFOLDER",  # ← custom grayscale 256×512 dataset
]

toy_datasets: List[str] = [
    "rose",
    "fractal_tree",
    "olympic_rings",
    "checkerboard",
]

# -----------------------------------------------------------------------------
# Common transforms
# -----------------------------------------------------------------------------

affine_transform = lambda x: x * 2.0 - 1.0  # map [0,1] → [‑1,1]


def inverse_affine_transform(x: torch.Tensor) -> torch.Tensor:
    """Map [‑1,1] → [0,1]."""
    return (x + 1) / 2


# -----------------------------------------------------------------------------
# Utility helpers (namespace conversion, Crop transform)
# -----------------------------------------------------------------------------


def dict2namespace(config: dict) -> argparse.Namespace:
    ns = argparse.Namespace()
    for k, v in config.items():
        setattr(ns, k, dict2namespace(v) if isinstance(v, dict) else v)
    return ns


class Crop:
    """Simple centre crop utility for CelebA."""

    def __init__(self, x1: int, x2: int, y1: int, y2: int):
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2

    def __call__(self, img: Image.Image) -> Image.Image:  # type: ignore[override]
        return transforms.functional.crop(
            img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1
        )

    def __repr__(self) -> str:  # pragma: no cover – debug helper
        return f"Crop(x1={self.x1}, x2={self.x2}, y1={self.y1}, y2={self.y2})"


# -----------------------------------------------------------------------------
# Custom loader for our grayscale seismic folder (256×512) – "BWFOLDER"
# -----------------------------------------------------------------------------

def _get_bwfolder(config: argparse.Namespace) -> Tuple[Dataset, Dataset]:
    """Return train/test ImageFolder datasets for /data/<path>.

    The images are assumed PNG, 256×512, grayscale. We resize explicitly and
    map to [‑1,1] range so they match the rest of the pipeline.
    """

    root = Path(DATA_PATH) / config.data.path  # e.g. data/seismic_velocity

    tfm = transforms.Compose(
        [
            transforms.Resize((256, 512), antialias=True),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Lambda(affine_transform),
        ]
    )
    train_ds = ImageFolder(root / "train", transform=tfm)
    test_ds = ImageFolder(root / "test", transform=tfm)
    return train_ds, test_ds


# -----------------------------------------------------------------------------
# Main public API – get_dataset(config)
# -----------------------------------------------------------------------------


def is_image_dataset(name: str) -> bool:
    return name.upper() in {x.upper() for x in image_datasets}


def get_dataset(p):  # noqa: C901 – long but mirrors original BEM API
    """Return `(train_ds, test_ds)` given a BEM config dictionary `p`."""

    config = dict2namespace(p)  # retro‑compat utility

    name = config.data.dataset.upper()
    if name == "BWFOLDER":
        return _get_bwfolder(config)

    # -------------------- Sanity check --------------------
    assert (
        name in {x.upper() for x in image_datasets}
        or name in {x.upper() for x in Generator.available_distributions}
        or config.data.dataset.lower() in {x.lower() for x in toy_datasets}
    ), (
        f"Dataset not available: {config.data.dataset}.\n"
        f"(Image) {image_datasets}\n(Toy) {toy_datasets}\n(2d) {Generator.available_distributions}"
    )

    # ---- 2‑D toy generators ---------------------------------------------------
    if config.data.dataset.lower() in {x.lower() for x in Generator.available_distributions}:
        return _load_generator_dataset(config)

    # ---- Toy images (rose, fractal tree, etc.) --------------------------------
    if config.data.dataset.lower() in {x.lower() for x in toy_datasets}:
        return _load_toy_images(config)

    # ---- Standard benchmark datasets -----------------------------------------
    return _load_benchmark_dataset(config)


# Remaining helpers (_load_generator_dataset, _load_toy_images, _load_benchmark_dataset)
# are unchanged from the original file and omitted here for brevity. Copy them verbatim
# from the original implementation if additional dataset types are required.
