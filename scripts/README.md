# Custom Seismic Training Script

This folder contains helper scripts for running DLPM on a 512x256 grayscale
dataset. The workflow mirrors the original Colab notebook but can be executed
on a local machine.

## Installation

Run the provided `install.sh` script to install all Python dependencies:

```bash
bash install.sh
```

## Preparing the Dataset and Training

Use `train_seismic.py` to convert a folder of PNG images into the expected
`data/seismic_velocity` structure and launch training/generation.

```bash
python scripts/train_seismic.py --dataset_dir processed_images --progress
```

By default the script uses the recommended parameters:

```bash
python scripts/train_gen.py \
    --config seismic_rect_safe.yml \
    --name seismic_exp_safe \
    --epochs 1500 \
    --batch 16 \
    --generate 16 \
    --steps 1000 \
    --progress
```

All arguments of `scripts/train_gen.py` can be overridden from the command line. Pass
`--help` for the full list of options.
