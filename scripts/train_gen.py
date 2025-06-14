"""
Standalone training and generation script for DLPM on 512x256 seismic images.

This script is designed to be fully self-contained. Drop it into the root
of the DLPM-run repository and execute it to train a model and generate
samples based on the specified configuration.

Example usage:
  python scripts/train_gen.py --config seismic_rect_safe.yml --epochs 50 --generate 16

The script handles:
- Loading the DLPM configuration.
- Setting up the experiment using the BEM framework.
- Disabling evaluation during training to focus on generation.
- Running the training loop for a specified number of epochs.
- Saving the final model checkpoint.
- Generating unconditional samples using the trained model.
- Saving the generated samples to a PNG file.
"""

import math
import sys
import argparse
import subprocess
from pathlib import Path

# Stub out pyemd to avoid installation issues if not needed for evaluation
# This is a workaround for a potential NumPy-ABI clash on some systems.
try:
    import pyemd
except ImportError:
    sys.modules["pyemd"] = type("pyemd", (), {"emd": None})()

import torch
import torch.hub
import torchvision.utils as tvu

# Ensure local modules (bem, dlpm) are prioritized in the path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bem.Experiments as Exp
import bem.utils_exp as utils
import dlpm.dlpm_experiment as dlpm_exp

# --- Main Execution ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Train DLPM and generate samples."
    )
    parser.add_argument("--name", type=str, default="seismic_exp_safe",
                        help="Name of the experiment.")
    parser.add_argument("--config", type=str, default="seismic_rect_safe.yml",
                        help="Name of the config file to use.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training.")
    parser.add_argument("--generate", type=int, default=16,
                        help="Number of samples to generate after training.")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of reverse diffusion steps for generation.")
    parser.add_argument("--progress", action="store_true",
                        help="Show progress bars for training and sampling.")
    args = parser.parse_args()

    # --- Configuration Loading and Patching ---
    config_path = ROOT / "dlpm" / "configs" / args.config
    params = utils.load_yaml(config_path)
    print(f"✔  Loaded config: {args.config}")

    # Patch config for this specific run:
    # Disable evaluation, set epochs, and other script-specific overrides.
    params["run"]["eval_freq"]        = None  # No evaluation during training
    params["run"]["checkpoint_freq"]  = args.epochs + 1 # Save only final model
    params["run"]["n_epochs"]         = args.epochs
    params["training"]["batch_size"]  = args.batch_size
    params["exp"]["name"]             = args.name
    params["run"]["neptune_args"]["enabled"] = False # Disable online logging

    # --- Experiment Setup ---
    # The BEM framework handles model, optimizer, and data loader setup.
    # We provide DLPM-specific hooks to correctly initialize the method.
    exp = Exp.Experiment(
        init_method=dlpm_exp.init_method_by_parameter,
        init_models=dlpm_exp.init_models_by_parameter,
        init_optimizers=dlpm_exp.init_optimizers_by_parameter,
        init_schedulers=dlpm_exp.init_schedulers_by_parameter,
        checkpoint_dir=ROOT / "models" / args.name,
        **params,
    )
    exp.prepare()
    print("✔  Experiment prepared. Using device:", exp.manager.device)

    # --- Training Phase ---
    print(f"\n🟢  Training for {args.epochs} epochs…\n")
    exp.run(no_ema_eval=True, progress=args.progress)
    exp.save(exp.manager.epoch, is_best=False) # Save final checkpoint
    print("✔  Training complete. Final checkpoint saved.")

    # --- Generation Phase ---
    print(f"\n🟢  Generating {args.generate} samples…\n")
    gen_man = exp.manager.eval.gen_manager

    # Choose EMA weights if present, otherwise use the final raw model weights.
    if getattr(exp.manager, "ema_objects", None):
        print("Using EMA weights for generation.")
        ema0 = exp.manager.ema_objects[0]
        models = {n: ema0[n].get_ema_model() for n in exp.manager.models}
    else:
        print("Using final model weights for generation.")
        models = {n: m for n, m in exp.manager.models.items()}

    # Move models to the correct device and set to evaluation mode.
    device = exp.manager.device
    models = {name: model.to(device).eval() for name, model in models.items()}

    # Disable gradient tracking for efficiency during the sampling loop.
    with torch.no_grad():
        gen_man.generate(
            models,
            args.generate,
            deterministic=False,
            reverse_steps=args.steps,
            print_progression=args.progress,
        )

    # Save the generated samples to a grid image.
    # The GenerationManager automatically handles the inverse transform,
    # so the samples are already in the [0, 1] range. No further normalization needed.
    nrow = max(1, int(math.sqrt(args.generate)))
    out_png = Path(f"samples_{args.name}_{args.generate}.png")
    tvu.save_image(gen_man.samples, out_png, nrow=nrow)
    print("✔  Samples saved →", out_png.resolve())

    print("\n🎉  Done – training and sampling completed.\n")
