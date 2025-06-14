#!/usr/bin/env python3
"""
Standalone training and generation script for DLPM.
This script is designed to be called by `train_seismic.py`.
"""
import argparse
import math
import os
import sys
from pathlib import Path

# pyemd kütüphanesini ve numpy uyumsuzluk hatasını atlamak için
try:
    import pyemd
except (ImportError, ValueError):
    sys.modules["pyemd"] = type("pyemd", (), {"emd": None})()

import torch
import torchvision.utils as tvu

# Ensure local modules (bem, dlpm) are in the Python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bem.Experiments as Exp
from bem.utils_exp import FileHandler
import dlpm.dlpm_experiment as dlpm_exp

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train DLPM and generate samples.")
    parser.add_argument("--name", type=str, default="seismic_exp_safe", help="Name of the experiment.")
    parser.add_argument("--config", type=str, default="seismic_rect_safe.yml", help="Name of the config file to use.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train for.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--generate", type=int, default=4, help="Number of samples to generate after training.")
    parser.add_argument("--steps", type=int, default=250, help="Number of reverse diffusion steps for generation.")
    parser.add_argument("--progress", action="store_true", help="Show progress bars for training and sampling.")
    args = parser.parse_args()

    # --- Configuration Loading and Patching ---
    config_dir = ROOT / "dlpm" / "configs"
    params = FileHandler.get_param_from_config(config_dir, args.config)
    print(f"✔ Loaded config: {args.config}")

    # Patch config to enable random cropping for robust training
    params["data"]["image_size"] = 256
    params["data"]["random_crop"] = True
    print("✔ Enabled random 256x256 cropping for training.")
    
    # Other patches for this specific run
    params["run"]["eval_freq"] = None
    params["run"]["checkpoint_freq"] = args.epochs + 1
    params["run"]["epochs"] = args.epochs
    params["run"]["progress"] = args.progress
    params["training"]["batch_size"] = args.batch_size
    params["eval"]["dlpm"]["reverse_steps"] = args.steps
    
    # --- Experiment Setup ---
    exp = Exp.Experiment(
        checkpoint_dir=ROOT / "models" / args.name,
        p=params,
        logger=None,
        exp_hash=dlpm_exp.exp_hash,
        init_method_by_parameter=dlpm_exp.init_method_by_parameter,
        init_models_by_parameter=dlpm_exp.init_models_by_parameter,
        reset_models=dlpm_exp.reset_models
    )
    exp.prepare()
    # --- DÜZELTME 1 ---
    # Cihaz bilgisine doğru yoldan ulaşıyoruz: exp.utils.p['device']
    device = exp.utils.p['device']
    print(f"✔ Experiment prepared. Using device: {device}")

    # --- Training ---
    print(f"\n🟢 Training for {args.epochs} epochs…\n")
    exp.run(no_ema_eval=True, progress=args.progress)
    exp.save(curr_epoch=exp.manager.epochs)
    print("✔ Training complete. Final checkpoint saved.")

    # --- Generation and Stitching ---
    if args.generate > 0:
        print(f"\n🟢 Generating {args.generate * 2} patches to create {args.generate} final 512x256 images…\n")
        patches_to_generate = args.generate * 2
        gen_man = exp.manager.eval.gen_manager

        if exp.manager.ema_objects:
            print("Using EMA weights for generation.")
            ema_obj = exp.manager.ema_objects[0]
            models = {name: ema_helper.get_ema_model() for name, ema_helper in ema_obj.items() if name != 'eval'}
        else:
            print("Using final model weights for generation.")
            models = exp.manager.models

        # --- DÜZELTME 2 ---
        # Cihaz bilgisini döngü içinde kullanmak için en başta aldığımız 'device' değişkenini kullanıyoruz
        for name, model in models.items():
            model.to(device).eval()
        
        with torch.no_grad():
            gen_man.generate(
                models,
                n_samples=patches_to_generate,
                deterministic=False,
                reverse_steps=args.steps,
                print_progression=args.progress,
            )
        print(f"✔ Generated {patches_to_generate} individual 256x256 patches.")

        print("\n🟢 Stitching patches into 512x256 images…")
        stitched_images = []
        for i in range(0, patches_to_generate, 2):
            patch_left = gen_man.samples[i]
            patch_right = gen_man.samples[i + 1]
            stitched_image = torch.cat([patch_left, patch_right], dim=2)
            stitched_images.append(stitched_image)

        final_output = torch.stack(stitched_images)
        nrow = max(1, int(math.sqrt(args.generate)))
        out_png = ROOT / f"samples_{args.name}_{args.epochs}epochs_512x256.png"
        tvu.save_image(final_output, out_png, nrow=nrow)
        print("✔ Stitched samples saved to:", out_png.resolve())

    print("\n🎉 Done – training and sampling completed.\n")

if __name__ == "__main__":
    main()
    main()
