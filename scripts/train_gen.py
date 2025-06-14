#!/usr/bin/env python3
"""
train_gen.py – Stand-alone DLPM trainer + sampler
for 512×256 grayscale seismic slices.

• disables evaluation metrics (no pyemd needed)
• trains for --epochs full passes of the dataset
• always samples from EMA weights if they exist
"""

import argparse, math, sys, types, torch
from pathlib import Path
import torchvision.utils as tvu

# ── stub out pyemd (metric library we’re not using) ──────────────
sys.modules["pyemd"] = types.ModuleType("pyemd")   # empty stub

# ── import local packages ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import bem.Experiments as Exp
from bem.utils_exp import FileHandler
import dlpm.dlpm_experiment as dlpm_exp

# ── CLI ──────────────────────────────────────────────────────────
P = argparse.ArgumentParser()
P.add_argument("--config",   default="seismic_rect_safe.yml",
               help="YAML in dlpm/configs/")
P.add_argument("--name",     default="seismic_exp_safe",
               help="sub-folder under models/ for checkpoints+png")
P.add_argument("--epochs",   type=int, default=1200,
               help="full dataset passes (≈ steps / 234)")
P.add_argument("--batch",    type=int, default=16)
P.add_argument("--generate", type=int, default=16)
P.add_argument("--steps",    type=int, default=1000,
               help="reverse diffusion steps at sampling")
P.add_argument("--progress", action="store_true")
args = P.parse_args()

# ── load & patch YAML ────────────────────────────────────────────
cfg = ROOT / "dlpm" / "configs" / args.config
params = FileHandler.get_param_from_config(cfg.parent, cfg.name)

params["run"].update(dict(
    eval_freq       = None,            # no FID/KID/EMD
    checkpoint_freq = args.epochs + 1, # save only final
    epochs          = args.epochs
))
params["training"]["batch_size"]  = args.batch
params["training"]["num_workers"] = params["training"].get("num_workers", 0)
params["exp"]["name"]             = args.name

# ── build experiment ─────────────────────────────────────────────
exp = Exp.Experiment(
        checkpoint_dir = ROOT / "models" / args.name,
        p              = params,
        logger         = None,
        exp_hash       = dlpm_exp.exp_hash,
        eval_hash      = None,
        init_method_by_parameter        = dlpm_exp.init_method_by_parameter,
        init_models_by_parameter        = dlpm_exp.init_models_by_parameter,
        init_optimizers_by_parameter    = dlpm_exp.init_optimizers_by_parameter,
        init_schedulers_by_parameter    = dlpm_exp.init_schedulers_by_parameter,
        reset_models                    = dlpm_exp.reset_models)
exp.prepare()

# ── training loop ────────────────────────────────────────────────
print(f"\n🟢  Training for {args.epochs} epochs …\n")
exp.run(no_ema_eval=True, progress=args.progress)
exp.save(exp.manager.epoch)      # final checkpoint
print("✔  training done.")

# ── unconditional generation ─────────────────────────────────────
print(f"\n🟢  Generating {args.generate} samples …\n")
gen_man = exp.manager.eval.gen_manager
device  = exp.manager.device

if getattr(exp.manager, "ema_objects", None):       # prefer EMA
    ema0   = exp.manager.ema_objects[0]
    models = {n: ema0[n].get_ema_model().eval().to(device)
              for n in exp.manager.models}
    print("Using EMA weights.")
else:
    models = {n: m.eval().to(device)
              for n, m in exp.manager.models.items()}
    print("Using final raw weights.")

with torch.no_grad():
    gen_man.generate(models,
                     args.generate,
                     reverse_steps=args.steps,
                     deterministic=False,
                     print_progression=args.progress)

png = ROOT / f"samples_{args.name}_{args.generate}.png"
tvu.save_image(gen_man.samples, png,
               nrow=max(1, int(math.sqrt(args.generate))),
               normalize=True, value_range=(-1, 1))
print("✔  samples saved →", png.resolve())
print("\n🎉  Finished.\n")
