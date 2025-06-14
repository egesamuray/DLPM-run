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

# Select EMA weights if available
if getattr(exp.manager, "ema_objects", None):
    ema = exp.manager.ema_objects[0]
    models = {k: ema[k].get_ema_model().to(device).eval() for k in exp.manager.models}
else:
    models = {k: m.to(device).eval() for k, m in exp.manager.models.items()}

# ---------------------------------------------------------------------
#  Disable gradients during sampling  ◄ NEW
# ---------------------------------------------------------------------
with torch.no_grad():
    gen_man.generate(models, args.generate,
                     deterministic=False,
                     reverse_steps=args.steps,
                     print_progression=True)

# ---------------------------------------------------------------------
#  Save as 0‑1 PNG (do **not** renormalise)  ◄ NEW
# ---------------------------------------------------------------------
grid_rows = max(1, int(math.sqrt(args.generate)))
out_png   = ROOT / f"samples_{args.generate}.png"
tvu.save_image(gen_man.samples, out_png, nrow=grid_rows)

print(f"✔  samples saved → {out_png.resolve()}\n🎉  Done.")
