#!/usr/bin/env python3
"""
Standalone trainer + sampler for DLPM on 512×256 seismic images.
"""

import argparse, math, os, sys, types, torch
from pathlib import Path
import torchvision.utils as tvu

# ────────────────────────── stub out pyemd ───────────────────────
def _make_dummy(name):
    m = types.ModuleType(name); m.__file__ = "<stub>"; return m
sys.modules["pyemd"] = _make_dummy("pyemd")

# ────────────────────────── import DLPM/BEM ──────────────────────
ROOT = Path(__file__).resolve().parent     # file location
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bem.Experiments as Exp
from bem.utils_exp import FileHandler
import dlpm.dlpm_experiment as dlpm_exp

# ─────────────────────────── CLI args ────────────────────────────
P = argparse.ArgumentParser()
P.add_argument("--name",     default="seismic_exp_safe")
P.add_argument("--config",   default="seismic_rect_safe.yml")
P.add_argument("--epochs",   type=int, default=1000)
P.add_argument("--batch",    type=int, default=16)
P.add_argument("--generate", type=int, default=16)
P.add_argument("--steps",    type=int, default=1000)
P.add_argument("--progress", action="store_true")
args = P.parse_args()

# ─────────────────────── load + patch YAML ───────────────────────
cfg = ROOT / "dlpm" / "configs" / args.config
params = FileHandler.get_param_from_config(cfg.parent, cfg.name)

params["run"]["eval_freq"]        = None                # no metrics
params["run"]["checkpoint_freq"]  = args.epochs + 1     # final only
params["run"]["epochs"]           = args.epochs
params["training"]["batch_size"]  = args.batch
params["exp"]["name"]             = args.name
params["training"]["num_workers"] = params["training"].get("num_workers", 0)

# ─────────────────────── build experiment ────────────────────────
exp = Exp.Experiment(
        checkpoint_dir = ROOT / "models" / args.name,
        p              = params,
        logger         = None,
        exp_hash       = dlpm_exp.exp_hash,
        eval_hash      = None,
        init_method_by_parameter  = dlpm_exp.init_method_by_parameter,
        init_models_by_parameter  = dlpm_exp.init_models_by_parameter,
        init_optimizers_by_param  = dlpm_exp.init_optimizers_by_parameter,
        init_schedulers_by_param  = dlpm_exp.init_schedulers_by_parameter,
        reset_models              = dlpm_exp.reset_models)

exp.prepare()
print("✔  Experiment ready – device:", exp.manager.device)

# ───────────────────────── training ──────────────────────────────
print(f"\n🟢  Training for {args.epochs} epochs …\n")
exp.run(no_ema_eval=True, progress=args.progress)
exp.save(exp.manager.epoch)        # final checkpoint
print("✔  Training complete and checkpoint saved.")

# ─────────────────────── unconditional sampling ──────────────────
print(f"\n🟢  Generating {args.generate} samples …\n")
gen_man = exp.manager.eval.gen_manager

# pick EMA models if present
if getattr(exp.manager, "ema_objects", None):
    ema = exp.manager.ema_objects[0]
    models = {n: ema[n].get_ema_model().eval().to(exp.manager.device)
              for n in exp.manager.models}
    print("Using EMA weights.")
else:
    models = {n: m.eval().to(exp.manager.device)
              for n, m in exp.manager.models.items()}
    print("Using final raw weights.")

with torch.no_grad():
    gen_man.generate(models,
                     args.generate,
                     deterministic=False,
                     reverse_steps=args.steps,
                     print_progression=args.progress)

grid = ROOT / f"samples_{args.name}_{args.generate}.png"
nrow = max(1, int(math.sqrt(args.generate)))
tvu.save_image(gen_man.samples, grid, nrow=nrow,
               normalize=True, value_range=(-1, 1))
print("✔  Samples saved →", grid.resolve())
print("\n🎉  Done – training + sampling finished.\n")
