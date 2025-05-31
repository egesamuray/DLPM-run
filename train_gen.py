#!/usr/bin/env python3
"""
train_gen.py  –  train a DLPM model **without** any evaluation stack,
                 then sample <N> unconditional 512×256 grayscale images.

The script is 100 % standalone – no edits elsewhere are required.
"""

import os, sys, argparse, types, importlib, pathlib

# ───────────────────────────── CLI ────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--config",   default="seismic_rect_safe.yml",
                help="YAML file inside dlpm/configs/")
ap.add_argument("--name",     default="seismic_exp_safe",
                help="folder under models/ for checkpoints / samples")
ap.add_argument("--epochs",   type=int, default=100)
ap.add_argument("--batch",    type=int, default=4)
ap.add_argument("--generate", type=int, default=64,
                help="number of images to sample after training")
ap.add_argument("--steps",    type=int, default=1000,
                help="reverse_steps for DLPM sampling")
ap.add_argument("--progress", action="store_true")
args = ap.parse_args()

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))                # make local packages importable
os.environ.setdefault("NEPTUNE_MODE", "offline")
os.environ.setdefault("NEPTUNE_DISABLE_TELEMETRY", "1")

# ── 1.  STUB-OUT **EVERY** EVALUATION IMPORT (pyemd, bem.evaluate…) ──
def _make_dummy(mod_name: str) -> types.ModuleType:
    dummy = types.ModuleType(mod_name)
    dummy.__dict__["__file__"] = "<stub>"
    return dummy

# pyemd causes the NumPy-ABI crash – nuke it up-front
sys.modules["pyemd"] = _make_dummy("pyemd")

# create hela dummy package chain bem.evaluate.EvaluationManager
eval_pkg = _make_dummy("bem.evaluate")
eval_pkg.__path__ = []                 # marks it as a *package*
eval_mgr_mod = _make_dummy("bem.evaluate.EvaluationManager")

class _NoEval:                         # placeholder class never called
    def __init__(self,*_,**__): pass
    def __getattr__(self,*_):  return self
    def __call__(self,*_,**__): return self
eval_mgr_mod.EvaluationManager = _NoEval

# register in sys.modules **before** anything imports bem
sys.modules.update({
    "bem.evaluate":                 eval_pkg,
    "bem.evaluate.EvaluationManager": eval_mgr_mod,
})
# link sub-module to parent so `import .evaluate.EvaluationManager` works
setattr(eval_pkg, "EvaluationManager", eval_mgr_mod)

# ── 2.  STANDARD DLPM IMPORTS  (safe now) ─────────────────────────
import yaml
import bem.Experiments as Exp           # no ImportError any more
from bem.utils_exp import FileHandler   # uses our stubs happily
import dlpm.dlpm_experiment as dlpm_exp

# ── 3.  LOAD & PATCH CONFIG  ───────────────────────────────────────
cfg_path = ROOT / "dlpm" / "configs" / args.config
params   = FileHandler.get_param_from_config(cfg_path.parent, cfg_path.name)

# 100 % disable evaluation during training
params["run"]["eval_freq"]       = None
# checkpoint only once at the very end
params["run"]["checkpoint_freq"] = args.epochs + 1

params["run"]["epochs"]                  = args.epochs
params["training"]["batch_size"]         = args.batch
params["training"]["num_workers"]        = params["training"].get("num_workers", 0)

# ── 4.  BUILD EXPERIMENT  ──────────────────────────────────────────
exp = Exp.Experiment(
        checkpoint_dir = ROOT / "models" / args.name,
        p              = params,
        logger         = None,
        exp_hash       = dlpm_exp.exp_hash,
        eval_hash      = None,
        init_method_by_parameter = dlpm_exp.init_method_by_parameter,
        init_models_by_parameter = dlpm_exp.init_models_by_parameter,
        reset_models             = dlpm_exp.reset_models)

exp.prepare()

print(f"\n🟢  Training for {args.epochs} epochs …\n")
# exp.run provides the needed checkpoint callback internally
exp.run(no_ema_eval=True, progress=args.progress)

ckpt_path = exp.save(curr_epoch=args.epochs)
print("✔  final checkpoint written to", ckpt_path)

# ── 5.  UNCONDITIONAL GENERATION  ──────────────────────────────────
print(f"\n🟢  Generating {args.generate} samples …\n")

# DLPM already created a GenerationManager at  exp.manager.gen_manager
gen_man   = exp.manager.gen_manager
models    = exp.manager.models          # dict of torch.nn.Module
# generation itself – **no metrics** are computed
gen_man.generate(models, args.generate,
                 reverse_steps=args.steps,
                 deterministic=False)    # use default sampler

# save to <script-folder>/samples_<N>.png
out_path = ROOT / f"samples_{args.generate}.png"
import torchvision.utils as tvu
tvu.save_image(gen_man.samples, out_path, nrow=8, normalize=True, value_range=(-1,1))
print("✔  samples saved →", out_path.resolve())

print("\n🎉  Done – training + sampling completed without evaluation stack.\n")
