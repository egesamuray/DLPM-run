#!/usr/bin/env python3
"""
train_gen.py – Train a DLPM model (with evaluation *disabled*) and
               sample N unconditional 512×256 grayscale images.

Fully standalone – drop into DLPM-run root and run.
"""

# ───────────────────────── Imports & CLI ─────────────────────────
import argparse, math, os, pathlib, sys, types, torch, torchvision.utils as tvu

P = argparse.ArgumentParser()
P.add_argument("--config",   default="seismic_rect_safe.yml",
               help="YAML file in dlpm/configs/")
P.add_argument("--name",     default="seismic_exp_safe",
               help="sub-folder under models/ for checkpoints + samples")
P.add_argument("--epochs",   type=int, default=100)
P.add_argument("--batch",    type=int, default=4)
P.add_argument("--generate", type=int, default=64)
P.add_argument("--steps",    type=int, default=1000)
P.add_argument("--progress", action="store_true")
args = P.parse_args()

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))           # local pkgs first
os.environ.setdefault("NEPTUNE_MODE", "offline")
os.environ.setdefault("NEPTUNE_DISABLE_TELEMETRY", "1")

# ── 1. stub out *only* pyemd (NumPy-ABI clash) ───────────────────
def _make_dummy(name):
    m = types.ModuleType(name)
    m.__file__ = "<stub>"
    return m
sys.modules["pyemd"] = _make_dummy("pyemd")

# ── 2. DLPM imports (safe now) ───────────────────────────────────
import bem.Experiments as Exp
from bem.utils_exp import FileHandler
import dlpm.dlpm_experiment as dlpm_exp

# ── 3. Load & patch config (turn evaluation off) ─────────────────
cfg = ROOT / "dlpm" / "configs" / args.config
params = FileHandler.get_param_from_config(cfg.parent, cfg.name)

params["run"]["eval_freq"]        = None            # disable FID/KID/EMD
params["run"]["checkpoint_freq"]  = args.epochs + 1 # save only final
params["run"]["epochs"]           = args.epochs
params["training"]["batch_size"]  = args.batch
params["training"]["num_workers"] = params["training"].get("num_workers", 0)

# ── 4. Build experiment ──────────────────────────────────────────
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

# ── 5. Train ─────────────────────────────────────────────────────
print(f"\n🟢  Training for {args.epochs} epochs …\n")
exp.run(no_ema_eval=True, progress=args.progress)

ckpt = exp.save(curr_epoch=args.epochs)
print("✔  final checkpoint written to", ckpt)

# ── 6. Unconditional generation (EMA weights) ────────────────────
print(f"\n🟢  Generating {args.generate} samples …\n")

gen_man = exp.manager.eval.gen_manager      # ← real GenerationManager

# choose EMA weights if present
if getattr(exp.manager, "ema_objects", None):
    ema0   = exp.manager.ema_objects[0]     # mu ≈ 0.9999
    models = {n: ema0[n].get_ema_model().eval()
              for n in exp.manager.models}
else:
    models = {n: m.eval() for n, m in exp.manager.models.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
for m in models.values():
    m.to(device)

gen_man.generate(models, args.generate,
                 deterministic=False,
                 reverse_steps=args.steps,
                 print_progression=True)

# grid save
nrow = max(1, int(math.sqrt(args.generate)))
out_png = ROOT / f"samples_{args.generate}.png"
tvu.save_image(gen_man.samples, out_png,
               nrow=nrow, normalize=True, value_range=(-1, 1))
print("✔  samples saved →", out_png.resolve())

print("\n🎉  Done – training + sampling completed.\n")

