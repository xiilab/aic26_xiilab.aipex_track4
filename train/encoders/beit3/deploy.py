#!/usr/bin/env python3
"""Copy the selected epoch checkpoint to the reproducible deployment location.

  beit3_tool.py eval : <heldout_run>  -> best epoch e*
  this script        : <all_run> e*   -> assets/model_rep/encoder/{beit3_v2|beit3_helip}

This is a full fine-tune, so a checkpoint is the single file `checkpoint-{N}.pth`. The deployed copy
is always named `checkpoint-best.pth` so consumers do not need to know the epoch number.

recipe -> deployment name
  v2     -> beit3_v2
  helip  -> beit3_helip
  stage1 is the init for helip and is not deployed.

usage (run from the repository root):
  python train/encoders/beit3/deploy.py <all_run> --recipe v2 --epoch 3
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

DST_ROOT = os.path.join(_REPO, "assets/model_rep/encoder")
RECIPE2NAME = {"v2": "beit3_v2", "helip": "beit3_helip"}


def _abs(p: str) -> str:
    """Resolve a relative path against the repository root."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def main():
    ap = argparse.ArgumentParser(
        description="deploy the selected epoch checkpoint to assets/model_rep/encoder/<name>")
    ap.add_argument("run", help="run directory trained on the full (all) data")
    ap.add_argument("--recipe", required=True, choices=list(RECIPE2NAME),
                    help="v2 → beit3_v2 · helip → beit3_helip")
    ap.add_argument("--epoch", type=int, required=True, help="best epoch chosen on the held-out bench")
    args = ap.parse_args()

    run = _abs(args.run)
    name = RECIPE2NAME[args.recipe]
    src = os.path.join(run, f"checkpoint-{args.epoch}.pth")
    if not os.path.exists(src):
        have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run, "checkpoint-*.pth")))
        raise SystemExit(
            f"[input check failed]\n  - checkpoint not found: {src}\n"
            f"      the best epoch is chosen on the held-out bench:\n"
            f"      python train/encoders/beit3/beit3_tool.py eval --run <heldout_run> --epochs 0-6 --with-best\n"
            f"      run directory contents: {have[:12]}")

    dst = os.path.join(DST_ROOT, name)
    out = os.path.join(dst, "checkpoint-best.pth")
    print(f"[deploy] {os.path.relpath(src, _REPO)}  →  {os.path.relpath(out, _REPO)}")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(src, out)
    print(f"         1 file · {os.path.getsize(out) / 2 ** 30:.2f} GiB  (recipe={args.recipe} ep{args.epoch})")


if __name__ == "__main__":
    main()
