#!/usr/bin/env python3
"""Copy the selected epoch checkpoint to the reproducible deployment location.

  eval_heldout_openclip.py : <heldout_run>    -> best epoch e*
  this script              : <all_ckpt_dir> e* -> assets/model_rep/encoder/metaclip_v1

This is a full fine-tune (open_clip), so a checkpoint is the single file `epoch_{N}.pt`. The
filename is kept as-is because the encoding scripts take it directly via `--checkpoint <path>`.

usage (run from the repository root):
  python train/encoders/metaclip_v1/deploy.py <all_run>/checkpoints --epoch 4
"""
from __future__ import annotations

import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

NAME = os.path.basename(HERE)
DST_ROOT = os.path.join(_REPO, "assets/model_rep/encoder")


def _abs(p: str) -> str:
    """Resolve a relative path against the repository root."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def main():
    ap = argparse.ArgumentParser(
        description=f"deploy the selected epoch checkpoint to assets/model_rep/encoder/{NAME}")
    ap.add_argument("ckpt_dir", help="checkpoints directory of the all-data run (holds epoch_N.pt)")
    ap.add_argument("--epoch", type=int, required=True, help="best epoch chosen on the held-out bench")
    args = ap.parse_args()

    d = _abs(args.ckpt_dir)
    fn = f"epoch_{args.epoch}.pt"
    src = os.path.join(d, fn)
    if not os.path.exists(src):
        have = sorted(x for x in os.listdir(d)) if os.path.isdir(d) else []
        raise SystemExit(
            f"[input check failed]\n  - checkpoint not found: {src}\n"
            f"      the best epoch is chosen on the held-out bench:\n"
            f"      python train/encoders/eval/eval_heldout_openclip.py "
            f"--model ViT-L-14-worldwide-xlmv --ckpt-dir {args.ckpt_dir} --epochs 1-10\n"
            f"      directory contents: {have[:12]}")

    dst = os.path.join(DST_ROOT, NAME)
    print(f"[deploy] {os.path.relpath(src, _REPO)}  →  {os.path.relpath(dst, _REPO)}/{fn}")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(src, os.path.join(dst, fn))
    print(f"         1 file · {os.path.getsize(os.path.join(dst, fn)) / 2 ** 30:.2f} GiB")


if __name__ == "__main__":
    main()
