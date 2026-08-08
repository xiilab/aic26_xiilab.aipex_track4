#!/usr/bin/env python3
"""Copy the SWA checkpoint of an all run to the reproduction deployment location.

  build_swa.py : <all_run> lo hi -> checkpoints/swa
  this script  : <all_run>       -> assets/model_rep/encoder/<name>

The name is this directory with _all removed (anchor_filip_all -> anchor_filip).
Only the files inference needs are copied (README.md and swa_meta.json are excluded).
An existing deployment is removed and rewritten.

usage (run from the repository root):
  python train/encoders/<enc>_all/deploy.py <all_run>
"""
from __future__ import annotations

import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

NAME = os.path.basename(HERE).removesuffix("_all")
DST_ROOT = os.path.join(_REPO, "assets/model_rep/encoder")
DEPLOY_FILES = ("adapter_config.json", "adapter_model.safetensors",
                "extras_state.pt", "meta.json")


def _abs(p: str) -> str:
    """Resolve a relative path against the repository root."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def main():
    ap = argparse.ArgumentParser(
        description=f"deploy the SWA checkpoint to assets/model_rep/encoder/{NAME}")
    ap.add_argument("run", help="all-split run directory (must hold checkpoints/swa)")
    args = ap.parse_args()

    run = _abs(args.run)
    src = os.path.join(run, "checkpoints", "swa")
    problems = []
    if not os.path.isdir(run):
        problems.append(f"path not found: {run}")
    elif not os.path.isdir(src):
        problems.append(f"SWA not found: {os.path.relpath(src, _REPO)}\n"
                        f"      build: python train/encoders/{os.path.basename(HERE)}"
                        f"/build_swa.py {args.run} <lo> <hi>")
    else:
        for fn in DEPLOY_FILES:
            if not os.path.exists(os.path.join(src, fn)):
                problems.append(f"swa/{fn} not found")
    if problems:
        raise SystemExit("[input check failed] fix the following and run again.\n"
                         + "\n".join(f"  - {p}" for p in problems))

    dst = os.path.join(DST_ROOT, NAME)
    print(f"[deploy] {os.path.relpath(src, _REPO)}  →  {os.path.relpath(dst, _REPO)}")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for fn in DEPLOY_FILES:
        shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    n = sum(os.path.getsize(os.path.join(dst, f)) for f in DEPLOY_FILES)
    print(f"         {len(DEPLOY_FILES)} files · {n / 2 ** 20:.1f} MiB")
    print(f"         {' '.join(DEPLOY_FILES)}")


if __name__ == "__main__":
    main()
