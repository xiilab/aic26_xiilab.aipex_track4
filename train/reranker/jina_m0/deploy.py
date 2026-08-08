#!/usr/bin/env python3
"""Copy the selected step checkpoint to the reproducible deployment location.

  eval_step.py : <run> -> best step t*   (by pair-acc)
  this script  : <run> t* -> assets/model_rep/reranker/jina_m0

`train.py` writes a DoRA adapter to `<run>/checkpoints/ex{NNNNNN}` every `CKPT_EVERY_EXAMPLES`
examples; one of those is picked.
eval_step.py's `--deploy-rep` hook does the same thing automatically; this script is the manual path.

usage (run from the repository root):
  python train/reranker/jina_m0/deploy.py <run> --step ex008000
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

NAME = os.path.basename(HERE)
DST_ROOT = os.path.join(_REPO, "assets/model_rep/reranker")
NEED_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _abs(p: str) -> str:
    """Resolve a relative path against the repository root."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def main():
    ap = argparse.ArgumentParser(
        description=f"deploy the selected step adapter to assets/model_rep/reranker/{NAME}")
    ap.add_argument("run", help="training run directory (contains checkpoints/ex{NNNNNN}/)")
    ap.add_argument("--step", required=True, help="best step chosen with eval_step.py (e.g. ex008000 or 8000)")
    args = ap.parse_args()

    run = _abs(args.run)
    tag = args.step if str(args.step).startswith("ex") else f"ex{int(args.step):06d}"
    src = os.path.join(run, "checkpoints", tag)
    problems = []
    if not os.path.isdir(run):
        problems.append(f"path not found: {run}")
    elif not os.path.isdir(src):
        have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run, "checkpoints", "ex*")))
        problems.append(f"checkpoint not found: {os.path.relpath(src, _REPO)}\n"
                        f"      pick the best step with:\n"
                        f"      python train/reranker/eval/eval_step.py --member jina --run {args.run} --steps <list>\n"
                        f"      run contents: {have[:12]}")
    else:
        for fn in NEED_FILES:
            if not os.path.exists(os.path.join(src, fn)):
                problems.append(f"{tag}/{fn} is missing")
    if problems:
        raise SystemExit("[input check failed] resolve the following and run again.\n"
                         + "\n".join(f"  - {p}" for p in problems))

    dst = os.path.join(DST_ROOT, NAME)
    print(f"[deploy] {os.path.relpath(src, _REPO)}  →  {os.path.relpath(dst, _REPO)}")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(DST_ROOT, exist_ok=True)
    shutil.copytree(src, dst)
    n = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
    print(f"         {len(os.listdir(dst))} files · {n / 2 ** 20:.1f} MiB")
    print(f"         reproduction cache: pipeline/S2_rerank/score_union_jina.py --rep --name {NAME}")


if __name__ == "__main__":
    main()
