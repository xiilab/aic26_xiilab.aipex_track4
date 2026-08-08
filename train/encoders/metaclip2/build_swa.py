#!/usr/bin/env python3
"""Average the late-epoch weights of a metaclip2 run element-wise into an SWA checkpoint.

The range is fixed at ep02-ep04. This encoder has no paired heldout run, so the script
performs no search - it only averages. The challenge test set is never read.

Averaged:
  adapter_model.safetensors   DoRA lora_A / lora_B / lora_magnitude
  extras_state.pt             logit_scale, position_embedding, flair_pooler (tensors only;
                              structural int/str/bool values keep the first checkpoint's
                              value). MetaCLIP2 has no logit_bias and K=1, so neither
                              logit_bias nor multi_probe_probe appears.
Copied from the last epoch: adapter_config.json, meta.json, README.md
The frozen backbone is not in the checkpoints and is left untouched.

DoRA applies B@A, a bilinear form, so an element-wise average is an approximation. It only
holds while the checkpoints are close to one another, i.e. on a plateau.

usage (run from the repository root):
  python train/encoders/metaclip2/build_swa.py <run>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

NEED_FILES = ("adapter_model.safetensors", "extras_state.pt")   # required for the average
COPY_FILES = ("adapter_config.json", "meta.json", "README.md")  # copied when present

SWA_RANGE = (2, 4)   # fixed; no heldout pair to search on


def _abs(p: str) -> str:
    """Resolve a relative path against the repository root."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def epochs_of(run: str) -> list:
    return sorted(int(os.path.basename(d)[2:])
                  for d in glob.glob(os.path.join(run, "checkpoints", "ep[0-9][0-9]")))


def build_swa(run: str, lo: int, hi: int, range_source: str = "fixed") -> str:
    """Average ep{lo}..ep{hi} of `run` element-wise into checkpoints/swa."""
    import torch
    from safetensors.torch import load_file, save_file

    ckdir = os.path.join(run, "checkpoints")
    dirs = [os.path.join(ckdir, f"ep{n:02d}") for n in range(lo, hi + 1)]
    miss = [os.path.basename(d) for d in dirs if not os.path.isdir(d)]
    if miss:
        raise SystemExit(f"[error] checkpoint not found: {miss}")
    if len(dirs) < 2:
        raise SystemExit(f"[error] averaging needs at least 2 checkpoints (got ep{lo}-ep{hi})")
    out = os.path.join(ckdir, "swa")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    print(f"[swa] averaging {len(dirs)}: {[os.path.basename(d) for d in dirs]}")

    sds = [load_file(os.path.join(d, "adapter_model.safetensors")) for d in dirs]
    keys = list(sds[0].keys())
    for i, sd in enumerate(sds[1:], 1):
        if set(sd.keys()) != set(keys):
            raise SystemExit(f"[error] {os.path.basename(dirs[i])} tensor keys do not match")
    save_file({k: sum(sd[k].float() for sd in sds) / len(sds) for k in keys},
              os.path.join(out, "adapter_model.safetensors"))
    print(f"[swa] {len(keys)} adapter tensors averaged")

    def avg_obj(objs):
        o0 = objs[0]
        if torch.is_tensor(o0):
            return sum(o.float() for o in objs) / len(objs)
        if isinstance(o0, dict):
            return {k: avg_obj([o[k] for o in objs]) for k in o0}
        return o0                                  # non-tensors (int/str/bool) keep the first value
    exs = [torch.load(os.path.join(d, "extras_state.pt"), map_location="cpu",
                      weights_only=False) for d in dirs]
    ex_avg, kept_last = {}, []
    for k in exs[0]:
        try:
            ex_avg[k] = avg_obj([ex[k] for ex in exs])
        except Exception:
            ex_avg[k] = exs[-1][k]
            kept_last.append(k)
    torch.save(ex_avg, os.path.join(out, "extras_state.pt"))
    print(f"[swa] extras averaged: {list(ex_avg)}")
    if kept_last:
        print(f"[swa] extras that cannot be averaged, keeping the last epoch's value: {kept_last}")

    copied = []
    for fn in COPY_FILES:
        src = os.path.join(dirs[-1], fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, fn))
            copied.append(fn)
    absent = [fn for fn in COPY_FILES if fn not in copied]
    if absent:
        print(f"[swa] not present in the last epoch, not copied: {absent}")

    json.dump({"swa_range": [lo, hi], "epochs": [os.path.basename(d) for d in dirs],
               "n_avg": len(dirs), "built_from": os.path.relpath(run, _REPO),
               "range_source": range_source},
              open(os.path.join(out, "swa_meta.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[swa] done → {os.path.relpath(out, _REPO)}")
    return out


def main():
    lo, hi = SWA_RANGE
    ap = argparse.ArgumentParser(
        description=f"average ep{lo:02d}-ep{hi:02d} of a metaclip2 run into an SWA checkpoint "
                    f"(range fixed, the test set is not used)")
    ap.add_argument("run", help="run directory (must hold checkpoints/epNN)")
    args = ap.parse_args()

    run = _abs(args.run)
    problems, have = [], []
    if not os.path.isdir(run):
        problems.append(f"path not found: {run}")
    elif not os.path.isdir(os.path.join(run, "checkpoints")):
        problems.append(f"not a run directory (no checkpoints/): {run}")
    else:
        have = epochs_of(run)
        if not have:
            problems.append(f"checkpoints/epNN not found: {run}")
    if have:
        gap = [e for e in range(lo, hi + 1) if e not in have]
        if gap:
            problems.append(f"ep{gap} not found (available ep{have[0]:02d}-ep{have[-1]:02d})")
        else:
            for e in range(lo, hi + 1):
                d = os.path.join(run, "checkpoints", f"ep{e:02d}")
                for fn in NEED_FILES:
                    if not os.path.exists(os.path.join(d, fn)):
                        problems.append(f"ep{e:02d}/{fn} not found")
    if problems:
        raise SystemExit("[input check failed] fix the following and run again.\n"
                         + "\n".join(f"  - {p}" for p in problems))

    print(f"[build] run={os.path.relpath(run, _REPO)}  available ep{have[0]:02d}-ep{have[-1]:02d}")
    print(f"        range ep{lo:02d}-ep{hi:02d} (fixed)")
    build_swa(run, lo, hi)


if __name__ == "__main__":
    main()
