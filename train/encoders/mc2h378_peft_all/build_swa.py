#!/usr/bin/env python3
"""Average the late-epoch weights of an all run element-wise into an SWA checkpoint.

The epoch range is found on the paired heldout run - <enc>_heldout/search_swa_range.py.
This script only averages; it reads no bench and never reads the challenge test set.

  search : <enc>_heldout/search_swa_range.py <heldout_run> -> [lo, hi]
  merge  : this script                       <all_run> lo hi -> checkpoints/swa

Averaged:
  adapter_model.safetensors   DoRA lora_A / lora_B / lora_magnitude
  extras_state.pt             logit_scale, logit_bias, position_embedding,
                              multi_probe_probe, flair_pooler (tensors only;
                              structural int/str/bool values keep the first checkpoint's value)
Copied from the last epoch: adapter_config.json, meta.json, README.md
The frozen backbone is not in the checkpoints and is left untouched.

DoRA applies B@A, a bilinear form, so an element-wise average is an approximation. It only
holds while the checkpoints are close to one another, i.e. on a plateau.

usage (run from the repository root):
  python train/encoders/<enc>_all/build_swa.py <all_run> <lo> <hi>
  python train/encoders/<enc>_all/build_swa.py <all_run>            # use DEFAULT_RANGE
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

# per-encoder default; normally pass the value found by <enc>_heldout/search_swa_range.py
# passing them is the rule; this table only covers the omitted-argument case.
DEFAULT_RANGE = {
    "anchor_filip_all": (8, 10),
    "anchor_tcap_all":  (8, 10),
    "mc2h378_peft_all": (2, 4),
}
_DEFAULT = DEFAULT_RANGE.get(os.path.basename(HERE))


def _abs(p: str) -> str:
    """Resolve a relative path against the repository root."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def epochs_of(run: str) -> list:
    return sorted(int(os.path.basename(d)[2:])
                  for d in glob.glob(os.path.join(run, "checkpoints", "ep[0-9][0-9]")))


def build_swa(run: str, lo: int, hi: int, range_source: str = "cli") -> str:
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
    ap = argparse.ArgumentParser(
        description="average the late-epoch weights of an all run into an SWA checkpoint (the test set is not used)")
    ap.add_argument("run", help="all-split run directory (must hold checkpoints/epNN)")
    _d = f" (default {_DEFAULT[0]}-{_DEFAULT[1]})" if _DEFAULT else " (no default)"
    ap.add_argument("lo", type=int, nargs="?", default=None,
                    help=f"range start epoch - the value found on the heldout run{_d}")
    ap.add_argument("hi", type=int, nargs="?", default=None,
                    help="range end epoch (give it together with lo)")
    args = ap.parse_args()

    run = _abs(args.run)
    problems, have = [], []
    if args.lo is None and args.hi is None:
        if _DEFAULT is None:
            problems.append(f"lo and hi must be given ({os.path.basename(HERE)} has no default range)")
            lo = hi = None
        else:
            lo, hi = _DEFAULT
            source = "default"
    elif args.lo is None or args.hi is None:
        problems.append("lo and hi must be given together")
        lo = hi = None
    else:
        lo, hi = args.lo, args.hi
        source = "cli"
    if not os.path.isdir(run):
        problems.append(f"path not found: {run}")
    elif not os.path.isdir(os.path.join(run, "checkpoints")):
        problems.append(f"not a run directory (no checkpoints/): {run}")
    else:
        have = epochs_of(run)
        if not have:
            problems.append(f"checkpoints/epNN not found: {run}")
    if lo is not None and lo >= hi:
        problems.append(f"lo must be < hi (got lo={lo}, hi={hi})")
    if have and lo is not None and lo < hi:
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
    print(f"        range ep{lo:02d}-ep{hi:02d} ({'given on the command line' if source == 'cli' else 'directory default'})")
    build_swa(run, lo, hi, range_source=source)


if __name__ == "__main__":
    main()
