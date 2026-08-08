#!/usr/bin/env python3
"""merge_union_slices — merge split-scoring parts (`{name}_q{s}_{e}_union_cache.pt`) into one file.

After splitting the query range across GPUs with `score_union_hf_4b.py --q-start/--q-end`, this
merges the parts back into the normal artifact (`{name}_union_cache.pt`). It writes only after
checking that the query ranges are contiguous and cover the whole union pool.

  python merge_union_slices.py --name internvl_r32 --rep
"""
import argparse
import glob
import os
import re

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser(description="Merge split-scoring parts into {name}_union_cache.pt")
ap.add_argument("--name", required=True)
ap.add_argument("--dir", default=None, help="location of the parts (default = cache[_rep]/s2_rerank)")
ap.add_argument("--suffix", default="union_cache")
ap.add_argument("--rep", action="store_true")
ap.add_argument("--keep", action="store_true", help="keep the part files after merging")
a = ap.parse_args()

d = a.dir or f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}/s2_rerank"
parts = sorted(glob.glob(f"{d}/{a.name}_q*_{a.suffix}.pt"),
               key=lambda p: int(re.search(rf"{a.name}_q(\d+)_", os.path.basename(p)).group(1)))
if not parts:
    raise SystemExit(f"[merge] no parts found: {d}/{a.name}_q*_{a.suffix}.pt")

pool = f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}/s1_base/union_pool.pt"
U = torch.load(pool, weights_only=False)
need = sum(len(u) for u in U["union"])

scores, qorder, covered = {}, [], []
for p in parts:
    dd = torch.load(p, weights_only=False)
    rng = re.search(rf"{a.name}_q(\d+)_(\d+)_", os.path.basename(p))
    s, e = int(rng.group(1)), int(rng.group(2))
    dup = set(scores) & set(dd["scores"])
    print(f"  {os.path.basename(p)}  queries {s}..{e}  {len(dd['scores'])} pairs"
          + (f"  duplicates: {len(dup)}" if dup else ""))
    scores.update(dd["scores"])
    qorder += list(dd.get("qorder", []))
    covered.append((s, e))

covered.sort()
gaps = [(covered[i][1], covered[i + 1][0]) for i in range(len(covered) - 1)
        if covered[i][1] != covered[i + 1][0]]
if covered[0][0] != 0 or gaps:
    raise SystemExit(f"[merge] query ranges are not contiguous — starts at {covered[0][0]}, gaps {gaps}")
if len(scores) != need:
    print(f"[merge] pairs {len(scores)} != union {need} (coverage {100*len(scores)/need:.1f}%) — check for gaps")

out = f"{d}/{a.name}_{a.suffix}.pt"
torch.save({"scores": scores, "qorder": qorder, "name": a.name, "nnew": len(scores)}, out)
print(f"[merge] {len(parts)} parts -> {out}  pairs {len(scores)} / union {need}")
if not a.keep:
    for p in parts:
        os.remove(p)
    print(f"  removed {len(parts)} part files (use --keep to retain them)")
