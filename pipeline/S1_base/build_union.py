#!/usr/bin/env python3
"""S1 (back end) — build the candidate union pool and merge reranker re-scores.

  union = 5-base union (champ/mapmax/r5/r10/r20, 28 cand/q) ∪ S1 base top-20 -> median 29 cand/q
  reranker scores = existing re-score cache ∪ anchor depbase cache
      The merged cache covers the richer union completely, so S2 does not need to call the VLMs
      again. (qwen3vl_2b covers only the top 5 = 23.8%; see `../../tools/README.md`.)

Output: `union_pool.pt` (this folder) + `../S2_rerank/{7 rerankers}_union_cache.pt`.
        The pipeline reads from `../../assets/cache/{s1_base,s2_rerank}/`, so move the files there
        after regenerating them.
Step 2 merges rather than regenerates: `REDUMP_SRC{,2}` are required inputs unless --no-merge.
UNION_SRC is a distributed artifact — the five sweep bases are not in final.json, so it cannot be
rebuilt here; --allow-no-src / --extra-base build a pool without it.

Usage: python pipeline/S1_base/build_union.py
       python pipeline/S1_base/build_union.py --no-merge --extra-base v1.pt:v2.pt
Environment: TRACK4 (artifact root) · UNION_SRC (existing union pool) · BASE_PT ·
             REDUMP_SRC/REDUMP_SRC2 (re-score sources) · OUT_UNION/OUT_RERANK ·
             PAB_TEST · QUERY_INDEX
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

sys.path[:0] = [os.path.join(ROOT, "pipeline"), os.environ.get("TRACK4_CODE", _REPO)]
from utils import gallery_norm as GN                      # noqa: E402

import argparse                                                      # noqa: E402
_ap = argparse.ArgumentParser(description="Build the candidate union pool and merge reranker re-scores")
_ap.add_argument("--overwrite", action="store_true",
    help="rebuild even if the artifact exists (default: skip)")
_ap.add_argument("--rep", action="store_true",
                 help="reproduction build: read and write under assets/cache_rep")
_ap.add_argument("--no-merge", action="store_true",
                 help="skip step 2 (merging the existing reranker caches) and build only the "
                      "union pool; use this when the rerankers are scored from scratch.")
_ap.add_argument("--extra-base", default="",
                 help="colon-separated base_score .pt files whose top-20 also join the union "
                      "(build them with `build_base.py --out X.pt` under different WEIGHTS)")
_ap.add_argument("--allow-no-src", action="store_true",
                 help="proceed when UNION_SRC is missing (pool = this base's top-20 + --extra-base)")
_a = _ap.parse_args()
CACHE = f"{_REPO}/assets/{'cache_rep' if _a.rep else 'cache'}"       # cache root
T4 = os.environ.get("TRACK4", f"{_REPO}/assets/cache/work")
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
QIDX = os.environ.get("QUERY_INDEX", f"{PAB_TEST}/query_index.txt")
UNION_SRC = os.environ.get("UNION_SRC", f"{T4}/union_pool.pt")          # existing 5-base union
REDUMP_SRC = os.environ.get("REDUMP_SRC", T4)                            # re-score source 1
REDUMP_SRC2 = os.environ.get("REDUMP_SRC2", f"{T4}/v28a_depbase_fullpipe")  # re-score source 2
OUT_RERANK = os.environ.get("OUT_RERANK", f"{CACHE}/s2_rerank")       # merge output location
OUT_UNION = os.environ.get("OUT_UNION", f"{CACHE}/s1_base/union_pool.pt")
for _d in (OUT_RERANK, os.path.dirname(OUT_UNION)):
    os.makedirs(_d, exist_ok=True)
RERANKERS = ["qwen3vl_2b", "8b", "internvl_r32", "pixtral", "llama", "ovis", "jina_m0"]

skip_if_exists(OUT_UNION, _a.overwrite)
qx = [l.strip() for l in open(QIDX) if l.strip()]
Q = len(qx)

# 1) richer union = existing 5-base union ∪ S1 base top-20 ∪ each --extra-base top-20
if os.path.exists(UNION_SRC):
    prev = torch.load(UNION_SRC, weights_only=False)
elif _a.allow_no_src or _a.extra_base:
    print(f"[S1] no 5-base union at {UNION_SRC} -> starting from an empty pool", flush=True)
    prev = {"union": [[] for _ in range(Q)], "tops": {}, "bases": []}
else:
    raise SystemExit(
        f"[build_union] 5-base union not found: {UNION_SRC}\n"
        f"  Stage it, or build the pool without it:\n"
        f"    --allow-no-src           base top-20 only\n"
        f"    --extra-base a.pt:b.pt   variant bases (build_base.py --out)")

base = GN.normalize(torch.load(os.environ.get("BASE_PT", f"{CACHE}/s1_base/base_score.pt"),
                               map_location="cpu", weights_only=False)).float().numpy()[:Q]
new_top = [np.argsort(-base[i])[:20].tolist() for i in range(Q)]
union = [set(prev["union"][i]) | {int(c) for c in new_top[i]} for i in range(Q)]

_extra_tops = {}
for _p in [p for p in _a.extra_base.split(":") if p]:
    _s = GN.normalize(torch.load(_p, map_location="cpu", weights_only=False)).float().numpy()[:Q]
    _t = [np.argsort(-_s[i])[:20].tolist() for i in range(Q)]
    for i in range(Q):
        union[i] |= {int(c) for c in _t[i]}
    _extra_tops[os.path.splitext(os.path.basename(_p))[0]] = _t
    print(f"[S1] +extra base {os.path.basename(_p)}", flush=True)
union = [sorted(s) for s in union]

NEW_BASE = "s1_base_top20"                       # top-20 of this base_score, under a key of its own
# Rename the inherited union labels to the current names. tops/bases are only iterated over
# (never looked up by key), so the candidate set is unchanged.
_RELABEL = {"v28a_depbase": "anchor_depbase"}
tops = {_RELABEL.get(k, k): v for k, v in prev["tops"].items()}
tops[NEW_BASE] = new_top
tops.update(_extra_tops)
bases = [_RELABEL.get(b, b) for b in prev.get("bases", []) if b != NEW_BASE] + [NEW_BASE]   # avoid duplicates on re-run
bases += [b for b in _extra_tops if b not in bases]
torch.save({"union": union, "qorder": qx, "tops": tops, "bases": bases},
           OUT_UNION)
print(f"[S1] union median {int(np.median([len(u) for u in union]))} cand/q -> {OUT_UNION}")

# Name the pairs the existing reranker caches lack, so a widened pool does not surprise 05/S3.
_total = sum(len(u) for u in union)
_gap = {}
for _nm in RERANKERS:
    _p = f"{OUT_RERANK}/{_nm}_union_cache.pt"
    if not os.path.exists(_p):
        _gap[_nm] = None; continue
    _sc = torch.load(_p, weights_only=False)["scores"]
    _gap[_nm] = sum(1 for i in range(Q) for c in union[i] if (qx[i], int(c)) not in _sc)
if any(v for v in _gap.values()):
    print("  existing union caches vs the new pool:", flush=True)
    for _nm, _m in _gap.items():
        if _m is None:
            print(f"    {_nm:12} no cache yet -> score the full pool")
        elif _m:
            print(f"    {_nm:12} {_m:>6} of {_total} pairs uncovered ({100*_m/_total:.2f}%) -> rescore, "
                  f"e.g. REUSE_EXTRA={OUT_RERANK}/{_nm}_union_cache.pt")
        else:
            print(f"    {_nm:12} covered")

# 2) merge the reranker re-scores from both sources to cover the richer union
if _a.no_merge:
    print("  [--no-merge] skipping the reranker cache merge; "
          f"score each reranker instead: pipeline/S2_rerank/score_union_*.py{' --rep' if _a.rep else ''}")
    raise SystemExit(0)
total = sum(len(u) for u in union)
for nm in RERANKERS:
    merged = dict(torch.load(f"{REDUMP_SRC}/{nm}_union_cache.pt", weights_only=False)["scores"])
    merged.update(torch.load(f"{REDUMP_SRC2}/{nm}_union_cache.pt", weights_only=False)["scores"])
    hit = sum(1 for i in range(Q) for c in union[i] if (qx[i], int(c)) in merged)
    torch.save({"scores": merged, "qorder": qx, "name": nm},
               f"{OUT_RERANK}/{nm}_union_cache.pt")
    print(f"  {nm:10} merged {len(merged)} pairs · union coverage {100 * hit / total:.1f}%")
