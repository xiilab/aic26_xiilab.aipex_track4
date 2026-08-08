#!/usr/bin/env python3
"""S1 (back end) — build the candidate union pool and merge reranker re-scores.

  union = 5-base union (champ/mapmax/r5/r10/r20, 28 cand/q) ∪ S1 base top-20 -> median 29 cand/q
  reranker scores = existing re-score cache ∪ anchor depbase cache
      The merged cache covers the richer union completely, so S2 does not need to call the VLMs
      again. (qwen3vl_2b covers only the top 5 = 23.8%; see `../../tools/README.md`.)

Output: `union_pool.pt` (this folder) + `../S2_rerank/{7 rerankers}_union_cache.pt`.
        The pipeline reads from `../../assets/cache/{s1_base,s2_rerank}/`, so move the files there
        after regenerating them.
This merges rather than regenerates: the existing caches (`UNION_SRC`, `REDUMP_SRC{,2}`) are
required inputs. To build from scratch, run `../S2_rerank/score_union_*.py` (VLM GPU) first.

Usage: python pipeline/S1_base/build_union.py
Environment: TRACK4 (artifact root) · UNION_SRC (existing union pool) ·
             REDUMP_SRC/REDUMP_SRC2 (re-score sources) · TRACK4_GALLERY · QUERY_INDEX
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

# 1) richer union = existing 5-base union ∪ S1 base top-20
prev = torch.load(UNION_SRC, weights_only=False)
base = GN.normalize(torch.load(os.environ.get("BASE_PT", f"{CACHE}/s1_base/base_score.pt"),
                               map_location="cpu", weights_only=False)).float().numpy()[:Q]
new_top = [np.argsort(-base[i])[:20].tolist() for i in range(Q)]
union = [sorted(set(prev["union"][i]) | {int(c) for c in new_top[i]}) for i in range(Q)]

NEW_BASE = "s1_base_top20"                       # top-20 of this base_score, under a key of its own
# Rename the inherited union labels to the current names. tops/bases are only iterated over
# (never looked up by key), so the candidate set is unchanged.
_RELABEL = {"v28a_depbase": "anchor_depbase"}
tops = {_RELABEL.get(k, k): v for k, v in prev["tops"].items()}
tops[NEW_BASE] = new_top
bases = [_RELABEL.get(b, b) for b in prev.get("bases", []) if b != NEW_BASE] + [NEW_BASE]   # avoid duplicates on re-run
torch.save({"union": union, "qorder": qx, "tops": tops, "bases": bases},
           OUT_UNION)
print(f"[S1] union median {int(np.median([len(u) for u in union]))} cand/q -> {OUT_UNION}")

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
