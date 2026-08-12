#!/usr/bin/env python3
"""build_recs — assemble base top-K candidates plus reranker scores into the `recs_*.pt` format.

S4a (`../S4_tail/tail_refinement.py`) falls back to this dump for any (q,c) missing from the union
cache. Both inputs (the base score and the reranker union scores) are already in the cache, so this
runs without any GPU work.

  recs[i] = {"qidx": query id, "cand": base top-K columns, "sim": their base scores,
             "scores": reranker scores}

The output file names are hard-coded by the consumers and kept as is:
  --name 8b          -> recs_8b_p3_k20.pt      (K=20)
  --name qwen3vl_2b  -> recs_2b_dora_k5_p3.pt  (K=5)

`cand` is the top-K of BASE_PT (default assets/cache/s1_base/base_score.pt). The distributed
recs_*.pt came from an earlier base in the greedy sweep, so a rebuild yields different candidate
lists — its top-20 overlaps the shipped dump for only 4.8% of queries as a set. That is expected:
the final answer is byte-identical either way (measured on the full test set), because cand/sim
feed S4a's fallback only for pairs the union caches already cover. Compare metrics, not the md5.

Usage:
  python build_recs.py --name 8b --rep
  python build_recs.py --name qwen3vl_2b --rep

Environment: S1_CACHE · QUERY_INDEX · PAB_TEST
"""
import argparse
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils import gallery_norm as GN                                # noqa: E402
from utils.artifacts import skip_if_exists                          # noqa: E402

# reranker -> (output file name, K, qwen tag); the consumers (tail_refinement, fuse) hard-code these names
TARGETS = {
    "8b":         ("recs_8b_p3_k20.pt", 20, "8b"),
    "qwen3vl_2b": ("recs_2b_dora_k5_p3.pt", 5, "2b"),
}

ap = argparse.ArgumentParser(description="Assemble base top-K plus reranker scores into a recs dump")
ap.add_argument("--name", required=True, choices=sorted(TARGETS), help="reranker (union cache name)")
ap.add_argument("--topk", type=int, default=None, help="number of candidates (default = the per-target K)")
ap.add_argument("--out", default=None)
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="operate on the reproduction cache (assets/cache_rep)")
a = ap.parse_args()

fname, K, qwen_tag = TARGETS[a.name]
K = a.topk or K
CACHE = f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}"
OUT = a.out or f"{CACHE}/s2_rerank/{fname}"
skip_if_exists(OUT, a.overwrite)

PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
QIDX = os.environ.get("QUERY_INDEX", f"{PAB_TEST}/query_index.txt")
BASE = os.environ.get("BASE_PT", f"{CACHE}/s1_base/base_score.pt")
UCACHE = f"{CACHE}/s2_rerank/{a.name}_union_cache.pt"

for p, what in ((BASE, "base score"), (UCACHE, f"{a.name} union cache")):
    if not os.path.exists(p):
        raise SystemExit(f"[build_recs] {what} not found: {p}")

qx = [l.strip() for l in open(QIDX) if l.strip()]
Q = len(qx)
base = GN.normalize(torch.load(BASE, map_location="cpu", weights_only=False)).float()[:Q]
scores = torch.load(UCACHE, weights_only=False)["scores"]
print(f"[build_recs] {a.name} · queries {Q} x top{K} · reranker pairs {len(scores)} -> {OUT}", flush=True)

recs, miss = [], 0
for i, q in enumerate(qx):
    top = base[i].topk(K)
    cand = top.indices.tolist()
    sim = [float(v) for v in top.values]
    sc = []
    for c in cand:
        v = scores.get((q, int(c)))
        if v is None:
            miss += 1
            v = 0.0                     # candidate outside the union; left at 0 since this is a fallback
        sc.append(float(v))
    recs.append({"qidx": q, "cand": cand, "sim": sim, "scores": sc})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({"recs": recs, "qwen": qwen_tag, "prompt": "p3", "K": K}, OUT)
cov = 100 * (1 - miss / max(Q * K, 1))
print(f"[build_recs] {len(recs)} queries · reranker score coverage {cov:.1f}% ({miss} pairs missing) -> {OUT}", flush=True)
