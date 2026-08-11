#!/usr/bin/env python3
"""check_reproduction — compare a reproduced answer against a reference, split by fusion margin.

`run_reproduce.sh` compares one md5, so a single flipped query reads the same as a broken
pipeline. It is not the same thing. The S3 fusion score

    comb = 0.55·z(sim) + 0.7·z(internvl_r32) + 0.4·z(pixtral) + 0.2·z(qwen3vl_2b) + 0.1·z(llama)

is decided inside a 20-candidate pool, and `z()` rescales each member to unit variance, so a 0.1 %
difference in a reranker score becomes ~0.1 in `comb` (measured median |Δcomb| = 0.110 when the
rerankers are re-scored with identical weights). Meanwhile the top1-top2 margin has a thick left
tail — median 2.22, but 254 of 1,978 queries sit below 0.5 and 7 below 0.01. Those queries flip on
noise that cannot be removed; the gallery holds near-duplicate frames of the same person, so rank 1
and rank 2 are often the same identity.

This script therefore reports agreement on two sets:

  stable   margin >= --tau   a disagreement here is a real regression
  boundary margin <  --tau   near-tie, expected to wobble between generations

Usage:
  python tools/check_reproduction.py --answer <new.txt> --reference <ref.txt> [--cache assets/cache]
  python tools/check_reproduction.py --answer <new.txt> --gt <gt.txt> --tau 0.5

`--cache` is the cache root whose `s2_rerank/` holds the recs dump and fuse_cache the margins are
computed from (default: assets/cache, or assets/cache_rep with --rep).
"""
import argparse
import json
import os

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IV_DIRS = {"internvl_r32": "internvl_r32", "pixtral": "pixtral", "llama": "llama32v"}
POOL = 20


def z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v * 0.0


def load_iv(fuse_dir, name):
    meta = json.load(open(f"{fuse_dir}/internvl_scores_{name}_meta.json"))
    sc = np.load(f"{fuse_dir}/internvl_scores_{name}_top20.npy")
    cd = np.load(f"{fuse_dir}/internvl_cand_{name}_top20.npy")
    return {(q, int(cd[i, j])): float(sc[i, j])
            for i, q in enumerate(meta["qorder"])
            for j in range(cd.shape[1]) if not np.isnan(sc[i, j])}


def margins(cache, weights):
    """top1-top2 margin of the S3 comb score, per query."""
    s2 = f"{cache}/s2_rerank"
    b8 = {r["qidx"]: r for r in torch.load(f"{s2}/recs_8b_p3_k20.pt", weights_only=False)["recs"]}
    do = torch.load(f"{s2}/recs_2b_dora_k5_p3.pt", weights_only=False)["recs"]
    qw = {(r["qidx"], c): float(s) for r in do for c, s in zip(r["cand"], r["scores"])}
    iv = {k: load_iv(f"{s2}/fuse_cache/{d}", d) for k, d in IV_DIRS.items()
          if os.path.isdir(f"{s2}/fuse_cache/{d}")}
    out = {}
    for q, rec in b8.items():
        cand = rec["cand"][:POOL]
        comb = weights["sim"] * z(rec["sim"][:POOL])
        comb = comb + weights["qwen3vl_2b"] * z([qw.get((q, c), 0.0) for c in cand])
        comb = comb + weights["8b"] * z(rec["scores"][:POOL])
        for k, tbl in iv.items():
            comb = comb + weights[k] * z([tbl.get((q, c), 0.0) for c in cand])
        srt = np.sort(np.asarray(comb, float))[::-1]
        out[q] = float(srt[0] - srt[1]) if len(srt) > 1 else float("inf")
    return out


def rows(path):
    return [l.split() for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser(description="margin-aware reproduction check")
    ap.add_argument("--answer", required=True, help="the reproduced answer")
    ap.add_argument("--reference", help="reference answer to agree with")
    ap.add_argument("--gt", help="ground truth (one gallery name per query) — scores R@1 instead")
    ap.add_argument("--cache", default=None, help="cache root for the margins (default assets/cache)")
    ap.add_argument("--rep", action="store_true", help="margins from assets/cache_rep")
    ap.add_argument("--tau", type=float, default=0.5, help="stable/boundary split on the margin")
    ap.add_argument("--query-index", default=None)
    a = ap.parse_args()
    if not a.reference and not a.gt:
        raise SystemExit("give --reference or --gt")

    cache = a.cache or f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}"
    qidx = a.query_index or f"{_REPO}/assets/data/raw/pab_test/query_index.txt"
    qx = [l.strip() for l in open(qidx) if l.strip()]
    w = json.load(open(f"{_REPO}/tools/ensemble/weights/final.json"))["comb"]

    mar = margins(cache, w)
    new = rows(a.answer)
    ref = rows(a.reference) if a.reference else None
    gt = None
    if a.gt:
        gt = [l.strip()[:-4] if l.strip().endswith(".jpg") else l.strip()
              for l in open(a.gt) if l.strip()]

    stable = boundary = s_ok = b_ok = 0
    bad = []
    for i, q in enumerate(qx):
        if i >= len(new):
            break
        m = mar.get(q, float("inf"))
        ok = (new[i][0] == ref[i][0]) if ref else (new[i][0] == gt[i])
        if m >= a.tau:
            stable += 1; s_ok += ok
            if not ok:
                bad.append((i, m))
        else:
            boundary += 1; b_ok += ok

    what = "reference" if ref else "GT"
    print(f"margin tau = {a.tau}   (comb top1-top2, cache={os.path.relpath(cache, _REPO)})")
    print(f"  stable   margin >= tau : {s_ok}/{stable} agree with {what}"
          f"  ({100 * s_ok / max(stable, 1):.2f}%)")
    print(f"  boundary margin <  tau : {b_ok}/{boundary} agree with {what}"
          f"  ({100 * b_ok / max(boundary, 1):.2f}%)")
    if bad:
        print(f"\n  regression candidates (stable set, {len(bad)}): "
              + " ".join(f"q{i}(m={m:.2f})" for i, m in bad[:20]))
        print("  → a disagreement at a wide margin is not near-tie noise; investigate these.")
    else:
        print("\n  no disagreement in the stable set — differences are confined to near-ties.")


if __name__ == "__main__":
    main()
