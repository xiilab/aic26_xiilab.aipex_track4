#!/usr/bin/env python3
"""Reranker-layer mass scalar.

  comb(α) = 0.55·z(sim) + α · Σ_k dir_k · z(r_k)        dir = max-normalized direction (max = 1.0)

Usage:
    /opt/conda/bin/python tools/ensemble/mass_pab.py
    /opt/conda/bin/python tools/ensemble/mass_pab.py --grid 0.36:0.84:0.04 --out tools/ensemble/weights/search/mass_pab.json

Environment: TRACK4 (artifact root) · RC_BENCH · RC_BASE · RC_REDUMP_DIR
Data       : `pab_ruleclean_bench.json` · `greedy_R10_v28_rc_exact_base.pt` ·
             `redump_{r32,pixtral,dora,llama}_ruleclean.pt`  (artifacts_bundle/ensemble_benches/)
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline")]
from utils import gallery_norm as GN                                        # noqa: E402
from ensemble import cosine, fmt                            # noqa: E402
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root; all default paths are relative to it

T4 = os.environ.get("TRACK4", f"{_REPO}/assets/data/benches")
BENCH = os.environ.get("RC_BENCH", f"{T4}/pab_ruleclean_bench.json")
BASE_PT = os.environ.get("RC_BASE", f"{T4}/greedy_R10_v28_rc_exact_base.pt")
RD = os.environ.get("RC_REDUMP_DIR", T4)
POOL, INJ_T = 20, 1.0
SIM_W = 0.55
DIRECTION = {"internvl_r32": 1.00, "pixtral": 0.40, "qwen3vl_2b": 0.40, "llama": 0.15}   # direction from the UCC+UCA search
FN = {"internvl_r32": "internvl_r32", "pixtral": "pixtral", "qwen3vl_2b": "qwen3vl_2b", "llama": "llama"}
ADOPTED = {"internvl_r32": 0.7, "pixtral": 0.4, "qwen3vl_2b": 0.2, "llama": 0.1}
COS_KEYS = ["internvl_r32", "pixtral", "qwen3vl_2b", "llama"]     # direction compared with the sim anchor at 1.0


def znorm(v):
    """1-D z-score (`ensemble.znorm` works row-wise on matrices; this is the vector form)."""
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v * 0.0


def load():
    B = json.load(open(BENCH))
    gidx = {p: i for i, p in enumerate(B["gallery"])}
    gt = np.array([gidx[q["img"]] for q in B["queries"]])
    qimg = [q["img"] for q in B["queries"]]
    Q = len(gt)
    bs = GN.normalize(torch.load(BASE_PT, map_location="cpu", weights_only=False)).float().numpy()[:Q]
    cand = [np.argsort(-bs[i])[:POOL].tolist() for i in range(Q)]
    zsim = [znorm([bs[i][c] for c in cand[i]]) for i in range(Q)]
    zrr, cov = {}, {}
    for mk, fn in FN.items():
        sc = torch.load(f"{RD}/redump_{fn}_ruleclean.pt", weights_only=False)["scores"]
        zrr[mk] = [znorm([sc.get((qimg[i], int(c)), 0.0) for c in cand[i]]) for i in range(Q)]
        cov[mk] = float(np.mean([sum((qimg[i], int(c)) in sc for c in cand[i]) / POOL for i in range(Q)]))
    return cand, gt, zsim, zrr, cov, Q


def evaluate(w, cand, gt, zsim, zrr, Q):
    """comb -> injective assignment -> mAP@10 / R@1, following the same rules as pipeline S3."""
    order, conf = [None] * Q, np.zeros(Q)
    for i in range(Q):
        c = SIM_W * zsim[i]
        for mk, v in w.items():
            if v:
                c = c + v * zrr[mk][i]
        c = np.asarray(c, float)
        order[i] = [cand[i][j] for j in np.argsort(-c, kind="stable")]
        p = np.exp((c - c.max()) / INJ_T)
        conf[i] = float((p / p.sum()).max())
    used, fin = set(), [None] * Q
    for q in np.argsort(-conf, kind="stable"):
        ch = next((c for c in order[q] if c not in used), order[q][0])
        used.add(ch)
        fin[q] = [ch] + [x for x in order[q] if x != ch]
    rk = [next((k for k, c in enumerate(fin[i]) if c == gt[i]), 999) for i in range(Q)]
    return (100 * np.mean([1.0 / (r + 1) if r < 10 else 0.0 for r in rk]),
            100 * np.mean([r == 0 for r in rk]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="0.36:0.84:0.04", help="alpha sweep as lo:hi:step")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    lo, hi, st = (float(x) for x in a.grid.split(":"))
    alphas = [round(lo + st * k, 4) for k in range(int(round((hi - lo) / st)) + 1)]

    cand, gt, zsim, zrr, cov, Q = load()
    print(f"[RC] Q={Q}  gallery {len(json.load(open(BENCH))['gallery'])}  "
          f"top-{POOL} reranker coverage " + " ".join(f"{k} {100 * v:.1f}%" for k, v in cov.items()))
    print(f"[dir] UCC+UCA {fmt(DIRECTION)}\n")

    print(f"{'alpha':>6} {'mass':>7} {'mAP@10':>8} {'R@1':>8}")
    best = None
    rows = []
    for al in alphas:
        w = {k: round(al * v, 3) for k, v in DIRECTION.items()}
        m, r = evaluate(w, cand, gt, zsim, zrr, Q)
        rows.append({"alpha": al, "mass": round(sum(w.values()), 3), "map10": m, "r1": r})
        mark = ""
        if best is None or m > best[1]:
            best, mark = (al, m, r, w), "  ←"
        print(f"{al:>6.2f} {sum(w.values()):>7.3f} {m:>8.3f} {r:>8.3f}{mark}")

    al, m, r, w = best
    print(f"\n★ alpha* = {al:.2f}  -> mass {sum(w.values()):.3f}   mAP@10 {m:.3f} / R@1 {r:.3f}")
    print(f"  weights {fmt(w)}")
    print(f"  adopted weights {fmt(ADOPTED)}  (mass {sum(ADOPTED.values()):.2f})")
    print(f"  cos = {cosine(w, ADOPTED, COS_KEYS):.4f}")
    if a.out:
        p = a.out if os.path.isabs(a.out) else os.path.join(HERE, a.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump({"bench": "pab_ruleclean", "direction": DIRECTION, "grid": a.grid,
                   "sweep": rows, "alpha": al, "mass": round(sum(w.values()), 3),
                   "comb": {"sim": SIM_W, **w, "8B": 0.0},
                   "cos_vs_adopted": round(cosine(w, ADOPTED, COS_KEYS), 4)}, open(p, "w"), indent=2)
        print(f"  → {p}")


if __name__ == "__main__":
    main()
