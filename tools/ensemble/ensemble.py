#!/usr/bin/env python3
"""Ensemble weight search and score combination core (shared by the pipeline).

S1 (`pipeline/S1_base/build_base.py`) uses `combine`/`mmnorm` from this module to build the
base ensemble score.

Both ensemble layers have the same form:

    score = w_anchor · f(anchor) + Σ_k w_k · f(member_k)

  · base layer   (encoders):       f = global min-max, anchor = v20,
                                   members = v28a, mc2, mc2h378, gme, eva02pre
  · rerank layer (cross-encoders): f = per-row z-score, anchor = base similarity (sim, fixed 0.55),
                                   members = r32, pixtral, DoRA, llama, 8B

Weights are found by per-axis grid coordinate ascent. The objective is supplied by the caller;
this file reads no data itself — it is a pure search/combination core.

CLI (data loading and objectives are borrowed from the eval_external modules):
    python tools/ensemble/ensemble.py --layer rerank            # reranker comb weight search
    python tools/ensemble/ensemble.py --layer base              # encoder ensemble weight search
        [--objective self|multi] [--mass 1.4] [--out w.json] [--score ens.pt]
    With --score, the searched weights are used to build and save the ensemble score matrix.

As a library:
    from ensemble import coord_ascent, combine, znorm, mmnorm, norm_mass
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

DEFAULT_GRID = [0.0, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


# ─────────────────────────────── normalization ───────────────────────────────
def znorm(a):
    """Per-row z-score — rerank layer convention (normalized within the 20 candidates)."""
    a = np.nan_to_num(np.asarray(a, float))
    s = a.std(axis=1, keepdims=True)
    return np.where(s > 1e-9, (a - a.mean(axis=1, keepdims=True)) / np.where(s > 1e-9, s, 1.0), 0.0)


def mmnorm(s):
    """Global min-max to [0,1] — base layer convention, same as `gallery_norm.normalize`.
    Works for both torch and numpy inputs.
    """
    mn, mx = s.min(), s.max()
    d = mx - mn
    if hasattr(d, "abs"):
        return (s - mn) / d if float(d.abs()) > 1e-12 else s * 0
    return (s - mn) / d if abs(d) > 1e-12 else s * 0


# ──────────────────────────── score combination ────────────────────────────
def combine(anchor, members, weights, anchor_w=1.0):
    """Ensemble score = anchor_w·anchor + Σ w_k·member_k.

    anchor and members must already be normalized with the same convention (znorm or mmnorm).
    """
    s = anchor_w * anchor if anchor_w != 1.0 else (anchor.clone() if hasattr(anchor, "clone") else anchor.copy())
    for k, w in weights.items():
        if w and k in members:
            s = s + w * members[k]
    return s


def norm_mass(weights, mass):
    tot = sum(weights.values())
    return {k: round(v * mass / tot, 3) for k, v in weights.items()} if tot > 0 else dict(weights)


# ───────────────────────────── weight search ─────────────────────────────
def coord_ascent(objective, members, grid=None, rounds=4, init=None, verbose=False):
    """Per-axis grid coordinate ascent.

    objective(weights: dict) -> float
    Returns: (weights, best_value)
    """
    grid = grid or DEFAULT_GRID
    w = dict(init) if init else {k: 0.0 for k in members}
    best = objective(w)
    for r in range(rounds):
        improved = False
        for k in members:
            value, g = max((objective({**w, k: g}), g) for g in grid)
            if value > best + 1e-9:
                if verbose:
                    print(f"    round{r} {k:9s} {w[k]} → {g}   obj={value:.5f}")
                w[k], best, improved = g, value, True
        if not improved:
            break
    return w, best


def cosine(a, b, keys, anchor=True):
    """Directional agreement between two weight vectors."""
    pre = [1.0] if anchor else []
    va = np.array(pre + [a.get(k, 0.0) for k in keys])
    vb = np.array(pre + [b.get(k, 0.0) for k in keys])
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))


def fmt(w, keys=None):
    keys = keys or list(w)
    return "{" + ", ".join(f"{k} {w.get(k, 0.0):.3f}" for k in keys) + "}"


# ══════════════════════════════════ CLI ══════════════════════════════════
def _run_rerank(args):
    """Rerank layer — search comb weights on the top-20 pools of the external benches
    (RSTPReid, UCA, UCC)."""
    from comb_search import Bench, SETS, MEMBERS, DEPLOYED, SIM_W, GRID   # data loading, scoring and grid

    benches = [Bench(n, c) for n, c in SETS.items()]
    if args.objective == "self":
        benches = [b for b in benches if b.has_self()]
    getm = (lambda b, w: b.score(w)[0]) if args.objective == "multi" else (lambda b, w: b.score_self(w)[0])
    base_val = {b.name: getm(b, {}) for b in benches}
    print(f"[rerank] benches {[b.name for b in benches]} · objective {args.objective}")

    def objective(w):
        return float(np.mean([getm(b, w) / base_val[b.name] for b in benches]))

    w, _ = coord_ascent(objective, MEMBERS, grid=GRID, rounds=args.rounds, verbose=True)
    wn = norm_mass(w, args.mass)
    print(f"  searched            {fmt(w, MEMBERS)}  mass={sum(w.values()):.2f}")
    print(f"  mass-{args.mass} normalized  {fmt(wn, MEMBERS)}")
    print(f"  cosine vs deployed  {cosine(w, DEPLOYED, MEMBERS, anchor=False):.4f}")
    return {"layer": "rerank", "objective": args.objective, "sim": SIM_W,
            "weights": w, "weights_massnorm": wn, "deployed": DEPLOYED}


def _run_base(args):
    """Base layer — search encoder ensemble weights on full-gallery UCC ranking."""
    from base import UCCBench, ENCODERS, DEPLOYED, ANCHOR

    bench = UCCBench(args.objective)
    print(f"[base] UCC N={bench.N} · anchor {ANCHOR} · objective {args.objective}")

    def objective(w):
        return bench.metrics(w)[0]

    w, best = coord_ascent(objective, ENCODERS, rounds=args.rounds, verbose=True)
    print(f"  anchor only       mAP@10 {bench.metrics({})[0]:.3f}")
    print(f"  deployed          mAP@10 {bench.metrics(DEPLOYED)[0]:.3f}   {fmt(DEPLOYED, ENCODERS)}")
    print(f"  cosine vs deployed  {cosine(w, DEPLOYED, ENCODERS):.4f}")
    if args.score:
        import torch
        torch.save(mmnorm(bench.combine(w)).cpu(), args.score)
        print(f"  ensemble score saved -> {args.score}  (UCC {bench.N}x{bench.N})")
    return {"layer": "base", "objective": args.objective, "anchor": ANCHOR,
            "weights": w, "deployed": DEPLOYED}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    ap = argparse.ArgumentParser(description="Ensemble weights on external datasets")
    ap.add_argument("--layer", choices=["base", "rerank"], required=True)
    ap.add_argument("--objective", choices=["self", "multi"], default="self",
                    help="self = 1:1 single ground truth (same structure as PAB, default) · multi = the bench native multi-positive")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--mass", type=float, default=1.4,
                    help="target total mass for rerank-layer member weights")
    ap.add_argument("--out", default=None, help="path to save the weights JSON")
    ap.add_argument("--score", default=None,
                    help="path to save the ensemble score built from the weights (base layer)")
    args = ap.parse_args()

    res = _run_base(args) if args.layer == "base" else _run_rerank(args)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"  weights saved -> {args.out}")
