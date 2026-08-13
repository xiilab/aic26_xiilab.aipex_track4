#!/usr/bin/env python3
"""Encoder ensemble (S1 base) weight search — objective design exploration.

`base.py` picks a bench and searches on it. This script explores the step before that: how the
objective should be composed. Four axes:

  1. lambda : weighted mix of the UCC (1:1) and UCA (same-video) objectives
  2. gallery: merge other datasets' galleries in as distractors to increase candidate competition
  3. ratio  : keep the gallery at full size and shrink the query set to approach the PAB
              query:gallery ratio of 1:18.6
  4. hard   : search only on queries the anchor (v20) alone gets wrong at top-1

Extra options: extend the grid upper bound (--grid-max) and report mass-normalized weights.

Usage:
    ENS_DEV=cuda:0 python tools/ensemble/base_search.py --mode all
    ENS_DEV=cuda:0 python tools/ensemble/base_search.py --mode hard --out w.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble import coord_ascent, cosine, fmt, mmnorm, norm_mass   # noqa: E402
import base as D                                                    # noqa: E402
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root; all default paths are relative to it

DEV, ENC, DEP, ANC, TTA = D.DEV, D.ENCODERS, D.DEPLOYED, D.ANCHOR, D.TTA
FEATS = os.environ.get("UCC_FEATS", f"{_REPO}/assets/data/benches")
DISC = 1.0 / torch.arange(1, 11, device=DEV).float()
DEP_MASS = sum(DEP.values())


def emb(prefix, name):
    """Unit-normalized (Q, G) embeddings. For a TTA view dump, the views are averaged."""
    d = torch.load(D.bench_path(f"{prefix}_{name}_feats.pt", prefix), map_location="cpu", weights_only=False)
    if "img" in d:
        v = [x for x in TTA[name] if x in d["img"]]
        G = F.normalize(torch.stack([F.normalize(d["img"][x].float(), dim=-1) for x in v], 0).mean(0), dim=-1)
        Q = F.normalize(d["txt"]["base"].float(), dim=-1)
    else:
        G, Q = F.normalize(d["G"].float(), dim=-1), F.normalize(d["Q"].float(), dim=-1)
    return Q.to(DEV), G.to(DEV)


def build_scores(distractors=()):
    """Per-encoder score matrix: UCC queries x [UCC gallery + distractor galleries]."""
    S = {}
    for name in [ANC] + ENC:
        Q, G = emb("ucc", name)
        S[name] = mmnorm(Q @ torch.cat([G] + [emb(p, name)[1] for p in distractors], 0).t())
    return S


def self_map(S, idx=None):
    """1:1 self-image mAP@10 objective (the positive for UCC query i is gallery item i)."""
    tgt = torch.arange(S[ANC].shape[0], device=DEV) if idx is None else idx

    def metric(w):
        s = S[ANC][tgt]
        for k, v in w.items():
            if v:
                s = s + v * S[k][tgt]
        lab = (s.topk(10, dim=1).indices == tgt[:, None]).float()
        return float(100 * (lab * DISC).sum(1).mean())
    return metric


def hard_idx(S):
    """Queries the anchor alone fails to rank at top-1."""
    n = S[ANC].shape[0]
    all_i = torch.arange(n, device=DEV)
    return torch.nonzero(S[ANC].topk(1, dim=1).indices.squeeze(1) != all_i).squeeze(1)


def main():
    ap = argparse.ArgumentParser(description="Encoder ensemble objective design exploration")
    ap.add_argument("--mode", default="all", choices=["all", "lambda", "gallery", "ratio", "hard"])
    ap.add_argument("--distractors", default="uca,rstp", help="datasets to merge into the gallery (comma-separated; empty means UCC only)")
    ap.add_argument("--grid-max", type=float, default=1.0, help="grid upper bound (values >1 append 1.2/1.5/2/2.5/3)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dist = tuple(x for x in a.distractors.split(",") if x)
    grid = D.GRID + ([1.2, 1.5, 2.0, 2.5, 3.0] if a.grid_max > 1.0 else [])
    tb = None
    print(f"baseline: deployed weights {fmt(DEP, ENC)}")

    results = {}

    def report(tag, w, S, extra=""):
        line = (f"  {tag:32s} {fmt(w, ENC):58s} cos {cosine(w, DEP, ENC):.4f}"
                f"/{cosine(w, DEP, ENC, anchor=False):.4f}(members only)")
        if tb:
            line += f"  test base {tb(w):.3f}"
        print(line + extra)
        results[tag] = w
        return w

    if a.mode in ("all", "lambda"):
        print("\n[1] objective weight lambda — UCC (1:1) vs UCA (same-video)")
        ucc, uca = D.UCCBench("self"), D.UCABench()
        a_ucc, a_uca = ucc.metrics({})[0], uca.metrics({})[0]
        for lam in (0.0, 0.2, 0.5, 1.0):
            obj = lambda w, l=lam: (1 - l) * ucc.metrics(w)[0] / a_ucc + l * uca.metrics(w)[0] / a_uca  # noqa: E731
            report(f"lambda={lam:.1f} (0=UCC only, 1=UCA only)", coord_ascent(obj, ENC, grid=grid, rounds=a.rounds)[0], None)
        del ucc, uca
        torch.cuda.empty_cache()

    if a.mode in ("all", "gallery", "ratio", "hard"):
        S_plain = build_scores()
        if a.mode in ("all", "gallery"):
            print("\n[2] gallery expansion (UCC queries fixed, distractors added)")
            report(f"UCC gallery only ({S_plain[ANC].shape[1]})",
                   coord_ascent(self_map(S_plain), ENC, grid=grid, rounds=a.rounds)[0], S_plain)
        S = build_scores(dist) if dist else S_plain
        G = S[ANC].shape[1]
        if a.mode in ("all", "gallery") and dist:
            report(f"+{'+'.join(dist)} distractor ({G})",
                   coord_ascent(self_map(S), ENC, grid=grid, rounds=a.rounds)[0], S)

        if a.mode in ("all", "ratio"):
            print(f"\n[3] query:gallery ratio (gallery {G} fixed, queries shrunk; PAB is 1:18.6)")
            rng = np.random.default_rng(0)
            for n in (2000, 1000, max(1, G // 18)):
                idx = torch.tensor(rng.choice(S[ANC].shape[0], n, replace=False), device=DEV)
                report(f"queries {n} (1:{G / n:.1f})",
                       coord_ascent(self_map(S, idx), ENC, grid=grid, rounds=a.rounds)[0], S)

        if a.mode in ("all", "hard"):
            hi = hard_idx(S)
            print(f"\n[4] hard cases — anchor {ANC} alone fails top-1 on {len(hi)}/{S[ANC].shape[0]} queries")
            w = coord_ascent(self_map(S, hi), ENC, grid=grid, rounds=a.rounds)[0]
            report("hard only", w, S)
            wn = norm_mass(w, DEP_MASS)
            report(f"hard + mass-{DEP_MASS:.2f} normalized", wn, S)

    if a.out:
        json.dump({"deployed": DEP, "results": results}, open(a.out, "w"), indent=2, default=float)
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
