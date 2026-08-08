#!/usr/bin/env python3
"""Encoder ensemble (S1 base) weight selection.

Usage:
    ENS_DEV=cuda:0 python tools/ensemble/base.py --sets ucc --objective self
    ENS_DEV=cuda:0 python tools/ensemble/base.py --sets ucc,uca --out tools/ensemble/weights/search/base.json

Environment: UCC_FEATS (encoder feats directory, default assets/data/benches) · ENS_DEV (default cuda:0)
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble import combine, coord_ascent as _coord_ascent, cosine, mmnorm
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root; all default paths are relative to it

DEV = os.environ.get("ENS_DEV", "cuda:0")
UCC = os.environ.get("UCC_FEATS", f"{_REPO}/assets/data/benches")     # uca_*_feats.pt is read from here too
ANCHOR = "anchor_filip"
ENCODERS = ["anchor_tcap", "metaclip2", "mc2h378_peft", "gme", "eva02_pre"]            # search targets, anchor excluded
DEPLOYED = {"anchor_tcap": 0.333, "metaclip2": 0.30, "mc2h378_peft": 0.25, "gme": 0.10, "eva02_pre": 0.08}
GRID = [0.0, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]
TTA = {"anchor_tcap": ("base", "hflip", "z090"), "anchor_filip": ("base", "hflip", "z080")}


def enc_scores(prefix="ucc"):
    """Per-encoder [Q, G] score matrix, min-max normalized. prefix = ucc | uca."""
    out = {}
    for name in [ANCHOR] + ENCODERS:
        d = torch.load(f"{UCC}/{prefix}_{name}_feats.pt", map_location="cpu", weights_only=False)
        if "img" in d:                                  # TTA view dump
            views = [v for v in TTA[name] if v in d["img"]]
            Qb = F.normalize(d["txt"]["base"].float(), dim=-1).to(DEV)
            Gc = F.normalize(torch.stack([F.normalize(d["img"][v].float(), dim=-1) for v in views], 0).mean(0), dim=-1).to(DEV)
        else:                                           # {G, Q} feats
            Qb = F.normalize(d["Q"].float(), dim=-1).to(DEV)
            Gc = F.normalize(d["G"].float(), dim=-1).to(DEV)
        out[name] = mmnorm(Qb @ Gc.t())
        del Qb, Gc
    return out


class UCCBench:
    """UCC — gallery 5,320 = query 5,320. objective: self (1:1) | multi (same-video)."""
    name = "UCC"

    def __init__(self, objective="self"):
        pool = torch.load(f"{UCC}/ucc_champion_pool.pt", weights_only=False)
        self.pids = torch.tensor(np.asarray(pool["pids"]), device=DEV)
        self.S = enc_scores("ucc")
        self.N = self.S[ANCHOR].shape[0]
        self.objective = objective
        self.disc = (1.0 / torch.arange(1, 11, device=DEV).float())

    def combine(self, w):
        return combine(self.S[ANCHOR], self.S, w)

    def metrics(self, w, objective=None):
        objective = objective or self.objective
        top = self.combine(w).topk(10, dim=1).indices                    # [N,10]
        if objective == "self":                                          # 1:1 (caption -> its own frame)
            lab = (top == torch.arange(self.N, device=DEV)[:, None]).float()
            ap, npos = (lab * self.disc).sum(1), torch.ones(self.N, device=DEV)
        else:                                                            # same-video multi-positive
            lab = (self.pids[top] == self.pids[:, None]).float()
            ap = (lab * lab.cumsum(1) * self.disc).sum(1)
            npos = torch.clamp(torch.bincount(self.pids, minlength=int(self.pids.max()) + 1)[self.pids].float(), max=10)
        return float(100 * (ap / npos).mean()), float(100 * lab[:, 0].mean())


class UCABench:
    """UCA — query 2,000 / gallery 4,267. Ground truth = same video (multi-positive),
    ranked over the full gallery. Query and gallery video ids come from uca_bench.json,
    whose order matches uca_pool.pt."""
    name = "UCA"

    def __init__(self, objective="multi"):
        import json
        b = json.load(open(f"{UCC}/uca_bench.json"))
        gv = {v: i for i, v in enumerate(sorted(set(b["gvideo"])))}
        self.gvid = torch.tensor([gv[v] for v in b["gvideo"]], device=DEV)              # [4267]
        self.qvid = torch.tensor([gv[q["video"]] for q in b["queries"]], device=DEV)    # [2000]
        self.S = enc_scores("uca")
        self.N = self.S[ANCHOR].shape[0]
        self.objective = "multi"          # UCA has no self-image correspondence
        self.disc = (1.0 / torch.arange(1, 11, device=DEV).float())
        self.npos = torch.clamp(torch.bincount(self.gvid, minlength=int(self.gvid.max()) + 1)[self.qvid].float(), max=10)

    def combine(self, w):
        return combine(self.S[ANCHOR], self.S, w)

    def metrics(self, w, objective=None):
        top = self.combine(w).topk(10, dim=1).indices
        lab = (self.gvid[top] == self.qvid[:, None]).float()
        ap = (lab * lab.cumsum(1) * self.disc).sum(1)
        return float(100 * (ap / self.npos).mean()), float(100 * lab[:, 0].mean())


def _metrics_subset(self, w, idx):
    """Score on a query subset (idx). The gallery stays complete."""
    s = self.combine(w)[idx]
    top = s.topk(10, dim=1).indices
    if self.objective == "self":
        lab = (top == idx[:, None]).float()
        ap, npos = (lab * self.disc).sum(1), torch.ones(len(idx), device=DEV)
    else:
        lab = (self.pids[top] == self.pids[idx][:, None]).float()
        ap = (lab * lab.cumsum(1) * self.disc).sum(1)
        npos = torch.clamp(torch.bincount(self.pids, minlength=int(self.pids.max()) + 1)[self.pids[idx]].float(), max=10)
    return float(100 * (ap / npos).mean()), float(100 * lab[:, 0].mean())


UCCBench.metrics_subset = _metrics_subset


def coord_ascent(bench, rounds=4, verbose=True):
    """Build an objective from one or more benches and delegate to the shared coordinate-ascent core.

    With several benches the objective is the mean of (mAP / anchor-only mAP) per bench, which
    combines benches of different scales fairly."""
    benches = bench if isinstance(bench, (list, tuple)) else [bench]
    anc = {b.name: b.metrics({})[0] for b in benches}
    print("    anchor only " + " · ".join(f"{k} {v:.3f}" for k, v in anc.items()))
    obj = (lambda w: benches[0].metrics(w)[0]) if len(benches) == 1 else \
          (lambda w: float(np.mean([b.metrics(w)[0] / anc[b.name] for b in benches])))
    return _coord_ascent(obj, ENCODERS, grid=GRID, rounds=rounds, verbose=verbose)


def _cos(a, b):
    return cosine(a, b, ENCODERS)


def fmt(w):
    return "{" + ", ".join(f"{k} {w.get(k, 0.0):.3f}" for k in ENCODERS) + "}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="ucc", help="benches: ucc | uca | ucc,uca")
    ap.add_argument("--objective", default="self", choices=["self", "multi"],
                    help="UCC scoring mode (UCA is always same-video multi-positive)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sets = [x.strip().lower() for x in a.sets.split(",") if x.strip()]

    benches = []
    for name in sets:
        benches.append(UCCBench(a.objective) if name == "ucc" else UCABench())
    for b in benches:
        ma = b.metrics({k: 0.0 for k in ENCODERS})[0]; md = b.metrics(DEPLOYED)[0]
        print(f"  {b.name:4s} N={b.N:5d}   anchor {ANCHOR} only {ma:6.3f}   deployed {md:6.3f}")
    w, _ = coord_ascent(benches, a.rounds)
    for b in benches:
        mw, rw = b.metrics(w); md = b.metrics(DEPLOYED)[0]
        print(f"  {b.name:4s} searched {mw:6.3f} (vs deployed {mw - md:+.3f})  R@1 {rw:6.3f}")
    print(f"  ★ searched weights  {fmt(w)}   cosine vs deployed {_cos(w, DEPLOYED):.4f}")

    if a.out:
        json.dump({"sets": sets, "objective": a.objective, "derived": w, "deployed": DEPLOYED},
                  open(a.out, "w"), indent=2)
        print(f"\nsaved -> {a.out}")
