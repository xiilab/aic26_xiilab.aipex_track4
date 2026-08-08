#!/usr/bin/env python3
"""comb fusion weight selection on external datasets.

Usage:
    python tools/ensemble/comb_search.py
    python tools/ensemble/comb_search.py --out w.json

Environment: EXT_DATA (cache directory, default ../benches)
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble import coord_ascent as _coord_ascent, cosine, norm_mass, znorm
from adopted import comb as _adopted_comb          # single source of adopted weights = weights/final.json

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("EXT_DATA", os.path.join(HERE, "..", "..", "assets", "data", "benches"))   # per-bench subdirs: rstp/ uca/ ucc/

def _d(fn):
    """Prefer `$EXT_DATA/<bench>/<file>`, else `$EXT_DATA/<file>`.
    The filename prefix (rstp_/uca_/ucc_) names the bench folder."""
    cands = (os.path.join(DATA, fn.split("_", 1)[0], fn), os.path.join(DATA, fn))
    return next((c for c in cands if os.path.exists(c)), cands[-1])

MEMBERS = ["internvl_r32", "pixtral", "qwen3vl_2b", "llama"]          # full coverage on every bench
MEMBERS_8B = MEMBERS + ["8b"]

def _deployed(variant="best"):
    """Convert the adopted comb (weights/final.json) to the comb_search form (members only).

    Read from config rather than hard-coded, so the current member weights carry over as is."""
    c = _adopted_comb(variant)                          # {sim, internvl_r32, pixtral, qwen3vl_2b, llama, 8b}
    return {k: v for k, v in c.items() if k != "sim"}

SIM_W = _adopted_comb("best").get("sim", 0.55)         # the base (sim) weight also comes from config
DEPLOYED = _deployed("best")                            # adopted (best) comb, from the single config source
GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

DORA_TOPK = int(os.environ.get("DORA_TOPK", "0"))       # 0 = no masking

SETS = {
    "RSTPReid": dict(pool="rstp_pool.pt", files={
        "internvl_r32": "rstp_internvl_r32_scores.npy", "pixtral": "rstp_pixtral_scores.npy",
        "qwen3vl_2b": "rstp_qwen3vl_2b_ex007000_scores.npy", "llama": "rstp_llama_scores.npy",
        "8b": "rstp_8b_scores.npy"}),
    "UCA": dict(pool="uca_pool.pt", files={
        "internvl_r32": "uca_internvl_r32_scores.npy", "pixtral": "uca_pixtral_scores.npy",
        "qwen3vl_2b": "uca_qwen3vl_2b_ex007000_scores.npy", "llama": "uca_llama_scores.npy",
        "8b": "uca_8b_scores.npy"}),
    "UCC": dict(pool="ucc_champion_pool.pt", files={
        "internvl_r32": "ucc_internvl_r32_scores.npy", "pixtral": "ucc_pixtral_scores.npy",
        "qwen3vl_2b": "ucc_qwen3vl_2b_ex007000_scores.npy", "llama": "ucc_llama_scores.npy",
        "8b": "ucc_8b_scores.npy"}),
}


class Bench:
    """One bench = base z + member z + multi-positive labels. Recomputes mAP@10 from weights quickly."""

    def __init__(self, name, cfg, members=None):
        members = members or MEMBERS
        pool = torch.load(_d(cfg["pool"]), weights_only=False)
        base = np.asarray(pool["base_score"], float)
        label = np.asarray([np.asarray(l) for l in pool["label"]], bool)
        mem, cov = {}, {}
        N = len(base)
        for k, f in cfg["files"].items():
            if k not in members:
                continue
            p = _d(f)
            if not os.path.exists(p):
                continue
            a = np.load(p)
            if k == "dora" and DORA_TOPK:               # reproduce the current coverage: zero outside the top K
                m = np.zeros_like(a); m[:, :DORA_TOPK] = a[:, :DORA_TOPK]; a = m
            mem[k] = a
            cov[k] = len(a)
            N = min(N, len(a))                      # truncate to the coverage shared by all members
        self.name, self.N = name, N
        self.zbase = znorm(base[:N])
        self.zmem = {k: znorm(v[:N]) for k, v in mem.items()}
        self.label = label[:N]
        self.npos = self.label.sum(1)
        self.cov = cov
        self.K = self.zbase.shape[1]
        # 1:1 self-image labels (same structure as PAB): each query's own image is the only positive.
        cand = np.asarray(pool["cand"])[:N]
        self.keep = None
        if name == "UCC":
            tgt = np.arange(N)
        elif name == "RSTPReid" and os.path.exists(_d("rstp_q_src.npy")):
            tgt = np.load(_d("rstp_q_src.npy"))[:N]
            seen = set()
            self.keep = np.array([j for j in range(N) if not (tgt[j] in seen or seen.add(tgt[j]))])
        else:
            tgt = None
        self.tgt = tgt
        self.label_self = None if tgt is None else (cand == tgt[:, None])
        self.disc = 1.0 / np.arange(1, min(10, self.K) + 1)

    def subset(self, idx):
        """View over a subset of queries. Labels and scores are sliced by idx and duplicate 1:1 targets
        are recomputed."""
        import copy
        o = copy.copy(self)
        idx = np.asarray(idx)
        o.N = len(idx)
        o.zbase = self.zbase[idx]
        o.zmem = {k: v[idx] for k, v in self.zmem.items()}
        o.label = self.label[idx]
        o.npos = self.npos[idx]
        if self.label_self is None:
            o.label_self, o.keep, o.tgt = None, None, None
        else:
            o.label_self = self.label_self[idx]
            o.tgt = self.tgt[idx]
            if self.keep is None:
                o.keep = None
            else:                                  # rebuild the distinct-target list
                seen = set()
                o.keep = np.array([j for j in range(o.N) if not (o.tgt[j] in seen or seen.add(o.tgt[j]))])
        return o

    def members(self):
        return [k for k in MEMBERS_8B if k in self.zmem]

    def _order(self, w):
        c = SIM_W * self.zbase
        for k, wk in w.items():
            if wk and k in self.zmem:
                c = c + wk * self.zmem[k]
        return np.argsort(-c, axis=1, kind="stable")

    def score(self, w):
        """multi-positive mAP@10 (normalized by min(npos, 10)) and R@1."""
        lab = np.take_along_axis(self.label, self._order(w), axis=1)[:, :10]
        hits = np.cumsum(lab, axis=1)
        ap = (lab * hits * self.disc).sum(1) / np.maximum(np.minimum(self.npos, 10), 1)
        return 100 * ap.mean(), 100 * lab[:, 0].mean()

    def score_self(self, w):
        """1:1 self-image mAP@10 and R@1, or None when unavailable."""
        if self.label_self is None:
            return None
        lab = np.take_along_axis(self.label_self, self._order(w), axis=1)[:, :10]
        if self.keep is not None:
            lab = lab[self.keep]
        rr = (lab * self.disc).sum(1)          # single positive, so AP = 1/rank
        return 100 * rr.mean(), 100 * lab[:, 0].mean()

    def has_self(self):
        return self.label_self is not None


def coord_ascent(benches, rounds=4, verbose=False, members=None, objective="multi"):
    members = members or MEMBERS
    sc = (lambda b, w: b.score(w)[0]) if objective == "multi" else (lambda b, w: b.score_self(w)[0])
    base_map = {b.name: sc(b, {}) for b in benches}
    obj = lambda w: float(np.mean([sc(b, w) / base_map[b.name] for b in benches]))
    w, best = _coord_ascent(obj, members, grid=GRID, rounds=rounds, verbose=verbose)
    return w, best, base_map


def _cos(a, b):
    return cosine(a, b, MEMBERS_8B, anchor=False)


def fmt(w, members=None):
    return "{" + ", ".join(f"{k} {w.get(k, 0.0):.2f}" for k in (members or MEMBERS)) + "}"


def report(tag, w, benches, base_map):
    row = []
    for b in benches:
        m, r1 = b.score(w)
        row.append(f"{b.name} {m:6.2f}({m - base_map[b.name]:+5.2f})")
    print(f"  {tag:26s} {fmt(w)}   " + " · ".join(row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="path to save the searched weights JSON")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--dora-topk", type=int, default=None,
                    help="mask DoRA as if only the top K candidates had been scored")
    ap.add_argument("--mass", type=float, default=1.4,
                    help="target total mass for the member weights (adopted = 0.7+0.4+0.2+0.1 = 1.4)")
    a = ap.parse_args()
    if a.dora_topk is not None:
        globals()["DORA_TOPK"] = a.dora_topk
    if DORA_TOPK:
        print(f"[DoRA masking] treating only the top {DORA_TOPK} candidates as scored")

    benches = [Bench(n, c) for n, c in SETS.items()]

    for b in benches:
        print(f"  {b.name:10s} N={b.N:5d} (member coverage {b.cov})")

    base_map = {b.name: b.score({})[0] for b in benches}
    print("\n[baseline] S1 base mAP@10: " + " · ".join(f"{k} {v:.2f}" for k, v in base_map.items()))

    print("\n[1] reference — score the adopted weights (config) on the external benches")
    report(f"deployed (config {fmt(DEPLOYED)})", DEPLOYED, benches, base_map)

    print("\n[2] all three external benches (coordinate ascent)")
    w_all, obj_all, _ = coord_ascent(benches, a.rounds, verbose=True)
    report("★ (RSTP+UCA+UCC)", w_all, benches, base_map)
    print(f"     objective {obj_all:.5f} · cosine vs deployed(best) {_cos(w_all, DEPLOYED):.4f}")

    print("\n[3] search on a single bench (as if only one dataset were available)")
    singles = {}
    for b in benches:
        w1, _, _ = coord_ascent([b], a.rounds)
        singles[b.name] = w1
        report(f"searched ({b.name} only)", w1, benches, base_map)
        print(f"     cosine vs deployed(best) {_cos(w1, DEPLOYED):.4f}")

    print("\n[4] leave-one-dataset-out (search on two, verify on the third)")
    lodo = {}
    for b in benches:
        rest = [x for x in benches if x is not b]
        w2, _, _ = coord_ascent(rest, a.rounds)
        lodo[b.name] = w2
        m, _ = b.score(w2); md, _ = b.score(DEPLOYED)
        print(f"  hold-out {b.name:10s} {fmt(w2)}  -> {b.name} mAP {m:6.2f} "
              f"(deployed {md:6.2f}, Δ {m - md:+.2f}) · cosine {_cos(w2, DEPLOYED):.4f}")

    selfable = [b for b in benches if b.has_self()]
    print(f"  1:1 scoring available on: {[b.name for b in selfable]} (UCA has no self mapping)")
    w_self, _, _ = coord_ascent(selfable, a.rounds, objective="self")
    for b in selfable:
        m_d = b.score_self(DEPLOYED)[0]; m_w = b.score_self(w_self)[0]
        print(f"    {b.name:10s} 1:1 mAP@10  {m_w:6.2f}  vs deployed {m_d:6.2f}")
    print(f"  ★ {fmt(w_self)}  mass={sum(w_self.values()):.2f} "
          f"· cosine {_cos(w_self, DEPLOYED):.4f}")
    w_self_n = norm_mass(w_self, a.mass)
    print(f"     mass-{a.mass} normalized -> {fmt(w_self_n)}")
    w_all_n = norm_mass(w_all, a.mass)
    print(f"     (multi normalized -> {fmt(w_all_n)})")

    b8 = [Bench(n, c, MEMBERS_8B) for n, c in SETS.items()]
    for b in b8:
        print(f"  {b.name:10s} N={b.N:5d}")
    w8, _, bm8 = coord_ascent(b8, a.rounds, members=MEMBERS_8B)
    print(f"  searched (+8B)             {fmt(w8, MEMBERS_8B)}   " +
          " · ".join(f"{b.name} {b.score(w8)[0]:6.2f}" for b in b8))
    print(f"     -> 8B weight {w8.get('8b', 0.0):.2f} (deployed 0.00)")

    print(f"  COMB_W='{{\"sim\":{SIM_W}," + ",".join(f'\"{k}\":{v}' for k, v in w_self_n.items()) + "}'")

    out = {"derived_external_all": w_all, "derived_self_objective": w_self,
           "derived_self_objective_massnorm": w_self_n, "derived_multi_massnorm": w_all_n,
           "derived_with_8b": w8, "single": singles, "lodo": lodo,
           "deployed_best": DEPLOYED, "sim": SIM_W}
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nsaved -> {a.out}")
