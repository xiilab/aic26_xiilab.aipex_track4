#!/usr/bin/env python3
"""eval_heldout_rerank — measure each fine-tuned reranker's contribution on the held-out bench.

  base(top-20) → comb = SIM_W·z(sim) + Σ w_k·z(rerank_k) → injective → mAP@10 / R@1
  solo = sim plus a single reranker at a shared weight   ·   LOO = comb with one member removed

The bench is PAB rule-clean (`assets/data/benches/ruleclean/`, overridable with `RC_DIR`). Only
precomputed caches are read, so no GPU is needed and no Track 4 test label is touched. Assignment and
softmax come from `pipeline/S3_assign/assign.py` unchanged.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [os.path.join(ROOT, "pipeline", "S3_assign"), os.path.join(ROOT, "pipeline")]
from assign import injective_assign, softmax_conf                    # noqa: E402
from utils import gallery_norm as GN                                            # noqa: E402

RC_DIR = os.environ.get("RC_DIR", os.path.join(ROOT, "assets", "data", "benches", "ruleclean"))
BENCH = os.environ.get("RC_BENCH", f"{RC_DIR}/pab_ruleclean_bench.json")
BASE = os.environ.get("RC_BASE", f"{RC_DIR}/greedy_R10_v28_rc_exact_base.pt")
POOL, SIM_W, INJ_T = 20, 0.55, 1.0
FT = ("internvl_r32", "qwen3vl_2b", "jina_m0")     # the others (pixtral, llama, 8b, ovis) are zero-shot, so there is no ckpt to select
PREFIX, SUFFIX = "redump_", "_ruleclean.pt"


def discover(rc_dir):
    """`redump_<name>_ruleclean.pt` → {name: path}. A newly dumped step is picked up automatically."""
    return {os.path.basename(p)[len(PREFIX):-len(SUFFIX)]: p
            for p in sorted(glob.glob(os.path.join(rc_dir, f"{PREFIX}*{SUFFIX}")))}


def z(v):
    a = np.asarray(v, float)
    s = a.std()
    return (a - a.mean()) / s if s > 1e-9 else a * 0.0


def load(paths):
    B = json.load(open(BENCH))
    gidx = {p: i for i, p in enumerate(B["gallery"])}
    gt = np.array([gidx[q["img"]] for q in B["queries"]])
    qimg = [q["img"] for q in B["queries"]]
    Q = len(gt)
    bs = GN.normalize(torch.load(BASE, map_location="cpu", weights_only=False)).float().numpy()[:Q]
    cand = [np.argsort(-bs[i])[:POOL].tolist() for i in range(Q)]
    zsim = [z([bs[i][c] for c in cand[i]]) for i in range(Q)]
    zr, cov = {}, {}
    for k, p in paths.items():
        sc = torch.load(p, weights_only=False)["scores"]
        zr[k] = [z([sc.get((qimg[i], int(c)), 0.0) for c in cand[i]]) for i in range(Q)]
        cov[k] = float(np.mean([sum((qimg[i], int(c)) in sc for c in cand[i]) / POOL for i in range(Q)]))
    return dict(Q=Q, gt=gt, cand=cand, zsim=zsim, zr=zr, cov=cov, gallery=len(B["gallery"]))


def ranks(D, w):
    """comb → injective assignment → the GT rank per query (0-based; 99 when outside the top 10)."""
    Q = D["Q"]
    order, conf = [None] * Q, np.zeros(Q)
    for i in range(Q):
        c = SIM_W * D["zsim"][i]
        for k, v in w.items():
            if v and k in D["zr"]:
                c = c + v * D["zr"][k][i]
        c = np.asarray(c, float)
        srt = np.argsort(-c, kind="stable")
        order[i] = [D["cand"][i][j] for j in srt]
        conf[i] = softmax_conf(c, INJ_T)
    fin = injective_assign(order, conf, 10)
    return np.array([next((j for j, x in enumerate(fin[i]) if x == D["gt"][i]), 99) for i in range(Q)])


def metrics(rk):
    return (100 * np.mean(np.where(rk < 10, 1.0 / (rk + 1), 0.0)), 100 * np.mean(rk == 0))


def run(D, w):
    return metrics(ranks(D, w))


def parse_comb(s, mem):
    """Parse "internvl_r32=0.7,qwen3vl_2b=0.2" into a weight dict; unlisted members get 0."""
    w = {k: 0.0 for k in mem}
    for part in s.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in mem:
            raise SystemExit(f"'{k}' in --comb-w is not being evaluated — members are {mem}")
        w[k] = float(v)
    return w


def main():
    avail = discover(RC_DIR)
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default=",".join(FT),
                    help=f"members to evaluate (default = the fine-tuned set {list(FT)} · `all` = every cache present). found {list(avail)}")
    ap.add_argument("--solo-w", type=float, default=0.7, help="shared weight used to measure each member's solo contribution")
    ap.add_argument("--comb-w", default=None, metavar="K=V,…",
                    help="explicit comb weights for the comb and LOO sections (default = --uniform-w for every member); useful for comparing against the deployed weights")
    ap.add_argument("--uniform-w", type=float, default=0.3, help="per-member weight of the default comb")
    ap.add_argument("--sweep", default=None, help="sweep this member's comb weight (e.g. jina)")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="bootstrap the queries N times and report the noise floor (the smallest resolvable difference)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --bootstrap")
    ap.add_argument("--out", default=None, help="path for the result JSON")
    a = ap.parse_args()

    if not avail:
        raise SystemExit(f"{RC_DIR} holds no {PREFIX}*{SUFFIX} cache (set RC_DIR)")
    mem = list(avail) if a.members == "all" else [m.strip() for m in a.members.split(",") if m.strip()]
    if bad := [m for m in mem if m not in avail]:
        raise SystemExit(f"no cache for {bad} — found in {RC_DIR}: {list(avail)}")

    D = load({k: avail[k] for k in mem})
    print(f"[RC] Q={D['Q']} · gallery {D['gallery']} · coverage "
          + " ".join(f"{k} {100 * v:.0f}%" for k, v in D['cov'].items()), flush=True)

    res = {"bench": "pab_ruleclean", "members": mem, "coverage": D["cov"]}
    b = run(D, {})
    res["base"] = {"mAP@10": b[0], "R@1": b[1]}
    print(f"\n{'configuration':30s}{'mAP@10':>9}{'R@1':>9}{'Δ':>9}")
    print(f"{'base only (no reranker)':30s}{b[0]:>9.3f}{b[1]:>9.3f}{'—':>9}")

    print(f"\n[1] solo contribution — the axis used for ckpt selection: sim {SIM_W} + member {a.solo_w}")
    res["solo"] = {}
    for k in mem:
        m, r = run(D, {k: a.solo_w})
        res["solo"][k] = {"mAP@10": m, "R@1": r, "delta": m - b[0]}
        print(f"  {k:28s}{m:>9.3f}{r:>9.3f}{m - b[0]:>+9.3f}")

    if a.comb_w:
        W, note = parse_comb(a.comb_w, mem), "explicit comb ⚠ if these are the deployed weights they derive from test data — not valid selection evidence"
    else:
        W, note = {k: a.uniform_w for k in mem}, "uniform comb (test-free)"
    full = run(D, W)
    res["comb"] = {"weights": W, "mAP@10": full[0], "R@1": full[1], "delta": full[0] - b[0]}
    tag = " + ".join(f"{k} {v}" for k, v in W.items() if v) or "(all zero)"
    print(f"\n[2] {note}\n  {tag:28s}{full[0]:>9.3f}{full[1]:>9.3f}{full[0] - b[0]:>+9.3f}")

    print("\n[3] LOO — removed from the comb above")
    res["loo"] = {}
    for k in [x for x in W if W[x]]:
        w = dict(W); w[k] = 0.0
        m, r = run(D, w)
        res["loo"][k] = {"mAP@10": m, "R@1": r, "delta": m - full[0]}
        print(f"  −{k:27s}{m:>9.3f}{r:>9.3f}{m - full[0]:>+9.3f}")

    if a.sweep and a.sweep in mem:
        print(f"\n[4] weight sweep for {a.sweep}")
        res["sweep"] = {}
        for v in [0.0, 0.1, 0.2, 0.4, 0.7]:
            w = dict(W); w[a.sweep] = v
            m, r = run(D, w)
            res["sweep"][str(v)] = {"mAP@10": m, "R@1": r}
            print(f"  {a.sweep}={v:<26}{m:>9.3f}{r:>9.3f}{m - full[0]:>+9.3f}")

    if a.bootstrap:
        # An approximation: resampling happens **after** assignment, and injective assignment couples
        # the queries, so the true SE is larger than this.
        rng = np.random.default_rng(a.seed)
        rk_full, best = ranks(D, W), max(mem, key=lambda k: res["solo"][k]["mAP@10"])
        rk_solo = {k: ranks(D, {k: a.solo_w}) for k in mem}
        idx = rng.integers(0, D["Q"], size=(a.bootstrap, D["Q"]))
        bm = np.array([metrics(rk_full[i])[0] for i in idx])
        se = float(bm.std(ddof=1))
        print(f"\n[5] noise floor — {a.bootstrap} query bootstrap rounds")
        print(f"  mAP@10 = {full[0]:.3f} · SE {se:.3f} · 95% CI "
              f"[{np.percentile(bm, 2.5):.3f}, {np.percentile(bm, 97.5):.3f}]")
        print(f"  1 query = {100 / D['Q']:.3f}pp (R@1) · resolvable difference ≈ 2·SE = {2 * se:.3f}")
        res["bootstrap"] = {"n": a.bootstrap, "se_mAP10": se,
                            "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
                            "min_detectable": 2 * se, "pairs": {}}
        for k in mem:                       # paired differences share the same queries, so their SE is much smaller
            if k == best:
                continue
            d = np.array([metrics(rk_solo[best][i])[0] - metrics(rk_solo[k][i])[0] for i in idx])
            lo, hi = np.percentile(d, [2.5, 97.5])
            res["bootstrap"]["pairs"][f"{best}-{k}"] = {"mean": float(d.mean()),
                                                        "ci95": [float(lo), float(hi)],
                                                        "significant": bool(lo > 0)}
            print(f"  solo Δ({best} − {k}) = {d.mean():+.3f} · 95% CI [{lo:+.3f}, {hi:+.3f}]"
                  f"  {'significant' if lo > 0 else 'not resolvable'}")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print(f"\n[save] {a.out}", flush=True)


if __name__ == "__main__":
    main()
