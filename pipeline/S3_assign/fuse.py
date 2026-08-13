#!/usr/bin/env python3
"""S3 fusion — cached base score + reranker rescores → a per-query comb vector.

This module is the **front half** of S3 (z-fusion); the back half (injective assignment) is
`assign.py`.

  comb_q = w_sim·z(base_q) + Σ_k w_k·z(rerank_k,q)        (candidates = base top-POOL)

Every operation is **deterministic**: ties go through `argsort(kind="stable")` and uncovered
candidates are pinned by the IMPUTE rule.
"""
import json
import os

import numpy as np
import torch

POOL = 20
MEMBER_W = {"qwen3vl_2b": 0.2, "8b": 0.3, "internvl_r32": 0.7, "pixtral": 0.3, "llama": 0.2}   # BASE=champion only
IV_CACHE = {"internvl_r32": "internvl_r32", "pixtral": "pixtral", "llama": "llama32v"}      # outputs/fuse_internvl/{dir}
REDUMP = {"qwen3vl_2b": "qwen3vl_2b", "8b": "8b", "internvl_r32": "internvl_r32", "pixtral": "pixtral", "llama": "llama"}


def z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v * 0.0


def _order_key(comb, cand):
    """COMB_EPS>0 rounds comb to that grid, so candidates within eps tie and the stable sort falls
    back to the base order. COMB_TIE=col ties by gallery column instead. Unset, ordering is unchanged."""
    eps = float(os.environ.get("COMB_EPS", "0") or 0)
    if eps <= 0:
        return comb
    r = np.round(np.asarray(comb, float) / eps) * eps
    if os.environ.get("COMB_TIE") == "col":
        order = np.argsort(np.argsort(np.asarray(cand)))          # column rank, ascending
        r = r - 1e-9 * order
    return r


def z_masked(vals, impute="zero"):
    """Handle uncovered (None) candidates.

    impute="zero" (default)  — fill 0.0 and z over everything. Identical to "neutral" when the base
                               has 100% coverage.
    impute="neutral"         — take the mean and std over covered values only, leaving uncovered at z=0.
    """
    if impute != "neutral":
        return z([0.0 if v is None else v for v in vals])
    a = np.array([np.nan if v is None else float(v) for v in vals])
    m = ~np.isnan(a)
    if m.sum() == 0:
        return np.zeros_like(a)
    mu, sd = a[m].mean(), a[m].std()
    out = np.zeros_like(a)
    if sd > 1e-9:
        out[m] = (a[m] - mu) / sd
    return out


def load_iv(track4, name):
    """InternVL-family top-20 score cache (.npy + meta.json)."""
    d = f"{track4}/outputs/fuse_internvl/{name}"
    sc = np.load(f"{d}/internvl_scores_{name}_top20.npy")
    cd = np.load(f"{d}/internvl_cand_{name}_top20.npy")
    qo = json.load(open(f"{d}/internvl_scores_{name}_meta.json"))["qorder"]
    iv = {}
    for i, q in enumerate(qo):
        for j in range(20):
            if not np.isnan(sc[i, j]):
                iv[(q, int(cd[i, j]))] = float(sc[i, j])
    return iv


def fuse_r10(track4, qx, weights, softmax_conf, inj_t=1.0, impute="zero"):
    """S1 recall base + the five re-dumped S2 rerankers.

    base      = `greedy_R10_base_score.pt`, the S1 output composed for R@10
    rerankers = `{name}_union_cache.pt`, rescores over the base top-20 pool
    weights   = {"sim","qwen3vl_2b","8b","internvl_r32","pixtral","llama"} — single source is
                `../../tools/ensemble/adopted.py`

    returns (cand_order, combs, conf):
      cand_order[q] comb-descending candidate gallery indices · combs[q] sorted comb values ·
      conf[q] softmax-max
    """
    bscore = torch.load(f"{track4}/greedy_R10_base_score.pt", map_location="cpu",
                        weights_only=False).float().numpy()
    RR = {mk: torch.load(f"{track4}/{fn}_union_cache.pt", weights_only=False)["scores"]
          for mk, fn in REDUMP.items()}
    Q = len(qx)
    cand_order, combs, conf = [None] * Q, [None] * Q, np.zeros(Q)
    for i, q in enumerate(qx):
        cand = np.argsort(-bscore[i])[:POOL].tolist()
        comb = weights["sim"] * z([bscore[i][c] for c in cand])
        for mk in ("qwen3vl_2b", "8b", "internvl_r32", "pixtral", "llama"):
            if weights[mk]:
                comb = comb + weights[mk] * z_masked([RR[mk].get((q, int(c))) for c in cand], impute)
        comb = np.asarray(comb, float)
        srt = np.argsort(-_order_key(comb, cand), kind="stable")
        cand_order[i] = [cand[j] for j in srt]
        combs[i] = comb[srt]
        conf[i] = softmax_conf(comb, inj_t)
    return cand_order, combs, conf


def fuse_champion(track4, qx, softmax_conf, inj_t=1.0):
    """Champion top-20 pool fused with `MEMBER_W`."""
    B8 = {r["qidx"]: r for r in torch.load(f"{track4}/recs_8b_p3_k20.pt", weights_only=False)["recs"]}
    DO = {r["qidx"]: r for r in torch.load(f"{track4}/recs_2b_dora_k5_p3.pt", weights_only=False)["recs"]}
    qwen3vl_2b = {}
    for q, r in DO.items():
        for c, s in zip(r["cand"], r["scores"]):
            qwen3vl_2b[(q, c)] = float(s)
    IV = {}
    for k, name in IV_CACHE.items():
        try:
            IV[k] = load_iv(track4, name)
        except Exception as e:
            print(f"[S3] no cache for member {k} ({name}) → excluded ({e})", flush=True)
    members = [k for k in MEMBER_W if k in ("qwen3vl_2b", "8b") or k in IV]
    print(f"[S3] BASE=champion | fusion: sim(0.55) + "
          f"{{{', '.join(f'{k}:{MEMBER_W[k]}' for k in members)}}}", flush=True)
    Q = len(qx)
    cand_order, conf = [None] * Q, np.zeros(Q)
    for i, q in enumerate(qx):
        cand = B8[q]["cand"][:POOL]
        comb = 0.55 * z(B8[q]["sim"][:POOL])
        comb = comb + MEMBER_W["qwen3vl_2b"] * z([qwen3vl_2b.get((q, c), 0.0) for c in cand])
        comb = comb + MEMBER_W["8b"] * z(B8[q]["scores"][:POOL])
        for k in IV:
            comb = comb + MEMBER_W[k] * z([IV[k].get((q, c), 0.0) for c in cand])
        comb = np.asarray(comb, float)
        cand_order[i] = [cand[j] for j in np.argsort(-comb, kind="stable")]
        conf[i] = softmax_conf(comb, inj_t)
    return cand_order, [None] * Q, conf


def vote_swap(track4, qx, cand_order, combs, vote_q, vote_frac=0.7):
    """Experimental, disabled by default: swap a low-margin top-2 by reranker supermajority.

    Fires where the margin is in the bottom `vote_q` quantile and support is >= `vote_frac`.
    vote_q=0 means the caller skips this entirely.
    """
    VR = {}
    for nm in ("internvl_r32", "qwen3vl_2b", "8b", "pixtral", "llama", "ovis", "jina_m0"):
        fp = f"{track4}/{nm}_union_cache.pt"
        if os.path.exists(fp):
            VR[nm] = torch.load(fp, weights_only=False)["scores"]
    Q = len(qx)
    marg = np.array([combs[i][0] - combs[i][1] for i in range(Q)])
    thr = np.quantile(marg, vote_q)
    fired = 0
    for i, q in enumerate(qx):
        if marg[i] > thr or len(cand_order[i]) < 2:
            continue
        c1, c2 = int(cand_order[i][0]), int(cand_order[i][1])
        sup = avail = 0
        for S_ in VR.values():
            a, b_ = S_.get((q, c1)), S_.get((q, c2))
            if a is None or b_ is None:
                continue
            avail += 1
            sup += (b_ > a)
        if avail and sup >= vote_frac * avail:
            cand_order[i][0], cand_order[i][1] = c2, c1
            fired += 1
    print(f"[S3] low-margin (q={vote_q}) supermajority (≥{vote_frac:.0%}) swapped {fired} queries", flush=True)
    return cand_order
