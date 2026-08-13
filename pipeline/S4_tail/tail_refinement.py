#!/usr/bin/env python3
"""S4 tail refinement — all five sub-stages in a single module.

Takes the S3 answer (noext stems, query_index order), refines it stage by stage and writes the
final answer.

  S4a overlay      freeze ranks 1-7, refill ranks 8-10 from the union pool by comb score; a
                   newcomer must beat the margin guard m to displace an incumbent
  S4b NN complete  insert the multi-encoder consensus 1-NN (cos >= tau) of a tail (rank 8-10)
                   candidate at rank 10
  S4c R@5 promote  promote a near-duplicate outside comb-top5 to rank 5 on ovis/internvl_r32/
                   jina_m0 agreement (ranks 1-4 frozen)
  S4d cons6        6 rerankers unanimously prefer rank2 over rank1 -> local augmenting path
                   re-assigns injectively
  S4e demote       move top-10 candidates whose image height < tau_px to the back, then
                   re-resolve the injective conflicts this creates

Usage:
    from tail_refinement import chain
    final, tag = chain(track4, tail_w=..., tau_px=..., final_pass="on")

    python tail_refinement.py --stage all            # every stage
    python tail_refinement.py --stage a --answer X   # a single stage
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."),
                os.environ.get("TRACK4_CODE", _REPO)]
from utils import gallery_norm as GN                                            # noqa: E402

T4 = os.environ.get("TRACK4", f"{_REPO}/assets/cache/work")
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
QUERY_INDEX = os.environ.get("QUERY_INDEX", f"{PAB_TEST}/query_index.txt")
HERE = os.path.dirname(os.path.abspath(__file__))

S3_NAME = "answer_r10_injective_fulltune_noext.txt"
S4A_NAME = "answer_tailoverlay_r10ft_m0.5_noext.txt"
S4B_NAME = "answer_tailoverlay_r10ft_m0.5_nn0.8c4_noext.txt"
S4C_NAME = "answer_tailoverlay_r10ft_m0.5_nn0.8c4_r5promote_v2_noext.txt"
S4D_NAME = "answer_tailoverlay_r5promote_top1prop_noext.txt"
S4E_NAME = "answer_final_noext.txt"

ENC = {"dfn": ("dfn_gallery_emb.pt", "emb"), "convnext": ("convnext_gallery_emb.pt", "emb"),
       "gme": ("gme_feats.pt", "G"), "qwen3vl_embed8b": ("qwen3vl_embed8b_feats.pt", "G"),
       "anchor5": ("anchor5_feats.pt", "G")}
W0_DEFAULT = {"sim": 0.55, "qwen3vl_2b": 0.2, "8b": 0.3, "internvl_r32": 0.7, "pixtral": 0.3, "llama": 0.2}


# ── shared ────────────────────────────────────────────────────────────────────
def _stem(t):
    return t[:-4] if t.endswith(".jpg") else t


class Ctx:
    """Gallery and query order plus answer I/O. Answer columns = sorted(listdir(gallery)), rows = query_index."""

    def __init__(self, track4=T4):
        self.t4 = track4
        self.gal = sorted(os.listdir(GN.GALLERY_DIR))
        self.gstem = [_stem(g) for g in self.gal]
        self.s2i = {s: i for i, s in enumerate(self.gstem)}
        self.qx = [l.strip() for l in open(QUERY_INDEX) if l.strip()]
        self.Q = len(self.qx)

    def read(self, name):
        rows = [l.split() for l in open(os.path.join(self.t4, name)) if l.strip()]
        assert len(rows) == self.Q and all(len(r) == 10 for r in rows), f"malformed answer: {name}"
        return [[self.s2i[_stem(t)] for t in r] for r in rows]

    def write(self, name, order):
        p = os.path.join(self.t4, name)
        with open(p, "w") as f:
            for row in order:
                f.write(" ".join(self.gstem[int(c)] for c in row[:10]) + "\n")
        return p


def _z(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v * 0.0


def _load_embs(track4):
    embs = {}
    for nm, (f, k) in ENC.items():
        o = torch.load(f"{track4}/{f}", map_location="cpu", weights_only=False)
        E = o[k].float()
        embs[nm] = E / E.norm(dim=1, keepdim=True)
    return embs


def _load_iv(track4, c):
    """Top-20 score cache in the InternVL fuse_cache layout."""
    d = f"{track4}/outputs/fuse_internvl/{c}"
    m = json.load(open(f"{d}/internvl_scores_{c}_meta.json"))
    sc = np.load(f"{d}/internvl_scores_{c}_top20.npy")
    cd = np.load(f"{d}/internvl_cand_{c}_top20.npy")
    o = {}
    for i, q in enumerate(m["qorder"]):
        for j in range(cd.shape[1]):
            if not np.isnan(sc[i, j]):
                o[(q, int(cd[i, j]))] = float(sc[i, j])
    return o


# ── S4a — overlay ─────────────────────────────────────────────────────────────
def s4a_overlay(ctx, answer=S3_NAME, tag="r10ft", sim="greedy_R10_base_score.pt",
                w=None, claim_theta=0.5, save_m=0.5):
    """Freeze ranks 1-7 and refill ranks 8-10 from the union pool by comb score.

    A newcomer has to overcome the `save_m` margin penalty (in z units) to displace an incumbent
    of ranks 8-10. With `claim_theta` > 0, candidates already claimed at rank 1 by another query
    with conf >= theta are dropped from the newcomers only; incumbents are exempt.
    """
    t4, Q = ctx.t4, ctx.Q
    top10 = ctx.read(answer)
    W0 = dict(W0_DEFAULT)
    if w:
        W0 = dict(zip(["sim", "qwen3vl_2b", "8b", "internvl_r32", "pixtral", "llama"], [float(x) for x in w.split(",")]))

    U = torch.load(f"{t4}/union_pool.pt")
    tops = U["tops"]
    uni = [set() for _ in range(Q)]
    for b in tops:
        for i in range(Q):
            uni[i].update(int(c) for c in tops[b][i][:20])
    for i in range(Q):
        uni[i].update(top10[i])                       # keep the current answer in the pool
    simM = torch.load(f"{t4}/{sim}", map_location="cpu").float().numpy()

    # signals: the union re-score cache when present, otherwise the champion recs dump
    B8 = {r["qidx"]: r for r in torch.load(f"{t4}/recs_8b_p3_k20.pt", weights_only=False)["recs"]}
    DO = {r["qidx"]: r for r in torch.load(f"{t4}/recs_2b_dora_k5_p3.pt", weights_only=False)["recs"]}
    champ = {"8b": {}, "qwen3vl_2b": {}}
    for q, r in B8.items():
        for c, s in zip(r["cand"], r["scores"]):
            champ["8b"][(q, int(c))] = float(s)
    for q, r in DO.items():
        for c, s in zip(r["cand"], r["scores"]):
            champ["qwen3vl_2b"][(q, int(c))] = float(s)
    champ.update({"internvl_r32": _load_iv(t4, "internvl_r32"), "pixtral": _load_iv(t4, "pixtral"),
                  "llama": _load_iv(t4, "llama32v")})
    cache = {}
    for key, nm in {"8b": "8b", "qwen3vl_2b": "qwen3vl_2b", "internvl_r32": "internvl_r32",
                    "pixtral": "pixtral", "llama": "llama"}.items():
        p = f"{t4}/{nm}_union_cache.pt"
        if os.path.exists(p):
            cache[key] = torch.load(p)["scores"]
    print(f"[S4a] cache {sorted(cache)} | W0={W0}", flush=True)

    def sig(name, q, c):
        if name in cache and (q, c) in cache[name]:
            return cache[name][(q, c)]
        return champ[name].get((q, c))

    comb, pools = [None] * Q, [None] * Q
    nmiss = ntot = 0
    for i, q in enumerate(ctx.qx):
        pool = sorted(uni[i])
        pools[i] = pool
        C = W0["sim"] * _z([simM[i, c] for c in pool])
        for nm in ("qwen3vl_2b", "8b", "internvl_r32", "pixtral", "llama"):
            vals = [sig(nm, q, c) for c in pool]
            nmiss += sum(v is None for v in vals)
            ntot += len(vals)
            C = C + W0[nm] * _z([0.0 if v is None else v for v in vals])
        comb[i] = {c: float(C[j]) for j, c in enumerate(pool)}
    print(f"[S4a] missing signals {nmiss}/{ntot} ({100 * nmiss / ntot:.2f}%) - expect ~0 once re-scoring is complete", flush=True)

    claims = {}
    if claim_theta > 0:
        for i in range(Q):
            v = np.array([comb[i][c] for c in pools[i]])
            e = np.exp(v - v.max())
            claims[top10[i][0]] = (i, float((e / e.sum()).max()))

    out, churn = [], 0
    for i in range(Q):
        frozen, tail = top10[i][:7], top10[i][7:10]
        newc = [c for c in pools[i] if c not in top10[i]]
        cand = ([(comb[i][c], comb[i][c], c) for c in tail]
                + [(comb[i][c] - save_m, comb[i][c], c) for c in newc])
        if claim_theta > 0:
            keep = [t for t in cand if t[2] in tail or not
                    (t[2] in claims and claims[t[2]][0] != i and claims[t[2]][1] >= claim_theta)]
            if len(keep) >= 3:
                cand = keep
        cand.sort(key=lambda x: -x[0])
        pick = sorted(cand[:3], key=lambda x: -x[1])   # select on the penalised score, place by raw comb
        nt = [c for _, _, c in pick]
        if nt != tail:
            churn += 1
        out.append(frozen + nt)

    name = f"answer_tailoverlay_{tag}_m{save_m:g}_noext.txt"
    ctx.write(name, out)
    print(f"[S4a] overlay(m={save_m}) -> {name}  (tail replaced in {churn} queries)", flush=True)
    return name


# ── S4b — NN completion ───────────────────────────────────────────────────────
def s4b_nn_completion(ctx, answer=S4A_NAME, seeds="8,9,10", tau=0.80, cons_min=4):
    """Insert the multi-encoder consensus 1-NN of a tail (rank 8-10) candidate at rank 10.

    Targets near-duplicate frame structure: the ground truth sitting next to an already
    retrieved neighbour frame.
    """
    t4, Q = ctx.t4, ctx.Q
    top10 = ctx.read(answer)
    seed_ranks = [int(x) - 1 for x in seeds.split(",")]
    embs = _load_embs(t4)
    print(f"[S4b] encoders {list(embs)}", flush=True)

    seeds_flat, owner = [], []
    for i in range(Q):
        for r in seed_ranks:
            seeds_flat.append(top10[i][r])
            owner.append(i)
    seeds_flat = torch.tensor(seeds_flat)

    NN, COS = {}, {}
    for nm, E in embs.items():
        nn_i = torch.empty(len(seeds_flat), dtype=torch.long)
        nn_c = torch.empty(len(seeds_flat))
        for b in range(0, len(seeds_flat), 512):
            sl = seeds_flat[b:b + 512]
            S = E[sl] @ E.T
            S[torch.arange(len(sl)), sl] = -2                        # exclude self
            v, ix = S.max(dim=1)
            nn_i[b:b + 512] = ix
            nn_c[b:b + 512] = v
        NN[nm], COS[nm] = nn_i, nn_c

    best = [None] * Q          # (edge_cos, cons, cand, seed_rank)
    for j in range(len(seeds_flat)):
        i = owner[j]
        votes = collections.Counter()
        for nm in embs:
            votes[int(NN[nm][j])] += 1
        for c, cn in votes.items():
            if c in top10[i]:
                continue
            ecos = float(np.mean([COS[nm][j] if int(NN[nm][j]) == c
                                  else float(embs[nm][seeds_flat[j]] @ embs[nm][c]) for nm in embs]))
            cand = (ecos, cn, c, j % len(seed_ranks))
            if best[i] is None or (cand[1], cand[0]) > (best[i][1], best[i][0]):
                best[i] = cand

    fired = [i for i in range(Q) if best[i] and best[i][1] >= cons_min and best[i][0] >= tau]
    out = [list(r) for r in top10]
    for i in fired:
        out[i][9] = best[i][2]
    name = answer.replace("_noext.txt", f"_nn{tau:g}c{cons_min}_noext.txt")
    ctx.write(name, out)
    print(f"[S4b] NN insert(tau={tau} cons>={cons_min}) -> {name}  (fired on {len(fired)} queries)", flush=True)
    return name


# ── S4c — R@5 near-dup promotion ──────────────────────────────────────────────
def s4c_r5_promote(ctx, answer=S4B_NAME, tau=0.80):
    """Promote a near-duplicate outside comb-top5 to rank 5.

    Condition: cos(c, top1) >= tau and [ (ovis<3 and internvl_r32<6) or (ovis<4 and jina_m0<3) ];
    the highest ovis score among the qualifiers wins.
    """
    t4, Q = ctx.t4, ctx.Q
    ansi = ctx.read(answer)
    R32 = _load_iv(t4, "internvl_r32")
    R32.update(torch.load(f"{t4}/internvl_r32_union_cache.pt", weights_only=False)["scores"])
    OV = torch.load(f"{t4}/ovis_union_cache.pt", weights_only=False)["scores"]
    JI = torch.load(f"{t4}/jina_m0_union_cache.pt", weights_only=False)["scores"]
    d = torch.load(f"{t4}/metaclip2_feats.pt", map_location="cpu", weights_only=False)
    Gv = F.normalize(d["G"].float(), dim=-1)
    gv2i = {_stem(str(x)): i for i, x in enumerate(d["G_base"])}

    def cos(a, b):
        if ctx.gstem[a] in gv2i and ctx.gstem[b] in gv2i:
            return float((Gv[gv2i[ctx.gstem[a]]] @ Gv[gv2i[ctx.gstem[b]]]).item())
        return -9

    out, fired = [], 0
    for i in range(Q):
        q = ctx.qx[i]
        row = list(ansi[i])
        top10 = row[:10]
        ovr = {c: r for r, c in enumerate(sorted(top10, key=lambda c: -OV.get((q, c), -99)))}
        r32r = {c: r for r, c in enumerate(sorted(top10, key=lambda c: -R32.get((q, c), -99)))}
        jir = {c: r for r, c in enumerate(sorted(top10, key=lambda c: -JI.get((q, c), -99)))}
        quals = [c for c in top10 if row.index(c) >= 5 and cos(c, row[0]) >= tau
                 and ((ovr[c] < 3 and r32r[c] < 6) or (ovr[c] < 4 and jir[c] < 3))]
        if quals:
            cand = max(quals, key=lambda c: OV.get((q, c), -99))
            row.remove(cand)
            row.insert(4, cand)
            fired += 1
        out.append(row)
    ctx.write(S4C_NAME, out)
    print(f"[S4c] R@5 promote -> {S4C_NAME}  (fired on {fired} queries)", flush=True)
    return S4C_NAME


# ── S4d — cons6 injective propagation ─────────────────────────────────────────
def s4d_propagate(ctx, answer=S4C_NAME, ncons=6, max_rounds=6):
    """Detect unanimous preference of rank2 over rank1 across 6 rerankers, then re-assign
    injectively along a local augmenting path.

    Only the displaced query moves to its best free candidate; the rest stay frozen. Detection
    and propagation repeat until convergence.
    """
    t4, Q = ctx.t4, ctx.Q
    A0 = ctx.read(answer)
    _drop = {x for x in os.environ.get("CONS_DROP", "").split(",") if x}
    RK = {k: torch.load(f"{t4}/{n}_union_cache.pt")["scores"]
          for k, n in [("ovis", "ovis"), ("jina_m0", "jina_m0"), ("internvl_r32", "internvl_r32"),
                       ("pixtral", "pixtral"), ("llama", "llama"), ("8b", "8b")]
          if k not in _drop and os.path.exists(f"{t4}/{n}_union_cache.pt")}
    if _drop:
        print(f"[S4d] cons vote: dropped {sorted(_drop)} -> {len(RK)} voters", flush=True)
    for nm, f in [("internvl_r32", "internvl_r32_nntail_cache.pt"), ("jina_m0", "jina_m0_nntail_cache.pt"),
                  ("llama", "llama_nntail_cache.pt")]:
        p = f"{t4}/{f}" if os.path.exists(f"{t4}/{f}") else os.path.join(HERE, f)
        if nm in RK and os.path.exists(p):
            RK[nm].update(torch.load(p)["scores"])

    def sc(q, c, k):
        return RK[k].get((ctx.qx[q], c))

    def cons(q, cand, over):
        return sum(1 for k in RK
                   if sc(q, cand, k) is not None and sc(q, over, k) is not None
                   and sc(q, cand, k) > sc(q, over, k))

    def propagate(A, swaps):
        A = [r[:] for r in A]
        top1 = [A[q][0] for q in range(Q)]
        holder = {top1[q]: q for q in range(Q)}
        for q in swaps:
            cur_q, target = q, A[q][1]
            for _ in range(20):
                old = top1[cur_q]
                if old == target:
                    break
                disp = holder.get(target)
                top1[cur_q] = target
                holder[target] = cur_q
                if target in A[cur_q]:
                    A[cur_q].remove(target)
                A[cur_q].insert(0, target)
                if old in holder and holder[old] == cur_q:
                    del holder[old]
                if disp is None:
                    break
                cur_q, target = disp, next((c for c in A[disp] if c not in holder), old)
        return A

    cur, done = [r[:] for r in A0], set()
    for rnd in range(1, max_rounds + 1):
        sw = [q for q in range(Q) if q not in done and cons(q, cur[q][1], cur[q][0]) >= ncons]
        if not sw:
            break
        cur = propagate(cur, sw)
        done |= set(sw)
        print(f"[S4d] round {rnd}: cons{ncons} detected {len(sw)} {sw}, propagated", flush=True)
    ctx.write(S4D_NAME, cur)
    print(f"[S4d] converged -> {S4D_NAME} | distinct top1 {len({r[0] for r in cur})}/{Q}", flush=True)
    return S4D_NAME


# ── S4e — final demotion pass ─────────────────────────────────────────────────
def s4e_demote(ctx, answer=S4D_NAME, tau_px=512, gallery=None):
    """Move top-10 candidates whose image height < tau_px to the back of the row (the set is
    unchanged, so R@10 cannot drop), then re-resolve the top1 conflicts this creates injectively
    (the query that originally held the top1 keeps it; up to 6 rounds).

    The default tau_px = 512 is the training input resolution of the anchor encoders
    (anchor_filip and anchor_tcap are siglip2-large-patch16-512 trained on train_jpg_512), so
    candidates stored smaller than that fall outside the scale those encoders were trained on.
    """
    from PIL import Image
    gal_dir = gallery or os.environ.get("GALLERY", GN.GALLERY_DIR)
    rows = [[_stem(a) for a in l.split()] for l in open(os.path.join(ctx.t4, answer)) if l.strip()]
    QR = len(rows)
    _h = {}

    def H(sm):
        if sm not in _h:
            _h[sm] = Image.open(os.path.join(gal_dir, sm + ".jpg")).size[1]
        return _h[sm]

    new = [[c for c in r[:10] if H(c) >= tau_px] + [c for c in r[:10] if H(c) < tau_px] for r in rows]
    orig_top1 = {rows[i][0]: i for i in range(QR)}
    for _ in range(6):
        dups = [k for k, v in collections.Counter(r[0] for r in new).items() if v > 1]
        if not dups:
            break
        for X in dups:
            qs = [q for q in range(QR) if new[q][0] == X]
            owner = orig_top1.get(X)
            losers = [q for q in qs if q != owner] if owner in qs else qs[1:]
            for q in losers:
                new[q] = [c for c in new[q] if c != X] + [X]
    p = os.path.join(ctx.t4, S4E_NAME)
    with open(p, "w") as f:
        for r in new:
            f.write(" ".join(r[:10]) + "\n")
    md5 = hashlib.md5(open(p, "rb").read()).hexdigest()[:8]
    print(f"[S4e] demote + re-resolve -> {S4E_NAME} md5={md5} (tau={tau_px}px)", flush=True)
    return S4E_NAME


# ── chain ─────────────────────────────────────────────────────────────────────
def write_s3_answer(track4, gstem, order, Q, K=10):
    """Write the S3 result under the fixed chain-input name and return its path."""
    path = f"{track4}/{S3_NAME}"
    with open(path, "w") as f:
        for q in range(Q):
            f.write(" ".join(gstem[int(c)] for c in order[q][:K]) + "\n")
    return path


def chain(track4=T4, tail_w=None, tau_px=None, final_pass="on", **_):
    """Run S4a through S4e in-process and return (final answer path, tag)."""
    ctx = Ctx(track4)
    a = s4a_overlay(ctx, S3_NAME, tag="r10ft", sim="greedy_R10_base_score.pt", w=tail_w,
                    claim_theta=0.5, save_m=0.5)
    b = s4b_nn_completion(ctx, a, seeds="8,9,10", tau=0.80, cons_min=4)
    c = s4c_r5_promote(ctx, b, tau=0.80)
    d = s4d_propagate(ctx, c)
    if str(final_pass).lower() in ("off", "none", "0", "false", "no"):
        return os.path.join(track4, d), "S3 -> S4a..S4d [final demotion pass skipped]"
    tau = int(tau_px or os.environ.get("TAU_PX", "512"))
    e = s4e_demote(ctx, d, tau_px=tau)
    return os.path.join(track4, e), f"S4e final pass: demote low-height candidates (tau={tau}px) + injective re-resolution"


def main():
    ap = argparse.ArgumentParser(description="S4 tail refinement (single module)")
    ap.add_argument("--stage", default="all", choices=["all", "a", "b", "c", "d", "e"])
    ap.add_argument("--answer", default=None, help="input answer for a single-stage run (default = that stage's standard input)")
    ap.add_argument("--tail-w", default=os.environ.get("TAIL_W"))
    ap.add_argument("--tau-px", type=int, default=int(os.environ.get("TAU_PX", "512")))
    ap.add_argument("--final-pass", default=os.environ.get("FINAL_PASS", "on"))
    g = ap.parse_args()
    if g.stage == "all":
        final, tag = chain(T4, g.tail_w, g.tau_px, g.final_pass)
        print(f"[S4] done -> {final}\n     {tag}", flush=True)
        return
    ctx = Ctx(T4)
    fn = {"a": lambda: s4a_overlay(ctx, g.answer or S3_NAME, w=g.tail_w),
          "b": lambda: s4b_nn_completion(ctx, g.answer or S4A_NAME),
          "c": lambda: s4c_r5_promote(ctx, g.answer or S4B_NAME),
          "d": lambda: s4d_propagate(ctx, g.answer or S4C_NAME),
          "e": lambda: s4e_demote(ctx, g.answer or S4D_NAME, tau_px=g.tau_px)}[g.stage]
    print(f"[S4] stage {g.stage} -> {fn()}", flush=True)


if __name__ == "__main__":
    main()
