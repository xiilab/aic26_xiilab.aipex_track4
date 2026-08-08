#!/usr/bin/env python3
"""S3 injective assignment — single implementation.

From the comb scores:
  (a) cross-query priority   conf = softmax-max(comb / T)
  (b) walking queries from the highest conf, fix the top candidate not already taken as top1
      (greedy injective)

The rule uses only the task's injectivity — one query has one ground truth.
Ties are pinned with `np.argsort(..., kind="stable")`, so every run gives the same answer.

Three assignment modes live here: `injective` (used in the pipeline), `sca` (visual-margin priority)
and `argmax` (no assignment, for ablation).
"""
import numpy as np


def softmax_conf(comb, T=1.0):
    """comb vector → softmax-max confidence, used as the cross-query priority."""
    comb = np.asarray(comb, float)
    p = np.exp((comb - comb.max()) / T)
    return float((p / p.sum()).max())


def injective_assign(cand_order, conf, K=10):
    """Walk queries by descending conf, each taking its highest candidate not yet taken.

    cand_order[q] : candidate gallery indices, comb-descending
    conf[q]       : softmax_conf(comb_q, T)
    returns order[q] : [taken candidate] + the rest in original order, first K — a python list

    A displaced query falls back to its own rank-2 candidate. If every candidate is already taken it
    keeps its original top1, which is the only case where a duplicate is allowed.
    """
    used = set()
    order = [None] * len(cand_order)
    for q in np.argsort(-np.asarray(conf, float), kind="stable"):
        co = cand_order[q]
        ch = next((c for c in co if c not in used), co[0])
        used.add(ch)
        order[q] = ([ch] + [x for x in co if x != ch])[:K]
    return order


def argmax_assign(cand_order, K=10):
    """Ablation control — comb-argmax top1 with no injectivity, so collisions are allowed."""
    return [list(co)[:K] for co in cand_order]


def sca_assign(cand_order, gme_score, K=10, tau=0.002):
    """
    Uses the top1-to-top2 visual margin as the priority instead of conf.
    """
    Q = len(cand_order)
    osub = np.array([list(co)[:K] for co in cand_order], np.int64)
    margin = np.array([gme_score[q, osub[q, 0]] - gme_score[q, osub[q, 1]] for q in range(Q)])
    taken, order = {}, osub.copy()
    for q in np.argsort(-margin):
        d0 = int(osub[q, 0])
        if d0 not in taken:
            taken[d0] = q
        elif margin[taken[d0]] - margin[q] >= tau:
            ch = next((int(osub[q, k]) for k in range(K) if int(osub[q, k]) not in taken), d0)
            if ch != d0:
                taken[ch] = q
                order[q] = np.array([ch] + [x for x in osub[q] if x != ch])
    return [list(row) for row in order]
