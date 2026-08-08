#!/usr/bin/env python3
"""run_submission.py — inference entrypoint, covering stages S1 to S4.

**A thin orchestrator.** It reads cached artifacts, calls the stage modules in order and writes the
submission file; the processing itself lives in the stages.

  S1  `pipeline/S1_base/`   base score + candidate union  (prebuilt; read only here)
  S2  `pipeline/S2_rerank/` reranker rescore caches       (prebuilt; read only here)
  S3  `pipeline/S3_assign/fuse.py`   z-fusion  →  `pipeline/S3_assign/assign.py`  assignment
  S4  `pipeline/S4_tail/tail_refinement.py`  tail refinement, one module (S4a-S4e)

The answer (top-10 per query) is produced from the test data alone: gallery, query_index and the
precomputed scores.

Environment variables:
  GALLERY      gallery directory (sorted listdir = the answer column convention)
  QUERY_INDEX  query order file, one per line
  TRACK4       artifact/intermediate root (base score, *_cache.pt, recs_*, outputs/fuse_internvl/*)
  OUT          output answer path
  COMB_W       override the S3 comb weights (JSON). Unset uses tools/ensemble/adopted.py
  COMB_VARIANT comb variant to select (default best)
  TAIL_W       S4a overlay weight
  POST         "full" (default, S3 then S4) | "none" (S3 core only)
  FINAL_PASS   "on" (default, through S4e) | "off" (stop after the first S4d pass)
  TAU_PX       S4e image-height threshold in px (default 512)
  METHOD       "injective" (default) | "sca" | "argmax" (ablation). POST=full forces injective
  BASE         "r10" (default) | "champion". POST=full forces r10
  INJ_T        injective softmax temperature (default 1.0)
  IMPUTE       candidates a reranker did not cover: "zero" (default) | "neutral"
  VOTE_Q       low-margin top-2 supermajority swap: bottom margin quantile (0 = off, default)
               together with VOTE_FRAC (0.7)
  EXT          extension appended to answer stems (default "" = no extension, as submitted)

Run:  bash run_reproduce.sh best          # recommended: loads weights, checks inputs, builds the workspace
      POST=none python run_submission.py  # S3 only
"""
import hashlib
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(_HERE, "pipeline", "S3_assign"),
                os.path.join(_HERE, "pipeline", "S4_tail")]
import fuse                                                          # noqa: E402  S3 front half
import tail_refinement                                               # noqa: E402  S4, one module
from assign import (argmax_assign, injective_assign,                 # noqa: E402  S3 back half
                    sca_assign, softmax_conf)

TRACK4 = os.environ.get("TRACK4", f"{_HERE}/assets/cache/work")
PAB_TEST = os.environ.get("PAB_TEST", f"{_HERE}/assets/data/raw/pab_test")
GALLERY = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
QUERY_INDEX = os.environ.get("QUERY_INDEX", f"{PAB_TEST}/query_index.txt")
OUT = os.environ.get("OUT", f"{TRACK4}/answer_submission.txt")
POST = os.environ.get("POST", "full").lower()
METHOD = os.environ.get("METHOD", "injective").lower()
BASE = os.environ.get("BASE", "r10").lower()
INJ_T = float(os.environ.get("INJ_T", "1.0"))
IMPUTE = os.environ.get("IMPUTE", "zero").lower()
EXT = os.environ.get("EXT", "")
K = 10

if POST == "full":
    BASE, METHOD = "r10", "injective"


def comb_weights():
    """S3 comb weights; single source is tools/ensemble/adopted.py. COMB_W (JSON) takes precedence."""
    cw = os.environ.get("COMB_W", "")
    if cw:
        w = json.loads(cw)
        print(f"[S3] comb weights from the COMB_W environment variable {w}", flush=True)
        return w
    try:
        sys.path.insert(0, os.path.join(_HERE, "tools", "ensemble"))
        from adopted import comb as _comb
        return _comb(os.environ.get("COMB_VARIANT", "best"))
    except Exception as e:                                    # fallback when the bundle is used without tools/
        print(f"[S3] weights/ unavailable → using the built-in defaults ({e})", flush=True)
        return {"sim": 0.55, "qwen3vl_2b": 0.2, "8b": 0.0, "internvl_r32": 0.7, "pixtral": 0.3, "llama": 0.1}


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_answer(path, gstem, order, Q):
    """Write the submission format: query_index order, top-10 stems (+EXT), space separated, one line per query."""
    with open(path, "w") as f:
        for q in range(Q):
            f.write(" ".join(gstem[int(c)] + EXT for c in order[q][:K]) + "\n")
    print(f"[run] ✅ answer saved → {path}  ({Q} queries, EXT='{EXT}')", flush=True)
    print(f"[run]    md5 = {md5_of(path)}  (metrics are in README.md)", flush=True)


def main():
    gstem = [f[:-4] if f.endswith(".jpg") else f for f in sorted(os.listdir(GALLERY))]
    qx = [l.strip() for l in open(QUERY_INDEX) if l.strip()]
    Q = len(qx)
    print(f"[run] gallery={len(gstem)} query={Q}", flush=True)

    # ── S3 front half: z-fusion ───────────────────────────────────────────────
    if BASE == "r10":
        W = comb_weights()
        print("[S3] BASE=r10 (recall-base) | fusion: "
              + " + ".join(f"{k}{W[k]}" for k in ("sim", "qwen3vl_2b", "internvl_r32", "pixtral", "llama") if W.get(k)),
              flush=True)
        cand_order, combs, conf = fuse.fuse_r10(TRACK4, qx, W, softmax_conf, INJ_T, IMPUTE)
    else:
        cand_order, combs, conf = fuse.fuse_champion(TRACK4, qx, softmax_conf, INJ_T)

    vote_q = float(os.environ.get("VOTE_Q", "0"))
    if vote_q > 0 and BASE == "r10":
        cand_order = fuse.vote_swap(TRACK4, qx, cand_order, combs, vote_q,
                                    float(os.environ.get("VOTE_FRAC", "0.7")))

    # ── S3 back half: assignment ──────────────────────────────────────────────
    if METHOD == "injective":
        order = injective_assign(cand_order, conf, K)
        tag = f"injective T={INJ_T}"
    elif METHOD == "argmax":
        order = argmax_assign(cand_order, K)
        tag = "argmax (no-injective)"
    elif METHOD == "sca":
        sys.path[:0] = [os.path.join(_HERE, "pipeline"), os.environ.get("TRACK4_CODE", TRACK4)]
        import torch
        from utils import gallery_norm as GN
        Sgme = GN.normalize(torch.load(f"{TRACK4}/base_qwen_gme_score.pt", map_location="cpu",
                                       weights_only=False)).float().numpy()[:Q]
        order = sca_assign(cand_order, Sgme, K)
        tag = "SCA τ=0.002"
    else:
        raise SystemExit(f"[run] METHOD='{METHOD}' is not supported (injective|sca|argmax)")

    # ── POST=none: write the S3 answer as the submission file ─────────────────
    if POST != "full":
        write_answer(OUT, gstem, order, Q)
        print(f"[run]    stage = S3 core, {tag}", flush=True)
        return

    # ── S4: tail chain (tail_refinement.py, one module) ───────────────────────
    s3_path = tail_refinement.write_s3_answer(TRACK4, gstem, order, Q, K)
    print(f"[run] S3 answer → {s3_path} (tag={tag}). Starting S4 tail refinement…", flush=True)
    final, final_tag = tail_refinement.chain(
        TRACK4, tail_w=os.environ.get("TAIL_W"), tau_px=os.environ.get("TAU_PX"),
        final_pass=os.environ.get("FINAL_PASS", "on"))

    rows = [l.split() for l in open(final) if l.strip()]
    with open(OUT, "w") as f:
        for row in rows:
            f.write(" ".join(s + EXT for s in row) + "\n")
    print(f"[run] ✅ answer saved → {OUT}  ({len(rows)} queries, EXT='{EXT}')", flush=True)
    print(f"[run]    md5 = {md5_of(OUT)}  (metrics are in README.md)", flush=True)
    print(f"[run]    stage = {final_tag}", flush=True)


if __name__ == "__main__":
    main()
