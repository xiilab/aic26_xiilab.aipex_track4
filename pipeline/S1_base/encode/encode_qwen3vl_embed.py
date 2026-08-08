#!/usr/bin/env python3
"""encode_qwen3vl_embed — Qwen3-VL-Embedding-8B (zero-shot) test global feature dump.

One of the neighbour-search encoders used by S4b tail-NN (`../../S4_tail/tail_refinement.py`):
`qwen3vl_embed8b_feats.pt` {G, Q, G_base, Q_idx}, 4096d. Being a decoder-VLM embedding, it has low
correlation with the CLIP-family members.

  gallery columns = sorted(listdir(GALLERY)) · query rows = query_text.json (matches query_index.txt)
  The gallery pass checkpoints to `<out>.ckpt`, so re-running resumes where it stopped.

Output: assets/cache/s4_nn/ by default, assets/cache_rep/s4_nn/ with `--rep`.
Usage: CUDA_VISIBLE_DEVICES=6 <py> encode_qwen3vl_embed.py --rep
Environment: QWEN3VL_EMBED (model path) · PAB_TEST · GALLERY · QUERY_TEXT · S4_NN
"""
import argparse
import json
import os
import sys
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402


MODEL = os.environ.get("QWEN3VL_EMBED", f"{_REPO}/assets/model/vlm_models/Qwen3-VL-Embedding-8B")
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GAL = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
QTEXT = os.environ.get("QUERY_TEXT", f"{PAB_TEST}/query_text.json")
QPROMPT = "Retrieve images or text relevant to the user's query."   # official QwenLM retrieval instruction

ap = argparse.ArgumentParser(description="Qwen3-VL-Embedding-8B zero-shot dump -> S4b tail-NN input")
ap.add_argument("--bs-img", type=int, default=32)
ap.add_argument("--bs-txt", type=int, default=64)
ap.add_argument("--ckpt-every", type=int, default=3200)
ap.add_argument("--out", default=None)
ap.add_argument("--limit", type=int, default=0, help="truncate the gallery for a smoke test")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction encoding -> cache_rep/s4_nn")
a = ap.parse_args()
if a.rep and a.limit:
    raise SystemExit("[encode_qwen3vl_embed] --limit cannot be combined with --rep (prevents truncated artifacts).")

S4NN = os.environ.get("S4_NN", f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}/s4_nn")
OUT = a.out or f"{S4NN}/qwen3vl_embed8b_feats.pt"
skip_if_exists(OUT, a.overwrite)
if not os.path.isdir(MODEL):
    raise SystemExit(f"[encode_qwen3vl_embed] model not found: {MODEL}  (set QWEN3VL_EMBED)")
sys.path.insert(0, f"{MODEL}/scripts")                 # Qwen3VLEmbedder shipped with the model

from qwen3_vl_embedding import Qwen3VLEmbedder          # noqa: E402

emb = Qwen3VLEmbedder(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
print("[loaded] bf16/sdpa", flush=True)


def _enc(items, bs, tag):
    """Batched encoding. On batch failure, retries item by item (length variance and OOM)."""
    out, t0 = [], time.time()
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        try:
            e = emb.process(chunk)
        except Exception as ex:
            print(f"  [{tag}] batch@{i} fail ({type(ex).__name__}) → per-item fallback", flush=True)
            e = torch.cat([emb.process([c]) for c in chunk], 0)
        out.append(e.float().cpu())
        if (i // bs) % 20 == 0:
            done = min(i + bs, len(items)); el = time.time() - t0
            print(f"  [{tag}] {done}/{len(items)} ({el:.0f}s, {done/max(el,1):.1f}/s)", flush=True)
    return torch.cat(out)[:len(items)]


rows = [json.loads(l) for l in open(QTEXT, encoding="utf-8") if l.strip()]
Q_idx = [r["query_index"] for r in rows]
Q = _enc([{"text": r["caption"], "instruction": QPROMPT} for r in rows], a.bs_txt, "query")
print(f"[query] {tuple(Q.shape)}", flush=True)

gal = sorted(os.listdir(GAL))
if a.limit:
    gal = gal[:a.limit]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
ckpt = OUT + ".ckpt"
G_parts, start = [], 0
if os.path.exists(ckpt):                                # resume from where it stopped
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    if c.get("n_total") == len(gal):
        G_parts, start = [c["G"]], c["done"]
        print(f"[resume] gallery from {start}/{len(gal)}", flush=True)
t0 = time.time()
for i in range(start, len(gal), a.bs_img):
    chunk = [{"image": f"{GAL}/{g}"} for g in gal[i:i + a.bs_img]]
    try:
        e = emb.process(chunk)
    except Exception as ex:
        print(f"  [gallery] batch@{i} fail ({type(ex).__name__}) → per-item", flush=True)
        e = torch.cat([emb.process([c]) for c in chunk], 0)
    G_parts.append(e.float().cpu())
    done = min(i + a.bs_img, len(gal))
    if (i // a.bs_img) % 20 == 0:
        el = time.time() - t0
        print(f"  [gallery] {done}/{len(gal)} ({el:.0f}s, {(done-start)/max(el,1):.1f}/s)", flush=True)
    if done % a.ckpt_every < a.bs_img:
        torch.save({"G": torch.cat(G_parts), "done": done, "n_total": len(gal)}, ckpt)
G = torch.cat(G_parts)[:len(gal)]

torch.save({"G": G, "Q": Q, "G_base": gal, "Q_idx": Q_idx}, OUT)
if os.path.exists(ckpt):
    os.remove(ckpt)
print(f"[save] {OUT}  G={tuple(G.shape)} Q={tuple(Q.shape)}", flush=True)
