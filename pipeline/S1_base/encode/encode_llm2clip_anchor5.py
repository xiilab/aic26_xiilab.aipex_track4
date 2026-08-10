#!/usr/bin/env python3
"""encode_llm2clip_anchor5 — LLM2CLIP (L-14-336) + anchor5 LoRA -> test feature dump.

One of the neighbour-search encoders used by S4b tail-NN (`../../S4_tail/tail_refinement.py`):
`anchor5_feats.pt` {G, Q, G_base, Q_idx}, 1280d.

Images: the gallery goes through the CLIP vision tower with the **v3 LoRA** merged into the base
        weights before anchor5 is attached. anchor5 targets `text_adapter.*` only, so without that
        merge the gallery would come out of the stock backbone.
Queries: LLM2CLIP first turns a caption into an LLM pooled hidden state (4096d) and then runs it
         through text_adapter. Producing that hidden state requires the 8B LLM, so it is
         precomputed and bundled (`<ADAPTER>/query_hidden.pt` {query_hidden[1978,4096], query_idx}).
         This script only runs hidden -> text_adapter (LoRA).

Output: assets/cache/s4_nn/ by default, assets/cache_rep/s4_nn/ with `--rep`.
Usage: CUDA_VISIBLE_DEVICES=6 <py> encode_llm2clip_anchor5.py --rep
Environment: LLM2CLIP_BASE (default microsoft/LLM2CLIP-Openai-L-14-336, HF cache) ·
             LLM2CLIP_V3_ADAPTER · ANCHOR5_ADAPTER · QUERY_HIDDEN · PAB_TEST · GALLERY · S4_NN
"""
import argparse
import glob
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Repo-local HF cache. huggingface_hub reads HF_HOME at import time, so it has to be set before
# transformers is imported below.
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402


ap = argparse.ArgumentParser(description="LLM2CLIP + anchor5 LoRA dump -> S4b tail-NN input")
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--out", default=None)
ap.add_argument("--limit", type=int, default=0, help="truncate the gallery for a smoke test")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction encoding -> cache_rep/s4_nn")
a = ap.parse_args()
if a.rep and a.limit:
    raise SystemExit("[encode_llm2clip_anchor5] --limit cannot be combined with --rep (prevents truncated artifacts).")

def _enc(name):
    """--rep takes the deployed adapter when there is one, else the shipped copy. Resolved per
    adapter, so deploying the text adapter alone still merges the shipped v3 vision LoRA."""
    rep = f"{_REPO}/assets/model_rep/encoder/{name}"
    return rep if a.rep and os.path.exists(rep) else f"{_REPO}/assets/model/encoder/{name}"


VIS = os.environ.get("LLM2CLIP_BASE", "microsoft/LLM2CLIP-Openai-L-14-336")
ADAPTER = os.environ.get("ANCHOR5_ADAPTER", _enc("llm2clip_anchor5"))
V3_ADAPTER = os.environ.get("LLM2CLIP_V3_ADAPTER", _enc("llm2clip_lora_v3_best"))
QHIDDEN = os.environ.get("QUERY_HIDDEN", f"{ADAPTER}/query_hidden.pt")
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GAL = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
S4NN = os.environ.get("S4_NN", f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}/s4_nn")
OUT = a.out or f"{S4NN}/anchor5_feats.pt"
skip_if_exists(OUT, a.overwrite)
DEV = os.environ.get("LLM2CLIP_DEV", "cuda:0")

for p, what in ((ADAPTER, "anchor5 adapter"), (QHIDDEN, "query hidden cache"),
                (V3_ADAPTER, "v3 vision LoRA (merged under anchor5)")):
    if not os.path.exists(p):
        raise SystemExit(f"[encode_llm2clip_anchor5] {what} not found: {p}")
    print(f"[weights] {what}: {os.path.relpath(p, _REPO)}", flush=True)


def log(*x):
    print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)


# The LLM2CLIP remote code probes for flash-attn3; stub the check so the import works without it.
import transformers.utils as _u                                  # noqa: E402
import transformers.utils.import_utils as _iu                    # noqa: E402


def _false(*_a, **_k):
    return False


for _m in (_iu, _u):
    if hasattr(_m, "is_flash_attn_3_available"):
        setattr(_m, "is_flash_attn_3_available", _false)
if hasattr(_iu, "_flash_attn_3_available"):
    _iu._flash_attn_3_available = False

import torchvision.transforms as _T                              # noqa: E402
from transformers import AutoModel                                # noqa: E402
from peft import PeftModel                                       # noqa: E402

# CLIP statistics · shortest-edge 336 bicubic · center crop. Substituting
# CLIPImageProcessor("openai/clip-vit-large-patch14-336") looks equivalent but takes a different
# resample path and shifts the features.
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)
TFM = _T.Compose([_T.Resize(336, interpolation=_T.InterpolationMode.BICUBIC),
                  _T.CenterCrop(336), _T.ToTensor(), _T.Normalize(_MEAN, _STD)])

log(f"loading base={VIS} + v3={V3_ADAPTER} + adapter={ADAPTER}")
base = AutoModel.from_pretrained(VIS, torch_dtype=torch.float32, trust_remote_code=True,
                                 attn_implementation="sdpa").to(DEV)
# Stacked in two steps: v3 is folded into the weights, then anchor5 is attached on top. Without
# merge_and_unload() peft applies only the active adapter when two are attached at once.
base = PeftModel.from_pretrained(base, V3_ADAPTER).merge_and_unload().float()
model = PeftModel.from_pretrained(base, ADAPTER).float().eval()
log("ready")

gallery_paths = sorted(glob.glob(f"{GAL}/*.jpg"))
if a.limit:
    gallery_paths = gallery_paths[:a.limit]
log(f"gallery {len(gallery_paths)}")


@torch.no_grad()
def enc_images(paths, bs):
    feats = []
    t0 = time.time()
    for i in range(0, len(paths), bs):
        imgs = []
        for p in paths[i:i + bs]:
            try:
                imgs.append(TFM(Image.open(p).convert("RGB")))
            except Exception:
                imgs.append(torch.zeros(3, 336, 336))     # unreadable image -> pre-normalisation zeros
        px = torch.stack(imgs).to(DEV).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.get_image_features(px)
        feats.append(F.normalize(f.float(), dim=-1).cpu())
        if (i // bs) % 20 == 0:
            log(f"  img {min(i+bs, len(paths))}/{len(paths)} ({time.time()-t0:.0f}s)")
    return torch.cat(feats)


G = enc_images(gallery_paths, a.bs)
G_base = [Path(p).stem for p in gallery_paths]
log(f"gallery {tuple(G.shape)}")

hc = torch.load(QHIDDEN, map_location="cpu", weights_only=False)   # precomputed LLM pooled hidden
qh = hc["query_hidden"].float().to(DEV)
q_idx = list(hc["query_idx"])
with torch.no_grad():
    Q = F.normalize(model.get_text_features(qh).float(), dim=-1).cpu()
log(f"query {tuple(Q.shape)}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({"G": G, "G_base": G_base, "Q": Q, "Q_idx": q_idx}, OUT)
log(f"saved -> {OUT}  G={tuple(G.shape)} Q={tuple(Q.shape)}")
