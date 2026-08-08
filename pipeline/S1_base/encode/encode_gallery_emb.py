#!/usr/bin/env python3
"""encode_gallery_emb — one-shot zero-shot CLIP embedding dump of the test gallery.

Produces the `{emb[N,d], stems[N]}` format that S4b tail-NN
(`../../S4_tail/tail_refinement.py`) uses to find near-duplicate neighbours. Gallery columns are
`sorted(listdir(GALLERY))`, the same ordering as every stage score matrix, so indices carry over.

  --enc dfn       DFN5B-CLIP-ViT-H-14-378   -> dfn_gallery_emb.pt
  --enc convnext  ConvNeXt-Large-320        -> convnext_gallery_emb.pt

Output: assets/cache/s4_nn/ by default, assets/cache_rep/s4_nn/ with `--rep`.

Usage (needs open_clip, i.e. a beit3/openclip interpreter):
  CUDA_VISIBLE_DEVICES=6 <py> encode_gallery_emb.py --enc dfn --rep

Weights default to the bundled `assets/model/vlm_models/<model>/open_clip_pytorch_model.bin`.
Environment: DFN_WEIGHTS · CONVNEXT_WEIGHTS · VLM_MODELS · PAB_TEST · GALLERY · S4_NN
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402


VLM = os.environ.get("VLM_MODELS", f"{_REPO}/assets/model/vlm_models")
BIN = "open_clip_pytorch_model.bin"

# member -> (open_clip model name, weights env var, bundled weights, output file name, image size)
ENCODERS = {
    "dfn":      ("ViT-H-14-378-quickgelu", "DFN_WEIGHTS",      f"{VLM}/DFN5B-CLIP-ViT-H-14-378/{BIN}",
                 "dfn_gallery_emb.pt",      378),
    "convnext": ("convnext_xxlarge",       "CONVNEXT_WEIGHTS", f"{VLM}/CLIP-convnext_xxlarge-laion2B/{BIN}",
                 "convnext_gallery_emb.pt", 256),
}

ap = argparse.ArgumentParser(description="Zero-shot gallery embedding dump -> S4b tail-NN input")
ap.add_argument("--enc", required=True, choices=sorted(ENCODERS))
ap.add_argument("--model", default=None, help="open_clip model name (default = the --enc preset)")
ap.add_argument("--weights", default=None, help="weights file (default = the environment variable)")
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--out", default=None)
ap.add_argument("--limit", type=int, default=0, help="truncate the gallery for a smoke test")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction encoding -> cache_rep/s4_nn")
a = ap.parse_args()
if a.rep and a.limit:
    raise SystemExit("[encode_gallery_emb] --limit cannot be combined with --rep (prevents truncated artifacts).")

model_name, wenv, bundled, fname, img_size = ENCODERS[a.enc]
model_name = a.model or model_name
weights = a.weights or os.environ.get(wenv) or bundled                  # default = the bundled weights
if not os.path.exists(weights):
    raise SystemExit(f"[encode_gallery_emb] {a.enc} weights not found: {weights}\n"
                     f"  If the bundled copy was removed, point --weights or {wenv} at one.")

PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GAL = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
S4NN = os.environ.get("S4_NN", f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}/s4_nn")
OUT = a.out or f"{S4NN}/{fname}"
skip_if_exists(OUT, a.overwrite)
DEV = "cuda:0"


def log(*x):
    print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)


import open_clip                                                   # noqa: E402

log(f"load {model_name}  weights={weights}")
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=weights)
model = model.to(DEV).eval()

gfiles = sorted(os.listdir(GAL))
if a.limit:
    gfiles = gfiles[:a.limit]
stems = [f[:-4] if f.endswith(".jpg") else f for f in gfiles]
log(f"gallery {len(gfiles)}")


class _DS(torch.utils.data.Dataset):
    def __len__(self):
        return len(gfiles)

    def __getitem__(self, i):
        try:
            return preprocess(Image.open(f"{GAL}/{gfiles[i]}").convert("RGB"))
        except Exception:
            return torch.zeros(3, img_size, img_size)


dl = torch.utils.data.DataLoader(_DS(), batch_size=a.bs, num_workers=a.workers, pin_memory=True)
embs, done, t0 = [], 0, time.time()
with torch.no_grad():
    for px in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embs.append(F.normalize(model.encode_image(px.to(DEV)).float(), dim=-1).cpu())
        done += px.size(0)
        if done % (a.bs * 20) == 0:
            log(f"  {done}/{len(gfiles)} ({done/(time.time()-t0):.0f}/s)")

G = torch.cat(embs)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({"emb": G, "stems": stems}, OUT)
log(f"done {tuple(G.shape)} → {OUT}")
