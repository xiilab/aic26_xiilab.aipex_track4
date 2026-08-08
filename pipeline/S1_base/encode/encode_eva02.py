"""EVA02-CLIP inference — builds the score matrix over the full 36,773-image gallery.

Usage:
  python encode_eva02.py
  python encode_eva02.py --checkpoint checkpoints/eva02_pab_ft/epoch_4.pt
"""
from __future__ import annotations
import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Repo-local HF cache. huggingface_hub reads HF_HOME at import time, so this has to precede the
# open_clip import; the base weights resolve to hub/models--timm--eva02_large_patch14_clip_336.*.
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"))

import numpy as np                                              # noqa: E402
import open_clip                                                # noqa: E402
import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
from PIL import Image                                           # noqa: E402
from tqdm import tqdm                                           # noqa: E402
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402


PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GALLERY_DIR  = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
QUERY_FILE   = os.environ.get("QUERY_TEXT", f"{PAB_TEST}/query_text.json")
SCORE_DIR    = os.environ.get("SCORE_DIR", os.environ.get("S1_MEMBERS", f"{_REPO}/assets/cache/s1_base/members"))
MODEL_NAME   = "EVA02-L-14-336"
PRETRAINED   = "merged2b_s6b_b61k"
BATCH_SIZE   = 128       # 336px has 2.25x the pixels of 224px, so the batch is reduced
MAX_TEXT_LEN = 77


def load_queries(path: str) -> list[str]:
    captions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            captions.append(json.loads(line)["caption"])
    print(f"Queries loaded: {len(captions)}")
    return captions


@torch.no_grad()
def encode_images(paths: list[str], model, preprocess, device: str) -> torch.Tensor:
    feats = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Images"):
        batch = [preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+BATCH_SIZE]]
        batch = torch.stack(batch).to(device)
        f = model.encode_image(batch)                    # fp32: AMP would shift the score by ~1e-3
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)  # [G, D]


@torch.no_grad()
def encode_texts(captions: list[str], model, tokenizer, device: str) -> torch.Tensor:
    feats = []
    for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="Texts"):
        batch = captions[i:i+BATCH_SIZE]
        tokens = tokenizer(batch).to(device)
        f = model.encode_text(tokens)                    # fp32, as above
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats, dim=0)  # [Q, D]


def main(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {MODEL_NAME}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED if not args.checkpoint else None
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        # Strip the DDP prefix from OpenCLIP checkpoint keys
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)

    model = model.to(device).eval()

    # Gallery image list, sorted: this is the score matrix column order
    gallery_files = sorted(os.listdir(GALLERY_DIR))
    gallery_paths = [os.path.join(GALLERY_DIR, f) for f in gallery_files]

    captions = load_queries(QUERY_FILE)

    print("Encoding gallery images...")
    img_feats = encode_images(gallery_paths, model, preprocess, device)  # [G, D]

    print("Encoding query texts...")
    txt_feats = encode_texts(captions, model, tokenizer, device)          # [Q, D]

    # score matrix: [Q, G]
    sims = txt_feats @ img_feats.t()

    tag = "eva02_ft" if args.checkpoint else "eva02_pre"
    score_dir = f"{_REPO}/assets/cache_rep/s1_base/members" if args.rep else SCORE_DIR
    os.makedirs(score_dir, exist_ok=True)
    score_path = os.path.join(score_dir, f"{tag}_score.pt")   # build_base LOADERS['eva02_pre'] = eva02_pre_score.pt
    torch.save(sims, score_path)
    print(f"Score matrix saved: {score_path}  {tuple(sims.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="fine-tuned checkpoint path; without it the pretrained model produces the eva02_pre member")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild even if the artifact exists (default: skip)")
    parser.add_argument("--rep", action="store_true",
                        help="reproduction encoding -> cache_rep/s1_base/members (pretrained member, model_rep not needed)")
    args = parser.parse_args()
    _sd = f"{_REPO}/assets/cache_rep/s1_base/members" if args.rep else SCORE_DIR
    _tag = "eva02_ft" if args.checkpoint else "eva02_pre"
    skip_if_exists(os.path.join(_sd, f"{_tag}_score.pt"), args.overwrite)
    main(args)
