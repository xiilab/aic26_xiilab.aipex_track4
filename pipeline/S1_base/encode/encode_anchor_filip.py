#!/usr/bin/env python3
"""anchor_filip (SigLIP2-L512 + DoRA, ep10) multi-view encoding -> views feats dump.

Encodes each base+hflip+zoom view once and stores them as the views file used by the ensemble.
Inference only: no training and no scoring.

Views: base / hflip / z088 / z080 / z094 / hf_z088   (zNNN = center-crop 0.NN -> resize)
Output: anchor_filip_tta_views.pt  {img:{view:G}, txt:{"base":Q}, G_base, Q_idx}
usage: python encode_anchor_filip.py --gpu 2
"""
import argparse, importlib.util, json, os, time
from pathlib import Path
import torch, torch.nn.functional as F
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = f"{_REPO}/train/encoders/anchor_filip_all/train.py"   # train module, which owns the eval functions
CKPT        = os.environ.get("ENCODER_CKPT", f"{_REPO}/assets/model/encoder/anchor_filip")   # bundled DoRA adapter
BEST_CKPT   = os.path.basename(CKPT)
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
TEST_ROOT   = PAB_TEST
T4          = os.environ.get("S1_MEMBERS", f"{_REPO}/assets/cache/s1_base/members")   # artifact output root
OUT_VIEWS   = f"{T4}/anchor_filip_tta_views.pt"


import anchor_infer as AI                                             # noqa: E402


def import_module_by_path(path):
    """Inject the train module into anchor_infer and return the module."""
    return AI.init(path)


def load_queries_gallery():
    """Load queries and gallery paths (no labels)."""
    queries = []
    with open(f"{TEST_ROOT}/query_text.json", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            queries.append({"query_index": str(r["query_index"]),
                            "caption": r["caption"], "change": r.get("change")})
    gallery_paths = sorted(str(p) for p in Path(f"{TEST_ROOT}/gallery").glob("*.jpg"))
    return queries, gallery_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--gallery-batch", type=int, default=48)
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild even if the artifact exists (default: skip)")
    ap.add_argument("--rep", action="store_true",
                    help="reproduction encoding with model_rep weights -> cache_rep/s1_base/members")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.rep:                              # rep: deployed weights -> reproduction cache
        ckpt_dir_rep = f"{_REPO}/assets/model_rep/encoder/anchor_filip"
        mem_rep = os.environ.get("S1_MEMBERS", f"{_REPO}/assets/cache_rep/s1_base/members")
        os.makedirs(mem_rep, exist_ok=True)
    args.out = args.out or ((f"{mem_rep}/anchor_filip_tta_views.pt") if args.rep else OUT_VIEWS)
    skip_if_exists(args.out, args.overwrite)

    import torchvision.transforms.v2 as Tv2
    device = "cuda:0"
    mod = import_module_by_path(MODULE_PATH); mod.EVAL_BATCH_SIZE = args.gallery_batch
    S = mod.IMAGE_SIZE
    ckpt_dir = ckpt_dir_rep if args.rep else CKPT
    print("="*64); print(f"[tta] ckpt={os.path.basename(ckpt_dir)} IMG={S}"); print("="*64, flush=True)

    queries, gallery_paths = load_queries_gallery()

    norm = [Tv2.ToDtype(torch.float32, scale=True), Tv2.Normalize(mean=[0.5]*3, std=[0.5]*3)]
    def zoom(r): return [Tv2.CenterCrop(int(round(S*r))), Tv2.Resize(S, antialias=True)]
    tfs = {
        "base":    Tv2.Compose(norm),
        "hflip":   Tv2.Compose([Tv2.RandomHorizontalFlip(1.0), *norm]),
        "z088":    Tv2.Compose([*zoom(0.88), *norm]),
        "z080":    Tv2.Compose([*zoom(0.80), *norm]),
        "z094":    Tv2.Compose([*zoom(0.94), *norm]),
        "hf_z088": Tv2.Compose([Tv2.RandomHorizontalFlip(1.0), *zoom(0.88), *norm]),
    }

    model, tok = AI.load_for_inference(mod.MODEL_NAME, ckpt_dir, device=device)
    img_views, G_base = {}, None
    for name, tf in tfs.items():
        t0 = time.time()
        G, G_base = AI.encode_test_gallery(model, gallery_paths, tf, device=device,
                                            batch_size=args.gallery_batch)
        img_views[name] = G.float().cpu()
        print(f"[img-view {name}] ({time.time()-t0:.0f}s)", flush=True)
    Q, Q_idx, _, _ = AI.encode_test_queries(model, queries, tok, device=device)

    torch.save({"img": img_views,
                "txt": {"base": Q.float().cpu()},
                "G_base": G_base,
                "Q_idx": Q_idx}, args.out)
    print(f"[save] {args.out}  (img views={list(img_views)}, txt=base)", flush=True)


if __name__ == "__main__":
    main()
