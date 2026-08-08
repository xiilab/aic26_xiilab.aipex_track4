#!/usr/bin/env python3
"""mc2h378_peft (MetaCLIP2 worldwide-huge-378, ViT-H/14@378, DoRA) -> global feature dump.

The huge-378 counterpart of `encode_metaclip2.py`: same structure, different train module and adapter.
Output: mc2h378_peft_score.pt {G, Q, G_base, Q_idx}  (member score = norm(Q) @ norm(G).T)

  default  rebuild the adopted cache -> assets/cache/s1_base/members/
  --rep    reproduction encoding with model_rep weights -> assets/cache_rep/s1_base/members/

Usage:
  python pipeline/S1_base/encode/encode_mc2h378.py --gpu 6
  python pipeline/S1_base/encode/encode_mc2h378.py --gpu 6 --rep
"""
import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

NAME = "mc2h378_peft"
# The train module is the single source for the architecture; the inference helpers are shared
# with metaclip2 (only the model differs).
MODULE = f"{_REPO}/train/encoders/{NAME}_all/train.py"
INFER = f"{_REPO}/pipeline/S1_base/encode/metaclip2_infer.py"

ap = argparse.ArgumentParser(description="MetaCLIP2 huge-378 DoRA -> feats dump")
ap.add_argument("--gpu", default="0")
ap.add_argument("--ckpt", default="ep03", help="epoch tag to use when --run is given")
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--run", default=None, help="run directory (default = the bundled adapter); set it to score a retrained model")
ap.add_argument("--out", default=None, help="output .pt (default = members/mc2h378_peft_score.pt)")
ap.add_argument("--limit", type=int, default=None, help="truncate gallery/query for a smoke test")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction encoding with model_rep weights -> cache_rep")
a = ap.parse_args()
if a.rep and a.limit:                                         # keep truncated artifacts out of the reproduction cache
    raise SystemExit("--limit cannot be combined with --rep (write smoke runs elsewhere with --out).")
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

MODEL_ROOT = os.environ.get(
    "ENCODER_CKPT_DIR",
    f"{_REPO}/assets/model_rep/encoder" if a.rep else f"{_REPO}/assets/model/encoder")
CKPT_DIR = os.path.join(a.run, "checkpoints", a.ckpt) if a.run else f"{MODEL_ROOT}/{NAME}"
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
MEMBERS = os.environ.get(
    "S1_MEMBERS",
    f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}/s1_base/members")
skip_if_exists(a.out or f"{MEMBERS}/{NAME}_score.pt", a.overwrite)

if not os.path.isdir(CKPT_DIR):
    raise SystemExit(
        f"[input check failed]\n  - adapter not found: {CKPT_DIR}\n"
        + (f"      --rep requires a prior deployment: "
           f"python train/encoders/{NAME}_all/deploy.py <all_run>\n" if a.rep else ""))

spec = importlib.util.spec_from_file_location("mc2infer", INFER)
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
M = E.import_train_module(MODULE)
dev = "cuda:0"
print(f"[mc2h378] MODEL={M.MODEL_NAME} IMG={M.IMAGE_SIZE} ckpt={CKPT_DIR}", flush=True)


def load_queries_gallery():
    queries = []
    with open(f"{PAB_TEST}/query_text.json", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            queries.append({"query_index": str(r["query_index"]), "caption": r["caption"],
                            "change": r.get("change")})
    return queries, sorted(str(p) for p in Path(f"{PAB_TEST}/gallery").glob("*.jpg"))


queries, gallery_paths = load_queries_gallery()
if a.limit:
    queries, gallery_paths = queries[:a.limit], gallery_paths[:a.limit]
tf = E.build_eval_transform(M)
model, tok = E.load_for_inference(M, CKPT_DIR, dev)
t0 = time.time()
G, G_base = E.encode_gallery(M, model, gallery_paths, tf, dev, a.bs)
print(f"gallery {tuple(G.shape)} {time.time() - t0:.0f}s", flush=True)
Q, Q_idx, Q_chg, Q_len = E.encode_queries(M, model, queries, tok, dev, a.bs)
print(f"query {tuple(Q.shape)}", flush=True)

OUT = a.out or f"{MEMBERS}/{NAME}_score.pt"                          # file name read by build_base LOADERS
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({"G": G, "Q": Q, "G_base": G_base, "Q_idx": Q_idx, "run": a.run, "ckpt": a.ckpt}, OUT)
print(f"[save] {OUT}", flush=True)
