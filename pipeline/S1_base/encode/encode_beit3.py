#!/usr/bin/env python3
"""beit3 (BEiT3-large-384 full FT) -> score [1978, 36773] dump.

Handles both recipes in one script; they differ only in the text token length.

  --recipe v2     beit3_v2     t64    -> beit3_v2_score.pt
  --recipe helip  beit3_helip  t128   -> beit3_helip_score.pt

The output is a score matrix (dot product of L2-normalized text and image embeddings), read by the
`score` loader in build_base. Columns follow `sorted(os.listdir(GALLERY))` and rows follow
query_text.json, the convention shared by every member.

  default  rebuild the adopted cache -> assets/cache/s1_base/members/
  --rep    reproduction encoding with model_rep weights -> assets/cache_rep/s1_base/members/

env: `track4_beit3` (needs torchscale and timm — see requirements/README.md).
Usage:
  python pipeline/S1_base/encode/encode_beit3.py --recipe v2    --gpu 6
  python pipeline/S1_base/encode/encode_beit3.py --recipe helip --gpu 6 --rep
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402


# recipe -> (deployment name, max text tokens)
RECIPES = {"v2": ("beit3_v2", 64), "helip": ("beit3_helip", 128)}

ap = argparse.ArgumentParser(description="BEiT3 full-FT -> score matrix dump")
ap.add_argument("--recipe", required=True, choices=list(RECIPES))
ap.add_argument("--gpu", default="0")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--ckpt", default=None, help="checkpoint-*.pth to use (default = the bundled deployment)")
ap.add_argument("--out", default=None, help="output .pt (default = members/<deployment name>_score.pt)")
ap.add_argument("--limit", type=int, default=None, help="truncate gallery/query for a smoke test")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction encoding with model_rep weights -> cache_rep")
a = ap.parse_args()
if a.rep and a.limit:                                         # keep truncated artifacts out of the reproduction cache
    raise SystemExit("--limit cannot be combined with --rep (write smoke runs elsewhere with --out).")
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

NAME, MAX_TOKENS = RECIPES[a.recipe]
MODEL_ROOT = f"{_REPO}/assets/model_rep/encoder" if a.rep else f"{_REPO}/assets/model/encoder"
CKPT = a.ckpt or os.environ.get("ENCODER_CKPT", f"{MODEL_ROOT}/{NAME}/checkpoint-best.pth")
SPM = os.environ.get("BEIT3_SPM", f"{_REPO}/assets/model/encoder/beit3_pre/beit3.spm")
BEIT3_SRC = os.environ.get("BEIT3_SRC", f"{_REPO}/third_party/beit3")
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GALLERY = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
QUERY_TEXT = os.environ.get("QUERY_TEXT", f"{PAB_TEST}/query_text.json")
MEMBERS = os.environ.get(
    "S1_MEMBERS",
    f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}/s1_base/members")
skip_if_exists(a.out or f"{MEMBERS}/{NAME}_score.pt", a.overwrite)

problems = [f"{k} not found: {v}" for k, v in
            (("checkpoint", CKPT), ("tokenizer", SPM), ("gallery", GALLERY), ("queries", QUERY_TEXT))
            if not os.path.exists(v)]
if problems:
    raise SystemExit("[input check failed] resolve the following and run again:\n"
                     + "\n".join(f"  - {p}" for p in problems)
                     + ("\n  --rep requires a prior deployment: "
                        f"python train/encoders/beit3/deploy.py <all_run> --recipe {a.recipe} --epoch <e*>"
                        if a.rep else ""))

import torch                                                         # noqa: E402  (keeps --help fast)
from PIL import Image                                                # noqa: E402
from timm import create_model                                        # noqa: E402
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD   # noqa: E402
from torchvision import transforms                                   # noqa: E402
from transformers import XLMRobertaTokenizer                         # noqa: E402

sys.path.insert(0, BEIT3_SRC)
sys.modules.pop("utils", None)

import modeling_finetune   
DEV = "cuda" if torch.cuda.is_available() else "cpu"
tfm = transforms.Compose([
    transforms.Resize((384, 384), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
])
tok = XLMRobertaTokenizer(vocab_file=SPM)


def to_tokens(text, max_len=MAX_TOKENS):
    ids = tok.convert_tokens_to_ids(tok.tokenize(text))[:max_len - 2]
    ids = [tok.bos_token_id] + ids + [tok.eos_token_id]
    n = len(ids)
    pad = [0] * n + [1] * (max_len - n)
    return torch.tensor(ids + [tok.pad_token_id] * (max_len - n)), torch.tensor(pad)


print(f"[beit3] recipe={a.recipe} name={NAME} t{MAX_TOKENS} ckpt={CKPT}", flush=True)
model = create_model("beit3_large_patch16_384_retrieval")
model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"])
model = model.to(DEV).eval()

gal = sorted(os.listdir(GALLERY))                                    # submission column convention
gpaths = [os.path.join(GALLERY, f) for f in gal]
caps = [json.loads(l)["caption"] for l in open(QUERY_TEXT, encoding="utf-8") if l.strip()]
if a.limit:
    gpaths, caps = gpaths[:a.limit], caps[:a.limit]
print(f"[beit3] gallery={len(gpaths)} query={len(caps)}", flush=True)


@torch.no_grad()
def encode_gallery():
    out = []
    for i in range(0, len(gpaths), a.bs):
        im = torch.stack([tfm(Image.open(p).convert("RGB")) for p in gpaths[i:i + a.bs]]).to(DEV)
        f, _ = model(image=im, only_infer=True)
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
        if (i // a.bs) % 50 == 0:
            print(f"  gallery {min(i + a.bs, len(gpaths))}/{len(gpaths)}", flush=True)
    return torch.cat(out)


@torch.no_grad()
def encode_queries():
    out = []
    for i in range(0, len(caps), a.bs):
        ids, pad = zip(*[to_tokens(t) for t in caps[i:i + a.bs]])
        _, f = model(text_description=torch.stack(ids).to(DEV),
                     padding_mask=torch.stack(pad).to(DEV), only_infer=True)
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out)


G = encode_gallery()
Q = encode_queries()
S = Q @ G.t()
print(f"[beit3] score {tuple(S.shape)}", flush=True)

OUT = a.out or f"{MEMBERS}/{NAME}_score.pt"                          # file name read by build_base LOADERS
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save(S, OUT)
print(f"[save] {OUT}", flush=True)
