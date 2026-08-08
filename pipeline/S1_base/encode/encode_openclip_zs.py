"""Zero-shot encoding with an open_clip backbone -> [Q, gallery] score.

Same ordering as eva02_pre: columns = sorted gallery, rows = query_index.txt order.
No fine-tuning, so the member is expected to be decorrelated from the fine-tuned members.

usage: python encode_openclip_zs.py --model ViT-H-14-378-quickgelu --pretrained dfn5b --out score_dfn_h378_zs.pt --gpu 6
"""
import argparse, json, os, time
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GAL = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
QJSON = os.environ.get("QUERY_TEXT", f"{PAB_TEST}/query_text.json")
QIDX = os.environ.get("QUERY_INDEX", f"{PAB_TEST}/query_index.txt")

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--pretrained", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--gpu", default="6")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction encoding -> cache_rep/s1_base/members (zero-shot)")
a = ap.parse_args()
T4 = os.environ.get("S1_MEMBERS",
                    f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}/s1_base/members")
os.makedirs(T4, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
dev = "cuda:0"
import open_clip
def log(*x): print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)

log(f"[load] {a.model} / {a.pretrained} ...")
model, _, pp = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained)
tok = open_clip.get_tokenizer(a.model)
model = model.to(dev).eval()

gfiles = sorted(os.listdir(GAL)); gstem = [f[:-4] if f.endswith(".jpg") else f for f in gfiles]
qrows = [json.loads(l) for l in open(QJSON) if l.strip()]
# Order by query_index.txt
qo = [l.strip() for l in open(QIDX) if l.strip()]
qcap_by_id = {str(r["query_index"]): r["caption"] for r in qrows}
qcaps = [qcap_by_id[q] for q in qo]
log(f"gallery {len(gfiles)} | queries {len(qcaps)}")

@torch.no_grad()
def enc_img():
    out = []
    for i in range(0, len(gfiles), a.batch):
        ims = torch.stack([pp(Image.open(f"{GAL}/{f}").convert("RGB")) for f in gfiles[i:i+a.batch]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f_ = model.encode_image(ims)
        out.append(F.normalize(f_.float(), dim=-1).cpu())
        if (i//a.batch) % 30 == 0: log(f"  img {i}/{len(gfiles)}")
    return torch.cat(out)

@torch.no_grad()
def enc_txt():
    out = []
    for i in range(0, len(qcaps), 256):
        tk = tok(qcaps[i:i+256]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f_ = model.encode_text(tk)
        out.append(F.normalize(f_.float(), dim=-1).cpu())
    return torch.cat(out)

t0 = time.time(); G = enc_img(); log(f"img done {G.shape} {time.time()-t0:.0f}s")
Q = enc_txt(); log(f"txt done {Q.shape}")
score = (Q @ G.t())                                  # [Q, gallery]: sorted-gallery columns, query_index rows
torch.save(score, f"{T4}/{a.out}")
log(f"[save] {a.out}  {tuple(score.shape)}")

