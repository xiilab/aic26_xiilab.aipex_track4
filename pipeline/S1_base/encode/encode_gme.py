"""GME-Qwen2-VL-2B zero-shot -> full-gallery feats `{G, Q, G_base, Q_idx}`.

Builds the `gme` member (weight 0.20) of the base ensemble. There is no training: the pretrained
`Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` is used as is.

The output `$TRACK4/gme_feats.pt` is the file read by `LOADERS["gme"]` in
[`../build_base.py`](../build_base.py) (G [36773,1536], Q [1978,1536]).

Setting `GME_ADAPTER` injects a LoRA adapter instead of running zero-shot; the output path changes
with it so the deployed dump is not overwritten.

Usage:
    CUDA_VISIBLE_DEVICES=7 HF_HOME=$HF_CACHE $PY_gme pipeline/S1_base/encode/encode_gme.py
"""
import os, json, glob, time, torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

os.environ["HF_HOME"] = os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"); os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import transformers.utils.versions as _v; _v.require_version = lambda *x, **k: None
import transformers; transformers.utils.require_version = lambda *x, **k: None

import argparse                                                   # noqa: E402
_ap = argparse.ArgumentParser(description="gme-Qwen2-VL-2B zero-shot -> feats dump")
_ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES")
_ap.add_argument("--out", default=None, help="output .pt (default = members/gme_feats.pt)")
_ap.add_argument("--overwrite", action="store_true",
                 help="rebuild even if the artifact exists (default: skip)")
_ap.add_argument("--rep", action="store_true",
                 help="reproduction encoding -> cache_rep/s1_base/members (zero-shot, model_rep not needed)")
_a = _ap.parse_args()
if _a.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = _a.gpu
T4 = os.environ.get("S1_MEMBERS",
                    f"{_REPO}/assets/{'cache_rep' if _a.rep else 'cache'}/s1_base/members")
BASE = os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache") + "/hub/models--Alibaba-NLP--gme-Qwen2-VL-2B-Instruct/snapshots/9cfa6413f704a7c1cf5064d240748e10c876b286"
ADAPTER = os.environ.get("GME_ADAPTER", "")      # default = unused (zero-shot); LoRA is injected only when set
OUT = _a.out or os.environ.get("GME_OUT", f"{T4}/gme_feats.pt")
skip_if_exists(OUT, _a.overwrite)
INSTR = "Find an image that matches the given text."
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
TEST = PAB_TEST
GAL = f"{TEST}/gallery"; QJSON = f"{TEST}/query_text.json"

from transformers import AutoModel
print(f"[load] base={BASE}", flush=True)
model = AutoModel.from_pretrained(BASE, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                  attn_implementation="sdpa").to("cuda:0").eval()
if ADAPTER:
    from peft import PeftModel
    print(f"[load] adapter={ADAPTER}  (not the deployed configuration)", flush=True)
    PeftModel.from_pretrained(model, ADAPTER)   # injected in place, so get_fused_embeddings picks it up
else:
    print("[load] adapter=none -> zero-shot (deployed configuration)", flush=True)
# Pixel budget: upscales small person crops. Same setting as the deployed dump; changing it makes
# the scores non-reproducible.
ip = model.processor.image_processor; ip.min_pixels = 64*28*28; ip.max_pixels = 256*28*28

_q = [json.loads(l) for l in open(QJSON) if l.strip()]
QCAP = [x["caption"] for x in _q]; QID = [x["query_index"] for x in _q]
gp = sorted(glob.glob(f"{GAL}/*.jpg")); gb = [os.path.basename(p)[:-4] for p in gp]
print(f"[data] gallery={len(gp)} query={len(QCAP)}", flush=True)

t0 = time.time()
with torch.no_grad():
    G = model.get_fused_embeddings(images=gp, is_query=False, batch_size=32, show_progress_bar=False).float().cpu()
    print(f"[G] {tuple(G.shape)} {time.time()-t0:.0f}s", flush=True)
    Q = model.get_fused_embeddings(texts=QCAP, instruction=INSTR, batch_size=64, show_progress_bar=False).float().cpu()
    print(f"[Q] {tuple(Q.shape)} {time.time()-t0:.0f}s", flush=True)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({"G": G, "Q": Q, "G_base": gb, "Q_idx": QID}, OUT)
print(f"[save] {OUT}  G={tuple(G.shape)} Q={tuple(Q.shape)}  total {time.time()-t0:.0f}s", flush=True)
if _a.rep and not _a.out:                       # S4b tail-NN uses the same feats (ENC["gme"])
    import shutil
    _s4 = f"{_REPO}/assets/cache_rep/s4_nn"
    os.makedirs(_s4, exist_ok=True)
    shutil.copy2(OUT, f"{_s4}/gme_feats.pt")
    print(f"[save] {_s4}/gme_feats.pt (tail-NN)", flush=True)
