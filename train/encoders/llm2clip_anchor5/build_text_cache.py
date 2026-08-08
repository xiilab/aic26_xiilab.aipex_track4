#!/usr/bin/env python3
"""Stage A cache for the vision-LoRA trainer: frozen text features.

Encodes training captions once through the full LLM2CLIP text path
(8B Llama-CC -> text_adapter), so the vision trainer never needs the 8B model
again. Training-split captions only — nothing test-derived goes in here.

Output: text_cache.pt {train_paths, train_text (N,1280)}
The cache used for the shipped adapter is bundled at
`assets/data/mining/llm2clip_text_cache.pt` — run this script only to rebuild it.

The 8B LLM is not bundled (it is only needed to build caches) — download
LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned into assets/model/vlm_models/ or point
LLM2CLIP_LLM at it. env: track4_llm2clip.

Usage:
  LLM2CLIP_DEV=cuda:6 python train/encoders/llm2clip_anchor5/build_text_cache.py
"""
import json
import os
import time

import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"))

DEV = os.environ.get("LLM2CLIP_DEV", "cuda:0")
LLM = os.environ.get("LLM2CLIP_LLM",
                     f"{_REPO}/assets/model/vlm_models/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned")
VIS = os.environ.get("LLM2CLIP_BASE", "microsoft/LLM2CLIP-Openai-L-14-336")
MANIFEST = os.environ.get("MANIFEST", f"{_REPO}/assets/data/manifest/pab_manifest_msr_v1.jsonl")
OUT = os.environ.get("OUT", f"{_REPO}/assets/data/mining/llm2clip_text_cache.pt")
N_TRAIN = int(os.environ.get("N_TRAIN", "150000"))

HELDOUT_DIR = os.environ.get("HELDOUT_DIR", f"{_REPO}/assets/data/heldout_v1")
EXCLUDE_HELDOUT = os.environ.get("EXCLUDE_HELDOUT", "1").lower() not in ("0", "false", "no")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


for p, what in ((LLM, "8B LLM (not bundled — see the docstring)"), (MANIFEST, "manifest")):
    if not os.path.exists(p):
        raise SystemExit(f"[text_cache] {what} not found: {p}")

heldout = set()
if EXCLUDE_HELDOUT:
    heldout = {l.strip() for l in open(f"{HELDOUT_DIR}/heldout_images.txt") if l.strip()}
    log(f"heldout exclusion on: {len(heldout):,} images")

train_paths, train_caps, seen = [], [], set()
with open(MANIFEST) as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        ip, caps = r.get("image"), r.get("captions")
        if not ip or not caps or ip in seen or ip in heldout:
            continue
        seen.add(ip)
        train_paths.append(ip)
        train_caps.append(caps[0])
        if len(train_paths) >= N_TRAIN:
            break
log(f"train pairs: {len(train_paths):,}")

# FA3 probing in the remote code breaks without the wheel; keep FA2.
import transformers.utils as _u                                   # noqa: E402
import transformers.utils.import_utils as _iu                     # noqa: E402
for _m in (_iu, _u):
    if hasattr(_m, "is_flash_attn_3_available"):
        setattr(_m, "is_flash_attn_3_available", lambda *a, **k: False)
if hasattr(_iu, "_flash_attn_3_available"):
    _iu._flash_attn_3_available = False
from transformers import AutoConfig, AutoModel, AutoTokenizer     # noqa: E402

log("loading the 8B text path (LLM + text_adapter) ...")
llm = AutoModel.from_pretrained(LLM, torch_dtype=torch.bfloat16,
                                config=AutoConfig.from_pretrained(LLM, trust_remote_code=True),
                                trust_remote_code=True, attn_implementation="flash_attention_2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained(LLM)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
clip = AutoModel.from_pretrained(VIS, torch_dtype=torch.float32, trust_remote_code=True,
                                 attn_implementation="sdpa").to(DEV).eval()


@torch.no_grad()
def enc(caps, bs=32, tag=""):
    out = []
    for i in range(0, len(caps), bs):
        inp = tok(caps[i:i + bs], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEV)
        o = llm(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        hid = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
        m = inp["attention_mask"].unsqueeze(-1).to(hid.dtype)
        pooled = (hid * m).sum(1) / m.sum(1).clamp(min=1)         # (B,4096) mean-pool
        feat = clip.get_text_features(pooled.float())             # hidden -> text_adapter (1280)
        out.append(F.normalize(feat.float(), dim=-1).cpu())
        if (i // bs) % 50 == 0:
            log(f"  {tag} {i}/{len(caps)}")
    return torch.cat(out)


t0 = time.time()
tf = enc(train_caps, tag="train")
log(f"train text {tuple(tf.shape)} {time.time() - t0:.0f}s")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({"train_paths": train_paths, "train_text": tf}, OUT)
log(f"saved -> {OUT}")
