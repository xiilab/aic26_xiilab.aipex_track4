#!/usr/bin/env python3
"""Multi-style 8B hidden cache for the text-adapter trainer.

Per training image, caches the pooled Llama-8B hidden state (4096d, pre-adaptor)
for K recaption presets; the text trainer then pulls paraphrases of the same
image together. Training-split rows only — this script reads nothing from the
test set. (The inference-side `query_hidden.pt` is a separate artifact, rebuilt
by `pipeline/S1_base/encode/build_llm2clip_query_hidden.py`.)

Output: OUT {train_paths, train_hidden_ms (N,K,4096), styles}

Styles are preset keys from the recap CSV. The default set puts the anchor
(source caption) first and adds four rewrite presets:
  p00_original · p07_telegraphic · p01_lexical · p03_clausal · p09_narrative

The cache used for the shipped adapter is bundled at
`assets/data/mining/llm2clip_hidden_cache_ms.pt` (148k x 5, same layout) — run
this script only to rebuild it. The anchor row reproduces exactly (p00_original
is the source caption); the four rewrite rows regenerate from the shipped
presets, which paraphrase the same axes the original rows were built from.

The 8B LLM is not bundled — see build_text_cache.py. env: track4_llm2clip.

Usage:
  LLM2CLIP_DEV=cuda:6 python train/encoders/llm2clip_anchor5/build_hidden_cache.py
"""
import csv
import json
import os
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"))
csv.field_size_limit(10 ** 7)

DEV = os.environ.get("LLM2CLIP_DEV", "cuda:0")
LLM = os.environ.get("LLM2CLIP_LLM",
                     f"{_REPO}/assets/model/vlm_models/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned")
RECAP = os.environ.get("RECAP_CSV", f"{_REPO}/assets/data/raw/recaption/train_msr_v1.csv")
MANIFEST = os.environ.get("MANIFEST", f"{_REPO}/assets/data/manifest/pab_manifest_msr_v1.jsonl")
OUT = os.environ.get("OUT", f"{_REPO}/assets/data/mining/llm2clip_hidden_cache_ms.pt")
N_IMG = int(os.environ.get("N_IMG", "148000"))   # matches the shipped cache size
STYLES = os.environ.get(
    "STYLES", "p00_original,p07_telegraphic,p01_lexical,p03_clausal,p09_narrative").split(",")

HELDOUT_DIR = os.environ.get("HELDOUT_DIR", f"{_REPO}/assets/data/heldout_v1")
EXCLUDE_HELDOUT = os.environ.get("EXCLUDE_HELDOUT", "1").lower() not in ("0", "false", "no")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


for p, what in ((LLM, "8B LLM (not bundled — see build_text_cache.py)"),
                (RECAP, "recap CSV"), (MANIFEST, "manifest")):
    if not os.path.exists(p):
        raise SystemExit(f"[hidden_cache] {what} not found: {p}")

heldout = set()
if EXCLUDE_HELDOUT:
    heldout = {l.strip() for l in open(f"{HELDOUT_DIR}/heldout_images.txt") if l.strip()}
    log(f"heldout exclusion on: {len(heldout):,} images")

# training image list, first N in manifest order
paths, seen = [], set()
with open(MANIFEST) as f:
    for line in f:
        if not line.strip():
            continue
        ip = json.loads(line).get("image")
        if not ip or ip in seen or ip in heldout:
            continue
        seen.add(ip)
        paths.append(ip)
        if len(paths) >= N_IMG:
            break
need = set(paths)
log(f"train images: {len(paths):,} × styles {STYLES}")

# (image, preset) -> caption from the recap CSV
want = {}
with open(RECAP, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ip, st = row.get("image_path"), row.get("style")
        if ip in need and st in STYLES:
            want[(ip, st)] = row.get("caption", "")
cover = sum(1 for p in paths if all((p, s) in want for s in STYLES))
log(f"style coverage: {cover:,}/{len(paths):,} images have all {len(STYLES)} presets")
if cover < 0.95 * len(paths):
    raise SystemExit("[hidden_cache] preset coverage below 95% — check RECAP_CSV/STYLES.")
paths = [p for p in paths if all((p, s) in want for s in STYLES)]

# FA3 probing in the remote code breaks without the wheel; keep FA2.
import transformers.utils as _u                                   # noqa: E402
import transformers.utils.import_utils as _iu                     # noqa: E402
for _m in (_iu, _u):
    if hasattr(_m, "is_flash_attn_3_available"):
        setattr(_m, "is_flash_attn_3_available", lambda *a, **k: False)
if hasattr(_iu, "_flash_attn_3_available"):
    _iu._flash_attn_3_available = False
from transformers import AutoConfig, AutoModel, AutoTokenizer     # noqa: E402

log("loading the 8B LLM ...")
llm = AutoModel.from_pretrained(LLM, torch_dtype=torch.bfloat16,
                                config=AutoConfig.from_pretrained(LLM, trust_remote_code=True),
                                trust_remote_code=True, attn_implementation="flash_attention_2").to(DEV).eval()
tok = AutoTokenizer.from_pretrained(LLM)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


@torch.no_grad()
def enc(caps, bs=32, tag=""):
    out = []
    for i in range(0, len(caps), bs):
        inp = tok(caps[i:i + bs], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEV)
        o = llm(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        hid = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
        m = inp["attention_mask"].unsqueeze(-1).to(hid.dtype)
        out.append(((hid * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())   # (B,4096)
        if (i // bs) % 50 == 0:
            log(f"  {tag} {i}/{len(caps)}")
    return torch.cat(out)


per_style = []
for s in STYLES:
    t0 = time.time()
    per_style.append(enc([want[(p, s)] for p in paths], tag=s))
    log(f"style {s} {time.time() - t0:.0f}s")
ms = torch.stack(per_style, dim=1)                                # (N,K,4096)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({"train_paths": paths, "train_hidden_ms": ms, "styles": STYLES}, OUT)
log(f"saved -> {OUT}  train_hidden_ms {tuple(ms.shape)}")
