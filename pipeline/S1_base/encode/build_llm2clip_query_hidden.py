#!/usr/bin/env python3
"""Rebuild the anchor5 inference input: pooled 8B hiddens of the test queries.

`encode_llm2clip_anchor5.py` never loads the 8B LLM — it reads the precomputed
`<ADAPTER>/query_hidden.pt` {query_hidden (1978,4096), query_idx}. That file is
bundled; this script regenerates it from `pab_test/query_text.json` when needed.
Encoding the queries is an inference-time operation, so it lives here and not in
`train/` (no training script reads the test set).

The default output goes to the reproduction tree (`assets/model_rep/`) so the
shipped file is never overwritten; `--rep` encode reads it from there. Copy it
into `assets/model/encoder/llm2clip_anchor5/` only if you mean to replace the
adopted input.

The 8B LLM is not bundled — download LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned
into assets/model/vlm_models/ or point LLM2CLIP_LLM at it. env: track4_llm2clip.

Usage:
  LLM2CLIP_DEV=cuda:6 python pipeline/S1_base/encode/build_llm2clip_query_hidden.py
"""
import json
import os
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"))

DEV = os.environ.get("LLM2CLIP_DEV", "cuda:0")
LLM = os.environ.get("LLM2CLIP_LLM",
                     f"{_REPO}/assets/model/vlm_models/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned")
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
QJSON = os.environ.get("QUERY_TEXT", f"{PAB_TEST}/query_text.json")
OUT = os.environ.get("OUT", f"{_REPO}/assets/model_rep/encoder/llm2clip_anchor5/query_hidden.pt")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


for p, what in ((LLM, "8B LLM (not bundled — see the docstring)"), (QJSON, "query_text.json")):
    if not os.path.exists(p):
        raise SystemExit(f"[query_hidden] {what} not found: {p}")

queries = [json.loads(l) for l in open(QJSON, encoding="utf-8") if l.strip()]
q_idx = [str(q["query_index"]) for q in queries]
q_caps = [str(q["caption"]) for q in queries]
log(f"queries: {len(q_caps)}")

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
def enc(caps, bs=32):
    out = []
    for i in range(0, len(caps), bs):
        inp = tok(caps[i:i + bs], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(DEV)
        o = llm(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        hid = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
        m = inp["attention_mask"].unsqueeze(-1).to(hid.dtype)
        out.append(((hid * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())   # (B,4096)
    return torch.cat(out)


t0 = time.time()
qh = enc(q_caps)
log(f"query hidden {tuple(qh.shape)} {time.time() - t0:.0f}s")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
torch.save({"query_hidden": qh, "query_idx": q_idx}, OUT)
log(f"saved -> {OUT}")
