#!/usr/bin/env python3
"""score_union_hf — re-score the union candidate pool with an HF yes/no VLM (internvl_r32, llama).

An existing dump passed with --reuse is reused, so the model is called only for new (q,c) pairs.
Output: dict {(q,c): score} saved as {name}_union.pt.

Usage: CUDA_VISIBLE_DEVICES=7 <py> score_union_hf_4b.py --model M [--adapter A] --name internvl_r32 --reuse internvl_r32
"""
import argparse, os, json, time, numpy as np, torch
from PIL import Image
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402
from utils.env import needs_moe, grouped_mm_ok                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--adapter", default=None); ap.add_argument("--reuse", default=None)
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction union re-scoring with model_rep weights -> cache_rep/s2_rerank")
ap.add_argument("--q-start", type=int, default=0, help="first query of the slice (for split runs)")
ap.add_argument("--q-end", type=int, default=0, help="last query of the slice (0 = all); parts are saved separately and merged")
a = ap.parse_args()
if a.rep:                                                       # force cache_rep/s2_rerank and the deployed adapter
    _CR = f"{_REPO}/assets/cache_rep"; os.environ["WORKDIR"] = f"{_CR}/s2_rerank"
    os.environ["OUT_SUFFIX"] = "union_cache"; os.environ["POOL_FILE"] = "../s1_base/union_pool.pt"
    os.makedirs(f"{_CR}/s2_rerank", exist_ok=True)
    _ad = f"{_REPO}/assets/model_rep/reranker/{a.name}"        # only trained rerankers have an adapter
    if a.adapter is None and os.path.isdir(_ad): a.adapter = _ad
    if not os.path.exists(f"{_CR}/s1_base/union_pool.pt"):
        print("  cache_rep/s1_base/union_pool.pt not found — run build_union with --rep first", flush=True)
T4 = os.environ.get("TRACK4", f"{_REPO}/assets/cache/work")
HERE = os.environ.get("WORKDIR", f"{T4}/rerank_work")          # candidate pool and output directory
# Skip recomputation when the artifact already exists (default; --overwrite forces a rebuild).
_final = f"{HERE}/{a.name}_" + (f"q{a.q_start}_{a.q_end}_" if getattr(a, "q_end", 0) else "") \
         + os.environ.get("OUT_SUFFIX", "union") + ".pt"
skip_if_exists(_final, a.overwrite)

PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GAL = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
QJSON = os.environ.get("QUERY_TEXT", f"{PAB_TEST}/query_text.json")
PROMPT = ('You are judging whether an image matches a text description of a person.\nDescription: "{cap}"\n'
          'Does this image show EXACTLY that person and situation — matching gender, clothing (color/type), the action/behavior, and the scene? Answer a single word: yes or no.')
YES = ["yes", "Yes", " yes", " Yes", "YES"]; NO = ["no", "No", " no", " No", "NO"]
gal = sorted(os.listdir(GAL)); gpath = [os.path.join(GAL, g) for g in gal]
cap = {json.loads(l)["query_index"]: json.loads(l)["caption"] for l in open(QJSON) if l.strip()}
U = torch.load(f"{HERE}/{os.environ.get('POOL_FILE','union_pool.pt')}"); union = U["union"]; qorder = U["qorder"]

reuse = {}
if a.reuse:
    d = f"{T4}/outputs/fuse_internvl/{a.reuse}"
    m = json.load(open(f"{d}/internvl_scores_{a.reuse}_meta.json"))
    sc = np.load(f"{d}/internvl_scores_{a.reuse}_top20.npy"); cd = np.load(f"{d}/internvl_cand_{a.reuse}_top20.npy")
    for i, q in enumerate(m["qorder"]):
        for j in range(cd.shape[1]):
            if not np.isnan(sc[i, j]): reuse[(q, int(cd[i, j]))] = float(sc[i, j])
    print(f"[reuse] {a.reuse}: {len(reuse)} pairs", flush=True)

# REUSE_EXTRA: reuse existing score caches so only new (q,c) pairs are scored.
for _rp in os.environ.get("REUSE_EXTRA","").split(":"):
    if _rp and __import__("os").path.exists(_rp):
        _d = torch.load(_rp, weights_only=False)["scores"]
        reuse.update({(q, int(c)): float(v) for (q, c), v in _d.items()})
        print(f"[reuse-extra] {_rp}: +{len(_d)}", flush=True)

# MoE models need torch._grouped_mm to work on this GPU. torch 2.8 defines it but supports only
# sm_90, so every forward fails on Blackwell; check before loading rather than after a long run.
if needs_moe(a.model) and not grouped_mm_ok():
    raise SystemExit(
        f"[score_union_hf] {a.model} is MoE and needs torch._grouped_mm, but the current\n"
        f"  interpreter ({_sys.executable}) fails on this GPU (torch 2.8 supports sm_90 only).\n"
        f"  Run it under a torch 2.11+cu130 environment — see requirements/README.md (track4_vllm)")

from transformers import AutoModelForImageTextToText, AutoProcessor
t0 = time.time(); proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
if a.adapter:
    import json as _j
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file as _lf
    _c = _j.load(open(a.adapter + "/adapter_config.json"))
    _lc = LoraConfig(r=_c["r"], lora_alpha=_c["lora_alpha"], lora_dropout=_c.get("lora_dropout", 0.0),
                     bias=_c.get("bias", "none"), target_modules=_c["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, _lc); _sd = _lf(a.adapter + "/adapter_model.safetensors")
    _rm = {k.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): v for k, v in _sd.items()}
    _r = model.load_state_dict(_rm, strict=False); assert len([k for k in _r.missing_keys if "lora" in k]) == 0
    model = model.eval(); print("[adapter] loaded", flush=True)
tok = proc.tokenizer
def ids(ws):
    s = set()
    for w in ws:
        e = tok.encode(w, add_special_tokens=False)
        if len(e) == 1: s.add(e[0])
    return s
yi, ni = ids(YES), ids(NO); print(f"[load] {time.time()-t0:.0f}s", flush=True); assert yi and ni
imgc = {}
@torch.no_grad()
def sp(c, txt):
    if c not in imgc: imgc[c] = Image.open(gpath[c]).convert("RGB")
    m = [{"role": "user", "content": [{"type": "image", "image": imgc[c]}, {"type": "text", "text": txt}]}]
    inp = proc.apply_chat_template(m, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    lg = model(**inp).logits[0, -1].float(); return max(lg[t].item() for t in yi) - max(lg[t].item() for t in ni)

out = {}; nnew = 0; t0 = time.time(); Q = len(qorder)
_s = max(0, a.q_start); _e = a.q_end if a.q_end else Q          # split run over [q_start, q_end)
_sl = (_s, _e) != (0, Q)
if _sl:
    print(f"[slice] scoring queries {_s}..{_e} ({_e-_s} of them) — save as a part, then merge", flush=True)
for i in range(_s, _e):
    q = qorder[i]
    ptxt = PROMPT.format(cap=cap[q])
    for c in union[i]:
        if (q, c) in reuse: out[(q, c)] = reuse[(q, c)]
        else: out[(q, c)] = sp(c, ptxt); nnew += 1
    if (i + 1 - _s) % 100 == 0:
        el = time.time() - t0
        print(f"  {i+1-_s}/{_e-_s} new={nnew} ({el:.0f}s, {nnew/max(el,1):.1f}/s)", flush=True)
_suf = os.environ.get("OUT_SUFFIX", "union")
_out = f"{HERE}/{a.name}_q{_s}_{_e}_{_suf}.pt" if _sl else f"{HERE}/{a.name}_{_suf}.pt"
torch.save({"scores": out, "qorder": qorder[_s:_e], "name": a.name, "nnew": nnew}, _out)
print(f"[done] {os.path.basename(_out)}  pairs={len(out)} new={nnew} ({time.time()-t0:.0f}s)", flush=True)
