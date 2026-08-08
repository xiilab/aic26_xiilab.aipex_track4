#!/usr/bin/env python3
"""score_union_jina — re-score the union candidate pool with jina-reranker-m0.

Usage: CUDA_VISIBLE_DEVICES=3 <py> score_union_jina.py [--limit N]   (env: requirements/README.md)
"""
import argparse, os, json, time, torch, numpy as np
from PIL import Image
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"))
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="jina_m0"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--chunk", type=int, default=10)
ap.add_argument("--ckpt", default=f"{_REPO}/assets/model/reranker/jina_m0",
                help="adapter to use (default = the bundled assets/model/reranker/)")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction union re-scoring with model_rep weights -> cache_rep/s2_rerank")
a = ap.parse_args()
if a.rep and a.limit:                                          # keep truncated artifacts out of the reproduction cache
    raise SystemExit("--limit cannot be combined with --rep (point WORKDIR elsewhere for smoke runs).")
if a.rep:                                                       # force cache_rep/s2_rerank and the deployed adapter
    _CR = f"{_REPO}/assets/cache_rep"; os.environ["WORKDIR"] = f"{_CR}/s2_rerank"
    os.environ["OUT_SUFFIX"] = "union_cache"; os.environ["POOL_FILE"] = "../s1_base/union_pool.pt"
    os.makedirs(f"{_CR}/s2_rerank", exist_ok=True)
    _ad = f"{_REPO}/assets/model_rep/reranker/{a.name}"
    if a.ckpt == f"{_REPO}/assets/model/reranker/jina_m0" and os.path.isdir(_ad): a.ckpt = _ad
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
JINA = "jinaai/jina-reranker-m0"; SCORE_TOKEN_ID = 100; MAXLEN = 2048
IMG_MIN, IMG_MAX = 4*28*28, 1024*28*28
def _fmt(cap): return f"**Document**:\n<|vision_start|><|image_pad|><|vision_end|>\n**Query**:\n{cap}"
gal = sorted(os.listdir(GAL)); gpath = [os.path.join(GAL, g) for g in gal]
cap = {json.loads(l)["query_index"]: json.loads(l)["caption"] for l in open(QJSON) if l.strip()}
U = torch.load(f"{HERE}/{os.environ.get('POOL_FILE','union_pool.pt')}"); union = U["union"]; qorder = U["qorder"]
print(f"[union] {len(qorder)}q total={sum(len(u) for u in union)} pairs", flush=True)

from transformers import AutoModel, AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
dev = "cuda:0"; t0 = time.time()
base = AutoModel.from_pretrained(JINA, trust_remote_code=True, torch_dtype=torch.bfloat16)
proc = AutoProcessor.from_pretrained(JINA, trust_remote_code=True, min_pixels=IMG_MIN, max_pixels=IMG_MAX)
proc.tokenizer.padding_side = "left"
sc_tok = int(getattr(base, "score_token_id", SCORE_TOKEN_ID))
base = base.to(dev).eval()
peft = PeftModel.from_pretrained(base, a.ckpt); peft.eval(); m_base = peft.get_base_model()
print(f"[load] {time.time()-t0:.0f}s sc_tok={sc_tok}", flush=True)
imgc = {}
@torch.no_grad()
def jscore(caption, cs):
    outs = []
    for s0 in range(0, len(cs), a.chunk):
        sub = cs[s0:s0+a.chunk]
        pils = [imgc.setdefault(c, Image.open(gpath[c]).convert("RGB")) for c in sub]
        prompts = [_fmt(caption)] * len(sub)
        b = proc(text=prompts, images=pils, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN-1)
        bs = b["input_ids"].size(0)
        b["input_ids"] = torch.cat([b["input_ids"], torch.full((bs, 1), sc_tok, dtype=b["input_ids"].dtype)], 1)
        b["attention_mask"] = torch.cat([b["attention_mask"], torch.ones((bs, 1), dtype=b["attention_mask"].dtype)], 1)
        if "mm_token_type_ids" in b:
            b["mm_token_type_ids"] = torch.cat([b["mm_token_type_ids"], torch.zeros((bs, 1), dtype=b["mm_token_type_ids"].dtype)], 1)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        out = Qwen2VLForConditionalGeneration.forward(m_base, output_hidden_states=True, use_cache=False, **b)
        outs += m_base.score(out.hidden_states[-1][:, -1]).squeeze(-1).float().cpu().numpy().tolist()
    return outs

res = {}; nnew = 0
for _rp in os.environ.get("REUSE_EXTRA","").split(":"):
    if _rp and os.path.exists(_rp):
        _d = torch.load(_rp, weights_only=False)["scores"]
        res.update({(qq,int(cc)):float(vv) for (qq,cc),vv in _d.items()})
        print(f"[reuse-extra] {_rp}: +{len(_d)}", flush=True)
t0 = time.time(); qs = qorder[:a.limit] if a.limit else qorder
for i, q in enumerate(qs):
    cs = [c for c in (int(x) for x in union[i]) if (q, c) not in res]
    if cs:
        sc = jscore(cap[q], cs)
        for c, s in zip(cs, sc): res[(q, c)] = float(s); nnew += 1
    if (i+1) % 25 == 0:
        el = time.time()-t0; print(f"  {i+1}/{len(qs)} pairs={nnew} ({el:.0f}s, {nnew/max(el,1):.2f}/s)", flush=True)
sfx = f"_limit{a.limit}" if a.limit else ""
torch.save({"scores": res, "qorder": list(qs), "name": a.name, "nnew": nnew}, f"{HERE}/{a.name}_{os.environ.get('OUT_SUFFIX','union')}.pt")
print(f"[done] {a.name}_union{sfx}.pt pairs={len(res)} ({time.time()-t0:.0f}s)", flush=True)
