#!/usr/bin/env python3
"""score_union_pixtral — re-score the union candidate pool with Pixtral-12B via vLLM.

Output: pixtral_union_cache.pt = {"scores": {(q,c): score}}.
"""
import os, json, time, base64, numpy as np, torch
from io import BytesIO
from PIL import Image
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

import argparse                                                   # noqa: E402
ap = argparse.ArgumentParser(description="Pixtral-12B zero-shot union re-scoring")
ap.add_argument("--name", default="pixtral")
ap.add_argument("--limit", type=int, default=0, help="limit the number of queries (smoke test)")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction union re-scoring -> cache_rep/s2_rerank (zero-shot)")
a = ap.parse_args()
if a.rep and a.limit:                                          # keep truncated artifacts out of the reproduction cache
    raise SystemExit("--limit cannot be combined with --rep (point WORKDIR elsewhere for smoke runs).")
if a.rep:                                                       # force cache_rep/s2_rerank
    _CR = f"{_REPO}/assets/cache_rep"; os.environ["WORKDIR"] = f"{_CR}/s2_rerank"
    os.environ["OUT_SUFFIX"] = "union_cache"; os.environ["POOL_FILE"] = "../s1_base/union_pool.pt"
    os.makedirs(f"{_CR}/s2_rerank", exist_ok=True)
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
M = os.environ.get("VLM_MODELS", f"{_REPO}/assets/model/vlm_models") + "/Pixtral-12B-2409"
PROMPT = ('You are judging whether an image matches a text description of a person.\nDescription: "{cap}"\n'
          'Does this image show EXACTLY that person and situation — matching gender, clothing (color/type), the action/behavior, and the scene? Answer a single word: yes or no.')
YES = {13059, 16860, 14842, 51935}; NO = {2649, 4753, 1836, 16071}
gal = sorted(os.listdir(GAL))
cap = {json.loads(l)["query_index"]: json.loads(l)["caption"] for l in open(QJSON) if l.strip()}
U = torch.load(f"{HERE}/{os.environ.get('POOL_FILE','union_pool.pt')}"); union = U["union"]; qorder = U["qorder"]
# Reuse an existing pixtral dump when present so only new pairs are scored; otherwise score
# everything. A reproduction run (--rep) must not pull in the adopted dump, so reuse is off there.
reuse = {}
d = os.environ.get("PIXTRAL_REUSE", "" if a.rep else f"{T4}/outputs/fuse_internvl/pixtral")
if d and os.path.exists(f"{d}/internvl_scores_pixtral_meta.json"):
    m = json.load(open(f"{d}/internvl_scores_pixtral_meta.json"))
    sc0 = np.load(f"{d}/internvl_scores_pixtral_top20.npy")
    cd0 = np.load(f"{d}/internvl_cand_pixtral_top20.npy")
    for i, q in enumerate(m["qorder"]):
        for j in range(cd0.shape[1]):
            if not np.isnan(sc0[i, j]):
                reuse[(q, int(cd0[i, j]))] = float(sc0[i, j])
    print(f"[reuse] pixtral: {len(reuse)}", flush=True)
else:
    print("[reuse] no pixtral dump found -> scoring everything", flush=True)
def b64(idx):
    im = Image.open(os.path.join(GAL, gal[idx])).convert("RGB"); buf = BytesIO(); im.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()
from vllm import LLM, SamplingParams
# REUSE_EXTRA: reuse existing score caches so only new (q,c) pairs are scored.
for _rp in os.environ.get("REUSE_EXTRA","").split(":"):
    if _rp and __import__("os").path.exists(_rp):
        _d = torch.load(_rp, weights_only=False)["scores"]
        reuse.update({(q, int(c)): float(v) for (q, c), v in _d.items()})
        print(f"[reuse-extra] {_rp}: +{len(_d)}", flush=True)

t0 = time.time()
# PIX_DETERMINISTIC=1 pins one sequence per forward step and drops the prefix cache, which is what
_det = {"max_num_seqs": 1, "enable_prefix_caching": False} if os.environ.get("PIX_DETERMINISTIC") == "1" else {}
llm = LLM(model=M, tokenizer_mode="mistral", load_format="mistral", config_format="mistral",
          limit_mm_per_prompt={"image": 1}, max_model_len=8192,
          gpu_memory_utilization=float(os.environ.get("PIX_GPU_UTIL", "0.28")), enforce_eager=True, **_det)
sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
print(f"[load] {time.time()-t0:.0f}s", flush=True)
def score_msgs(msgs):
    outs = llm.chat(msgs, sp, use_tqdm=False); res = []
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        y = max([v.logprob for k, v in lp.items() if k in YES], default=-20.0)
        n = max([v.logprob for k, v in lp.items() if k in NO], default=-20.0)
        res.append(y - n)
    return res
out = {}; nnew = 0; imgc = {}; buf_msgs = []; buf_loc = []; t0 = time.time()
def flush():
    global buf_msgs, buf_loc, nnew
    if not buf_msgs: return
    r = score_msgs(buf_msgs)
    for (q, c), val in zip(buf_loc, r): out[(q, c)] = val; nnew += 1
    buf_msgs = []; buf_loc = []
qs = qorder[:a.limit] if a.limit else qorder      # --limit was declared but never applied: a smoke
for i, q in enumerate(qs):                        # run scored all 1978 queries (~1 h) instead of 8
    ptxt = PROMPT.format(cap=cap[q])
    for c in union[i]:
        if (q, c) in reuse: out[(q, c)] = reuse[(q, c)]
        else:
            if c not in imgc: imgc[c] = b64(c)
            buf_msgs.append([{"role": "user", "content": [{"type": "text", "text": ptxt},
                              {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgc[c]}"}}]}])
            buf_loc.append((q, c))
    if len(buf_msgs) >= 800:
        flush()
        el = time.time() - t0; print(f"  {i+1}/{len(qs)} new={nnew} ({el:.0f}s)", flush=True)
        if len(imgc) > 3000: imgc.clear()
flush()
torch.save({"scores": out, "qorder": qs, "name": a.name, "nnew": nnew}, f"{HERE}/{a.name}_{os.environ.get('OUT_SUFFIX','union_cache')}.pt")
print(f"[done] pixtral_union_cache.pt pairs={len(out)} new={nnew} ({time.time()-t0:.0f}s)", flush=True)
