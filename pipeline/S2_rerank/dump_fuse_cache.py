#!/usr/bin/env python3
"""dump_fuse_cache — score the top-20 candidates with yes/no logits into the S3 fuse_cache.

Produces `fuse_cache/<dir>/internvl_scores_<dir>_top20.npy` (plus `_cand_` and `_meta.json`), read
by the `IV_CACHE` members of S3 (`../S3_assign/fuse.py`): internvl_r32, pixtral, llama32v.
`score_union_*.py` re-scores the whole union pool, whereas this produces the fixed top-20-column
format that S3 consumes.

Output:
  default  assets/cache/s2_rerank/fuse_cache/<name>/
  --rep    assets/cache_rep/s2_rerank/fuse_cache/<name>/

Usage (a VLM, so it needs a vllm environment — track4_vllm in `requirements/README.md`):
  PY=$(conda info --base)/envs/track4_vllm/bin/python
  $PY dump_fuse_cache.py --model <HF path> --name internvl_r32 [--adapter <LoRA>] [--rep]
  $PY dump_fuse_cache.py --model $VLM_MODELS/Llama-3.2-11B-Vision-Instruct --name llama32v --rep

"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402


ap = argparse.ArgumentParser(description="top-20 yes/no re-scoring -> S3 fuse_cache")
ap.add_argument("--model", default=None, help="HF VLM path (not needed with --from-union)")
ap.add_argument("--name", required=True, help="fuse_cache directory name (internvl_r32, pixtral, llama32v)")
ap.add_argument("--adapter", default=None, help="LoRA adapter (optional)")
ap.add_argument("--from-union", action="store_true",
                help="assemble from the union re-score cache without calling the VLM again (same model and prompt)")
ap.add_argument("--union-name", default=None,
                help="union cache member name (default: the fuse_cache directory name; llama32v -> llama)")
ap.add_argument("--allow-missing", action="store_true",
                help="with --from-union, leave pairs the union does not cover as NaN and continue (default: abort)")
ap.add_argument("--engine", default="hf", choices=["hf", "vllm"],
                help="hf = AutoModelForImageTextToText · vllm = Mistral format (Pixtral) only")
ap.add_argument("--topk", type=int, default=20)
ap.add_argument("--max-q", type=int, default=0, help="truncate the query list for a smoke test")
ap.add_argument("--out", default=None, help="output directory override")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction output -> cache_rep/s2_rerank/fuse_cache")
a = ap.parse_args()
if a.rep and a.max_q:
    raise SystemExit("[dump_fuse_cache] --max-q cannot be combined with --rep (prevents truncated artifacts).")

CACHE = f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}"
OUTDIR = a.out or f"{CACHE}/s2_rerank/fuse_cache/{a.name}"
skip_if_exists(f"{OUTDIR}/internvl_scores_{a.name}_meta.json", a.overwrite,
               label=f"fuse_cache/{a.name}")
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GAL = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
QJSON = os.environ.get("QUERY_TEXT", f"{PAB_TEST}/query_text.json")
RECS = os.environ.get("RECS", f"{CACHE}/s2_rerank/recs_8b_p3_k20.pt")   # CACHE already branches on --rep
# Source of the 20 candidate columns: cand[:topk] of the recs dump (buildable with build_recs.py),
# which holds the base top-20 set. The consumer (_load_iv) turns it into a (q,c) -> score dict, so
# the column order is not part of the contract.

PROMPT = ('You are judging whether an image matches a text description of a person.\nDescription: "{cap}"\n'
          'Does this image show EXACTLY that person and situation — matching gender, clothing '
          '(color/type), the action/behavior, and the scene? Answer a single word: yes or no.')
YES = ["yes", "Yes", " yes", " Yes", "YES"]
NO = ["no", "No", " no", " No", "NO"]

gal = sorted(os.listdir(GAL))
gpath = [os.path.join(GAL, g) for g in gal]
cap = {json.loads(l)["query_index"]: json.loads(l)["caption"] for l in open(QJSON) if l.strip()}
if not os.path.exists(RECS):
    raise SystemExit(f"[dump_fuse_cache] candidates not found: {RECS}\n"
                     f"  Set RECS, or place assets/cache/s2_rerank/recs_8b_p3_k20.pt.")
B8 = {r["qidx"]: r for r in torch.load(RECS, weights_only=False)["recs"]}
qids = list(B8.keys())
cand = {q: B8[q]["cand"][:a.topk] for q in qids}
if a.max_q:
    qids = qids[:a.max_q]
print(f"[fuse_cache] {a.name} · queries {len(qids)} x top{a.topk} -> {OUTDIR}", flush=True)

if a.from_union:
    # The union re-score already covers these (q,c) pairs with the same model and prompt, and the
    # recs candidates (base top-K) are a subset of the union pool, so no VLM call is needed.
    _member = a.union_name or ("llama" if a.name == "llama32v" else a.name)
    _uc = f"{CACHE}/s2_rerank/{_member}_union_cache.pt"
    if not os.path.exists(_uc):
        raise SystemExit(f"[fuse_cache] union cache not found: {_uc} (set the member name with --union-name)")
    _sc = torch.load(_uc, weights_only=False)["scores"]
    sc = np.full((len(qids), a.topk), np.nan, np.float32)
    cd = np.zeros((len(qids), a.topk), np.int64)
    miss = 0
    for i, q in enumerate(qids):
        cols = cand[q]
        cd[i, :len(cols)] = cols
        for j, c in enumerate(cols):
            v = _sc.get((q, int(c)))
            if v is None:
                miss += 1
            else:
                sc[i, j] = float(v)
    tot = len(qids) * a.topk
    if miss and not a.allow_missing:
        # Missing pairs stay NaN and the consumer (_load_iv) skips them silently, so abort here
        # rather than degrade quietly.
        raise SystemExit(
            f"[fuse_cache] the union cache does not cover all candidates: {miss}/{tot} pairs missing "
            f"({100*miss/tot:.2f}%)\n"
            f"  union: {_uc}\n"
            f"  recs : {RECS}\n"
            f"  Check that both come from the same base generation.\n"
            f"  Pass --allow-missing to leave the missing pairs as NaN and continue.")
    os.makedirs(OUTDIR, exist_ok=True)
    np.save(f"{OUTDIR}/internvl_scores_{a.name}_top{a.topk}.npy", sc)
    np.save(f"{OUTDIR}/internvl_cand_{a.name}_top{a.topk}.npy", cd)
    json.dump({"ckpt": a.name, "model": f"from-union:{_member}", "source_union": _uc,
               "source_recs": RECS, "K": a.topk, "Q": len(qids), "qorder": qids},
              open(f"{OUTDIR}/internvl_scores_{a.name}_meta.json", "w"))
    print(f"[fuse_cache] assembled from union · coverage {100*(tot-miss)/tot:.2f}% ({miss} pairs missing) -> {OUTDIR}",
          flush=True)
    raise SystemExit(0)

if not a.model:
    raise SystemExit("[fuse_cache] --model is required (use --from-union for assembly mode).")

t0 = time.time()
if a.engine == "vllm":
    # Pixtral uses the Mistral format, which HF AutoProcessor cannot open, so it is served by vLLM.
    # The yes/no token ids are fixed values of the Mistral tokenizer (same as in the scorer).
    from vllm import LLM, SamplingParams                              # noqa: E402
    import base64                                                     # noqa: E402
    from io import BytesIO                                            # noqa: E402
    YES_ID = {13059, 16860, 14842, 51935}; NO_ID = {2649, 4753, 1836, 16071}
    # PIX_DETERMINISTIC=1 pins one sequence per step and drops the prefix cache (re-run reproducible).
    _det = {"max_num_seqs": 1, "enable_prefix_caching": False} if os.environ.get("PIX_DETERMINISTIC") == "1" else {}
    llm = LLM(model=a.model, tokenizer_mode="mistral", load_format="mistral",
              config_format="mistral", limit_mm_per_prompt={"image": 1}, max_model_len=8192,
              gpu_memory_utilization=float(os.environ.get("PIX_GPU_UTIL", "0.55")),
              enforce_eager=True, **_det)
    _sp = SamplingParams(max_tokens=1, temperature=0, logprobs=20)
    print(f"[load] {time.time()-t0:.0f}s (vllm)", flush=True)

    def _b64(idx):
        im = Image.open(gpath[idx]).convert("RGB"); buf = BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    def score_batch(msgs):
        res = []
        for o in llm.chat(msgs, _sp, use_tqdm=False):
            lp = o.outputs[0].logprobs[0]
            y = max([v.logprob for k, v in lp.items() if k in YES_ID], default=-20.0)
            nn = max([v.logprob for k, v in lp.items() if k in NO_ID], default=-20.0)
            res.append(y - nn)
        return res
else:
    from transformers import AutoModelForImageTextToText, AutoProcessor    # noqa: E402
    proc = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True).eval()
    if a.adapter:
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file as _lf
        _c = json.load(open(a.adapter + "/adapter_config.json"))
        _lc = LoraConfig(r=_c["r"], lora_alpha=_c["lora_alpha"], lora_dropout=_c.get("lora_dropout", 0.0),
                         bias=_c.get("bias", "none"), target_modules=_c["target_modules"], task_type="CAUSAL_LM")
        model = get_peft_model(model, _lc)
        _sd = _lf(a.adapter + "/adapter_model.safetensors")
        _rm = {k.replace(".lora_A.weight", ".lora_A.default.weight")
                .replace(".lora_B.weight", ".lora_B.default.weight"): v for k, v in _sd.items()}
        _r = model.load_state_dict(_rm, strict=False)
        assert not [k for k in _r.missing_keys if "lora" in k], "adapter mismatch"
        model = model.eval()
        print(f"[adapter] {a.adapter}", flush=True)
    tok = proc.tokenizer


    def _ids(ws):
        s = set()
        for w in ws:
            e = tok.encode(w, add_special_tokens=False)
            if len(e) == 1:
                s.add(e[0])
        return s


    yi, ni = _ids(YES), _ids(NO)
    print(f"[load] {time.time()-t0:.0f}s yes={sorted(yi)} no={sorted(ni)}", flush=True)
    assert yi and ni, "could not resolve yes/no to single tokens — check the tokenizer"


    @torch.no_grad()
    def score_pair(img, txt):
        m = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}]
        inp = proc.apply_chat_template(m, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(model.device)
        lg = model(**inp).logits[0, -1].float()
        return max(lg[t].item() for t in yi) - max(lg[t].item() for t in ni)


sc = np.full((len(qids), a.topk), np.nan, np.float32)
cd = np.zeros((len(qids), a.topk), np.int64)
imgc = {}
t0 = time.time()
if a.engine == "vllm":                       # batch the messages together for vLLM throughput
    buf_msgs, buf_loc = [], []

    def _flush():
        if not buf_msgs:
            return
        for (qi, j), val in zip(buf_loc, score_batch(buf_msgs)):
            sc[qi, j] = val
        buf_msgs.clear(); buf_loc.clear()

    for i, q in enumerate(qids):
        cols = cand[q]
        cd[i, :len(cols)] = cols
        ptxt = PROMPT.format(cap=cap[q])
        for j, c in enumerate(cols):
            if c not in imgc:
                imgc[c] = _b64(c)
            buf_msgs.append([{"role": "user", "content": [
                {"type": "text", "text": ptxt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{imgc[c]}"}}]}])
            buf_loc.append((i, j))
        if len(buf_msgs) >= 800:
            _flush()
            el = time.time() - t0
            print(f"  {i+1}/{len(qids)} ({el:.0f}s, {el/(i+1):.2f}s/q)", flush=True)
            if len(imgc) > 3000:
                imgc.clear()
    _flush()
else:
    for i, q in enumerate(qids):
        cols = cand[q]
        cd[i, :len(cols)] = cols
        ptxt = PROMPT.format(cap=cap[q])
        for j, c in enumerate(cols):
            if c not in imgc:
                imgc[c] = Image.open(gpath[c]).convert("RGB")
            sc[i, j] = score_pair(imgc[c], ptxt)
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(qids)} ({el:.0f}s, {el/(i+1):.2f}s/q)", flush=True)
            if len(imgc) > 4000:
                imgc.clear()

os.makedirs(OUTDIR, exist_ok=True)
np.save(f"{OUTDIR}/internvl_scores_{a.name}_top{a.topk}.npy", sc)
np.save(f"{OUTDIR}/internvl_cand_{a.name}_top{a.topk}.npy", cd)
json.dump({"ckpt": a.name, "model": a.model, "K": a.topk, "Q": len(qids), "qorder": qids},
          open(f"{OUTDIR}/internvl_scores_{a.name}_meta.json", "w"))
print(f"[done] {time.time()-t0:.0f}s → {OUTDIR}", flush=True)
