#!/usr/bin/env python3
"""score_union_qwen — re-score the union candidate pool with Qwen3-VL-Reranker (8B/2B [+DoRA]).

An existing recs dump passed with --reuse-recs is reused, so the model is called only for new
(q,c) pairs. Output: dict {(q,c): score} saved as {name}_union_cache.pt.

Usage: CUDA_VISIBLE_DEVICES=6 <py> score_union_qwen_4b.py --qwen 8b --name 8b --reuse-recs recs_8b_p3_k20.pt
       (DoRA) --qwen 2b --adapter assets/model/reranker/qwen3vl_2b --name qwen3vl_2b --reuse-recs recs_2b_dora_k5_p3.pt
"""
import argparse, os, sys, json, time, torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--qwen", default="8b", choices=["2b", "8b"]); ap.add_argument("--name", required=True)
ap.add_argument("--adapter", default=None); ap.add_argument("--reuse-recs", default=None)
ap.add_argument("--pool", default="union_pool.pt")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction union re-scoring with model_rep weights -> cache_rep/s2_rerank")
ap.add_argument("--q-start", type=int, default=0, help="first query of the slice (for split runs)")
ap.add_argument("--q-end", type=int, default=0, help="last query of the slice (0 = all); merge the parts with merge_union_slices.py")
a = ap.parse_args()
if a.rep:                                                       # force cache_rep/s2_rerank and the deployed adapter
    _CR = f"{_REPO}/assets/cache_rep"; os.environ["WORKDIR"] = f"{_CR}/s2_rerank"
    os.environ["OUT_SUFFIX"] = "union_cache"; os.environ["POOL_FILE"] = "../s1_base/union_pool.pt"
    os.makedirs(f"{_CR}/s2_rerank", exist_ok=True)
    _ad = f"{_REPO}/assets/model_rep/reranker/{a.name}"        # only trained rerankers have an adapter (8b is zero-shot)
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
SNAP = {"2b": os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache") + "/hub/models--Qwen--Qwen3-VL-Reranker-2B/snapshots/4bd860ac4f15ad1897a214615cccc700f8f71818",
        "8b": os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache") + "/hub/models--Qwen--Qwen3-VL-Reranker-8B/snapshots/b212dc8c91a8164aef1ea2de9c1a867611e75c04"}[a.qwen]
INSTR = ("Retrieve the frame showing the described person performing the described action. "
         "Prioritize the action/behavior and scene, then clothing color/type and gender.")   # p3
gal = sorted(os.listdir(GAL)); gpath = [os.path.join(GAL, g) for g in gal]
cap = {json.loads(l)["query_index"]: json.loads(l)["caption"] for l in open(QJSON) if l.strip()}
U = torch.load(f"{HERE}/{os.environ.get('POOL_FILE','union_pool.pt')}"); union = U["union"]; qorder = U["qorder"]

reuse = {}
if a.reuse_recs:
    for r in torch.load(f"{T4}/{a.reuse_recs}", weights_only=False)["recs"]:
        for c, s in zip(r["cand"], r["scores"]): reuse[(r["qidx"], int(c))] = float(s)
    print(f"[reuse] {a.reuse_recs}: {len(reuse)} pairs", flush=True)

sys.path.insert(0, os.path.join(SNAP, "scripts"))
from qwen3_vl_reranker import Qwen3VLReranker
class PatchedReranker(Qwen3VLReranker):
    def tokenize(self, pairs, **kw):
        inp = super().tokenize(pairs, **kw)
        mt = inp.get("mm_token_type_ids")
        if isinstance(mt, list): inp["mm_token_type_ids"] = torch.tensor(mt, dtype=torch.long)
        return inp
# REUSE_EXTRA: reuse existing score caches so only new (q,c) pairs are scored.
for _rp in os.environ.get("REUSE_EXTRA","").split(":"):
    if _rp and __import__("os").path.exists(_rp):
        _d = torch.load(_rp, weights_only=False)["scores"]
        reuse.update({(q, int(c)): float(v) for (q, c), v in _d.items()})
        print(f"[reuse-extra] {_rp}: +{len(_d)}", flush=True)

t0 = time.time(); reranker = PatchedReranker(SNAP, torch_dtype=torch.bfloat16)
if a.adapter:
    from peft import PeftModel
    reranker.model = PeftModel.from_pretrained(reranker.model, a.adapter); reranker.model.eval()
    print(f"[adapter] {a.adapter}", flush=True)
print(f"[load] {time.time()-t0:.0f}s", flush=True)

out = {}; nnew = 0; t0 = time.time(); Q = len(qorder)
_s = max(0, a.q_start); _e = a.q_end if a.q_end else Q          # split run over [q_start, q_end)
_sl = (_s, _e) != (0, Q)
if _sl:
    print(f"[slice] scoring queries {_s}..{_e} ({_e-_s} of them) — save as a part, then merge", flush=True)
for i in range(_s, _e):
    q = qorder[i]
    need = [c for c in union[i] if (q, c) not in reuse]
    for c in union[i]:
        if (q, c) in reuse: out[(q, c)] = reuse[(q, c)]
    if need:
        docs = [{"image": gpath[c]} for c in need]
        sc = reranker.process({"query": {"text": cap[q]}, "documents": docs, "instruction": INSTR})
        for c, s in zip(need, sc): out[(q, c)] = float(s); nnew += 1
    if (i + 1 - _s) % 100 == 0:
        el = time.time() - t0
        print(f"  {i+1-_s}/{_e-_s} new={nnew} ({el:.0f}s, {nnew/max(el,1):.1f}/s)", flush=True)
_suf = os.environ.get("OUT_SUFFIX", "union")
_out = f"{HERE}/{a.name}_q{_s}_{_e}_{_suf}.pt" if _sl else f"{HERE}/{a.name}_{_suf}.pt"
torch.save({"scores": out, "qorder": qorder[_s:_e], "name": a.name, "nnew": nnew}, _out)
print(f"[done] {os.path.basename(_out)} pairs={len(out)} new={nnew} ({time.time()-t0:.0f}s)", flush=True)
