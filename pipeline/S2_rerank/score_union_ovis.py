#!/usr/bin/env python3
"""score_union_ovis — yes/no re-scoring of the union candidate pool with Ovis2.5-9B (AIMv2+Qwen3),
written to ovis_union_cache.pt.

Ovis2.5 exposes its own API: preprocess_inputs(messages) -> (input_ids, pixel_values, grid_thws),
then forward -> .logits.

Usage: CUDA_VISIBLE_DEVICES=7 <py> score_union_ovis.py [--limit N] [--topk K]
"""
import argparse, os, sys, json, time, torch
from PIL import Image
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.environ.get("VLM_MODELS", f"{_REPO}/assets/model/vlm_models") + "/Ovis2.5-9B")
ap.add_argument("--name", default="ovis")
ap.add_argument("--limit", type=int, default=0, help="limit the number of queries (smoke test)")
ap.add_argument("--topk", type=int, default=0, help="max candidates per query (0 = the full union)")
ap.add_argument("--max_pixels", type=int, default=896*896)
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="reproduction union re-scoring -> cache_rep/s2_rerank (zero-shot)")
ap.add_argument("--q-start", type=int, default=0, help="first query of the slice (for split runs)")
ap.add_argument("--q-end", type=int, default=0, help="last query of the slice (0 = all); merge the parts with merge_union_slices.py")
a = ap.parse_args()
if a.rep and a.limit:                                          # keep truncated artifacts out of the reproduction cache
    raise SystemExit("--limit cannot be combined with --rep (point WORKDIR elsewhere for smoke runs).")
if a.rep:                                                       # force cache_rep/s2_rerank (zero-shot, so model_rep is not needed)
    _CR = f"{_REPO}/assets/cache_rep"; os.environ["WORKDIR"] = f"{_CR}/s2_rerank"
    os.environ["OUT_SUFFIX"] = "union_cache"; os.environ["POOL_FILE"] = "../s1_base/union_pool.pt"
    os.makedirs(f"{_CR}/s2_rerank", exist_ok=True)
    if not os.path.exists(f"{_CR}/s1_base/union_pool.pt"):
        print("  cache_rep/s1_base/union_pool.pt not found — run build_union with --rep first", flush=True)

T4 = os.environ.get("TRACK4", f"{_REPO}/assets/cache/work")
HERE = os.environ.get("WORKDIR", f"{T4}/rerank_work")   
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
print(f"[union] {len(qorder)} queries, sizes ~{sum(len(u) for u in union)/len(union):.1f}/q total={sum(len(u) for u in union)}", flush=True)

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM   # noqa: E402
from transformers.dynamic_module_utils import get_class_from_dynamic_module   # noqa: E402

t0 = time.time()
# Work around the Ovis2.5 remote code being loaded twice: auto_map's AutoConfig builds the config
# from a module copy containing only configuration_*.py, while modeling_*.py calls AutoModel.register
# with classes from its own copy. The class names match but their identities differ, so
# `AutoModel.from_config(vit_config)` fails as Unrecognized. Registering the config instance's actual
# class once more links the two copies.
_cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
_Ovis = get_class_from_dynamic_module("modeling_ovis2_5.Ovis2_5", a.model)   # load and register the modeling copy
_mm = sys.modules[_Ovis.__module__]
AutoModel.register(type(_cfg.vit_config), _mm.Siglip2NavitModel, exist_ok=True)
AutoModelForCausalLM.register(type(_cfg), _Ovis, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(a.model, config=_cfg, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True).cuda().eval()
tok = model.text_tokenizer
def ids(ws):
    s = set()
    for w in ws:
        e = tok.encode(w, add_special_tokens=False)
        if len(e) == 1: s.add(e[0])
    return s
yi, ni = ids(YES), ids(NO); print(f"[load] {time.time()-t0:.0f}s  yes={yi} no={ni}", flush=True); assert yi and ni
imgc = {}
@torch.no_grad()
def sp(c, txt):
    if c not in imgc: imgc[c] = Image.open(gpath[c]).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image", "image": imgc[c]}, {"type": "text", "text": txt}]}]
    input_ids, pixel_values, grid_thws = model.preprocess_inputs(messages=msgs, max_pixels=a.max_pixels,
                                                                 add_generation_prompt=True, enable_thinking=False)
    input_ids = input_ids.cuda(); attn = torch.ones_like(input_ids)
    pv = pixel_values.cuda().to(model.dtype) if pixel_values is not None else None
    gt = grid_thws.cuda() if grid_thws is not None else None
    lg = model(input_ids=input_ids, attention_mask=attn, pixel_values=pv, grid_thws=gt).logits[0, -1].float()
    return max(lg[t].item() for t in yi) - max(lg[t].item() for t in ni)

out = {}; nnew = 0
for _rp in os.environ.get("REUSE_EXTRA","").split(":"):
    if _rp and os.path.exists(_rp):
        _d = torch.load(_rp, weights_only=False)["scores"]
        out.update({(qq,int(cc)):float(vv) for (qq,cc),vv in _d.items()})
        print(f"[reuse-extra] {_rp}: +{len(_d)}", flush=True)
t0 = time.time()
_s = max(0, a.q_start); _e = a.q_end if a.q_end else (a.limit if a.limit else len(qorder))
_sl = (_s, _e) != (0, len(qorder))
if _sl:
    print(f"[slice] scoring queries {_s}..{_e} ({_e-_s} of them) — save as a part, then merge", flush=True)
_tot = sum(len(union[j]) for j in range(_s, _e))
for i in range(_s, _e):
    q = qorder[i]
    ptxt = PROMPT.format(cap=cap[q]); cs = union[i][:a.topk] if a.topk else union[i]
    for c in cs:
        _k = (q, int(c))
        if _k not in out: out[_k] = sp(int(c), ptxt); nnew += 1
    if (i + 1 - _s) % 25 == 0:
        el = time.time() - t0
        print(f"  {i+1-_s}/{_e-_s} pairs={nnew} ({el:.0f}s, {nnew/max(el,1):.2f}/s, "
              f"eta {(_tot-nnew)/max(nnew/max(el,1),1e-6)/60:.0f}m)", flush=True)
_suf = os.environ.get("OUT_SUFFIX", "union")
_out = f"{HERE}/{a.name}_q{_s}_{_e}_{_suf}.pt" if _sl else f"{HERE}/{a.name}_{_suf}.pt"
torch.save({"scores": out, "qorder": qorder[_s:_e], "name": a.name, "nnew": nnew}, _out)
print(f"[done] {os.path.basename(_out)}  pairs={len(out)} ({time.time()-t0:.0f}s)", flush=True)
