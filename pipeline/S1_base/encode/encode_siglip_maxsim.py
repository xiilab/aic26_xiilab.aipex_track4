#!/usr/bin/env python3
"""siglip_maxsim member = SigLIP2-L pooled base + BEiT3-helip token MaxSim rerank.

The member is built in two stages that need different environments, so this
script runs one stage at a time:

  --stage base    (env track4_llm2clip)  SigLIP2-L DoRA pooled score [1978, 36773]
                                         -> members/siglip_maxsim_base.pt
  --stage maxsim  (env track4_beit3)     BEiT3-helip token<->patch MaxSim over the
                                         base's own top-K, blended in place
                                         -> members/siglip_maxsim_score.pt

  blend: score <- (1-ALPHA)*base + ALPHA*rowminmax(MaxSim) on the top-K columns;
         everything outside the top-K keeps its base score.
  MaxSim(q,d) = sum_i max_j cos(t_i, v_j), text tokens t (t64) x image patches v (576).

Provenance (traced to the original dumps, not just reverse-engineered):
  base   = SigLIP2-L DoRA ep8 pooled score 
  MaxSim = BEiT3-helip tokens              
                                             
  ALPHA=0.05 · row min-max · K=10         
  NOTE the member name says "siglip" for its *base*; the MaxSim tokens come from
  BEiT3-helip. Both checkpoints ship with the repository.

  default  rebuild the adopted cache      -> assets/cache/s1_base/members/
  --rep    encode from assets/model_rep/  -> assets/cache_rep/s1_base/members/

Usage:
  python pipeline/S1_base/encode/encode_siglip_maxsim.py --stage base   --gpu 6   # track4_llm2clip
  python pipeline/S1_base/encode/encode_siglip_maxsim.py --stage maxsim --gpu 6   # track4_beit3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Repo-local HF cache. huggingface_hub reads HF_HOME at import time, so it has to be set before
# transformers is imported in stage_base; the trainer's own setdefault runs too late for that.
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"))
sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

NAME = "siglip_maxsim"

ap = argparse.ArgumentParser(description="SigLIP2-L base + BEiT3-helip MaxSim -> member score")
ap.add_argument("--stage", required=True, choices=["base", "maxsim"])
ap.add_argument("--gpu", default="0")
ap.add_argument("--bs", type=int, default=48, help="encode batch size")
ap.add_argument("--ibs", type=int, default=48, help="MaxSim patch-encode batch size")
ap.add_argument("--topk", type=int, default=10, help="rerank candidate count K")
ap.add_argument("--alpha", type=float, default=0.05, help="blend weight (adopted 0.05)")
ap.add_argument("--ckpt", default=None, help="stage-specific checkpoint override")
ap.add_argument("--base", default=None,
                help="maxsim stage: base score .pt (default = members/siglip_maxsim_base.pt)")
ap.add_argument("--out", default=None, help="output .pt override")
ap.add_argument("--limit", type=int, default=None, help="truncate gallery/queries for a smoke run")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true", help="encode from model_rep -> cache_rep")
a = ap.parse_args()
if a.rep and a.limit:                       # keep truncated output away from the reproduction cache
    raise SystemExit("--limit cannot be combined with --rep (smoke-test into --out instead).")
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

MODEL_ROOT = f"{_REPO}/assets/model_rep/encoder" if a.rep else f"{_REPO}/assets/model/encoder"
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GALLERY = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
QUERY_TEXT = os.environ.get("QUERY_TEXT", f"{PAB_TEST}/query_text.json")
MEMBERS = os.environ.get(
    "S1_MEMBERS", f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}/s1_base/members")
BASE_PT = a.base or f"{MEMBERS}/{NAME}_base.pt"
OUT = a.out or (f"{MEMBERS}/{NAME}_base.pt" if a.stage == "base" else f"{MEMBERS}/{NAME}_score.pt")
skip_if_exists(OUT, a.overwrite)

gal = sorted(os.listdir(GALLERY))                              # answer-column convention
gpaths = [os.path.join(GALLERY, f) for f in gal]
caps = [json.loads(l)["caption"] for l in open(QUERY_TEXT, encoding="utf-8") if l.strip()]
if a.limit:
    gpaths, caps = gpaths[:a.limit], caps[:a.limit]

import torch                                                   # noqa: E402
import torch.nn.functional as F                                # noqa: E402
from PIL import Image                                          # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ── stage 1: SigLIP2-L pooled base ──────────────────────────────────────────
def stage_base():
    """Load the DoRA checkpoint through the training package's surgery chain and
    dump the pooled cosine score. The architecture pieces are imported from the
    trainer so the surgery has a single source."""
    from torchvision import transforms as T                    # noqa: E402
    from transformers import AutoModel, AutoTokenizer          # noqa: E402
    from peft import PeftModel                                 # noqa: E402

    spec = importlib.util.spec_from_file_location(
        "siglip_trainer", f"{_REPO}/train/encoders/siglip_maxsim_all/train.py")
    M = importlib.util.module_from_spec(spec)
    sys.modules["siglip_trainer"] = M
    spec.loader.exec_module(M)

    ckpt = a.ckpt or os.environ.get("ENCODER_CKPT", f"{MODEL_ROOT}/{NAME}")
    for f in ("adapter_model.safetensors", "extras_state.pt"):
        if not os.path.exists(f"{ckpt}/{f}"):
            raise SystemExit(f"[base] missing {f} under {ckpt}"
                             + ("\n  --rep needs a deployment first: "
                                "python train/encoders/siglip_maxsim_all/deploy.py <all_run>" if a.rep else ""))

    meta = json.load(open(f"{ckpt}/meta.json")) if os.path.exists(f"{ckpt}/meta.json") else {}
    if meta and int(meta.get("multi_probe_count", M.NUM_PROBES)) != M.NUM_PROBES:
        raise SystemExit(f"[base] meta multi_probe_count={meta.get('multi_probe_count')} "
                         f"!= trainer NUM_PROBES={M.NUM_PROBES}")

    print(f"[base] {M.MODEL_NAME} + DoRA <- {ckpt}", flush=True)
    model = AutoModel.from_pretrained(M.MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(M.MODEL_NAME)
    # surgery in training order: PI -> split pooler MHA -> multi-probe head -> adapter
    M.apply_position_interpolation(model, pretrain_len=M.POSITION_PI_PRETRAIN_LEN,
                                   target_len=M.POSITION_PI_TARGET_LEN)
    M._replace_pooler_mha_with_split(model)
    M._replace_head_with_multi_probe(model, num_probes=M.NUM_PROBES,
                                     perturb_std=M.MULTI_PROBE_PERTURB_STD,
                                     aggregator_init=M.MULTI_PROBE_AGG_INIT,
                                     perturb_mode=M.MULTI_PROBE_PERTURB_MODE,
                                     block_scale=M.MULTI_PROBE_BLOCK_SCALE)
    model = PeftModel.from_pretrained(model, ckpt)

    # extras_state.pt holds the base params PEFT does not save
    extras = torch.load(f"{ckpt}/extras_state.pt", map_location="cpu", weights_only=False)
    base = model.get_base_model()
    pe = extras["position_embedding"]
    if pe.shape[0] != M.POSITION_PI_PRETRAIN_LEN:
        raise SystemExit(f"[base] position_embedding rows {pe.shape[0]} "
                         f"!= PI pretrain_len {M.POSITION_PI_PRETRAIN_LEN}")
    base.text_model.embeddings.position_embedding.weight.data.copy_(pe)
    if "logit_scale" in extras and hasattr(base, "logit_scale"):
        base.logit_scale.data.copy_(extras["logit_scale"].to(base.logit_scale.data))
    if "logit_bias" in extras and hasattr(base, "logit_bias"):
        base.logit_bias.data.copy_(extras["logit_bias"].to(base.logit_bias.data))
    head = base.vision_model.head
    head.probe.data.copy_(extras["multi_probe_probe"])
    agg_type = extras.get("multi_probe_agg_type", getattr(head, "agg_type", "attention"))
    if agg_type == "linear":
        head.aggregator.weight.data.copy_(extras["multi_probe_aggregator"])
    else:
        head.agg_query.data.copy_(extras["multi_probe_agg_query"])
        head.agg_init_bias.data.copy_(extras["multi_probe_agg_init_bias"])
        head.agg_proj.weight.data.copy_(extras["multi_probe_agg_proj"])
        if "multi_probe_attn_temp" in extras:
            head.attn_temp = float(extras["multi_probe_attn_temp"])
    model = model.to(DEV).eval()
    tok = M._wrap_tokenizer_with_lowercase(tok)                # same preprocessing as training

    tf = T.Compose([T.Resize((M.IMAGE_SIZE, M.IMAGE_SIZE), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    G, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(gpaths), a.bs):
            px = torch.stack([tf(Image.open(p).convert("RGB")) for p in gpaths[i:i + a.bs]]).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                v = model.get_image_features(pixel_values=px)
            v = v[0] if isinstance(v, (tuple, list)) else v
            G.append(F.normalize(v.float(), dim=-1).cpu())
            if (i // a.bs) % 50 == 0:
                print(f"  gallery {min(i + a.bs, len(gpaths))}/{len(gpaths)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        G = torch.cat(G)
        Q = []
        for i in range(0, len(caps), a.bs):
            enc = tok([M._siglip2_normalize_text(c) for c in caps[i:i + a.bs]], padding="max_length",
                      max_length=M.MAX_TEXT_LENGTH, truncation=True, return_tensors="pt")
            ids = enc["input_ids"].to(DEV)
            mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"])).to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda"):
                t = model.get_text_features(input_ids=ids, attention_mask=mask)
            t = t[0] if isinstance(t, (tuple, list)) else t
            Q.append(F.normalize(t.float(), dim=-1).cpu())
        Q = torch.cat(Q)
    S = Q @ G.t()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(S, OUT)
    print(f"[save] {OUT}  {tuple(S.shape)}", flush=True)


# ── stage 2: BEiT3-helip token MaxSim + blend ───────────────────────────────
def stage_maxsim():
    """Token<->patch MaxSim with the BEiT3-helip checkpoint over the base's own
    top-K columns, blended as (1-a)*base + a*rowminmax(MaxSim)."""
    from timm import create_model                              # noqa: E402
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD   # noqa: E402
    from torchvision import transforms as T                    # noqa: E402
    from transformers import XLMRobertaTokenizer               # noqa: E402
    # the vendored BEiT3 tree does `import utils` for its own utils.py; drop the cached
    # pipeline/utils package so that import resolves inside third_party/beit3
    for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
        del sys.modules[k]
    sys.path.insert(0, os.environ.get("BEIT3_SRC", f"{_REPO}/third_party/beit3"))
    import modeling_finetune                                   # noqa: E402,F401

    MAXT = 64                                                  # the original extraction used t64
    ckpt = a.ckpt or os.environ.get("MAXSIM_CKPT", f"{MODEL_ROOT}/beit3_helip/checkpoint-best.pth")
    spm = os.environ.get("BEIT3_SPM", f"{_REPO}/assets/model/encoder/beit3_pre/beit3.spm")
    for p, what in ((ckpt, "beit3_helip checkpoint"), (spm, "tokenizer"),
                    (BASE_PT, "base score (run --stage base first)")):
        if not os.path.exists(p):
            raise SystemExit(f"[maxsim] missing {what}: {p}")

    print(f"[maxsim] tokens=beit3_helip t{MAXT} <- {ckpt}  K={a.topk} alpha={a.alpha}", flush=True)
    m = create_model("beit3_large_patch16_384_retrieval")
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    m = m.to(DEV).eval()
    tok = XLMRobertaTokenizer(vocab_file=spm)
    tf = T.Compose([T.Resize((384, 384), interpolation=3), T.ToTensor(),
                    T.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)])

    base = torch.nan_to_num(torch.load(BASE_PT, map_location="cpu", weights_only=False).float(),
                            neginf=-1e9)
    if a.limit:
        base = base[:len(caps), :len(gpaths)]
    topi = base.topk(a.topk, 1).indices

    def tok_batch(texts):
        ib, pb = [], []
        for c in texts:
            ids = ([tok.bos_token_id] + tok.convert_tokens_to_ids(tok.tokenize(c)[:MAXT - 2])
                   + [tok.eos_token_id])
            n = len(ids)
            pb.append([0] * n + [1] * (MAXT - n))
            ib.append(ids + [tok.pad_token_id] * (MAXT - n))
        return torch.tensor(ib), torch.tensor(pb)

    t0 = time.time()
    with torch.no_grad():
        qtoks = []
        for s in range(0, len(caps), 64):
            ids, pad = tok_batch(caps[s:s + 64])
            out = m.beit3(textual_tokens=ids.to(DEV), visual_tokens=None,
                          text_padding_position=pad.to(DEV))["encoder_out"]
            t = F.normalize(m.language_head(out), dim=-1).cpu()
            for i in range(t.shape[0]):
                qtoks.append(t[i][pad[i] == 0].half())
        print(f"[maxsim] query tokens {len(qtoks)} ({time.time() - t0:.0f}s)", flush=True)

        uniq = sorted(set(topi.flatten().tolist()))
        pemb = {}
        for s in range(0, len(uniq), a.ibs):
            gis = uniq[s:s + a.ibs]
            px = torch.stack([tf(Image.open(gpaths[gi]).convert("RGB")) for gi in gis]).to(DEV)
            out = m.beit3(textual_tokens=None, visual_tokens=px)["encoder_out"]
            p = F.normalize(m.vision_head(out[:, 1:, :]), dim=-1).cpu().half()
            for j, gi in enumerate(gis):
                pemb[gi] = p[j]
            if (s // a.ibs) % 20 == 0:
                print(f"  patches {min(s + a.ibs, len(uniq))}/{len(uniq)} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        mx = torch.zeros(base.shape[0], a.topk)
        for qi in range(base.shape[0]):
            q = qtoks[qi].to(DEV).float()
            imgs = torch.stack([pemb[int(topi[qi, k])] for k in range(a.topk)]).to(DEV).float()
            sim = torch.einsum("td,kpd->ktp", q, imgs)         # (K, Tq, 576)
            mx[qi] = sim.max(dim=2).values.sum(dim=1).cpu()

    lo = mx.min(1, keepdim=True).values
    rng = (mx.max(1, keepdim=True).values - lo).clamp_min(1e-12)
    blend = (1 - a.alpha) * base.gather(1, topi) + a.alpha * ((mx - lo) / rng)   # per-row min-max
    S = base.clone().scatter_(1, topi, blend)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    torch.save(S, OUT)
    print(f"[save] {OUT}  {tuple(S.shape)}", flush=True)


(stage_base if a.stage == "base" else stage_maxsim)()
