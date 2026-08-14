#!/usr/bin/env python3
"""jina_m0/train — bidirectional DoRA fine-tuning of jina-reranker-m0.

Its fusion weight is 0, but the S4 tail (near-duplicate promotion) uses it, so it is a required
deployment member. The structure matches `../qwen3vl_2b/train.py`:
  Path A (attribute grounding)  anchor=image X · pos=orig_cap · neg=flip_caps
  Path B (cross-image ranking)  anchor=styled recap query · pos=image X
                                · neg = img_negs (appearance) + action_negs (behaviour)
  loss = CE_A + λB·CE_B. Path B queries are sampled uniformly over the 12 styles, from train
  captions only, so no test caption is involved.

jina specifics:
  · JinaVLForRanking (Qwen2-VL-2B, trust_remote_code) — lm_head=Identity plus an MLP score head.
    The deployment score is sigmoid(MLP(hidden[-1]) - LOGIT_BIAS); training needs the **raw logit
    before the sigmoid** in order to use CE, which is why the forward pass is replicated below.
  · The score MLP head stays frozen — only the DoRA adapter on the text decoder is trained.

Input  NEG_CACHE = the `build_negcache.py` output (shared with qwen3vl_2b).
Run    python -u train.py --gpu 6
       Hyper-parameters are edited directly in the `1. Config` constants below.
"""
import argparse, json, math, os, random, sys, time
from collections import Counter, defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from PIL import Image
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repository root — all default paths are relative to it

# ---- 1. Config ----
RUN_TAG         = "full"
RUN_NOTE        = os.path.basename(os.path.dirname(os.path.abspath(__file__)))   # run name = {timestamp}_{directory name}
STYLE_CSV       = os.environ.get("RECAP_CSV", f"{_REPO}/assets/data/raw/recaption/train_msr_v2.csv")
STYLE_LIST      = ["p00_original", "p01_lexical", "p02_phrasal", "p03_clausal", "p04_diathesis", "p05_involved", "p06_informal", "p07_telegraphic", "p08_compact", "p09_narrative", "p10_formal", "p11_compound"]
STYLE_UNIFORM   = True
SEED            = 42
HF_CACHE        = os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache")
JINA_REPO       = "jinaai/jina-reranker-m0"
MINING_DIR      = f"{_REPO}/assets/data/mining"                                            # negcache input — read only
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", f"{_REPO}/assets/runs/rerank_jina")         # the only write target (never overwrite the shared tree)
RUNS_ROOT       = f"{OUTPUT_DIR}/runs"

# Vision token budget (Qwen2-VL uses a 28x28 effective patch). Near-duplicate discrimination depends
# on resolution, so the upper bound is generous.
IMG_MIN_PIXELS  = 4 * 28 * 28
IMG_MAX_PIXELS  = 1024 * 28 * 28
MAX_LENGTH      = 2048            # caption + image tokens (~1024) + overhead
INSTRUCTION = ("Given a textual description of a person, retrieve the image that "
               "depicts exactly that person (matching gender, clothing color/type, and action).")

# ---- negcache (the same file is shared with qwen3vl_2b) ----
N_IMG_NEG     = 4
N_ACTION_NEG  = 4
NEG_CACHE   = f"{MINING_DIR}/negcache_action_top8a6.pt"   # point at OUTPUT_DIR if you rebuilt it yourself
# The cache stores image paths relative to this root, so the dataset can sit anywhere.
DATA_ROOT     = os.environ.get("PAB_TRAIN", f"{_REPO}/assets/data/raw/pab_train")
IMG_ROOT_WEBP = f"{DATA_ROOT}/train_webp"


def img_abs(p):
    """Cache path -> readable path. Caches written before the paths were made relative hold an
    absolute path already; those are passed through so an old cache keeps working."""
    return p if os.path.isabs(p) else os.path.join(IMG_ROOT_WEBP, p)

# ---- DoRA (r16/α32/dropout0.10) ----
DORA_RANK       = 16
DORA_ALPHA      = 32
DORA_DROPOUT    = 0.10
DORA_TARGET_SET = "text"
DORA_TARGETS_TEXT   = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DORA_TARGETS_VISION = ["qkv", "proj", "mlp.0", "mlp.2"]   # unused while DORA_TARGET_SET="text"

# ---- training (checkpoint period is counted in examples seen; step selection is ../eval/eval_step.py) ----
EPOCHS          = 1
SAMPLES_PER_EPOCH = 0
SAMPLE_MODE     = "fresh"
CKPT_EVERY_EXAMPLES = 1000
GRAD_ACCUM      = 16
LR              = 1e-4
WEIGHT_DECAY    = 0.0
BETAS           = (0.9, 0.95)
WARMUP_FRAC     = 0.05
WARMUP_STEPS    = 80
MAX_GRAD_NORM   = 1.0
N_FLIP_NEG      = 3
LAMBDA_B        = 2.0
GRAD_CKPT       = True
LOG_EVERY       = 20
CKPT_LAST_NAME  = "last"

# Taken from jina's modeling.py — must be updated together with upstream
LOGIT_BIAS      = 2.65
SCORE_TOKEN_ID  = 100


# ---- 2. logging and utilities ----
def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_global_dirs():
    for d in (OUTPUT_DIR, RUNS_ROOT):
        os.makedirs(d, exist_ok=True)


def make_run_dir(note):
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{run_tag}_{note}" if note else run_tag
    run_dir = f"{RUNS_ROOT}/{name}"
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{run_dir}/tb", exist_ok=True)
    return run_dir, name


class TeeStdout:
    def __init__(self, log_path, also_stderr=True, timestamp=True):
        self._file = open(log_path, "a", buffering=1, encoding="utf-8")
        self._stdout = sys.stdout
        self._timestamp = timestamp
        self._line_start = True

    def write(self, msg):
        if not msg:
            return
        if self._timestamp:
            out = []
            for piece in msg.splitlines(keepends=True):
                if self._line_start and piece.strip():
                    out.append(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {piece}")
                else:
                    out.append(piece)
                self._line_start = piece.endswith("\n")
            decorated = "".join(out)
        else:
            decorated = msg
        self._stdout.write(msg); self._file.write(decorated)

    def flush(self):
        self._stdout.flush(); self._file.flush()

    def isatty(self):
        return self._stdout.isatty()


class MetricWriter:
    def __init__(self, path):
        self._f = open(path, "a", buffering=1, encoding="utf-8")

    def log(self, **fields):
        fields["_ts"] = time.time()
        self._f.write(json.dumps(fields, ensure_ascii=False, default=float) + "\n")

    def close(self):
        try: self._f.flush(); self._f.close()
        except Exception: pass


def collect_env_info():
    import platform, socket
    info = {"timestamp": datetime.now().isoformat(timespec="seconds"),
            "hostname": socket.gethostname(), "python": sys.version.split()[0],
            "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda if torch.cuda.is_available() else None}
    for pkg in ("transformers", "peft"):
        try:
            info[pkg] = getattr(__import__(pkg), "__version__", "unknown")
        except Exception:
            info[pkg] = None
    return info


def compute_grad_norm(params, norm_type=2.0):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.stack([g.detach().float().norm(norm_type) for g in grads]).norm(norm_type).item())


def gpu_mem_stats():
    if not torch.cuda.is_available():
        return {}
    return {"gpu_mem_alloc_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "gpu_mem_peak_gb":  round(torch.cuda.max_memory_allocated() / 1024**3, 2)}


# jina formatting, replicated from modeling.py to avoid depending on its import surface
def formatting_prompt(query_text, query_type="text", doc_type="image"):
    query_part = ("**Query**:\n<|vision_start|><|image_pad|><|vision_end|>"
                  if query_type == "image" else f"**Query**:\n{query_text}")
    doc_part = ("**Document**:\n<|vision_start|><|image_pad|><|vision_end|>"
                if doc_type == "image" else f"**Document**:\n{query_text}")
    return doc_part + "\n" + query_part


# ---- 3. load the cache, then train both paths ----
def main():
    ap = argparse.ArgumentParser(description="bidirectional DoRA fine-tuning of jina-reranker-m0 (single GPU)")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--run-note", default="", help="run directory suffix (default: none → {timestamp}_{directory name})")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    set_seed(SEED)
    rng = random.Random(SEED)
    device = "cuda:0"; world_size = 1

    assert os.path.isfile(NEG_CACHE), (
        f"NEG_CACHE not found: {NEG_CACHE}\n  build the cache first with `python build_negcache.py`.")
    cache = torch.load(NEG_CACHE, map_location="cpu")
    examples = cache["examples"]
    rng.shuffle(examples); pool_n = len(examples)

    # ---- Path B query style map (train recap only, never test) ----
    import csv as _csv
    def norm_key(p):
        parts = p.replace("\\", "/").split("/")
        for i, seg in enumerate(parts):
            if seg.startswith("imgs_"):
                return os.path.splitext("/".join(parts[i:]))[0]
        return os.path.splitext(parts[-1])[0]
    pos_keys = {norm_key(e["image_path"]) for e in examples}
    style_map = {}
    _csv.field_size_limit(10**7)
    with open(STYLE_CSV, newline="") as f:
        for row in _csv.DictReader(f):
            k = norm_key(row["image_path"]); st = row["style"]
            if k in pos_keys and st in STYLE_LIST:
                cap = (row.get("caption") or "").strip()
                if cap:
                    style_map.setdefault(k, {})[st] = cap
    n_full = sum(1 for k in pos_keys if len(style_map.get(k, {})) >= len(STYLE_LIST))
    print(f"[qstyle] pos images={len(pos_keys):,} | style map={len(style_map):,} | all styles present={n_full:,}/{len(STYLE_LIST)} styles uniform={STYLE_UNIFORM}", flush=True)
    def sample_query(e, _rng):
        m = style_map.get(norm_key(e["image_path"]))
        if not m:
            return e["pos"]
        st = _rng.choice([s for s in STYLE_LIST if s in m])
        return m[st]

    # ---- run directory and logging ----
    ensure_global_dirs()
    note = f"{RUN_NOTE}_{args.run_note}" if args.run_note else RUN_NOTE
    run_dir, _ = make_run_dir(f"{note}_{RUN_TAG}_w{world_size}")
    sys.stdout = TeeStdout(os.path.join(run_dir, "train.log"))
    mw = MetricWriter(os.path.join(run_dir, "metrics.jsonl"))
    from torch.utils.tensorboard import SummaryWriter
    tb = SummaryWriter(os.path.join(run_dir, "tb"))
    print(f"[run] {run_dir}", flush=True)
    print(f"[cache] {NEG_CACHE} | meta={cache.get('meta')}", flush=True)
    json.dump(collect_env_info(), open(os.path.join(run_dir, "env.json"), "w"), indent=2)
    cfg = {k: v for k, v in globals().items()
           if k.isupper() and isinstance(v, (int, float, str, bool, tuple, list))}
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2, default=str)

    N = SAMPLES_PER_EPOCH if 0 < SAMPLES_PER_EPOCH < pool_n else pool_n
    uniq = min(EPOCHS * N, pool_n) if SAMPLE_MODE == "fresh" else N
    print(f"[dataset] working_pool={pool_n:,} | per_epoch={N:,} | mode={SAMPLE_MODE} | EPOCHS={EPOCHS} "
          f"→ total unique≈{uniq:,} | img_neg={N_IMG_NEG} act_neg={N_ACTION_NEG}", flush=True)
    mw.log(event="dataset", working_pool=pool_n, per_epoch=N, mode=SAMPLE_MODE, epochs=EPOCHS,
           total_unique=uniq, n_img_neg=N_IMG_NEG, n_action_neg=N_ACTION_NEG, n_flip=N_FLIP_NEG, lambda_b=LAMBDA_B,
           axis_dist=dict(Counter(a for e in examples for a in e["axes"])))

    # ---- jina reranker + DoRA ----
    from transformers import AutoModel, AutoProcessor, Qwen2VLForConditionalGeneration
    try:
        model = AutoModel.from_pretrained(JINA_REPO, trust_remote_code=True, torch_dtype=torch.bfloat16)
    except Exception:
        model = AutoModel.from_pretrained(JINA_REPO, trust_remote_code=True).to(torch.bfloat16)
    processor = AutoProcessor.from_pretrained(JINA_REPO, trust_remote_code=True,
                                              min_pixels=IMG_MIN_PIXELS, max_pixels=IMG_MAX_PIXELS)
    sc_tok = int(getattr(model, "score_token_id", SCORE_TOKEN_ID))
    # the score head stays frozen — only the adapter is trained
    for p in model.score.parameters():
        p.requires_grad_(False)
    targets = {"text": DORA_TARGETS_TEXT, "vision": DORA_TARGETS_VISION,
               "both": DORA_TARGETS_TEXT + DORA_TARGETS_VISION}[DORA_TARGET_SET]
    model = get_peft_model(model, LoraConfig(
        r=DORA_RANK, lora_alpha=DORA_ALPHA, lora_dropout=DORA_DROPOUT,
        use_dora=True, bias="none", target_modules=targets, task_type=None))
    model = model.to(device)
    if GRAD_CKPT:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model.train(); model.print_trainable_parameters()
    base = model.get_base_model()      # JinaVLForRanking with peft unwrapped — needed for the grandparent forward and the score head
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)

    def score_logit(caption, img_path):
        """Replicates jina's forward and returns the raw logit before the sigmoid. doc=image, query=caption."""
        prompt = formatting_prompt(caption, query_type="text", doc_type="image")
        pil = Image.open(img_path).convert("RGB")
        batch = processor(text=[prompt], images=[pil], return_tensors="pt",
                          padding=True, truncation=True, max_length=MAX_LENGTH - 1)
        bs = batch["input_ids"].size(0)
        st = torch.full((bs, 1), sc_tok, dtype=batch["input_ids"].dtype)
        batch["input_ids"] = torch.cat([batch["input_ids"], st], dim=1)
        batch["attention_mask"] = torch.cat(
            [batch["attention_mask"], torch.ones((bs, 1), dtype=batch["attention_mask"].dtype)], dim=1)
        if "mm_token_type_ids" in batch:
            batch["mm_token_type_ids"] = torch.cat(
                [batch["mm_token_type_ids"], torch.zeros((bs, 1), dtype=batch["mm_token_type_ids"].dtype)], dim=1)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outputs = Qwen2VLForConditionalGeneration.forward(
            base, output_hidden_states=True, use_cache=False, **batch)
        hidden = outputs.hidden_states[-1][:, -1]
        return base.score(hidden).squeeze(-1).squeeze()      # pre-sigmoid raw logit

    # ---- LR schedule ----
    steps_per_epoch = max(1, N // GRAD_ACCUM); total_steps = steps_per_epoch * EPOCHS
    warmup = WARMUP_STEPS if WARMUP_STEPS > 0 else max(1, int(total_steps * WARMUP_FRAC))
    def lr_at(st):
        if st < warmup:
            return LR * st / warmup
        return LR * 0.5 * (1 + math.cos(math.pi * min(1.0, (st - warmup) / max(1, total_steps - warmup))))
    print(f"[train] per_epoch={N:,} steps/ep={steps_per_epoch:,} total={total_steps:,} warmup={warmup} "
          f"accum={GRAD_ACCUM} flip={N_FLIP_NEG} imgneg={N_IMG_NEG} actneg={N_ACTION_NEG} λB={LAMBDA_B} "
          f"target={DORA_TARGET_SET}", flush=True)

    def slice_ep(ep):
        if N >= pool_n:
            return examples
        if SAMPLE_MODE == "fixed":
            return examples[:N]
        st = (ep - 1) * N
        if st + N <= pool_n:
            return examples[st:st + N]
        p = examples[:]; random.Random(SEED + ep).shuffle(p); return p[:N]

    def fresh():
        return defaultdict(float), defaultdict(lambda: [0, 0, 0.0])
    CE = lambda lg: F.cross_entropy(lg.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=device))
    gstep = 0; ep_lossA = ep_lossB = 0.0; gt0 = time.time()
    ex_seen = 0; next_ckpt = CKPT_EVERY_EXAMPLES
    for ep in range(1, EPOCHS + 1):
        ex_ep = slice_ep(ep); rng.shuffle(ex_ep)
        opt.zero_grad(set_to_none=True)
        run, run_ax = fresh(); ep_, ep_ax = fresh(); t0 = time.time(); nan_skips = 0
        print(f"\n===== epoch {ep}/{EPOCHS}  examples={len(ex_ep):,} =====", flush=True)
        for qi, e in enumerate(ex_ep):
            X, cap = img_abs(e["image_path"]), e["pos"]
            capQ = sample_query(e, rng)
            s_posA = score_logit(cap, X)
            logA = torch.stack([s_posA] + [score_logit(f, X) for f in e["flip_negs"]]).float()
            s_posB = s_posA if capQ == cap else score_logit(capQ, X)
            negsB = e["img_negs"][:N_IMG_NEG] + e.get("action_negs", [])[:N_ACTION_NEG]
            logB = torch.stack([s_posB] + [score_logit(capQ, img_abs(ni)) for ni in negsB]).float()
            lossA, lossB = CE(logA), CE(logB)
            loss = lossA + LAMBDA_B * lossB
            if not torch.isfinite(loss):
                nan_skips += 1; opt.zero_grad(set_to_none=True); continue
            (loss / GRAD_ACCUM).backward()
            ex_seen += 1
            with torch.no_grad():
                accA = int(torch.argmax(logA).item() == 0); accB = int(torch.argmax(logB).item() == 0)
                mA = (logA[0] - logA[1:].max()).item(); mB = (logB[0] - logB[1:].max()).item()
            for w, wa in ((run, run_ax), (ep_, ep_ax)):
                w["lossA"] += lossA.item(); w["lossB"] += lossB.item()
                w["accA"] += accA; w["accB"] += accB; w["mA"] += mA; w["mB"] += mB; w["seen"] += 1
                for j, ax in enumerate(e["axes"][:len(e["flip_negs"])]):
                    s = wa[ax]; s[0] += 1; s[1] += int((logA[0] > logA[1+j]).item()); s[2] += (logA[0] - logA[1+j]).item()

            if (qi + 1) % GRAD_ACCUM == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(gstep)
                gnorm = compute_grad_norm(trainable)
                torch.nn.utils.clip_grad_norm_(trainable, MAX_GRAD_NORM)
                opt.step(); opt.zero_grad(set_to_none=True); gstep += 1
                if ex_seen >= next_ckpt:
                    tag = f"ex{ex_seen:06d}"
                    for nm in (tag, CKPT_LAST_NAME):
                        ck = os.path.join(run_dir, "checkpoints", nm); os.makedirs(ck, exist_ok=True)
                        model.save_pretrained(ck)
                    mw.log(event="checkpoint", ex_seen=ex_seen, epoch=ep, step=gstep,
                           path=os.path.join(run_dir, "checkpoints", tag))
                    print(f"  [ckpt] ex={ex_seen:,} → checkpoints/{tag} (+{CKPT_LAST_NAME})", flush=True)
                    next_ckpt += CKPT_EVERY_EXAMPLES
                if gstep % LOG_EVERY == 0:
                    n = max(1, int(run["seen"])); el = time.time() - t0
                    axwin = {a: round(100*s[1]/max(1, s[0]), 1) for a, s in run_ax.items()}
                    print(f"  ep{ep} step{gstep}/{total_steps} q{qi+1}/{len(ex_ep)} "
                          f"lossA={run['lossA']/n:.3f} lossB={run['lossB']/n:.3f} "
                          f"accA={100*run['accA']/n:.1f}% accB={100*run['accB']/n:.1f}% "
                          f"mA={run['mA']/n:+.2f} mB={run['mB']/n:+.2f} grad={gnorm:.2f} "
                          f"lr={lr_at(gstep):.2e} {el/(qi+1):.2f}s/q axwin={axwin}", flush=True)
                    mw.log(event="step", epoch=ep, step=gstep, lossA=run["lossA"]/n, lossB=run["lossB"]/n,
                           accA=100*run["accA"]/n, accB=100*run["accB"]/n, marginA=run["mA"]/n, marginB=run["mB"]/n,
                           grad_norm=gnorm, lr=lr_at(gstep), examples_per_s=(qi+1)/el, axis_winrate=axwin)
                    for k, v in (("lossA", run["lossA"]/n), ("lossB", run["lossB"]/n),
                                 ("accB", 100*run["accB"]/n), ("marginB", run["mB"]/n), ("grad_norm", gnorm)):
                        tb.add_scalar(f"train/{k}", v, gstep)
                    run, run_ax = fresh()

        n = max(1, int(ep_["seen"]))
        ep_lossA, ep_lossB = ep_["lossA"]/n, ep_["lossB"]/n
        ax_sum = {a: {"winrate": round(100*s[1]/max(1, s[0]), 2), "margin": round(s[2]/max(1, s[0]), 4), "n": s[0]}
                  for a, s in ep_ax.items()}
        dur = time.time() - t0
        print(f"  [train-progress] ep{ep}: lossA={ep_lossA:.4f} lossB={ep_lossB:.4f} "
              f"accA={100*ep_['accA']/n:.1f}% accB={100*ep_['accB']/n:.1f}% "
              f"mA={ep_['mA']/n:+.3f} mB={ep_['mB']/n:+.3f} nan={nan_skips} "
              f"({dur:.0f}s, {n/dur:.2f}ex/s) {gpu_mem_stats()}", flush=True)
        mw.log(event="train_progress", epoch=ep, lossA=ep_lossA, lossB=ep_lossB,
               accA=100*ep_["accA"]/n, accB=100*ep_["accB"]/n, marginA=ep_["mA"]/n, marginB=ep_["mB"]/n,
               nan_skips=nan_skips, sec=round(dur, 1), ex_per_s=round(n/dur, 3), axis=ax_sum, gpu=gpu_mem_stats())
        ck = os.path.join(run_dir, "checkpoints", CKPT_LAST_NAME); os.makedirs(ck, exist_ok=True)
        model.save_pretrained(ck)
        print(f"  [epoch {ep} done] ex_seen={ex_seen:,}, last updated", flush=True)

    mw.log(event="summary", epochs=EPOCHS, final_lossA=ep_lossA, final_lossB=ep_lossB,
           total_sec=round(time.time()-gt0, 1), run_dir=run_dir)
    tb.close(); mw.close(); print("[done]", flush=True)


if __name__ == "__main__":
    main()
