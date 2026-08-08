#!/usr/bin/env python3
"""qwen3vl_2b/train — bidirectional DoRA fine-tuning of Qwen3-VL-Reranker-2B.

The last of the three reranker data steps: manifest → negcache → **training**.

Both directions are trained together so the score function s(text, image) is calibrated from either
side:
  Path A (attribute grounding)  anchor=image X · pos=orig_cap · neg=flip_cap (colour, clothing, action)
  Path B (cross-image ranking)  anchor=caption C · pos=image X · neg=the anchor encoder's top-K
                                similar images, i.e. the same distribution as deployment inference;
                                negcache mines them in advance
  loss = CE_A + λB·CE_B. The shared positive s(C,X) is forwarded once and reused by both paths.

Input  NEG_CACHE (.pt) from `build_negcache.py`; its absence stops the run at an assert.
       The cache exists so that peft and the anchor encoder run in separate processes.
Run    Single GPU. Hyper-parameters are edited directly in the `1. Config` constants below.
        python -u train.py --gpu 6
"""
import argparse, json, math, os, random, sys, time
from collections import Counter, defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from huggingface_hub import snapshot_download
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repository root — all default paths are relative to it

# ---- 1. Config ----
RUN_TAG         = "full"
RUN_NOTE        = os.path.basename(os.path.dirname(os.path.abspath(__file__)))   # run name = {timestamp}_{directory name}
SEED            = 42
HF_CACHE        = os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache")   # base VLM — read only
QWEN_REPO       = "Qwen/Qwen3-VL-Reranker-2B"
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", f"{_REPO}/assets/runs/rerank_qw2b")   # the only write target
RUNS_ROOT       = f"{OUTPUT_DIR}/runs"

# smart-resize token budget, shared by training and inference. A 1024-pixel webp is kept near its
# original size (~1024 vision tokens). Near-duplicate discrimination depends on resolution, so
# lowering the cap costs accuracy; it is also the main speed lever.
IMG_MIN_PIXELS  = 4 * 32 * 32
IMG_MAX_PIXELS  = 1280 * 32 * 32
INSTRUCTION = ("Given a textual description of a person, retrieve the image that "
               "depicts exactly that person (matching gender, clothing color/type, and action).")

# ---- negcache — these values must match build_negcache.py, or the path differs and the assert fires ----
ANCHOR_CKPT   = "ep06"
NEG_POOL_SIZE = 0          # 0 = the whole pool (same as build_negcache)
N_IMG_NEG     = 5
NEARDUP_TAU   = 0.85
POOL_TAG      = "full" if not NEG_POOL_SIZE else str(NEG_POOL_SIZE)
MINING_DIR    = os.environ.get("MINE_DIR", f"{_REPO}/assets/data/mining")   # negcache input — read only
NEG_CACHE   = f"{MINING_DIR}/negcache_hardimg_{ANCHOR_CKPT}_pool{POOL_TAG}_top{N_IMG_NEG}_tau{NEARDUP_TAU}.pt"   # point at OUTPUT_DIR if you rebuilt it yourself

# ---- DoRA (r16/α32/dropout0.10) ----
DORA_RANK       = 16
DORA_ALPHA      = 32
DORA_DROPOUT    = 0.10
DORA_TARGET_SET = "text"                    # text | vision | both
DORA_TARGETS_TEXT   = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DORA_TARGETS_VISION = ["qkv", "proj", "linear_fc1", "linear_fc2"]

# ---- training ----
# The checkpoint period counts **examples seen** (CKPT_EVERY_EXAMPLES), not epochs: one epoch takes
# roughly 25 h, so epoch granularity would leave no steps to choose between. Each example costs
# (1 + flip + img_neg) ~= 9 forwards, about 4.7 s at webp 1024.
# Step selection happens outside training, in `../eval/eval_step.py`.
EPOCHS          = 1
SAMPLES_PER_EPOCH = 0                         # 0 = the whole cache
SAMPLE_MODE     = "fresh"
CKPT_EVERY_EXAMPLES = 1000
GRAD_ACCUM      = 8
LR              = 1e-4
WEIGHT_DECAY    = 0.0
BETAS           = (0.9, 0.95)
WARMUP_FRAC     = 0.05                        # used only when WARMUP_STEPS=0
WARMUP_STEPS    = 80                          # fixed warmup, so even an early ckpt is past peak LR (~640 examples, about 1 h)
MAX_GRAD_NORM   = 1.0
N_FLIP_NEG      = 3                          # cap on Path A flip negatives (index-aligned with the axes logging)
LAMBDA_B        = 1.0                        # weight on CE_B
GRAD_CKPT       = True
LOG_EVERY       = 20
CKPT_LAST_NAME  = "last"


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
        self._also_stderr = also_stderr
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
            "cuda": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}
    if torch.cuda.is_available():
        info["gpus"] = [{"idx": i, "name": torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())]
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


# ---- 3. load the cache, then train both paths ----
def main():
    ap = argparse.ArgumentParser(description="bidirectional DoRA fine-tuning of Qwen3-VL-Reranker-2B (single GPU)")
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
    examples = cache["examples"]   # {image_path(webp), pos, flip_negs, axes, img_negs:[webp...]}
    rng.shuffle(examples); pool_n = len(examples)

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
    print(f"[dataset] working_pool={pool_n:,} | per_epoch={N:,} | mode={SAMPLE_MODE} | "
          f"EPOCHS={EPOCHS} → total unique≈{uniq:,} | img_neg/example={N_IMG_NEG}", flush=True)
    mw.log(event="dataset", working_pool=pool_n, per_epoch=N, mode=SAMPLE_MODE, epochs=EPOCHS,
           total_unique=uniq, n_img_neg=N_IMG_NEG, n_flip=N_FLIP_NEG, lambda_b=LAMBDA_B,
           axis_dist=dict(Counter(a for e in examples for a in e["axes"])))

    # reranker + DoRA (the score head stays frozen — only the adapter is trained)
    snap = snapshot_download(QWEN_REPO, cache_dir=os.path.join(HF_CACHE, "hub"))
    sys.path.insert(0, os.path.join(snap, "scripts"))
    from qwen3_vl_reranker import Qwen3VLReranker

    class PatchedReranker(Qwen3VLReranker):
        def tokenize(self, pairs, **kw):
            inp = super().tokenize(pairs, **kw)
            mt = inp.get("mm_token_type_ids")
            if isinstance(mt, list):
                inp["mm_token_type_ids"] = torch.tensor(mt, dtype=torch.long)
            return inp

    rr = PatchedReranker(snap, torch_dtype=torch.bfloat16,
                         min_pixels=IMG_MIN_PIXELS, max_pixels=IMG_MAX_PIXELS)
    targets = {"text": DORA_TARGETS_TEXT, "vision": DORA_TARGETS_VISION,
               "both": DORA_TARGETS_TEXT + DORA_TARGETS_VISION}[DORA_TARGET_SET]
    rr.model = get_peft_model(rr.model, LoraConfig(
        r=DORA_RANK, lora_alpha=DORA_ALPHA, lora_dropout=DORA_DROPOUT,
        use_dora=True, bias="none", target_modules=targets, task_type=None))
    for p in rr.score_linear.parameters():
        p.requires_grad_(False)
    if GRAD_CKPT:
        rr.model.gradient_checkpointing_enable(); rr.model.enable_input_require_grads()
    rr.model.train(); rr.model.print_trainable_parameters()
    trainable = [p for p in rr.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY, betas=BETAS)

    def score_logit(caption, img_path):
        pair = rr.format_mm_instruction(caption, None, None, None, img_path, None, instruction=INSTRUCTION)
        inputs = rr.tokenize([pair]).to(device)
        hidden = rr.model(**inputs).last_hidden_state[:, -1]
        return rr.score_linear(hidden).squeeze()

    steps_per_epoch = max(1, N // GRAD_ACCUM); total_steps = steps_per_epoch * EPOCHS
    warmup = WARMUP_STEPS if WARMUP_STEPS > 0 else max(1, int(total_steps * WARMUP_FRAC))
    def lr_at(st):
        if st < warmup:
            return LR * st / warmup
        return LR * 0.5 * (1 + math.cos(math.pi * min(1.0, (st - warmup) / max(1, total_steps - warmup))))
    print(f"[train] per_epoch={N:,} steps/ep={steps_per_epoch:,} total={total_steps:,} warmup={warmup} "
          f"accum={GRAD_ACCUM} flip={N_FLIP_NEG} imgneg={N_IMG_NEG} λB={LAMBDA_B} target={DORA_TARGET_SET}", flush=True)

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
            X, cap = e["image_path"], e["pos"]
            s_pos = score_logit(cap, X)                                       # shared by both paths (one forward)
            logA = torch.stack([s_pos] + [score_logit(f, X) for f in e["flip_negs"]]).float()     # A: contrast over text
            logB = torch.stack([s_pos] + [score_logit(cap, ni) for ni in e["img_negs"]]).float()  # B: contrast over images
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
                        rr.model.save_pretrained(ck)
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
        print(f"  [axis] " + " | ".join(f"{a}: win={v['winrate']}% m={v['margin']:+.3f}(n={v['n']})"
                                        for a, v in ax_sum.items()), flush=True)
        mw.log(event="train_progress", epoch=ep, lossA=ep_lossA, lossB=ep_lossB,
               accA=100*ep_["accA"]/n, accB=100*ep_["accB"]/n, marginA=ep_["mA"]/n, marginB=ep_["mB"]/n,
               nan_skips=nan_skips, sec=round(dur, 1), ex_per_s=round(n/dur, 3), axis=ax_sum, gpu=gpu_mem_stats())
        # refresh only `last` at the end of an epoch, so the tail (<CKPT_EVERY_EXAMPLES) is not lost
        ck = os.path.join(run_dir, "checkpoints", CKPT_LAST_NAME); os.makedirs(ck, exist_ok=True)
        rr.model.save_pretrained(ck)
        print(f"  [epoch {ep} done] ex_seen={ex_seen:,}, last updated", flush=True)

    mw.log(event="summary", epochs=EPOCHS, final_lossA=ep_lossA, final_lossB=ep_lossB,
           total_sec=round(time.time()-gt0, 1), run_dir=run_dir)
    tb.close(); mw.close(); print("[done]", flush=True)


if __name__ == "__main__":
    main()
