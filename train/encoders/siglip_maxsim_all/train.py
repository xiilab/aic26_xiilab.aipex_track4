"""SigLIP2-L/16-384 + DoRA + PI(64->128), Multi-Probe K=2 + soft labels, full split, single GPU.

Same recipe and hyperparameters as siglip_maxsim_heldout, without the held-out exclusion.
The two runs differ only in the training data, which isolates the effect of the held-out split.

usage:
  python train.py --gpu <id>
  python train.py --gpu <id> --resume <ckpt>"""

import os
import sys
import json
import math
import random
import time
import socket
import platform
import shutil
import traceback
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import v2 as Tv2
from PIL import Image
from tqdm import tqdm

import cv2
cv2.setNumThreads(1)  # avoids DataLoader fork workers clashing with cv2 threads

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
HF_CACHE_DIR = "assets/model/hf_cache"   # single HF cache, relative to the repository root
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)

# every path below is relative to the repository root; run from there
IMG_ROOT        = "assets/data/raw/pab_train/train_jpg_512"  # Part 1~10 (512x512 jpg)

MANIFEST_DIR    = "assets/data/manifest"  # read-only consumer of the gen_manifest.py output

# caption source CSV; train/gen/gen_manifest.py builds the manifest (11 preset styles, 1.01M images, 11.15M rows)
RECAP_CSV_PATH      = os.environ.get("RECAP_CSV", "assets/data/raw/recaption/train_msr_v1.csv")  # manifest source; only read for the staleness check
MANIFEST_TAG        = "v1_scene"                  # msr v1 captions plus scene, for the soft labels
MANIFEST_PATH       = f"{MANIFEST_DIR}/pab_manifest_msr_{MANIFEST_TAG}.jsonl"

USE_UNIQUE_IMAGE_MODE = True
RUNS_ROOT       = "assets/runs"
RUN_NOTE        = os.path.basename(os.path.dirname(os.path.abspath(__file__)))  # run name = {timestamp}_{directory name}

MODEL_NAME      = "google/siglip2-large-patch16-384"  # large-384 (hidden 1024, 24L, 576 patch)
IMAGE_SIZE      = 384  # 24x24 grid (patch 16)
MAX_TEXT_LENGTH = 128  # PI maps into [0..63]
POSITION_EMBED_NEW_SIZE = 128  # *logical* target length; PI does not add rows

DORA_RANK       = 16
DORA_ALPHA      = 32  # alpha = 2 * rank
DORA_RANK_TEXT  = 16
DORA_ALPHA_TEXT = 32
DORA_DROPOUT    = 0.10
DORA_TARGETS    = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
FREEZE_VISION_DORA = False
FREEZE_POOLER_DORA = False

NUM_PROBES               = 2  # K=2 multi-probe
MULTI_PROBE_PERTURB_STD  = 0.01  # effective = std * probe.std() (relative)
MULTI_PROBE_PERTURB_MODE = "relative"
MULTI_PROBE_AGG_INIT     = "identity_warmup"
MULTI_PROBE_BLOCK_SCALE  = 0.1
TRAIN_MULTI_PROBE_PARAMS = True
LR_MULTI_PROBE           = 7e-5  # 1e-4 * sqrt(128/256)
WD_MULTI_PROBE           = 0.0

MULTI_PROBE_AGG_TYPE     = "attention"  # "linear" | "attention" (default)
MULTI_PROBE_ATTN_TEMP    = 1.0

LR_MULTI_PROBE_PROBE_SPLIT = True
LR_MULTI_PROBE_PROBE       = 3.5e-5  # 5e-5 * sqrt(128/256)
LR_MULTI_PROBE_AGG         = 7e-5  # 1e-4 * sqrt(128/256)

ENABLE_PROBE_DIVERSITY_REG    = True
PROBE_DIVERSITY_REG_LAMBDA    = 2e-3
PROBE_DIVERSITY_WARMUP_EPOCHS = 2  # disabled for the first 2 epochs

# GC=False peaks at ~155GB and OOMs; GC=True is required
N_POSITIVES_PER_IMAGE    = 2

SEED            = 42
EPOCHS          = 10
# flat at eta_min past the horizon; training continues to EPOCHS
SCHEDULE_HORIZON_EPOCHS = 15
BATCH_SIZE      = 128  # 384px with multi-positive K=2 gives 256 texts, so GC is required
REF_BATCH       = 128  # batch the LR was tuned at; lr_scale = sqrt(BATCH/REF)
NUM_WORKERS     = 4  # /dev/shm is shared, kept low for concurrent runs
PREFETCH_FACTOR = 2  # batches prefetched per worker (shm use = workers * prefetch * batch)
EVAL_NUM_WORKERS = 4
EVAL_BATCH_SIZE = 512
LR              = 7e-5  # 1e-4 * sqrt(128/256)
LR_LOGIT        = 1e-5

FINEGRAIN_ENABLED     = False  # FILIP off; multi-probe pooling is used instead
FINEGRAIN_WEIGHT      = 0.5  # lambda: L = L_global + lambda*L_filip
FINEGRAIN_DIM         = 256
FINEGRAIN_PATCH_POOL  = 4  # keeps the pooled patch count near 64 (512px = 32 grid -> pool 4)
FINEGRAIN_LOGIT_SCALE = 20.0  # fixed temperature (not a parameter, so no DDP logit sync)
FINEGRAIN_TOK_MAXTHR  = 0  # >0 enables the OOM guard (0 = off)
FINEGRAIN_AGG_RATIO   = 0.001  # k = max(1, round(Lpxratio)) = 1 -> top-1 mean = amax
                                   # mathematically identical to sim.amax(dim=-1)
FINEGRAIN_TARGET_LP   = 64  # FILIP late-interaction resolution (8x8); changing it is a different experiment
if FINEGRAIN_ENABLED:
    _fg_grid = IMAGE_SIZE // 16
    _fg_Lp   = (math.ceil(_fg_grid / FINEGRAIN_PATCH_POOL)) ** 2
    assert _fg_Lp == FINEGRAIN_TARGET_LP, (
        f"FILIP Lp={_fg_Lp} (grid {_fg_grid}/pool {FINEGRAIN_PATCH_POOL}) != TARGET {FINEGRAIN_TARGET_LP}. "
        f"Check that IMAGE_SIZE and the pool agree - if the change was intentional, update FINEGRAIN_TARGET_LP.")
LR_FINEGRAIN          = 2e-4
JIGSAW_ENABLED        = False

# inference drops the pooler and falls back to global v, keeping the gallery cache; train only
FLAIR_ENABLED      = False  # FLAIR off; mutually exclusive with FILIP
FLAIR_WEIGHT       = 0.5  # lambda_flair in L = L_global + lambda_flair * L_tcs (target after warmup)
FLAIR_WEIGHT_WARMUP_EPOCHS = 2  # linear warmup of lambda, protecting the randomly initialised pooler; 0 = none
FLAIR_WEIGHT_WARMUP_START  = 0.0
FLAIR_DIM          = 256
FLAIR_NUM_HEADS    = 4  # multi-head (FLAIR_DIM % FLAIR_NUM_HEADS == 0)
FLAIR_INPUT_LN     = True  # input LayerNorm; damps attention peakiness at init
FLAIR_LOGIT_SCALE  = 20.0  # fixed temperature (not a parameter, so no DDP logit sync)
FLAIR_PATCH_POOL   = 4  # v_loc 2D avgpool (grid/pool -> Lp)
FLAIR_TARGET_LP    = 64  # intended Lp after pooling (8x8); used by the fail-fast check
FLAIR_SYMMETRIC    = True  # symmetric text->image + image->text CE (handles K>1 multi-positive)
FLAIR_LOSS_ABORT_THRESH = 15.0  # divergence abort threshold (healthy CE is about log(Ni) = 4.8)
LR_FLAIR           = 2e-4
_FG_OR_FLAIR = FINEGRAIN_ENABLED or FLAIR_ENABLED  # gate for the fine-grained path shared by both modes

assert not (FLAIR_ENABLED and FINEGRAIN_ENABLED), \
    "FLAIR_ENABLED and FINEGRAIN_ENABLED cannot both be on - pick one fine-grained aggregator."
assert (not FLAIR_ENABLED) or (FLAIR_DIM % FLAIR_NUM_HEADS == 0), \
    "FLAIR_DIM must be a multiple of FLAIR_NUM_HEADS."
if FLAIR_ENABLED:
    _flair_grid = IMAGE_SIZE // 16
    _flair_Lp   = (math.ceil(_flair_grid / FLAIR_PATCH_POOL)) ** 2
    assert _flair_Lp == FLAIR_TARGET_LP, (
        f"FLAIR Lp={_flair_Lp} (grid {_flair_grid}/pool {FLAIR_PATCH_POOL}) != TARGET {FLAIR_TARGET_LP}.")
LR_POSITION     = 5e-5  # no effect when POSITION_PRETRAINED_FREEZE=True (the grad hook zeroes it)
# LR constants assume REF_BATCH (=128); build_optimizer applies lr_scale
# the EMA half-life is epoch-based, so no extra correction
LR_BATCH_SCALING      = "sqrt"  # "sqrt" | "linear" | "none"
POSITION_PRETRAINED_FREEZE = True
POSITION_PRETRAINED_SIZE   = 64  # SigLIP2 pretrained max_position_embeddings
# Chen et al. 2023, the standard long-context LLM extension.
# scale = (PRETRAIN_LEN-1)/(TARGET_LEN-1), endpoint-aligned
POSITION_PI_ENABLED      = True
POSITION_PI_PRETRAIN_LEN = 64
POSITION_PI_TARGET_LEN   = 128  # = MAX_TEXT_LENGTH, the usable length after the PI extension
LR_MIN_RATIO    = 0.02
WEIGHT_DECAY    = 0.01
GRAD_CLIP       = 1.0
# grad_norm is 0.3-1.6 when healthy and 12-17 just before collapse
GRAD_SPIKE_SKIP   = False
GRAD_SPIKE_THRESH = 5.0
USE_AMP         = True
AMP_DTYPE       = torch.bfloat16
DROP_LAST       = True
WARMUP_RATIO    = 0.05

EARLY_STOP_ENABLED   = True
EARLY_STOP_PATIENCE  = 3
EARLY_STOP_MIN_DELTA = 0.15
EARLY_STOP_METRIC    = "score"  # no early stopping

LOSS_EXPLOSION_THRESHOLD = 8.0
LOSS_EXPLOSION_EP1_SCALE = 2.5
LOSS_NAN_ABORT           = True

GRADIENT_CHECKPOINTING = True  # at 512px even batch 128 OOMs without GC

LOGIT_SCALE_CLAMP_ENABLED = False
LOGIT_SCALE_MAX           = 50.0  # upper bound on exp(logit_scale) via clamp_(max=log(50))
LOGIT_SCALE_FREEZE        = False

CROSS_GPU_NEG_GATHER = False  # single GPU, so the negatives are the local NxN batch

UNIFORMITY_REG_ENABLED = False
UNIFORMITY_REG_LAMBDA  = 0.01
UNIFORMITY_REG_T       = 2.0  # Wang & Isola t (default 2)

EMA_ENABLED          = True
EMA_HALF_LIFE_EPOCHS = 1.76  # half-life; the actual decay is derived from steps_per_epoch
EMA_START_EPOCH      = 2  # ep1 is paused (init residue), so reset the shadow when ep2 starts
# EMA: the shadow is refreshed every epoch; the ckpt holds the shadow
EMA_DIAGNOSTIC_RAW_EVAL = False  # unused

USE_SOFT_LABEL              = True
SOFT_LABEL_SAME_AS          = 0.55
SOFT_LABEL_SAME_ACTION      = 0.25  # unused
SOFT_LABEL_SAME_SCENE       = 0.0  # unused
SOFT_LABEL_MISSING_FALLBACK = 0.0  # unused
SOFT_LABEL_DIAG_ENABLED     = True

IMG_AUG_COLOR_JITTER  = (0.4, 0.4, 0.4, 0.1)
IMG_AUG_HFLIP_PROB    = 0.0  # blocks the left/right mismatch present in 12% of captions
IMG_AUG_GRAYSCALE_P   = 0.0  # 95% of captions name a colour, so a mismatch would hurt
IMG_AUG_GNOISE_P      = 0.20
IMG_AUG_GNOISE_STD    = (0.005, 0.025)
IMG_AUG_RRC_ENABLED   = True  # off: deterministic Resize+CenterCrop, same as inference
IMG_AUG_RRC_SCALE     = (0.90, 1.0)  # only used when RRC is on
IMG_AUG_RRC_RATIO     = (0.85, 1.18)  # only used when RRC is on
IMG_AUG_ERASE_P       = 0.15
IMG_AUG_ERASE_SCALE   = (0.02, 0.05)

# train anomaly:normal is 1.69:1; the batch is forced to 1:1
BALANCE_LABEL_TYPE  = True
# anomaly action long-tail (Falling 36%) -> inverse-freq^alpha
BALANCE_ACTION      = True
ACTION_SMOOTH_ALPHA = 0.5  # 0 = raw / 0.5 = sqrt / 1.0 = fully balanced
EPOCH_LEN_FACTOR    = 1.0  # samples per epoch = factor * N_train

LOG_EVERY_STEPS  = 50
DIAG_EVERY_STEPS = 200
EVAL_EVERY_EP    = 1
KEEP_LAST_N_CKPT = -1  # -1 = keep every epoch, >0 = rolling last N, 0 = no epoch ckpt

SUBSET_MODE     = False
SUBSET_TRAIN_N  = 50_000

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    """Reproducibility first: deterministic=True, benchmark=False"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_global_dirs():
    """Create only the global directories shared by every run."""
    for d in [MANIFEST_DIR, RUNS_ROOT]:
        os.makedirs(d, exist_ok=True)

def make_run_dir(name: str):
    """Create the per-run directory. Aborts if it already exists."""
    run_dir = f"{RUNS_ROOT}/{name}"
    if os.path.exists(run_dir):
        raise SystemExit(f"[error] the run directory already exists: {run_dir}\n"
                         f"  to train again under the same name, delete this path first")
    os.makedirs(run_dir)
    os.makedirs(f"{run_dir}/checkpoints")
    os.makedirs(f"{run_dir}/tb")
    return run_dir, name

def map_annotation_path_to_local(ann_path: str, root: str = IMG_ROOT) -> str:
    """"train/imgs_8/full/10.jpg" -> ".../train_jpg/Part 2/imgs_8/full/10.jpg" """
    parts = ann_path.split("/")  # ['train','imgs_8','full','10.jpg']
    assert parts[0] == "train" and len(parts) == 4, ann_path
    n = int(parts[1].replace("imgs_", ""))
    part_no = n // 8 + 1
    fname = parts[3]
    return f"{root}/Part {part_no}/{parts[1]}/{parts[2]}/{fname}"

_SCENE_ALIAS = {
    "indoor setting":     "indoor",
    "indoor gym":         "indoor gym",
    "indoor flea market": "indoor flea market",
    "outdoor setting":    "outdoor",
    "grassy area":        "lawn",
    "grass field":        "lawn",
    "grassy field":       "lawn",
    "field of grass":     "lawn",
    "grass":              "lawn",
    "parking":            "parking lot",
    "park lot":           "parking lot",
    "soccer pitch":       "soccer field",
    "baseball stadium":   "baseball field",
    "basketball gym":     "basketball court",
}

def normalize_scene(s: str) -> str:
    """scene string -> normalized cluster key (lowercase, strip, alias mapping)."""
    if not s:
        return ""
    s = s.strip().lower()
    return _SCENE_ALIAS.get(s, s)

def normalize_action(s: str) -> str:
    """action label normalization (lowercase + strip)."""
    if not s:
        return ""
    return s.strip().lower()

class TeeStdout:
    """Copy stdout (and stderr) to a file as well, with a timestamped prefix."""
    def __init__(self, log_path, also_stderr=True, timestamp=True):
        self._file = open(log_path, "a", buffering=1, encoding="utf-8")
        self._stdout = sys.stdout
        self._stderr = sys.stderr if also_stderr else None
        self._also_stderr = also_stderr
        self._timestamp = timestamp
        self._line_start = True

    def write(self, msg):
        if not msg:
            return
        if self._timestamp:
            out_lines = []
            for piece in msg.splitlines(keepends=True):
                if self._line_start and piece.strip():
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    out_lines.append(f"[{ts}] {piece}")
                else:
                    out_lines.append(piece)
                self._line_start = piece.endswith("\n")
            decorated = "".join(out_lines)
        else:
            decorated = msg
        self._stdout.write(msg)
        self._file.write(decorated)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def isatty(self):
        return self._stdout.isatty()

    def install(self):
        sys.stdout = self
        if self._also_stderr:
            self._saved_stderr = sys.stderr
            sys.stderr = self

    def restore(self):
        sys.stdout = self._stdout
        if self._also_stderr and hasattr(self, "_saved_stderr"):
            sys.stderr = self._saved_stderr
        try:
            self._file.flush(); self._file.close()
        except Exception:
            pass

class MetricWriter:
    """JSONL writer, one line per metric event, for later plotting and statistics."""
    def __init__(self, path):
        self._f = open(path, "a", buffering=1, encoding="utf-8")

    def log(self, **fields):
        fields["_ts"] = time.time()
        self._f.write(json.dumps(fields, ensure_ascii=False, default=float) + "\n")

    def close(self):
        try: self._f.flush(); self._f.close()
        except Exception: pass

def collect_env_info():
    """Snapshot of the environment at training time, for reproduction and debugging."""
    info = {
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "hostname":      socket.gethostname(),
        "python":        sys.version.split()[0],
        "platform":      platform.platform(),
        "torch":         torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda":          torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn":         torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_count":     torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["gpus"] = [
            {
                "idx":  i,
                "name": torch.cuda.get_device_name(i),
                "mem_total_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    for pkg in ["transformers", "peft", "torchvision", "PIL", "numpy"]:
        try:
            mod = __import__(pkg)
            info[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            info[pkg] = None
    try:
        import subprocess
        info["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        info["git_hash"] = None
    return info

def snapshot_config():
    """Snapshot the module-level UPPER_CASE config variables into a dict."""
    import types
    g = globals()
    cfg = {}
    for k, v in g.items():
        if not k.isupper() or k.startswith("_"):
            continue
        if isinstance(v, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type)):
            continue
        try:
            json.dumps(v, default=str)
            cfg[k] = v if not isinstance(v, torch.dtype) else str(v)
        except Exception:
            cfg[k] = str(v)
    return cfg

def compute_grad_norm(params, norm_type=2.0):
    """Measure the gradient norm before clipping. Promoting to fp32 keeps bf16 params safe."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    norms = torch.stack([g.detach().float().norm(norm_type) for g in grads])
    return float(norms.norm(norm_type).item())

@torch.no_grad()
def diagnose_multi_probe(model):
    """Measure multi-probe divergence and the aggregator block weight distribution (called at epoch end)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        head = base.vision_model.head
    except AttributeError:
        return {}

    if not hasattr(head, "probe"):
        return {}
    probe_full = head.probe.detach().float()  # (1, K, D)
    if probe_full.shape[1] <= 1:
        return {}
    K = int(probe_full.shape[1])
    E = int(probe_full.shape[2])
    probe_vec = probe_full.squeeze(0)

    probe_norms = probe_vec.norm(dim=-1)
    # eps=1e-6 keeps a near-zero norm from producing NaN (diagnostics only)
    probe_normalized = F.normalize(probe_vec, dim=-1, eps=1e-6)
    sim_mat = probe_normalized @ probe_normalized.t()
    mask = ~torch.eye(K, dtype=torch.bool, device=sim_mat.device)
    off_diag = sim_mat[mask]

    out = {
        "probe_offdiag_sim_mean": off_diag.mean().item(),
        "probe_offdiag_sim_std":  off_diag.std().item() if off_diag.numel() > 1 else 0.0,
        "probe_norm_mean":        probe_norms.mean().item(),
        "probe_norm_std":         probe_norms.std().item() if K > 1 else 0.0,
        "probe_0_norm":           probe_norms[0].item(),
    }
    if K > 1:
        out["probe_rest_norm_mean"] = probe_norms[1:].mean().item()
        out["probe_rest_norm_std"]  = (probe_norms[1:].std().item()
                                       if probe_norms[1:].numel() > 1 else 0.0)

    agg_type = getattr(head, "agg_type", "linear")
    if agg_type == "linear":
        if not hasattr(head, "aggregator") or head.aggregator is None:
            return out
        W = head.aggregator.weight.detach().float()  # (D, K*D)
        block_norms = []
        for k in range(K):
            n = W[:, k*E:(k+1)*E].norm().item()
            out[f"agg_block_{k}_norm"] = n
            block_norms.append(n)
        import statistics
        if len(block_norms) > 1:
            mean_norm = statistics.mean(block_norms)
            std_norm  = statistics.stdev(block_norms)
            out["agg_block_norm_cv"] = std_norm / max(mean_norm, 1e-8)
            rest_mean = statistics.mean(block_norms[1:])
            out["agg_block_0_to_rest_ratio"] = block_norms[0] / max(rest_mean, 1e-8)
    elif agg_type == "attention":
        if hasattr(head, "agg_query"):
            out["agg_query_norm"] = head.agg_query.detach().float().norm().item()
        if hasattr(head, "agg_init_bias"):
            bias = head.agg_init_bias.detach().float()
            for k in range(K):
                out[f"agg_init_bias_{k}"] = bias[k].item()
            sm = bias.softmax(dim=-1)
            for k in range(K):
                out[f"agg_softmax_bias_{k}"] = sm[k].item()
        if hasattr(head, "agg_proj") and head.agg_proj is not None:
            W = head.agg_proj.weight.detach().float()
            I = torch.eye(E, device=W.device, dtype=W.dtype)
            out["agg_proj_dev_from_I"] = (W - I).norm().item()
    return out

@torch.no_grad()
def compute_sim_diagnostics(v, t, image_ids=None):
    """SigLIP positive vs negative similarity statistics."""
    v = F.normalize(v.float(), dim=-1)
    t = F.normalize(t.float(), dim=-1)
    sim = v @ t.t()
    n_img, n_txt = sim.shape

    if image_ids is not None and n_img != n_txt:
        pos_mask = (image_ids.unsqueeze(0) ==
                    torch.arange(n_img, device=sim.device).unsqueeze(1))
        pos = sim[pos_mask]
        neg = sim[~pos_mask]
    else:
        n = sim.size(0)
        eye = torch.eye(n, device=sim.device, dtype=torch.bool)
        pos = sim[eye]
        neg = sim[~eye]
    return {
        "pos_sim_mean":  pos.mean().item(),
        "pos_sim_min":   pos.min().item(),
        "neg_sim_mean":  neg.mean().item(),
        "neg_sim_max":   neg.max().item(),
        "margin":        (pos.mean() - neg.mean()).item(),
        "v_norm_mean":   v.norm(dim=-1).mean().item(),
        "t_norm_mean":   t.norm(dim=-1).mean().item(),
    }

def gpu_mem_stats():
    if not torch.cuda.is_available():
        return {}
    return {
        "gpu_mem_alloc_gb": torch.cuda.memory_allocated() / 1024**3,
        "gpu_mem_peak_gb":  torch.cuda.max_memory_allocated() / 1024**3,
    }








def _diag_image_group_sizes(all_rows: list):
    """Image key consistency check: an exact base/recap image_path match should yield mostly size-2 groups."""
    sizes = Counter(Counter(r["image"] for r in all_rows).values())
    n_imgs = sum(sizes.values())
    n1 = sizes.get(1, 0)
    n2 = sizes.get(2, 0)
    other = {k: v for k, v in sizes.items() if k not in (1, 2)}
    print(f"  image group size distribution: "
          f"size=1: {n1:,} ({100.0*n1/max(1,n_imgs):.1f}%), "
          f"size=2: {n2:,} ({100.0*n2/max(1,n_imgs):.1f}%), "
          f"other: {other}")
    if n_imgs > 0 and n1 > n_imgs * 0.05:
        print(f"  [warn] {n1:,} images ({100.0*n1/n_imgs:.1f}%) only have 1 caption "
              f"- possible base/recap image key mismatch (whitespace or trailing newline). Check the data.")


def load_manifest():
    """Load the per-image manifest (built by train/gen/gen_manifest.py)."""
    rows = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["image_path"] = map_annotation_path_to_local(r["image"])
            r["scene_normalized"]  = normalize_scene(r.get("scene", ""))
            r["action_normalized"] = normalize_action(r.get("action_label", ""))
            rows.append(r)
    n_cap = sum(len(r["captions"]) for r in rows)
    print(f"  [manifest] loaded {len(rows):,} image records ({n_cap:,} captions, "
          f"mean {n_cap/max(1,len(rows)):.1f}/img; image_path -> {IMG_ROOT})")
    return rows

class GaussianNoiseAug:
    """Gaussian noise in the tensor domain (sim2real sensor noise)."""
    def __init__(self, p=0.20, std_range=(0.005, 0.025)):
        self.p = p
        self.std_range = std_range

    def __call__(self, tensor):
        # (C, H, W), [-1, 1] after Normalize
        if random.random() >= self.p:
            return tensor
        std = random.uniform(self.std_range[0], self.std_range[1])
        noise = torch.randn_like(tensor) * std
        return (tensor + noise).clamp(-1.0, 1.0)

def _cv2_load_image(path, target_size: Optional[int] = None) -> torch.Tensor:
    """cv2 JPEG decode plus BGR->RGB. Applies cv2.resize(INTER_AREA) when target_size is given."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR uint8 (H, W, 3)
    if img is None:
        raise IOError(f"cv2.imread failed: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size is not None and (img.shape[0] != target_size or img.shape[1] != target_size):
        img = cv2.resize(img, (target_size, target_size),
                         interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(np.ascontiguousarray(img))  # (H, W, 3) uint8
    return t.permute(2, 0, 1).contiguous()  # (3, H, W) uint8

def build_train_transform(image_size=IMAGE_SIZE):
    """Train augmentation: cv2 decode plus torchvision v2 tensor-domain ops."""
    transforms = []
    if IMG_AUG_RRC_ENABLED:
        transforms.append(Tv2.RandomResizedCrop(
            image_size,
            scale=IMG_AUG_RRC_SCALE,
            ratio=IMG_AUG_RRC_RATIO,
            interpolation=Tv2.InterpolationMode.BICUBIC,
            antialias=True,
        ))
    if IMG_AUG_HFLIP_PROB > 0:
        transforms.append(Tv2.RandomHorizontalFlip(IMG_AUG_HFLIP_PROB))
    transforms.append(Tv2.ColorJitter(*IMG_AUG_COLOR_JITTER))
    if IMG_AUG_GRAYSCALE_P > 0:
        transforms.append(Tv2.RandomGrayscale(p=IMG_AUG_GRAYSCALE_P))
    transforms.append(Tv2.ToDtype(torch.float32, scale=True))  # uint8 -> float [0,1]
    transforms.append(Tv2.Normalize(mean=[0.5]*3, std=[0.5]*3))  # -> [-1, 1]
    if IMG_AUG_GNOISE_P > 0:
        transforms.append(GaussianNoiseAug(
            p=IMG_AUG_GNOISE_P, std_range=IMG_AUG_GNOISE_STD,
        ))
    if IMG_AUG_ERASE_P > 0:
        transforms.append(Tv2.RandomErasing(
            p=IMG_AUG_ERASE_P,
            scale=IMG_AUG_ERASE_SCALE,
            ratio=(0.5, 2.0),
        ))
    return Tv2.Compose(transforms)

class PABDataset(Dataset):
    """PAB row -> (pixel_values, caption_text, image_id)."""

    def __init__(self, rows, transform, is_training=False,
                 unique_image_mode: bool = False,
                 n_positives: int = 1):
        self.transform = transform
        self.is_training = is_training
        self.unique_image_mode = unique_image_mode
        self.n_positives = max(1, int(n_positives))

        self._image_level = bool(rows) and isinstance(rows[0].get("captions"), list)
        if unique_image_mode and self._image_level:
            self.rows = rows  # one record = one image (with a captions list)
            self.unique_images = [r["image"] for r in rows]
            self.unique_rows = rows
            self.image_groups = None
        elif unique_image_mode:
            self.image_groups = defaultdict(list)
            for r in rows:
                self.image_groups[r["image"]].append(r)
            self.unique_images = sorted(self.image_groups.keys())
            self.unique_rows = [self.image_groups[img][0] for img in self.unique_images]
            self.rows = self.unique_rows
        else:
            self.rows = rows
            self.image_groups = None
            self.unique_images = None
            self.unique_rows = None

        # with persistent_workers every worker accumulates, so the print is rate-limited
        self._img_read_fail_count = 0
        self._img_read_fail_print_at = 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        r_rep = rec
        K = self.n_positives
        if isinstance(rec.get("captions"), list):
            pool = rec["captions"]
        elif self.unique_image_mode:
            pool = [r["caption"] for r in self.image_groups[rec["image"]]]
        else:
            pool = [rec["caption"]]
        if len(pool) >= K:
            sel_caps = random.sample(pool, K)
        else:
            sel_caps = [random.choice(pool) for _ in range(K)]

        # with RRC on, RandomResizedCrop handles it; load at native resolution
        _target = None if IMG_AUG_RRC_ENABLED else IMAGE_SIZE
        try:
            img_t = _cv2_load_image(r_rep["image_path"], target_size=_target)
        except Exception as e:
            self._img_read_fail_count += 1
            if self._img_read_fail_count >= self._img_read_fail_print_at:
                print(f"  [warn] image read fail #{self._img_read_fail_count} "
                      f"(worker_local count): {r_rep['image_path']} ({e})")
                self._img_read_fail_print_at *= 10
            _sz = IMAGE_SIZE if _target is None else _target
            img_t = torch.zeros(3, _sz, _sz, dtype=torch.uint8)
        pixel_values = self.transform(img_t)

        texts: list = list(sel_caps)

        return {
            "pixel_values":      pixel_values,
            "texts":             texts,
            "text":              texts[0],
            "image":             r_rep["image"],
            "label_type":        r_rep["label_type"],
            "action_label":      r_rep["action_label"],
            "scene_normalized":  r_rep.get("scene_normalized") or normalize_scene(r_rep.get("scene", "")),
            "action_normalized": r_rep.get("action_normalized") or normalize_action(r_rep.get("action_label", "")),
        }

# HF model card: trained on lowercased text, and GemmaTokenizerFast does not lowercase
def _siglip2_normalize_text(text: Optional[str]) -> str:
    """.strip() + .casefold() (Unicode-safe lowercase). Safe on None and the empty string."""
    if text is None:
        return ""
    return text.strip().casefold()

def _assert_lowercase(texts, where: str = "?"):
    """Lowercase consistency assertion right before the tokenizer call (stripped under python -O)."""
    if not __debug__:
        return
    for t in texts:
        if t is None:
            continue
        if t != t.casefold():
            sample = t[:80]
            raise RuntimeError(
                f"[lowercase-defense] uppercase leaked into tokenizer at {where!r}: "
                f"{sample!r} (casefold mismatch). _siglip2_normalize_text may be missing."
            )

class _LowercaseTokenizerWrapper:
    """SigLIP2 lowercase wrapper: normalize automatically before every __call__."""
    def __init__(self, tokenizer):
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(self, "_lowercase_wrapped", True)

    def __call__(self, text=None, text_pair=None, *args, **kwargs):
        if text is not None:
            if isinstance(text, str):
                text = _siglip2_normalize_text(text)
            elif isinstance(text, (list, tuple)):
                text = [_siglip2_normalize_text(t) if isinstance(t, str) else t for t in text]
        if text_pair is not None:
            if isinstance(text_pair, str):
                text_pair = _siglip2_normalize_text(text_pair)
            elif isinstance(text_pair, (list, tuple)):
                text_pair = [_siglip2_normalize_text(t) if isinstance(t, str) else t
                             for t in text_pair]
        return self._tokenizer(text=text, text_pair=text_pair, *args, **kwargs)

    def __getattr__(self, name):
        # unpickling bypasses __init__ and leaves __dict__ empty; recursion guard
        if name.startswith("_") and name in ("_tokenizer", "_lowercase_wrapped"):
            raise AttributeError(name)
        tok = self.__dict__.get("_tokenizer")
        if tok is None:
            raise AttributeError(name)
        return getattr(tok, name)

    def __setattr__(self, name, value):
        if name in ("_tokenizer", "_lowercase_wrapped"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._tokenizer, name, value)

    def __repr__(self):
        return f"_LowercaseTokenizerWrapper({self._tokenizer!r})"

    def __getstate__(self):
        return {"_tokenizer": self._tokenizer, "_lowercase_wrapped": True}

    def __setstate__(self, state):
        object.__setattr__(self, "_tokenizer", state["_tokenizer"])
        object.__setattr__(self, "_lowercase_wrapped", state.get("_lowercase_wrapped", True))

def _wrap_tokenizer_with_lowercase(tokenizer):
    """Wrap the tokenizer in _LowercaseTokenizerWrapper (idempotent)."""
    if isinstance(tokenizer, _LowercaseTokenizerWrapper):
        return tokenizer
    if getattr(tokenizer, "_lowercase_wrapped", False):
        return tokenizer
    return _LowercaseTokenizerWrapper(tokenizer)

_LOWERCASE_VERIFY_FLAGS = {"train_collate": False, "eval_query": False}

def _first_batch_lowercase_verify(texts, where: str):
    """Print a sample on the first call. Silent for worker_id != 0 (avoids duplicates when NUM_WORKERS>0)."""
    if _LOWERCASE_VERIFY_FLAGS.get(where, True):
        return
    try:
        from torch.utils.data import get_worker_info
        winfo = get_worker_info()
        if winfo is not None and winfo.id != 0:
            _LOWERCASE_VERIFY_FLAGS[where] = True
            return
    except Exception:
        pass
    sample = [t for t in texts[:3] if t is not None]
    all_ok = all(t == t.casefold() for t in sample)
    status = "✓ all lowercase" if all_ok else "⚠ UPPERCASE DETECTED"
    print(f"  [lowercase-verify @ {where}] {status}, sample={sample}")
    _LOWERCASE_VERIFY_FLAGS[where] = True

def _assign_cluster_id(value: str, mapping: dict) -> int:
    """Assign batch-local cluster ids. An empty string maps to -1."""
    if not value:
        return -1
    if value not in mapping:
        mapping[value] = len(mapping)
    return mapping[value]

class _TrainCollator:
    """Module-level picklable collate callable."""

    def __init__(self, tokenizer, max_length=MAX_TEXT_LENGTH, n_positives=1):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.K = max(1, int(n_positives))

    def __call__(self, batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        N = len(batch)
        K = self.K

        texts: list = []
        for b in batch:
            ts = b.get("texts")
            if ts is None:
                ts = [b["text"]] * K
            if len(ts) != K:
                if len(ts) < K:
                    ts = ts + [ts[0]] * (K - len(ts))
                else:
                    ts = ts[:K]
            texts.extend(ts)
        assert len(texts) == N * K, (
            f"collate text count mismatch: got {len(texts)} expected {N*K} (N={N}, K={K})"
        )

        image_ids = torch.arange(N, dtype=torch.long).repeat_interleave(K)

        texts = [_siglip2_normalize_text(t) for t in texts]
        _assert_lowercase(texts, where="train_collate")
        _first_batch_lowercase_verify(texts, where="train_collate")
        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        scene_to_id, action_to_id = {}, {}
        scene_ids_img_list, action_ids_img_list = [], []
        for b in batch:
            scene_ids_img_list.append(
                _assign_cluster_id(b.get("scene_normalized")  or "", scene_to_id))
            action_ids_img_list.append(
                _assign_cluster_id(b.get("action_normalized") or "", action_to_id))

        text_scene_ids  = [scene_ids_img_list[i]  for i in range(N) for _ in range(K)]
        text_action_ids = [action_ids_img_list[i] for i in range(N) for _ in range(K)]

        return {
            "pixel_values":   pixel_values,
            "input_ids":      tok["input_ids"],
            "attention_mask": tok.get("attention_mask", torch.ones_like(tok["input_ids"])),
            "image_ids":      image_ids,
            "images":         [b["image"] for b in batch],
            "label_types":    [b["label_type"] for b in batch],
            "actions":        [b["action_label"] for b in batch],
            "scene_ids_img":   torch.tensor(scene_ids_img_list,  dtype=torch.long),
            "action_ids_img":  torch.tensor(action_ids_img_list, dtype=torch.long),
            "scene_ids_txt":   torch.tensor(text_scene_ids,  dtype=torch.long),
            "action_ids_txt":  torch.tensor(text_action_ids, dtype=torch.long),
            "n_positives_per_image": K,
        }

def make_collate(tokenizer, max_length=MAX_TEXT_LENGTH, n_positives=1):
    """Picklable collate factory; returns a _TrainCollator instance."""
    return _TrainCollator(tokenizer, max_length=max_length, n_positives=n_positives)

def compute_sample_weights(rows,
                           balance_label=BALANCE_LABEL_TYPE,
                           balance_action=BALANCE_ACTION,
                           action_alpha=ACTION_SMOOTH_ALPHA):
    """Compute the sampling weight of each sample (an orthogonal two-layer policy)."""
    label_count  = Counter(r["label_type"]   for r in rows)
    action_count = Counter(r["action_label"] for r in rows)

    weights = np.ones(len(rows), dtype=np.float64)
    if balance_action and action_alpha > 0:
        for i, r in enumerate(rows):
            weights[i] = 1.0 / max(1.0, action_count[r["action_label"]]) ** action_alpha

    if balance_label:
        per_label_sum = defaultdict(float)
        for i, r in enumerate(rows):
            per_label_sum[r["label_type"]] += weights[i]
        K = len(per_label_sum)
        for i, r in enumerate(rows):
            weights[i] *= (1.0 / K) / max(1e-12, per_label_sum[r["label_type"]])

    weights = weights / weights.sum() * len(rows)
    return weights, label_count, action_count

def build_train_sampler(rows, generator):
    """WeightedRandomSampler (replacement=True). Returns None when every BALANCE_* is False."""
    from torch.utils.data import WeightedRandomSampler
    if not (BALANCE_LABEL_TYPE or BALANCE_ACTION):
        return None, None, None
    weights, label_count, action_count = compute_sample_weights(rows)
    num_samples = max(1, int(round(len(rows) * EPOCH_LEN_FACTOR)))
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=num_samples,
        replacement=True,
        generator=generator,
    )
    return sampler, label_count, action_count

def worker_init_fn(worker_id):
    """Deterministic worker seeding (reproducible and spread across workers)."""
    base_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)

def _verify_unique_image_invariants(train_ds):
    """Verify the sampler invariants of USE_UNIQUE_IMAGE_MODE."""
    if getattr(train_ds, "image_groups", None) is None:
        n_unique = len(train_ds.unique_images)
        n_cap = sum(len(r["captions"]) for r in train_ds.rows)
        print(f"  [manifest] unique_image_mode: {n_unique:,} images, {n_cap:,} captions "
              f"(mean {n_cap/max(1,n_unique):.2f}/img) - invariant guaranteed at build time")
        return
    n_unique = len(train_ds.unique_images)
    n_total_rows = sum(len(g) for g in train_ds.image_groups.values())
    print(f"  unique_image_mode=True: dataset size = {n_unique:,} (unique images), "
          f"row pool = {n_total_rows:,} (mean {n_total_rows/n_unique:.2f} captions per image)")
    n_label_mismatch  = sum(1 for g in train_ds.image_groups.values()
                            if len({r["label_type"] for r in g}) > 1)
    n_action_mismatch = sum(1 for g in train_ds.image_groups.values()
                            if len({r["action_label"] for r in g}) > 1)
    if n_label_mismatch or n_action_mismatch:
        raise RuntimeError(
            f"[unique_image_mode] sampler invariant violated: "
            f"{n_label_mismatch} images have inconsistent label_type, "
            f"{n_action_mismatch} have inconsistent action_label across captions. "
            f"The recap CSV and the base manifest may disagree - check the data."
        )
    print(f"  unique_image_mode invariant ✓ label_type / action_label image-consistent "
          f"({n_unique:,} images verified)")

def build_train_loader(train_rows, tokenizer, train_tf):
    """The training DataLoader. With USE_UNIQUE_IMAGE_MODE=True the sampler weights are per image (representative row)."""
    train_ds = PABDataset(train_rows, train_tf, is_training=True,
                          unique_image_mode=USE_UNIQUE_IMAGE_MODE,
                          n_positives=N_POSITIVES_PER_IMAGE)
    collate = make_collate(tokenizer, MAX_TEXT_LENGTH,
                            n_positives=N_POSITIVES_PER_IMAGE)
    print(f"  positives per image: K={N_POSITIVES_PER_IMAGE} "
          f"({'multi-positive' if N_POSITIVES_PER_IMAGE > 1 else 'single positive'})")
    if USE_UNIQUE_IMAGE_MODE:
        _verify_unique_image_invariants(train_ds)

    sampler_gen = torch.Generator()
    sampler_gen.manual_seed(SEED)

    sampler_rows = train_ds.unique_rows if USE_UNIQUE_IMAGE_MODE else train_rows
    sampler, label_count, action_count = build_train_sampler(sampler_rows, sampler_gen)
    use_shuffle = sampler is None
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=use_shuffle,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        collate_fn=collate,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        drop_last=DROP_LAST,
        worker_init_fn=worker_init_fn if NUM_WORKERS > 0 else None,
        generator=sampler_gen if sampler is None else None,
    )
    return loader, label_count, action_count

# p' = p * (P-1)/(T-1) in [0, P-1]
# endpoint-aligned (Chen et al. 2023 PI variant)

def _compute_pi_scale(pretrain_len: int, target_len: int) -> float:
    """Endpoint-aligned PI scale: pos p in [0, T-1] -> p' = p * scale in [0, P-1]."""
    if target_len <= 1:
        return 0.0
    return (pretrain_len - 1) / (target_len - 1)

def _pi_text_embeddings_forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
    """PI version of Siglip2TextEmbeddings.forward, installed as a bound method."""
    seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

    if inputs_embeds is None:
        inputs_embeds = self.token_embedding(input_ids)

    pe_weight = self.position_embedding.weight  # (P, D), P=pi_pretrain_len
    P = int(pe_weight.shape[0])
    D = int(pe_weight.shape[1])
    device = inputs_embeds.device

    if position_ids is not None:
        pos = position_ids.to(device=device, dtype=torch.float32)
        if pos.dim() == 2:
            pos = pos[0]  # (L,); assumed identical across the batch (Siglip2 default)
    else:
        scale = float(self._pi_scale)
        pos = torch.arange(seq_length, device=device, dtype=torch.float32) * scale

    # PI lookup: clamp to [0,P-1], weighted blend of the floor/ceil rows
    pos = pos.clamp(min=0.0, max=float(P - 1))
    p_lo = pos.floor().long()
    p_hi = (p_lo + 1).clamp(max=P - 1)
    w_hi = (pos - p_lo.to(pos.dtype)).unsqueeze(-1)  # (L, 1)
    w_lo = 1.0 - w_hi

    e_lo = pe_weight[p_lo]  # (L, D)
    e_hi = pe_weight[p_hi]  # (L, D)
    # under autocast pe_weight is bf16 and w is fp32, so cast explicitly
    w_lo_c = w_lo.to(e_lo.dtype)
    w_hi_c = w_hi.to(e_hi.dtype)
    position_embeddings = w_lo_c * e_lo + w_hi_c * e_hi  # (L, D)

    # (1, L, D) -> (B, L, D) broadcast
    embeddings = inputs_embeds + position_embeddings.unsqueeze(0)
    return embeddings

def _install_position_freeze_hook(text, n_freeze: int, dim: int, total_rows: int):
    """Register a grad hook on rows [0..n_freeze-1] (zero gradient + AdamW with WD=0 = fully frozen)."""
    weight = text.embeddings.position_embedding.weight

    def _freeze_pretrained_rows_hook(grad, n=n_freeze):
        if grad is None:
            return grad
        masked = grad.clone()
        masked[:n].zero_()
        return masked

    weight.register_hook(_freeze_pretrained_rows_hook)
    text.embeddings.register_buffer(
        "_pretrained_pos_snapshot",
        weight[:n_freeze].detach().clone().cpu(),
        persistent=False,
    )
    text.embeddings._pretrained_pos_n_freeze = int(n_freeze)

    n_train_rows = max(0, total_rows - n_freeze)
    print(f"  [pos-pi] pretrained rows [0..{n_freeze-1}] FROZEN; "
          f"trainable rows = {n_train_rows} (usually 0 under PI)")
    if n_train_rows > 0:
        print(f"  [pos-pi] effective trainable position params: "
              f"{n_train_rows * dim:,} (LR_POSITION={LR_POSITION:.0e})")

def apply_position_interpolation(base_model,
                                  pretrain_len: int = POSITION_PI_PRETRAIN_LEN,
                                  target_len: int = POSITION_PI_TARGET_LEN):
    """Apply Position Interpolation to the SigLIP2 text encoder."""
    text = base_model.text_model
    emb = text.embeddings
    pe = emb.position_embedding.weight.data  # (P, D)
    n_old, dim = pe.shape

    if n_old != pretrain_len:
        raise RuntimeError(
            f"[pos-pi] position_embedding rows = {n_old} but POSITION_PI_PRETRAIN_LEN={pretrain_len}. "
            f"PI cannot be applied to a checkpoint that was already extended (e.g. 128 rows) - "
            f"it must go on the pretrained base (rows=64)."
        )

    scale = _compute_pi_scale(pretrain_len, target_len)
    print(f"  [pos-pi] Position Interpolation: rows={n_old} (unchanged) | "
          f"target_len={target_len} | scale={scale:.6f} "
          f"(token pos p → p×{scale:.4f} ∈ [0, {pretrain_len-1}])")

    # idempotency: skip if already applied
    if getattr(emb, "_pi_applied", False):
        print(f"  [pos-pi] skip: PI already applied (safe re-entry)")
        return base_model

    import types as _types
    emb.forward = _types.MethodType(_pi_text_embeddings_forward, emb)
    emb._pi_scale = float(scale)
    emb._pi_pretrain_len = int(pretrain_len)
    emb._pi_target_len = int(target_len)
    emb._pi_applied = True

    base_model.config.text_config.max_position_embeddings = target_len

    norm_mean = pe.norm(dim=-1).mean().item()
    print(f"  [pos-pi] PI applied. pe rows={n_old}, dim={dim}, mean_row_norm={norm_mean:.3f}")
    print(f"  [pos-pi] config.text_config.max_position_embeddings = {target_len}")

    if POSITION_PRETRAINED_FREEZE and POSITION_PRETRAINED_SIZE > 0:
        n_freeze = min(POSITION_PRETRAINED_SIZE, n_old)
        _install_position_freeze_hook(text, n_freeze, dim, total_rows=n_old)
    else:
        print("  [pos-pi] POSITION_PRETRAINED_FREEZE=False -> 64 rows train alongside DoRA")

    return base_model

# compatibility alias; callers use extend_position_embedding
def extend_position_embedding(base_model, new_size: int = POSITION_EMBED_NEW_SIZE):
    """Compat shim; handled by Position Interpolation."""
    return apply_position_interpolation(base_model,
                                         pretrain_len=POSITION_PI_PRETRAIN_LEN,
                                         target_len=new_size)

def verify_pretrained_position_unchanged(model, where: str = "?", atol: float = 1e-6):
    """Verify that the pretrained position rows [0..N-1] still match the snapshot."""
    if not POSITION_PRETRAINED_FREEZE:
        return
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        emb = base.text_model.embeddings
        snapshot = getattr(emb, "_pretrained_pos_snapshot", None)
        n_freeze = getattr(emb, "_pretrained_pos_n_freeze", None)
        if snapshot is None or n_freeze is None:
            raise HaltPositionDrift(
                f"[pos-freeze-verify @ {where}] snapshot/n_freeze buffers missing - "
                f"the freeze path in apply_position_interpolation never ran, or the hook is detached."
            )
        current = base.text_model.embeddings.position_embedding.weight[:n_freeze].detach().cpu()
        # the snapshot is a register_buffer and follows .to(cuda), so align on cpu explicitly
        diff = (current - snapshot.cpu()).abs().max().item()
        if diff > atol:
            raise HaltPositionDrift(
                f"[pos-freeze-verify @ {where}] change detected in the pretrained position rows [0..{n_freeze-1}]! "
                f"max|delta|={diff:.3e} > atol={atol:.0e}. The hook may be detached, or something wrote to it externally."
            )
    except AttributeError as e:
        raise HaltPositionDrift(f"[pos-freeze-verify @ {where}] could not reach the model structure: {e}")

# nn.MultiheadAttention has a combined in_proj_weight, which PEFT cannot wrap
# replaced by numerically equivalent split Q/K/V/O Linear layers
# named q/k/v/out_proj so DORA_TARGETS matches automatically
# inference must apply the same surgery *before* PeftModel.from_pretrained()
# pooler_mha_split=True in meta.json signals that requirement
class SplitMHAModule(nn.Module):
    """A split-projection version compatible with the nn.MultiheadAttention forward signature."""
    def __init__(self, embed_dim, num_heads, bias=True, dropout=0.0,
                 batch_first=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None):
        super().__init__()
        assert (kdim is None or kdim == embed_dim), "split MHA: a separate kdim is not supported"
        assert (vdim is None or vdim == embed_dim), "split MHA: a separate vdim is not supported"
        assert not add_bias_kv, "split MHA: add_bias_kv is not supported"
        assert not add_zero_attn, "split MHA: add_zero_attn is not supported"
        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.head_dim    = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be a multiple of num_heads"
        self.dropout     = dropout
        self.batch_first = batch_first
        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, query, key, value,
                key_padding_mask=None, need_weights=False, attn_mask=None,
                average_attn_weights=True, is_causal=False):
        if not self.batch_first:
            query = query.transpose(0, 1); key = key.transpose(0, 1); value = value.transpose(0, 1)

        B, Lq, _ = query.shape
        Lk = key.shape[1]
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(query).view(B, Lq, H, Dh).transpose(1, 2)
        k = self.k_proj(key  ).view(B, Lk, H, Dh).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, H, Dh).transpose(1, 2)

        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Lq, self.embed_dim)
        out = self.out_proj(attn_out)

        if not self.batch_first:
            out = out.transpose(0, 1)
        # returns (out, attn_weights) to match the MHA signature; weights unused
        return out, None

def _replace_pooler_mha_with_split(base_model):
    """Replace every nn.MultiheadAttention in vision_model with a SplitMHAModule."""
    targets = []
    for parent_name, parent in base_model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.MultiheadAttention):
                targets.append((parent_name, parent, child_name, child))
    if not targets:
        print("  [pooler-mha-split] no nn.MultiheadAttention found in model — skip")
        return base_model

    print(f"  [pooler-mha-split] found {len(targets)} nn.MultiheadAttention module(s)")
    for parent_name, parent, child_name, old_mha in targets:
        E   = old_mha.embed_dim
        H   = old_mha.num_heads
        bias_flag = (old_mha.in_proj_bias is not None)
        full_name = f"{parent_name}.{child_name}" if parent_name else child_name
        print(f"  [pooler-mha-split] replacing {full_name}: embed_dim={E}, num_heads={H}, bias={bias_flag}")

        new_mha = SplitMHAModule(
            embed_dim=E, num_heads=H, bias=bias_flag,
            dropout=old_mha.dropout,
            batch_first=old_mha.batch_first,
        )
        old_dev   = old_mha.in_proj_weight.device
        old_dtype = old_mha.in_proj_weight.dtype
        new_mha = new_mha.to(device=old_dev, dtype=old_dtype)

        with torch.no_grad():
            W = old_mha.in_proj_weight.data  # (3E, E)
            new_mha.q_proj.weight.data.copy_(W[0:E    ])
            new_mha.k_proj.weight.data.copy_(W[E:2*E  ])
            new_mha.v_proj.weight.data.copy_(W[2*E:3*E])
            if bias_flag:
                b = old_mha.in_proj_bias.data  # (3E,)
                new_mha.q_proj.bias.data.copy_(b[0:E    ])
                new_mha.k_proj.bias.data.copy_(b[E:2*E  ])
                new_mha.v_proj.bias.data.copy_(b[2*E:3*E])
            new_mha.out_proj.weight.data.copy_(old_mha.out_proj.weight.data)
            if old_mha.out_proj.bias is not None and new_mha.out_proj.bias is not None:
                new_mha.out_proj.bias.data.copy_(old_mha.out_proj.bias.data)

        prev_train_old = old_mha.training
        prev_train_new = new_mha.training
        old_mha.eval(); new_mha.eval()
        fork_devices = [old_dev] if old_dev.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
            torch.manual_seed(0)
            if old_dev.type == "cuda":
                torch.cuda.manual_seed(0)
            if old_mha.batch_first:
                q_in = torch.randn(2, 4, E, device=old_dev, dtype=old_dtype)
                k_in = torch.randn(2, 6, E, device=old_dev, dtype=old_dtype)
            else:
                q_in = torch.randn(4, 2, E, device=old_dev, dtype=old_dtype)
                k_in = torch.randn(6, 2, E, device=old_dev, dtype=old_dtype)
            v_in = k_in.clone()
            old_out, _ = old_mha(q_in, k_in, v_in, need_weights=False)
            new_out, _ = new_mha(q_in, k_in, v_in, need_weights=False)
            max_diff = (old_out - new_out).abs().max().item()
            assert max_diff < 1e-4, (
                f"SplitMHA numerical equivalence failed at {full_name}: "
                f"max_diff={max_diff:.2e} (>1e-4). Possible weight mapping error."
            )
            print(f"  [pooler-mha-split] {full_name}: numerical eq OK fp32 (max_diff={max_diff:.2e})")

        # called before model.to(DEVICE); moves to GPU temporarily only on CUDA
        if torch.cuda.is_available() and old_dev.type == "cuda":
            bf_dev = old_dev
            old_for_bf, new_for_bf = old_mha, new_mha
            q_bf, k_bf, v_bf = q_in, k_in, v_in
        elif torch.cuda.is_available():
            bf_dev = torch.device("cuda")
            try:
                old_for_bf = old_mha.to(bf_dev)
                new_for_bf = new_mha.to(bf_dev)
                q_bf = q_in.to(bf_dev); k_bf = k_in.to(bf_dev); v_bf = v_in.to(bf_dev)
            except Exception as _mv_e:
                print(f"  [pooler-mha-split] {full_name}: bf16 check skipped (cuda move failed: {_mv_e})")
                try:
                    old_mha.to(old_dev); new_mha.to(old_dev)
                except Exception:
                    pass
                bf_dev = None
        else:
            bf_dev = None
            print(f"  [pooler-mha-split] {full_name}: bf16 check skipped (no CUDA available)")

        if bf_dev is not None:
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    old_out_bf, _ = old_for_bf(q_bf, k_bf, v_bf, need_weights=False)
                    new_out_bf, _ = new_for_bf(q_bf, k_bf, v_bf, need_weights=False)
                bf_diff = (old_out_bf.float() - new_out_bf.float()).abs().max().item()
                print(f"  [pooler-mha-split] {full_name}: bf16 numerical gap (informational): "
                      f"{bf_diff:.2e} ({'⚠ large' if bf_diff > 5e-2 else 'within bf16 precision'})")
            except Exception as _bf_e:
                print(f"  [pooler-mha-split] {full_name}: bf16 check skipped ({_bf_e})")
            finally:
                if old_dev.type == "cpu":
                    try:
                        old_mha.to(old_dev); new_mha.to(old_dev)
                    except Exception:
                        pass
        old_mha.train(prev_train_old)
        new_mha.train(prev_train_new)

        setattr(parent, child_name, new_mha)

    return base_model

class MultiProbePoolingHead(nn.Module):
    """K-probe + aggregator wrapper (unified design)."""
    def __init__(self, orig_head, num_probes=4, perturb_std=0.01,
                 aggregator_init="identity_warmup",
                 perturb_mode="relative",
                 block_scale=0.1,
                 agg_type="linear",
                 attn_temp=1.0):
        super().__init__()
        E = orig_head.probe.shape[-1]
        self.K = num_probes
        self.embed_dim = E
        self._aggregator_init = aggregator_init
        self._block_scale = block_scale
        self.agg_type = agg_type
        self.attn_temp = float(attn_temp)

        probes = orig_head.probe.data.repeat(1, num_probes, 1)  # (1, K, D)
        probe_scale = orig_head.probe.data.std().item()
        if num_probes > 1:
            if perturb_mode == "relative":
                effective_std = perturb_std * max(probe_scale, 1e-8)
            elif perturb_mode == "absolute":
                effective_std = perturb_std
            else:
                raise ValueError(f"unknown perturb_mode: {perturb_mode}")
            with torch.no_grad():
                probes[:, 1:].add_(effective_std * torch.randn_like(probes[:, 1:]))
            self._effective_perturb_std = effective_std
        else:
            self._effective_perturb_std = 0.0
        self.probe = nn.Parameter(probes)
        self._probe_init_scale = probe_scale

        self.attention = orig_head.attention
        self.layernorm = orig_head.layernorm
        self.mlp       = orig_head.mlp
        self.num_heads = orig_head.num_heads if hasattr(orig_head, "num_heads") \
                         else orig_head.attention.num_heads

        if agg_type == "linear":
            # norm_factor = 1 + block_scale * (K-1) restores the init magnitude
            self.aggregator = nn.Linear(E * num_probes, E, bias=False)

            with torch.no_grad():
                self.aggregator.weight.data.zero_()
                if aggregator_init == "identity_warmup":
                    norm_factor = 1.0 + block_scale * (num_probes - 1) if num_probes > 1 else 1.0
                    self.aggregator.weight.data[:, 0:E].copy_(torch.eye(E) / norm_factor)
                    for k in range(1, num_probes):
                        self.aggregator.weight.data[:, k*E:(k+1)*E].copy_(
                            (block_scale / norm_factor) * torch.eye(E))
                    self._init_norm_factor = float(norm_factor)
                elif aggregator_init == "uniform":
                    if abs(block_scale - 0.1) > 1e-8:
                        print(f"  [multi-probe] warn: aggregator_init='uniform' with "
                              f"block_scale={block_scale} ignored.")
                    for k in range(num_probes):
                        self.aggregator.weight.data[:, k*E:(k+1)*E].copy_(torch.eye(E) / num_probes)
                    self._init_norm_factor = float(num_probes)
                elif aggregator_init == "identity":
                    self.aggregator.weight.data[:, 0:E].copy_(torch.eye(E))
                    self._init_norm_factor = 1.0
                else:
                    raise ValueError(f"unknown aggregator_init: {aggregator_init}")
        elif agg_type == "attention":
            self.aggregator = None  # linear aggregator disabled; kept as None for state_dict compatibility
            self.agg_query = nn.Parameter(torch.empty(E))
            with torch.no_grad():
                self.agg_query.copy_(orig_head.probe.data.squeeze(0).squeeze(0).detach())
            self.agg_init_bias = nn.Parameter(torch.zeros(num_probes))
            with torch.no_grad():
                self.agg_init_bias[0] = 1.0
            self.agg_proj = nn.Linear(E, E, bias=False)
            with torch.no_grad():
                self.agg_proj.weight.copy_(torch.eye(E))
            self._init_norm_factor = 1.0
        else:
            raise ValueError(f"unknown agg_type: {agg_type}")

    def forward(self, hidden_state, attention_mask=None):
        """SigLIP2 head.forward compatible. Input (B, L, D) -> output (B, D)."""
        B = hidden_state.shape[0]
        probe = self.probe.expand(B, -1, -1)  # (B, K, D)

        # the pooler is currently only ever called with mask=None
        if attention_mask is not None:
            try:
                from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
                target_len = probe.shape[1]
                attention_mask = _prepare_4d_attention_mask(
                    attention_mask, hidden_state.dtype, target_len)
                attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1).contiguous()
            except Exception:
                attention_mask = None

        attn_out, _ = self.attention(probe, hidden_state, hidden_state,
                                      need_weights=False, attn_mask=attention_mask)
        residual = attn_out
        h = self.layernorm(attn_out)
        h = residual + self.mlp(h)  # (B, K, D)

        if self.agg_type == "linear":
            h_flat = h.reshape(B, self.K * self.embed_dim)
            pooled = self.aggregator(h_flat)
        elif self.agg_type == "attention":
            logits = (h * self.agg_query).sum(dim=-1) / self.attn_temp
            logits = logits + self.agg_init_bias
            attn_weights = logits.softmax(dim=-1)  # (B, K)
            weighted = (attn_weights.unsqueeze(-1) * h).sum(dim=1)  # (B, D)
            pooled = self.agg_proj(weighted)
        else:
            raise RuntimeError(f"unknown agg_type at forward: {self.agg_type}")
        return pooled

def _replace_head_with_multi_probe(base_model, num_probes=None,
                                    perturb_std=None,
                                    aggregator_init=None,
                                    perturb_mode=None,
                                    block_scale=None,
                                    agg_type=None,
                                    attn_temp=None):
    """Wrap vision_model.head in a MultiProbePoolingHead."""
    if num_probes is None:       num_probes = NUM_PROBES
    if perturb_std is None:      perturb_std = MULTI_PROBE_PERTURB_STD
    if aggregator_init is None:  aggregator_init = MULTI_PROBE_AGG_INIT
    if perturb_mode is None:     perturb_mode = MULTI_PROBE_PERTURB_MODE
    if block_scale is None:      block_scale = MULTI_PROBE_BLOCK_SCALE
    if agg_type is None:         agg_type = MULTI_PROBE_AGG_TYPE
    if attn_temp is None:        attn_temp = MULTI_PROBE_ATTN_TEMP

    if num_probes <= 1:
        print(f"  [multi-probe] NUM_PROBES={num_probes} → single-probe, skip")
        return base_model

    vision = base_model.vision_model
    if not hasattr(vision, "head"):
        print("  [multi-probe] no vision_model.head — skip")
        return base_model

    orig_head = vision.head
    # idempotency guard: a second call must not expand K again
    if isinstance(orig_head, MultiProbePoolingHead):
        if orig_head.K == num_probes:
            print(f"  [multi-probe] head is already MultiProbePoolingHead with K={orig_head.K} — skip")
            return base_model
        raise RuntimeError(
            f"[multi-probe] head is already MultiProbePoolingHead but K mismatch: "
            f"existing K={orig_head.K}, requested K={num_probes}. Reload the base model and retry."
        )
    if not hasattr(orig_head, "probe"):
        print("  [multi-probe] vision_model.head has no 'probe' attribute — skip")
        return base_model

    E = orig_head.probe.shape[-1]
    old_dev   = orig_head.probe.device
    old_dtype = orig_head.probe.dtype
    print(f"  [multi-probe] wrapping head with K={num_probes} probes "
          f"(perturb_std={perturb_std}, perturb_mode={perturb_mode}, "
          f"agg_init={aggregator_init}, block_scale={block_scale}, dim={E}, "
          f"agg_type={agg_type}, attn_temp={attn_temp})")

    new_head = MultiProbePoolingHead(orig_head, num_probes=num_probes,
                                      perturb_std=perturb_std,
                                      aggregator_init=aggregator_init,
                                      perturb_mode=perturb_mode,
                                      block_scale=block_scale,
                                      agg_type=agg_type,
                                      attn_temp=attn_temp)
    new_head = new_head.to(device=old_dev, dtype=old_dtype)
    print(f"  [multi-probe] probe_init_scale={new_head._probe_init_scale:.4e}, "
          f"effective_perturb_std={new_head._effective_perturb_std:.4e}")

    prev_train_orig = orig_head.training
    prev_train_new  = new_head.training
    orig_head.eval(); new_head.eval()
    orig_out = new_out = None
    fork_devices = [old_dev] if old_dev.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=fork_devices), torch.no_grad():
            torch.manual_seed(0)
            if old_dev.type == "cuda":
                torch.cuda.manual_seed(0)
            # patches = (IMAGE_SIZE // patch_size)^2
            try:
                _patch_size = int(base_model.config.vision_config.patch_size)
            except AttributeError:
                _patch_size = 16  # fallback (SigLIP2 default)
            _n_patches = (IMAGE_SIZE // _patch_size) ** 2
            dummy = torch.randn(2, _n_patches, E, device=old_dev, dtype=old_dtype)
            orig_out = orig_head(dummy)
            new_out  = new_head(dummy)
    except Exception as _fwd_e:
        print(f"  [multi-probe] ⚠ numerical eq check skipped (forward failed: {_fwd_e})")
    finally:
        orig_head.train(prev_train_orig); new_head.train(prev_train_new)

    if orig_out is not None and new_out is not None:
        max_diff = (orig_out - new_out).abs().max().item()
        orig_scale = orig_out.abs().max().item()
        relative_diff = max_diff / max(orig_scale, 1e-8)

        if agg_type == "attention":
            print(f"  [multi-probe] @ init (attention agg): max_diff={max_diff:.2e}, "
                  f"rel_diff={relative_diff:.2%}  "
                  f"(init bias [+1,0,...] -> softmax weights; not exactly equal to the original)")
        elif aggregator_init == "identity":
            assert max_diff < 1e-4, (
                f"MultiProbeHead init eq (identity) failed: max_diff={max_diff:.2e}."
            )
            print(f"  [multi-probe] numerical eq @ init (strict): max_diff={max_diff:.2e} ✓")
        elif aggregator_init == "identity_warmup":
            if block_scale == 0:
                assert max_diff < 1e-4, (
                    f"MultiProbeHead init eq (warmup, scale=0) failed: max_diff={max_diff:.2e}."
                )
                print(f"  [multi-probe] numerical eq @ init (warmup, scale=0): "
                      f"max_diff={max_diff:.2e} ✓")
            else:
                upper_bound = block_scale * num_probes * orig_scale * 5.0
                if max_diff >= upper_bound:
                    print(f"  [multi-probe] ⚠ approx eq @ init (warmup, scale={block_scale}) "
                          f"diff larger than expected: max_diff={max_diff:.2e} > "
                          f"upper_bound={upper_bound:.2e} (training continues; check the design intent)")
                else:
                    print(f"  [multi-probe] approx eq @ init (warmup, scale={block_scale}): "
                          f"max_diff={max_diff:.2e}, rel_diff={relative_diff:.2%}")
        elif aggregator_init == "uniform":
            print(f"  [multi-probe] @ init (uniform): max_diff={max_diff:.2e}, "
                  f"rel_diff={relative_diff:.2%} (no equivalence by design)")

    vision.head = new_head
    return base_model

def freeze_vision_encoder(base_model):
    """Permanently freeze the vision encoder base params."""
    n_frozen = 0
    for _, p in base_model.vision_model.named_parameters():
        p.requires_grad_(False)
        n_frozen += p.numel()
    print(f"  [vision-frozen] {n_frozen:,} params frozen")

def _is_vision_lora_param(name: str) -> bool:
    """Identify the LoRA wrapper parameters PEFT attached to vision_model."""
    nl = name.lower()
    return ("vision_model" in nl) and (
        ".lora_a." in nl or ".lora_b." in nl or "lora_magnitude_vector" in nl
    )

def _is_pooler_lora_param(name: str) -> bool:
    """Identify the LoRA wrappers on vision_model.head (pooler MHA + MLP)."""
    nl = name.lower()
    return ("vision_model.head" in nl) and (
        ".lora_a." in nl or ".lora_b." in nl or "lora_magnitude_vector" in nl
    )

def _is_multi_probe_param(name: str) -> bool:
    """Multi-probe params (unified): probe + aggregator."""
    if "lora" in name.lower():
        return False
    return (
        name.endswith("vision_model.head.probe") or
        name.endswith("vision_model.head.aggregator.weight") or
        name.endswith("vision_model.head.agg_query") or
        name.endswith("vision_model.head.agg_init_bias") or
        name.endswith("vision_model.head.agg_proj.weight")
    )

def _is_multi_probe_probe_param(name: str) -> bool:
    """The probe alone (the LR_MULTI_PROBE_PROBE group)."""
    if "lora" in name.lower():
        return False
    return name.endswith("vision_model.head.probe")

def freeze_vision_dora(peft_model):
    """Block the vision DoRA adapter from training (PEFT also wraps a frozen base)."""
    n_frozen = 0
    n_kept = 0
    for name, p in peft_model.named_parameters():
        if _is_vision_lora_param(name):
            p.requires_grad_(False)
            n_frozen += p.numel()
        elif p.requires_grad:
            n_kept += p.numel()
    print(f"  [vision-dora-frozen] {n_frozen:,}  |  still trainable: {n_kept:,}")
    return n_frozen

def _build_text_dora_patterns(num_text_layers: int, rank: int, alpha: int) -> tuple:
    """Build the per-layer LoRA rank/alpha override pattern dict for the text encoder."""
    rank_pattern, alpha_pattern = {}, {}
    suffixes = [f"self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "out_proj")] + \
               [f"mlp.{p}"       for p in ("fc1", "fc2")]
    for li in range(num_text_layers):
        for suf in suffixes:
            key = f"text_model.encoder.layers.{li}.{suf}"
            rank_pattern[key]  = rank
            alpha_pattern[key] = alpha
    return rank_pattern, alpha_pattern

def _verify_dora_ranks(peft_model, *, expected_text: int, expected_vision: int, mode: str):
    """Check that the actual lora_A rank of text/vision layer 0 q_proj matches the intent."""
    r_text, r_vis = None, None
    for name, module in peft_model.named_modules():
        if not name.endswith(".lora_A.default"):
            continue
        base_name = name[:-len(".lora_A.default")]
        if r_text is None and "text_model.encoder.layers.0.self_attn.q_proj" in base_name:
            r_text = module.weight.shape[0]
        if r_vis is None and "vision_model.encoder.layers.0.self_attn.q_proj" in base_name:
            r_vis = module.weight.shape[0]
        if r_text is not None and r_vis is not None:
            break
    print(f"  [rank-verify] text   q_proj (layer 0) lora_A rank: {r_text}  (expected {expected_text})")
    print(f"  [rank-verify] vision q_proj (layer 0) lora_A rank: {r_vis}  (expected {expected_vision})")
    assert r_text == expected_text, \
        f"text rank mismatch: actual={r_text}, expected={expected_text}. rank_pattern was not applied."
    assert r_vis == expected_vision, \
        f"vision rank mismatch: actual={r_vis}, expected={expected_vision}."
    print(f"  [rank-verify] {mode} rank ok (text r={expected_text}, vision r={expected_vision})")

def _enable_gradient_checkpointing(peft_model):
    """Gradient checkpointing on a PeftModel: unwrap base_model first, then enable (use_reentrant=False)."""
    if not GRADIENT_CHECKPOINTING:
        return
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    gc_ok = False
    try:
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        if hasattr(base, "gradient_checkpointing_enable"):
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            gc_ok = True
            print(f"  [gradient-checkpointing] ENABLED on {type(base).__name__} "
                  f"(use_reentrant=False)")
        else:
            print(f"  [gradient-checkpointing] {type(base).__name__} has no "
                  f"gradient_checkpointing_enable - skipped (OOM risk)")
    except Exception as e:
        print(f"  [gradient-checkpointing] failed to enable ({e}) - OOM risk")
    if not gc_ok:
        print("  [gradient-checkpointing] large model with GC off - "
              "lower BATCH_SIZE or check GRADIENT_CHECKPOINTING=False")

def _unfreeze_pooler_dora(peft_model):
    """Set the pooler DoRA wrapper trainable/frozen, with a matching sanity guard."""
    n_total = 0
    for name, p in peft_model.named_parameters():
        if _is_pooler_lora_param(name):
            n_total += p.numel()
            p.requires_grad_(not FREEZE_POOLER_DORA)
    state = "FROZEN" if FREEZE_POOLER_DORA else "TRAINABLE"
    print(f"\n[step 4.5] Pooler DoRA wrapper {state} ({n_total:,} params)")
    if n_total == 0 and not FREEZE_POOLER_DORA:
        head_modules = sorted({
            n.split(".")[2] for n, _ in peft_model.named_parameters()
            if n.startswith("base_model.model.vision_model.") and "lora" in n.lower()
        })
        raise RuntimeError(
            f"[step 4.5] FREEZE_POOLER_DORA=False but the 'vision_model.head' substring matched nothing. "
            f"LoRA wrap locations under vision_model: {head_modules}. "
            f"Adjust the _is_pooler_lora_param substring to match the model structure."
        )

def _unfreeze_multi_probe_params(peft_model) -> dict:
    """Explicitly unfreeze the multi-probe unified params (probe + aggregator.weight)."""
    if not (TRAIN_MULTI_PROBE_PARAMS and NUM_PROBES > 1):
        print(f"\n[step 4.7] Multi-probe params skipped "
              f"(TRAIN_MULTI_PROBE_PARAMS={TRAIN_MULTI_PROBE_PARAMS}, NUM_PROBES={NUM_PROBES})")
        return {"n_probe": 0, "n_agg": 0, "n_total": 0}

    counts = {"n_probe": 0, "n_agg": 0}
    for name, p in peft_model.named_parameters():
        if _is_multi_probe_param(name):
            p.requires_grad_(True)
            key = "n_probe" if name.endswith("vision_model.head.probe") else "n_agg"
            counts[key] += p.numel()
    n_total = sum(counts.values())
    counts["n_total"] = n_total

    if counts["n_probe"] == 0 or counts["n_agg"] == 0:
        head_params = sorted({
            n.replace("base_model.model.", "")
            for n, _ in peft_model.named_parameters()
            if "vision_model.head" in n
        })
        raise RuntimeError(
            f"[step 4.7] NUM_PROBES={NUM_PROBES} but a component matched zero unified params: "
            f"probe={counts['n_probe']:,}, agg={counts['n_agg']:,}. "
            f"vision_model.head params: {head_params[:8]}..."
        )

    try:
        base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
        head = base.vision_model.head
        probe = head.probe
        if probe.dim() != 3 or probe.shape[1] != NUM_PROBES:
            raise RuntimeError(
                f"[step 4.7] probe.shape={tuple(probe.shape)} != (1, {NUM_PROBES}, D)"
            )
    except AttributeError as ae:
        raise RuntimeError(f"[step 4.7] could not reach head.probe ({ae}).")

    print(f"\n[step 4.7] Multi-probe TRAINABLE (unified): "
          f"probe={counts['n_probe']:,} + agg={counts['n_agg']:,} = {n_total:,} params, "
          f"shape: probe={tuple(probe.shape)}")
    return counts

def _setup_text_position_emb(peft_model):
    """Text position_embedding trainable, vision pos_emb frozen."""
    n_text, n_vision = 0, 0
    for name, p in peft_model.named_parameters():
        if "position_embedding" not in name:
            continue
        if "vision_model" in name:
            p.requires_grad_(False)
            n_vision += p.numel()
        else:
            p.requires_grad_(True)
            n_text += p.numel()
    print(f"  position_embedding trainable (text):   {n_text:,} params")
    print(f"  position_embedding frozen    (vision): {n_vision:,} params")

def _unfreeze_logit_params(peft_model):
    """Make logit_scale and logit_bias trainable (their own LR group)."""
    for name, p in peft_model.named_parameters():
        if "logit_scale" in name:
            p.requires_grad_(not LOGIT_SCALE_FREEZE)
        elif "logit_bias" in name:
            p.requires_grad_(True)
    if LOGIT_SCALE_FREEZE:
        print("  [logit] LOGIT_SCALE_FREEZE=True -> logit_scale frozen at init, only logit_bias trains")

def _count_trainable_breakdown(peft_model) -> dict:
    """Return a dict of trainable parameter counts per LR group."""
    c = {"pos": 0, "dora_text": 0, "dora_vision_enc": 0, "dora_vision_head": 0,
         "multi_probe": 0, "logit": 0}
    for name, p in peft_model.named_parameters():
        if not p.requires_grad:
            continue
        if "position_embedding" in name:
            c["pos"] += p.numel()
        elif "logit_scale" in name or "logit_bias" in name:
            c["logit"] += p.numel()
        elif _is_vision_lora_param(name):
            key = "dora_vision_head" if "vision_model.head" in name else "dora_vision_enc"
            c[key] += p.numel()
        elif _is_multi_probe_param(name):
            c["multi_probe"] += p.numel()
        else:
            c["dora_text"] += p.numel()
    c["dora_vision"] = c["dora_vision_enc"] + c["dora_vision_head"]
    c["n_trainable"] = c["pos"] + c["dora_text"] + c["dora_vision"] + c["multi_probe"] + c["logit"]
    c["n_total"]     = sum(p.numel() for p in peft_model.parameters())
    return c

def _print_trainable_breakdown(c: dict):
    """Print the _count_trainable_breakdown result per LR group in readable form."""
    print(f"\n  Trainable summary (vision DoRA {'OFF' if FREEZE_VISION_DORA else 'ON'}, "
          f"pooler DoRA {'OFF' if FREEZE_POOLER_DORA else 'ON'}, "
          f"multi-probe K={NUM_PROBES}):")
    if POSITION_PI_ENABLED:
        if POSITION_PRETRAINED_FREEZE:
            print(f"    position_emb (text, PI)          : {c['pos']:>10,}  (LR {LR_POSITION:.0e}) "
                  f"PI rows=[0..{POSITION_PI_PRETRAIN_LEN-1}] ALL FROZEN, "
                  f"target_len={POSITION_PI_TARGET_LEN} (DoRA adapts the row meaning)")
        else:
            print(f"    position_emb (text, PI)          : {c['pos']:>10,}  (LR {LR_POSITION:.0e}) "
                  f"PI rows={POSITION_PI_PRETRAIN_LEN} (TRAINABLE), target_len={POSITION_PI_TARGET_LEN}")
    elif POSITION_PRETRAINED_FREEZE and POSITION_PRETRAINED_SIZE > 0:
        pos_dim   = 768
        eff_rows  = max(0, POSITION_EMBED_NEW_SIZE - POSITION_PRETRAINED_SIZE)
        eff_par   = eff_rows * pos_dim
        print(f"    position_emb (text)              : {c['pos']:>10,}  (LR {LR_POSITION:.0e}) "
              f"[0..{POSITION_PRETRAINED_SIZE-1}] FROZEN, "
              f"[{POSITION_PRETRAINED_SIZE}..{POSITION_EMBED_NEW_SIZE-1}] active "
              f"(effective {eff_par:,} params)")
    else:
        print(f"    position_emb (text)              : {c['pos']:>10,}  (LR {LR_POSITION:.0e})")
    print(f"    DoRA         (text)              : {c['dora_text']:>10,}  (LR {LR:.0e})")
    print(f"    DoRA         (vision enc)        : {c['dora_vision_enc']:>10,}  (LR {LR:.0e})")
    print(f"    DoRA         (vision head/pooler): {c['dora_vision_head']:>10,}  (LR {LR:.0e})  "
          f"{'(frozen)' if FREEZE_POOLER_DORA else '(TRAINABLE)'}")
    print(f"    multi-probe (probe + agg)        : {c['multi_probe']:>10,}  "
          f"(LR {LR_MULTI_PROBE:.0e}, WD {WD_MULTI_PROBE})")
    print(f"    logit                            : {c['logit']:>10,}  (LR {LR_LOGIT:.0e})")
    print("    ─────────────────────────────────────")
    print(f"    Total                            : {c['n_trainable']:>10,} / {c['n_total']:,}  "
          f"({100*c['n_trainable']/c['n_total']:.2f}%)")

def _verify_model_invariants(peft_model, tokenizer):
    """Verify the invariants that must hold when build_model() returns."""
    print(f"\n  [verify] tokenizer class: {type(tokenizer).__name__}")
    locations = {"encoder": 0, "pooler/head": 0, "other": 0}
    for name, p in peft_model.named_parameters():
        if not p.requires_grad or "lora" not in name.lower():
            continue
        if ".encoder.layers." in name:
            locations["encoder"] += 1
        elif "pooler" in name.lower() or ".head." in name:
            locations["pooler/head"] += 1
        else:
            locations["other"] += 1
    print(f"  [verify] DoRA wrappers — encoder: {locations['encoder']}, "
          f"pooler/head: {locations['pooler/head']}, other: {locations['other']}")
    if locations["pooler/head"] == 0:
        print("  [verify] pooler/head DoRA wrapper = 0 (expected > 0).")

    non_lora_vision = [
        (name, p.numel()) for name, p in peft_model.named_parameters()
        if p.requires_grad and "vision" in name.lower() and "lora" not in name.lower()
        and not _is_multi_probe_param(name)
    ]
    if non_lora_vision:
        print("  [verify] non-LoRA trainable found in the vision base (violates the frozen policy):")
        for name, n in non_lora_vision[:5]:
            print(f"           {name}: {n:,}")
    else:
        print("  [verify] vision base fully frozen (nothing trainable outside LoRA + multi-probe)")

    if NUM_PROBES > 1:
        base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
        head = base.vision_model.head
        if not isinstance(head, MultiProbePoolingHead):
            raise RuntimeError(
                f"[verify] NUM_PROBES={NUM_PROBES} but the head is not a MultiProbePoolingHead "
                f"(type: {type(head).__name__}). The step 1.6 surgery failed silently."
            )
        print(f"  [verify] ✓ vision_model.head = MultiProbePoolingHead (K={head.K})")
        diag = diagnose_multi_probe(peft_model)
        if diag:
            diag_str = ", ".join(f"{k}={v:.4f}" for k, v in diag.items())
            print(f"  [verify] init multi-probe diag (0 training steps): {diag_str}")
            init_sim = diag.get("probe_offdiag_sim_mean")
            if init_sim is not None and init_sim < 0.99:
                print(f"  [verify] ⚠ init offdiag_sim={init_sim:.4f} < 0.99 — "
                      f"the relative perturbation may not be working.")

def build_model():
    """SigLIP2 + DoRA + Multi-Probe Pooling Head build."""
    from transformers import AutoModel, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"[model] loading {MODEL_NAME} ...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n[step 1] text position embedding - applying Position Interpolation ...")
    apply_position_interpolation(base_model,
                                  pretrain_len=POSITION_PI_PRETRAIN_LEN,
                                  target_len=POSITION_PI_TARGET_LEN)

    # must run before the vision freeze and get_peft_model
    print("\n[step 1.5] Vision pooler MHA → split Q/K/V/O Linear ...")
    _replace_pooler_mha_with_split(base_model)

    if NUM_PROBES > 1:
        print(f"\n[step 1.6] Vision head → MultiProbePoolingHead "
              f"(K={NUM_PROBES}, perturb={MULTI_PROBE_PERTURB_MODE}, "
              f"agg={MULTI_PROBE_AGG_INIT}, block_scale={MULTI_PROBE_BLOCK_SCALE}, "
              f"agg_type={MULTI_PROBE_AGG_TYPE}, attn_temp={MULTI_PROBE_ATTN_TEMP}) ...")
    else:
        print(f"\n[step 1.6] vision head stays single-probe (NUM_PROBES={NUM_PROBES}) ...")
    _replace_head_with_multi_probe(base_model,
                                    num_probes=NUM_PROBES,
                                    perturb_std=MULTI_PROBE_PERTURB_STD,
                                    aggregator_init=MULTI_PROBE_AGG_INIT,
                                    perturb_mode=MULTI_PROBE_PERTURB_MODE,
                                    block_scale=MULTI_PROBE_BLOCK_SCALE,
                                    agg_type=MULTI_PROBE_AGG_TYPE,
                                    attn_temp=MULTI_PROBE_ATTN_TEMP)

    print("\n[step 2] Vision encoder base frozen ...")
    freeze_vision_encoder(base_model)

    rank_mode = "symmetric" if DORA_RANK_TEXT == DORA_RANK else "asymmetric"
    print(f"\n[step 3] attaching DoRA adapters ({rank_mode}: vision r={DORA_RANK} / text r={DORA_RANK_TEXT}) ...")
    num_text_layers = len(base_model.text_model.encoder.layers)
    print(f"  num_text_layers={num_text_layers}, "
          f"text override modules = {num_text_layers} × 6 = {num_text_layers*6}")
    text_rank_pattern, text_alpha_pattern = _build_text_dora_patterns(
        num_text_layers, DORA_RANK_TEXT, DORA_ALPHA_TEXT,
    )
    dora_cfg = LoraConfig(
        r=DORA_RANK, lora_alpha=DORA_ALPHA, lora_dropout=DORA_DROPOUT,
        bias="none", use_dora=True, target_modules=DORA_TARGETS, modules_to_save=None,
        rank_pattern=text_rank_pattern, alpha_pattern=text_alpha_pattern,
    )
    peft_model = get_peft_model(base_model, dora_cfg)
    try:
        _verify_dora_ranks(peft_model, expected_text=DORA_RANK_TEXT,
                           expected_vision=DORA_RANK, mode=rank_mode)
    except AssertionError:
        raise
    except Exception as e:
        print(f"  [rank-verify] skipped (non-fatal): {type(e).__name__}: {e}")

    _enable_gradient_checkpointing(peft_model)

    if FREEZE_VISION_DORA:
        print("\n[step 4] Vision DoRA wrapper FROZEN (text-only mode) ...")
        freeze_vision_dora(peft_model)
    else:
        print("\n[step 4] Vision DoRA wrapper TRAINABLE ...")
        n_vision_dora = 0
        for name, p in peft_model.named_parameters():
            if _is_vision_lora_param(name):
                p.requires_grad_(True)
                n_vision_dora += p.numel()
        print(f"  [vision-dora-trainable] {n_vision_dora:,} params")

    _unfreeze_pooler_dora(peft_model)
    _unfreeze_multi_probe_params(peft_model)

    print("\n[step 5] Position embedding trainable (text only) ...")
    _setup_text_position_emb(peft_model)
    _unfreeze_logit_params(peft_model)

    _print_trainable_breakdown(_count_trainable_breakdown(peft_model))
    _verify_model_invariants(peft_model, tokenizer)

    tokenizer = _wrap_tokenizer_with_lowercase(tokenizer)
    print("  [verify] ✓ tokenizer wrapped with auto-lowercase")
    return peft_model, tokenizer

def get_logit_params(model):
    """Handles on logit_scale and logit_bias (their own LR group)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    out = []
    if hasattr(base, "logit_scale") and isinstance(base.logit_scale, nn.Parameter):
        out.append(base.logit_scale)
    if hasattr(base, "logit_bias") and isinstance(base.logit_bias, nn.Parameter):
        out.append(base.logit_bias)
    return out

def _to_tensor(x):
    """Extract the tensor when get_*_features returns a BaseModelOutputWithPooling."""
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        if hasattr(x, attr):
            v = getattr(x, attr)
            if isinstance(v, torch.Tensor):
                return v
    raise RuntimeError(f"unexpected feature output type: {type(x).__name__}")

def sigmoid_pair_loss(v, t, image_ids,
                       action_ids_txt=None, action_ids_img=None,
                       scene_ids_txt=None, scene_ids_img=None,
                       logit_scale=None, logit_bias=None,
                       p_same_as: float = 0.0,
                       p_same_action: float = 0.0,
                       p_same_scene: float = 0.0,
                       return_diag: bool = False,
                       row_normalizer: int = None):
    """Unified Sigmoid pair loss (multi-positive + optional soft label)."""
    v = F.normalize(v, dim=-1)
    t = F.normalize(t, dim=-1)
    scale = logit_scale.exp() if logit_scale is not None else torch.tensor(10.0, device=v.device)
    bias  = logit_bias if logit_bias is not None else 0.0

    logits = scale * (t @ v.t()) + bias  # (N_txt, N_img)
    N_txt = logits.shape[0]

    p = torch.zeros_like(logits)
    use_soft = (p_same_as > 0 or p_same_action > 0 or p_same_scene > 0) and \
               action_ids_txt is not None and action_ids_img is not None and \
               scene_ids_txt is not None and scene_ids_img is not None
    if use_soft:
        at = action_ids_txt.view(-1, 1)
        ai = action_ids_img.view(1, -1)
        st = scene_ids_txt.view(-1, 1)
        si = scene_ids_img.view(1, -1)
        valid_a = (at >= 0) & (ai >= 0)
        valid_s = (st >= 0) & (si >= 0)
        same_action = (at == ai) & valid_a
        same_scene  = (st == si) & valid_s
        if p_same_scene > 0:
            p = torch.where(same_scene, torch.full_like(p, p_same_scene), p)
        if p_same_action > 0:
            p = torch.where(same_action, torch.full_like(p, p_same_action), p)
        if p_same_as > 0:
            p = torch.where(same_action & same_scene, torch.full_like(p, p_same_as), p)
    row_idx = torch.arange(N_txt, device=v.device)
    p[row_idx, image_ids] = 1.0

    logits_fp = logits.float()
    log_p_pos = F.logsigmoid(logits_fp)
    log_p_neg = F.logsigmoid(-logits_fp)
    pair_loss = -(p * log_p_pos + (1.0 - p) * log_p_neg)

    denom = row_normalizer if row_normalizer is not None else N_txt
    loss = pair_loss.sum() / denom

    if not return_diag:
        return loss

    with torch.no_grad():
        n_pos    = (p == 1.0).sum().item()
        n_soft   = ((p > 0.0) & (p < 1.0)).sum().item()
        n_neg    = (p == 0.0).sum().item()
        n_total  = p.numel()
        avg_p    = p.mean().item()
        avg_soft_p = p[(p > 0.0) & (p < 1.0)].mean().item() if n_soft > 0 else 0.0
        diag = {
            "n_pos":   n_pos,
            "n_soft":  n_soft,
            "n_neg":   n_neg,
            "ratio_soft": n_soft / max(1, n_total),
            "avg_p":      avg_p,
            "avg_soft_p": avg_soft_p,
            "loss_pos":  -(log_p_pos[p == 1.0]).mean().item() if n_pos > 0 else 0.0,
            "loss_soft": pair_loss[(p > 0) & (p < 1)].mean().item() if n_soft > 0 else 0.0,
            "loss_neg":  -(log_p_neg[p == 0.0]).mean().item() if n_neg > 0 else 0.0,
        }
    return loss, diag

def probe_diversity_loss(model) -> torch.Tensor:
    """Probe off-diagonal cosine similarity penalty."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    try:
        head = base.vision_model.head
    except AttributeError:
        return torch.zeros((), device=next(model.parameters()).device)
    if not hasattr(head, "probe"):
        return torch.zeros((), device=next(model.parameters()).device)
    probe = head.probe.squeeze(0)  # (K, D)
    K = probe.shape[0]
    if K <= 1:
        return torch.zeros((), device=probe.device, dtype=probe.dtype)
    p_n = F.normalize(probe.float(), dim=-1, eps=1e-6)
    sim = p_n @ p_n.t()  # (K, K)
    off = sim - torch.eye(K, device=sim.device, dtype=sim.dtype)
    return (off ** 2).sum() / (K * (K - 1))

def _init_soft_diag_accumulator() -> dict:
    return {"n_pos": 0, "n_soft": 0, "n_neg": 0, "avg_p_sum": 0.0,
            "ratio_soft_sum": 0.0, "loss_pos_sum": 0.0,
            "loss_soft_sum": 0.0, "loss_neg_sum": 0.0, "n_diag": 0}

def _accumulate_soft_diag(accum: dict, diag: dict):
    """Accumulate the per-batch sigmoid_pair_loss diagnostic into the epoch accumulator."""
    accum["n_pos"]          += diag["n_pos"]
    accum["n_soft"]         += diag["n_soft"]
    accum["n_neg"]          += diag["n_neg"]
    accum["avg_p_sum"]      += diag["avg_p"]
    accum["ratio_soft_sum"] += diag["ratio_soft"]
    accum["loss_pos_sum"]   += diag["loss_pos"]
    accum["loss_soft_sum"]  += diag["loss_soft"]
    accum["loss_neg_sum"]   += diag["loss_neg"]
    accum["n_diag"]         += 1

def uniformity_loss(x, t: float = 2.0) -> torch.Tensor:
    """Wang & Isola (2020) uniformity: log E_{i!=j}[ exp(-t * ||x_i - x_j||^2) ]."""
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device)
    x = F.normalize(x.float(), dim=-1)
    sq_pdist = torch.pdist(x, p=2).pow(2)  # (N*(N-1)/2,)
    return sq_pdist.mul(-t).exp().mean().log()

def compute_step_loss(v, t, image_ids,
                      action_ids_txt, action_ids_img,
                      scene_ids_txt, scene_ids_img,
                      *, logit_scale, logit_bias,
                      want_soft_diag: bool,
                      row_normalizer: int = None):
    """Call the unified sigmoid_pair_loss. With USE_SOFT_LABEL=False every p_* is 0 (vanilla multi-positive)."""
    p_kwargs = {
        "p_same_as":     SOFT_LABEL_SAME_AS     if USE_SOFT_LABEL else 0.0,
        "p_same_action": SOFT_LABEL_SAME_ACTION if USE_SOFT_LABEL else 0.0,
        "p_same_scene":  SOFT_LABEL_SAME_SCENE  if USE_SOFT_LABEL else 0.0,
    }
    # with USE_SOFT_LABEL=False the p matrix is 0 and the diagnostic is meaningless
    want_soft_diag = want_soft_diag and USE_SOFT_LABEL
    out = sigmoid_pair_loss(
        v, t, image_ids,
        action_ids_txt, action_ids_img,
        scene_ids_txt, scene_ids_img,
        logit_scale=logit_scale, logit_bias=logit_bias,
        return_diag=want_soft_diag,
        row_normalizer=row_normalizer,
        **p_kwargs,
    )
    if want_soft_diag:
        loss, diag = out
        return loss, diag
    return out, None

class TrainHalt(Exception):
    """Base class for a training abort. The single except in main catches every subtype."""
    reason: str = "halt"

class HaltLossNaN(TrainHalt):
    reason = "loss_nan_inf"

class HaltLossExplosion(TrainHalt):
    reason = "loss_explosion"

class HaltPositionDrift(TrainHalt):
    reason = "position_drift"

class HaltEarlyStop(TrainHalt):
    """Graceful: training has plateaued. Closer to a normal exit than a failure."""
    reason = "early_stop"

class HaltFlairLossExplosion(TrainHalt):
    """Abort training if loss_tcs diverges through FLAIR pooler self-reinforcement."""
    reason = "flair_loss_explosion"

def _check_loss_safety(loss_val: float, global_step: int, threshold_scale: float = 1.0, nan_val=None):
    """Automatic abort guard for NaN/Inf and explosion (detects position embedding collapse)."""
    eff_threshold = LOSS_EXPLOSION_THRESHOLD * max(1.0, threshold_scale)
    # explosion is judged on loss_main, NaN/Inf on combined
    _nanv = loss_val if nan_val is None else nan_val
    if LOSS_NAN_ABORT and (math.isnan(_nanv) or math.isinf(_nanv)):
        print(f"\n  🚨 [ABORT] step {global_step}: loss = {_nanv} (NaN/Inf)")
        raise HaltLossNaN(f"Loss NaN/Inf at step {global_step}")
    if loss_val > eff_threshold:
        print(f"\n  🚨 [ABORT] step {global_step}: loss = {loss_val:.3f} > {eff_threshold:.3f}")
        raise HaltLossExplosion(f"Loss explosion at step {global_step}")

class ParamEMA:
    """Trainable parameter exponential moving average."""
    def __init__(self, parameters, *, decay: float):
        assert 0.0 < decay < 1.0, f"EMA decay must be in (0,1), got {decay}"
        self.decay = decay
        self.shadow = {id(p): p.detach().clone() for p in parameters if p.requires_grad}
        self.params = [p for p in parameters if p.requires_grad]
        self._backup = None
        self._frozen_rows = {}

    @torch.no_grad()
    def register_frozen_rows(self, param, snapshot, n_freeze):
        """Protect rows [0..n_freeze-1] of a param from EMA drift."""
        pid = id(param)
        if pid not in self.shadow:
            print("  [ParamEMA] ⚠ register_frozen_rows: param not tracked (skipped)")
            return
        snap = snapshot.detach().to(device=param.device, dtype=param.dtype).clone()
        self._frozen_rows[pid] = (snap, int(n_freeze))
        self.shadow[pid][:n_freeze].copy_(snap[:n_freeze])

    @torch.no_grad()
    def _sync_frozen_in_shadow(self):
        for pid, (snap, n) in self._frozen_rows.items():
            if pid in self.shadow:
                self.shadow[pid][:n].copy_(snap[:n])

    @torch.no_grad()
    def _sync_frozen_in_params(self):
        for p in self.params:
            pid = id(p)
            if pid in self._frozen_rows:
                snap, n = self._frozen_rows[pid]
                p.data[:n].copy_(snap[:n])

    @torch.no_grad()
    def update(self):
        d = self.decay
        for p in self.params:
            self.shadow[id(p)].mul_(d).add_(p.detach(), alpha=1 - d)
        self._sync_frozen_in_shadow()

    @torch.no_grad()
    def reset_to_current(self):
        """Reset the shadow to the current weights, clearing the init residue."""
        for p in self.params:
            self.shadow[id(p)].copy_(p.detach())
        self._sync_frozen_in_shadow()

    @torch.no_grad()
    def apply_shadow(self):
        self._backup = {id(p): p.detach().clone() for p in self.params}
        for p in self.params:
            p.data.copy_(self.shadow[id(p)])
        self._sync_frozen_in_params()

    @torch.no_grad()
    def restore(self):
        if self._backup is None:
            return
        for p in self.params:
            p.data.copy_(self._backup[id(p)])
        self._backup = None
        self._sync_frozen_in_params()

def get_position_emb_params(model):
    """Handle on the position embedding (its own LR group)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    out = []
    for n, p in base.named_parameters():
        if "position_embedding" in n and p.requires_grad:
            out.append(p)
    return out

def _partition_trainable_params(model) -> dict:
    """Sort the trainable params into per-LR-group lists (no duplicates, requires_grad=True only)."""
    groups = {
        "position_emb":      get_position_emb_params(model),
        "logit":             get_logit_params(model),
        "multi_probe":       [],
        "multi_probe_probe": [],
        "multi_probe_agg":   [],
        "finegrain":         [],  # FineGrainedHead projection (new trainable)
        "flair":             [],  # TextConditionedPooler (new trainable)
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "flair_pooler" in name:
            groups["flair"].append(p)
            continue
        if "finegrain_head" in name:
            groups["finegrain"].append(p)
            continue
        if _is_multi_probe_param(name):
            if LR_MULTI_PROBE_PROBE_SPLIT:
                if _is_multi_probe_probe_param(name):
                    groups["multi_probe_probe"].append(p)
                else:
                    groups["multi_probe_agg"].append(p)
            else:
                groups["multi_probe"].append(p)

    excluded_ids = {id(p) for params in groups.values() for p in params}
    groups["dora"] = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in excluded_ids
    ]
    return groups

def build_optimizer(model, lr_dora: float = None, lr_scale: float = 1.0):
    """5 LR group AdamW optimizer."""
    params_by_group = _partition_trainable_params(model)
    lr_dora_effective = lr_dora if lr_dora is not None else LR

    group_specs = [
        ("position_emb",       LR_POSITION,             0.0),
        ("dora",               lr_dora_effective,       WEIGHT_DECAY),
        ("multi_probe_probe",  LR_MULTI_PROBE_PROBE,    WD_MULTI_PROBE),
        ("multi_probe_agg",    LR_MULTI_PROBE_AGG,      WD_MULTI_PROBE),
        ("multi_probe",        LR_MULTI_PROBE,          WD_MULTI_PROBE),
        ("logit",              LR_LOGIT,                0.0),
        ("finegrain",          LR_FINEGRAIN,            WEIGHT_DECAY),
        ("flair",              LR_FLAIR,                WEIGHT_DECAY),
    ]
    param_groups = [
        {"params": params_by_group[name], "lr": lr * lr_scale, "weight_decay": wd, "name": name}
        for name, lr, wd in group_specs
        if params_by_group[name]
    ]

    optim = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
    print(f"  [optimizer] {len(param_groups)} LR groups (lr_scale={lr_scale:.4f}):")
    for g in param_groups:
        n_p = sum(p.numel() for p in g["params"])
        print(f"    {g['name']:22s}: {n_p:>10,} params, lr={g['lr']:.3e}, wd={g['weight_decay']}")
    return optim

def build_scheduler(optim, total_steps, warmup_steps: int):
    """LambdaLR cosine + warmup + horizon clamp."""
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = min(warmup_steps, max(1, total_steps - 1))
    decay_steps = max(1, total_steps - warmup_steps)

    max_group_lr = max(g.get("lr", LR) for g in optim.param_groups)
    eta_min_abs = max_group_lr * LR_MIN_RATIO
    print(f"  [scheduler] cosine+clamp: warmup={warmup_steps}, decay={decay_steps}, "
          f"final_min_lr={eta_min_abs:.2e} ({LR_MIN_RATIO*100:.0f}% of largest group LR={max_group_lr:.0e})")

    def _make_lambda(base_lr):
        group_eta_min = min(eta_min_abs, base_lr)
        def f(step):
            if step < warmup_steps:
                return 0.01 + 0.99 * step / max(1, warmup_steps)
            progress = min(1.0, (step - warmup_steps) / decay_steps)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            target_lr = group_eta_min + (base_lr - group_eta_min) * cos_factor
            return target_lr / base_lr
        return f

    lr_lambdas = [_make_lambda(g.get("lr", LR)) for g in optim.param_groups]
    sched = LambdaLR(optim, lr_lambda=lr_lambdas)
    return sched

def build_ckpt_extra(epoch, **overrides) -> dict:
    """The common fields of the save_checkpoint extra dict (model_name / dora / seed / pooler_mha_split)."""
    base = {
        "epoch":            epoch,
        "model_name":       MODEL_NAME,
        "dora": {"r_vision": DORA_RANK, "alpha_vision": DORA_ALPHA,
                 "r_text":   DORA_RANK_TEXT, "alpha_text": DORA_ALPHA_TEXT,
                 "targets":  DORA_TARGETS},
        "seed":             SEED,
        "pooler_mha_split": True,
        "flair_enabled":    FLAIR_ENABLED,
        "position_pi_enabled":      POSITION_PI_ENABLED,
        "position_pi_pretrain_len": POSITION_PI_PRETRAIN_LEN,
        "position_pi_target_len":   POSITION_PI_TARGET_LEN,
    }
    base.update(overrides)
    return base

def _collect_logit_and_position(base, extras: dict):
    """Collect logit_scale / logit_bias / the text position_embedding into extras."""
    if hasattr(base, "logit_scale") and isinstance(base.logit_scale, nn.Parameter):
        extras["logit_scale"] = base.logit_scale.detach().cpu()
    if hasattr(base, "logit_bias") and isinstance(base.logit_bias, nn.Parameter):
        extras["logit_bias"] = base.logit_bias.detach().cpu()
    try:
        pe_weight = base.text_model.embeddings.position_embedding.weight.detach().cpu()
        extras["position_embedding"]      = pe_weight
        extras["position_embedding_size"] = int(pe_weight.shape[0])
    except AttributeError:
        pass

def _collect_multi_probe_state(head, extras: dict):
    """Multi-probe (unified): collect probe (1,K,D) and the aggregator into extras."""
    if hasattr(head, "probe"):
        probe = head.probe.detach().cpu()
        extras["multi_probe_probe"]           = probe
        extras["multi_probe_count"]           = int(probe.shape[1])
        extras["multi_probe_design_version"]  = "unified"

    agg_type = getattr(head, "agg_type", "linear")
    extras["multi_probe_agg_type"] = agg_type
    if agg_type == "linear":
        if not hasattr(head, "aggregator") or head.aggregator is None:
            return
        agg = head.aggregator.weight.detach().cpu()
        extras["multi_probe_aggregator"]              = agg
        extras["multi_probe_aggregator_in_features"]  = int(head.aggregator.in_features)
        extras["multi_probe_aggregator_out_features"] = int(head.aggregator.out_features)
    elif agg_type == "attention":
        if hasattr(head, "agg_query"):
            extras["multi_probe_agg_query"]     = head.agg_query.detach().cpu()
        if hasattr(head, "agg_init_bias"):
            extras["multi_probe_agg_init_bias"] = head.agg_init_bias.detach().cpu()
        if hasattr(head, "agg_proj") and head.agg_proj is not None:
            extras["multi_probe_agg_proj"]      = head.agg_proj.weight.detach().cpu()
        extras["multi_probe_attn_temp"]         = float(getattr(head, "attn_temp", 1.0))

def _pack_extras_state(base) -> dict:
    """Collect the base params PEFT does not save (logit / position / multi-probe / finegrain)."""
    extras = {"pooler_mha_split": True}
    _collect_logit_and_position(base, extras)
    try:
        head = base.vision_model.head
    except AttributeError:
        head = None
    if head is not None:
        _collect_multi_probe_state(head, extras)
    # new trainables outside the PEFT adapter; saved manually or the ckpt is broken
    fg = getattr(base, "finegrain_head", None)
    if fg is not None:
        extras["finegrain_head"] = {k: v.detach().cpu() for k, v in fg.state_dict().items()}
    # FLAIR pooler; saved manually
    fl = getattr(base, "flair_pooler", None)
    if fl is not None:
        extras["flair_pooler"] = {k: v.detach().cpu() for k, v in fl.state_dict().items()}
    return extras

def save_checkpoint(model, ckpt_dir: str, tag: str, extra: dict = None):
    """Save the PEFT DoRA adapter, extras_state.pt (base params PEFT does not save) and meta.json."""
    verify_pretrained_position_unchanged(model, where=f"save_checkpoint(tag={tag})")

    path = f"{ckpt_dir}/{tag}"
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    extras_state = _pack_extras_state(base)
    if extras_state:
        torch.save(extras_state, f"{path}/extras_state.pt")

    # meta.json: marks whether inference needs the surgery, plus the probe count
    extra = dict(extra or {})
    extra.setdefault("multi_probe_count", NUM_PROBES)
    extra.setdefault("pooler_mha_split", True)
    with open(f"{path}/meta.json", "w") as f:
        json.dump(extra, f, indent=2, default=str)
    print(f"  [ckpt] saved → {path}")

def _serialize_ema_shadow(ema):
    """ParamEMA.shadow (id-keyed) -> a list[dict(shape, tensor)] in ema.params order."""
    if ema is None:
        return None
    return [
        {"shape": tuple(ema.shadow[id(p)].shape),
         "tensor": ema.shadow[id(p)].detach().cpu().clone()}
        for p in ema.params
    ]

def _restore_ema_shadow(ema, saved_list):
    """saved_list (in ema.params order) -> ema.shadow. Checks the shapes, then re-syncs the frozen rows."""
    if ema is None or saved_list is None:
        return
    if len(saved_list) != len(ema.params):
        raise ValueError(f"EMA shadow size mismatch: saved={len(saved_list)}, "
                         f"current params={len(ema.params)}. The surgery code may have changed - "
                         f"resume is not compatible.")
    for i, (p, item) in enumerate(zip(ema.params, saved_list)):
        # backward compatible: a plain list of tensors is also accepted
        if torch.is_tensor(item):
            s = item
        else:
            saved_shape = tuple(item.get("shape", ()))
            s = item["tensor"]
            cur_shape = tuple(p.shape)
            if saved_shape and saved_shape != cur_shape:
                raise ValueError(
                    f"EMA shadow shape mismatch at index {i}: "
                    f"saved={saved_shape} current={cur_shape}. "
                    f"The surgery code, the multi-probe K or the DoRA rank may have changed - not compatible."
                )
        ema.shadow[id(p)].copy_(s.to(device=p.device, dtype=p.dtype))
    if hasattr(ema, "_sync_frozen_in_shadow"):
        ema._sync_frozen_in_shadow()

def save_resume_state(*, ckpt_dir: str, epoch: int, global_step: int,
                     best_score: float, best_epoch: int, best_metrics: dict,
                     early_stop_counter: int, last_reset_score: float,
                     optimizer, scheduler, scaler, ema):
    """Save every non-model state needed to resume into last/resume_state.pt."""
    last_dir = os.path.join(ckpt_dir, "last")
    os.makedirs(last_dir, exist_ok=True)

    state = {
        "_version":           1,
        "epoch":              int(epoch),  # last completed epoch
        "global_step":        int(global_step),
        "best_score":         float(best_score),
        "best_epoch":         int(best_epoch),
        "best_metrics":       dict(best_metrics or {}),
        "early_stop_counter": int(early_stop_counter),
        "last_reset_score":   float(last_reset_score),
        "optimizer":          optimizer.state_dict() if optimizer is not None else None,
        "scheduler":          scheduler.state_dict() if scheduler is not None else None,
        "scaler":             (scaler.state_dict() if (scaler is not None and scaler.is_enabled()) else None),
        "ema_shadow":         _serialize_ema_shadow(ema),
        "rng": {
            "python": random.getstate(),
            "numpy":  np.random.get_state(),
            "torch":  torch.get_rng_state(),
            "cuda":   (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        },
    }
    out_path = os.path.join(last_dir, "resume_state.pt")
    torch.save(state, out_path)
    print(f"  [resume] state saved → {out_path} (epoch={epoch}, step={global_step})")

def _resolve_resume_ckpt_dir(arg: str) -> str:
    """Normalize the --resume argument to an absolute path to a 'last/' ckpt directory."""
    if not arg:
        return ""
    p = os.path.abspath(arg)
    cand_last = os.path.join(p, "checkpoints", "last")
    if os.path.isdir(cand_last):
        return cand_last
    if os.path.basename(p) == "last" and os.path.isfile(os.path.join(p, "resume_state.pt")):
        return p
    if os.path.isdir(p):
        if os.path.isfile(os.path.join(p, "resume_state.pt")):
            return p
    raise FileNotFoundError(f"resume_state.pt not found under the --resume path: {arg}")

def load_model_weights_from_ckpt(peft_model, ckpt_path: str):
    """Load the adapter weights and extras from ckpt_path into an existing peft_model, in place."""
    from safetensors.torch import load_file as _st_load
    from peft.utils.save_and_load import set_peft_model_state_dict

    adapter_path = os.path.join(ckpt_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"adapter_model.safetensors not found: {adapter_path}")
    adapter_state = _st_load(adapter_path)
    set_peft_model_state_dict(peft_model, adapter_state)
    print(f"  [resume] adapter weight loaded ({len(adapter_state)} tensors)")

    extras_path = None
    for cand in ("extras_state.pt", "logit_scalars.pt"):
        p = os.path.join(ckpt_path, cand)
        if os.path.exists(p):
            extras_path = p
            break
    if extras_path is None:
        raise FileNotFoundError(f"extras_state.pt / logit_scalars.pt not found in {ckpt_path}")

    extras = torch.load(extras_path, map_location="cpu", weights_only=False)
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model

    if "position_embedding" in extras:
        new_pe = extras["position_embedding"]
        base.text_model.embeddings.position_embedding.weight.data.copy_(
            new_pe.to(base.text_model.embeddings.position_embedding.weight))
        print(f"  [resume] position_embedding restored shape={tuple(new_pe.shape)}")
    if "logit_scale" in extras and hasattr(base, "logit_scale"):
        base.logit_scale.data.copy_(extras["logit_scale"].to(base.logit_scale.data))
    if "logit_bias" in extras and hasattr(base, "logit_bias"):
        base.logit_bias.data.copy_(extras["logit_bias"].to(base.logit_bias.data))

    if NUM_PROBES > 1:
        head = base.vision_model.head
        if hasattr(head, "probe") and "multi_probe_probe" in extras:
            head.probe.data.copy_(extras["multi_probe_probe"].to(head.probe.data))
        if hasattr(head, "agg_query") and "multi_probe_agg_query" in extras:
            head.agg_query.data.copy_(extras["multi_probe_agg_query"].to(head.agg_query.data))
        if hasattr(head, "agg_init_bias") and "multi_probe_agg_init_bias" in extras:
            head.agg_init_bias.data.copy_(
                extras["multi_probe_agg_init_bias"].to(head.agg_init_bias.data))
        if (hasattr(head, "agg_proj") and head.agg_proj is not None
                and "multi_probe_agg_proj" in extras):
            head.agg_proj.weight.data.copy_(
                extras["multi_probe_agg_proj"].to(head.agg_proj.weight.data))
        if (hasattr(head, "aggregator") and head.aggregator is not None
                and "multi_probe_aggregator" in extras):
            head.aggregator.weight.data.copy_(
                extras["multi_probe_aggregator"].to(head.aggregator.weight.data))
    # attach runs before resume, so base.finegrain_head exists
    fg = getattr(base, "finegrain_head", None)
    if fg is not None and "finegrain_head" in extras:
        dev = next(fg.parameters()).device
        fg.load_state_dict({k: v.to(dev) for k, v in extras["finegrain_head"].items()})
        print("  [resume] finegrain_head restored")
    # attach runs before resume, so base.flair_pooler exists
    fl = getattr(base, "flair_pooler", None)
    if fl is not None and "flair_pooler" in extras:
        dev = next(fl.parameters()).device
        fl.load_state_dict({k: v.to(dev) for k, v in extras["flair_pooler"].items()})
        print("  [resume] flair_pooler restored")
    print(f"  [resume] extras restored ← {extras_path}")

def load_resume_state(*, ckpt_path: str, optimizer, scheduler, scaler, ema, device):
    """Restore resume_state.pt. Applies optimizer/scheduler/scaler/ema/RNG/counters in place."""
    resume_file = os.path.join(ckpt_path, "resume_state.pt")
    if not os.path.exists(resume_file):
        raise FileNotFoundError(f"resume_state.pt not found: {resume_file}")
    print(f"  [resume] loading state ← {resume_file}")
    state = torch.load(resume_file, map_location="cpu", weights_only=False)

    if state.get("optimizer") is not None and optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        for grp_state in optimizer.state.values():
            for k, v in list(grp_state.items()):
                if torch.is_tensor(v):
                    grp_state[k] = v.to(device)
    if state.get("scheduler") is not None and scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if (state.get("scaler") is not None and scaler is not None
            and getattr(scaler, "is_enabled", lambda: False)()):
        scaler.load_state_dict(state["scaler"])
    if state.get("ema_shadow") is not None and ema is not None:
        _restore_ema_shadow(ema, state["ema_shadow"])

    rng = state.get("rng") or {}
    try:
        if "python" in rng: random.setstate(rng["python"])
        if "numpy"  in rng: np.random.set_state(rng["numpy"])
        if "torch"  in rng:
            t = rng["torch"]
            torch.set_rng_state(t.cpu() if torch.is_tensor(t) else t)
        if rng.get("cuda") is not None and torch.cuda.is_available():
            cuda_states = [t.cpu() if torch.is_tensor(t) else t for t in rng["cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception as e:
        print(f"  [resume] warn: RNG restore partial ({type(e).__name__}: {e})")

    return {
        "start_epoch":        int(state["epoch"]) + 1,
        "global_step":        int(state["global_step"]),
        "best_score":         float(state["best_score"]),
        "best_epoch":         int(state["best_epoch"]),
        "best_metrics":       dict(state.get("best_metrics") or {}),
        "early_stop_counter": int(state["early_stop_counter"]),
        "last_reset_score":   float(state["last_reset_score"]),
    }

def _setup_run_paths(run_dir: str) -> dict:
    """The per-run artifact paths. Every path lives under run_dir."""
    return {
        "log":      f"{run_dir}/train.log",
        "metrics":  f"{run_dir}/metrics.jsonl",
        "config":   f"{run_dir}/config.json",
        "env":      f"{run_dir}/env.json",
        "summary":  f"{run_dir}/summary.json",
        "ckpt":     f"{run_dir}/checkpoints",
        "tb":       f"{run_dir}/tb",
    }

def print_run_banner(run_name: str, run_dir: str):
    """Startup banner listing every major hyperparameter."""
    print("=" * 64)
    print("  Track4 Baseline v19 L — SigLIP2-Large + DoRA + Multi-Probe Pooling Head")
    print(f"  run = {run_name}")
    print("=" * 64)
    print(f"  device={DEVICE}  amp={USE_AMP}  amp_dtype={AMP_DTYPE}  seed={SEED}")
    print(f"  model={MODEL_NAME}  image={IMAGE_SIZE}  max_text={MAX_TEXT_LENGTH}")
    print(f"  batch={BATCH_SIZE}  epochs={EPOCHS}  warmup_ratio={WARMUP_RATIO}")
    print(f"  Multi-positive: K={N_POSITIVES_PER_IMAGE} (positives per image)")
    print(f"  DoRA: vision r={DORA_RANK}/α={DORA_ALPHA} | text r={DORA_RANK_TEXT}/α={DORA_ALPHA_TEXT} "
          f"dropout={DORA_DROPOUT} targets={DORA_TARGETS}")
    print(f"  Multi-Probe: K={NUM_PROBES} (perturb={MULTI_PROBE_PERTURB_MODE}, "
          f"agg={MULTI_PROBE_AGG_INIT}, block_scale={MULTI_PROBE_BLOCK_SCALE})")
    if POSITION_PI_ENABLED:
        _pi_scale = _compute_pi_scale(POSITION_PI_PRETRAIN_LEN, POSITION_PI_TARGET_LEN)
        print(f"  Position: PI (rows={POSITION_PI_PRETRAIN_LEN} unchanged, "
              f"target_len={POSITION_PI_TARGET_LEN}, scale={_pi_scale:.4f}), "
              f"freeze_all_rows={POSITION_PRETRAINED_FREEZE} (LR_POSITION={LR_POSITION:.0e})")
    else:
        print(f"  Position: 64 → {POSITION_EMBED_NEW_SIZE}, "
              f"pretrained [0..{POSITION_PRETRAINED_SIZE-1}] "
              f"{'FROZEN' if POSITION_PRETRAINED_FREEZE else 'TRAINABLE'} (LR_POSITION={LR_POSITION:.0e})")
    print(f"  Vision base: frozen | Vision DoRA: {'OFF' if FREEZE_VISION_DORA else 'ON'} | "
          f"Pooler DoRA: {'OFF' if FREEZE_POOLER_DORA else 'ON'} | GC={GRADIENT_CHECKPOINTING}")
    print(f"  LR groups: pos={LR_POSITION:.0e} dora={LR:.0e} "
          f"multi_probe={LR_MULTI_PROBE:.0e}/{WD_MULTI_PROBE} logit={LR_LOGIT:.0e}")
    print(f"  Schedule: horizon={SCHEDULE_HORIZON_EPOCHS}ep cosine, "
          f"EarlyStop patience={EARLY_STOP_PATIENCE} min_delta={EARLY_STOP_MIN_DELTA}")
    print(f"  EMA: enabled={EMA_ENABLED}, half_life={EMA_HALF_LIFE_EPOCHS}ep, start_ep={EMA_START_EPOCH}")
    print(f"  Soft Label: USE={USE_SOFT_LABEL} (off → vanilla multi-pos SigLIP)")
    print(f"  [logit] logit_scale clamp: enabled={LOGIT_SCALE_CLAMP_ENABLED} max_exp={LOGIT_SCALE_MAX} "
          f"freeze={LOGIT_SCALE_FREEZE}")
    print(f"  [gather] cross-GPU neg gather: {CROSS_GPU_NEG_GATHER} (single GPU - always off)")
    print(f"  [uniform] Uniformity reg: enabled={UNIFORMITY_REG_ENABLED} "
          f"λ={UNIFORMITY_REG_LAMBDA} t={UNIFORMITY_REG_T} (rank-local v,t)")
    print(f"  Ckpt rolling: keep_last={'ALL (keep every epoch)' if KEEP_LAST_N_CKPT<0 else KEEP_LAST_N_CKPT} (plus best/last)")
    print(f"  Diag freq: log_every={LOG_EVERY_STEPS}  sim_diag_every={DIAG_EVERY_STEPS}")
    print(f"  Image Aug: HFlip={IMG_AUG_HFLIP_PROB} Gray={IMG_AUG_GRAYSCALE_P} "
          f"RRC={'ON' + str(IMG_AUG_RRC_SCALE) if IMG_AUG_RRC_ENABLED else 'OFF (Resize+CenterCrop)'} "
          f"Erase={IMG_AUG_ERASE_P}/{IMG_AUG_ERASE_SCALE} "
          f"GNoise={IMG_AUG_GNOISE_P}/{IMG_AUG_GNOISE_STD} "
          f"Color={IMG_AUG_COLOR_JITTER[0]}")
    print(f"  Data: unique_image_mode={USE_UNIQUE_IMAGE_MODE} "
          f"subset_mode={SUBSET_MODE}")
    print(f"  Safety: loss_explosion>{LOSS_EXPLOSION_THRESHOLD}, NaN_abort={LOSS_NAN_ABORT}")
    print(f"  [flair] FLAIR: enabled={FLAIR_ENABLED} λ={FLAIR_WEIGHT} dim={FLAIR_DIM} "
          f"heads={FLAIR_NUM_HEADS} pool={FLAIR_PATCH_POOL} symmetric={FLAIR_SYMMETRIC} "
          f"input_ln={FLAIR_INPUT_LN} warmup_ep={FLAIR_WEIGHT_WARMUP_EPOCHS} "
          f"(train-only, FINEGRAIN={FINEGRAIN_ENABLED}, UNIFORMITY={UNIFORMITY_REG_ENABLED})")
    print(f"  run_dir={run_dir}")

def setup_run_logging(paths: dict, run_name: str, run_dir: str):
    """tee + metric_writer + a config/env JSON snapshot. Returns (tee, metric_writer)."""
    tee = TeeStdout(paths["log"], also_stderr=True, timestamp=True)
    tee.install()
    metric_writer = MetricWriter(paths["metrics"])
    cfg_snapshot = snapshot_config()
    env_snapshot = collect_env_info()
    with open(paths["config"], "w") as f:
        json.dump(cfg_snapshot, f, indent=2, default=str)
    with open(paths["env"], "w") as f:
        json.dump(env_snapshot, f, indent=2, default=str)
    metric_writer.log(event="run_start", run_name=run_name, run_dir=run_dir)
    print(f"  [logger] tee → {paths['log']}")
    print(f"  [logger] metrics → {paths['metrics']}")
    print("  [logger] config / env saved")
    return tee, metric_writer

def _dump_grad_spike(metric_writer, epoch, gstep, grad_norm, loss_val, batch, image_ids):
    """Forensic dump of a batch skipped by the gradient spike guard, to identify the toxic batch."""
    from collections import Counter
    imgs = batch.get("images", []); acts = batch.get("actions", [])
    img_counts = Counter(imgs); dup = {k: c for k, c in img_counts.items() if c > 1}
    top_acts = Counter(acts).most_common(5)
    n_txt = int(image_ids.numel()) if hasattr(image_ids, "numel") else len(image_ids)
    print(f"\n  [GRAD-SPIKE SKIP] ep{epoch} g{gstep}: grad_norm={grad_norm:.2f} loss={loss_val:.3f}")
    print(f"     batch: images={len(imgs)} unique={len(img_counts)} dup={len(dup)} texts={n_txt}")
    if dup:
        print(f"     dup_images(top5): {list(dup.items())[:5]}")
    print(f"     action_concentration(top5): {top_acts}")
    metric_writer.log(event="grad_spike_skip", epoch=epoch, global_step=gstep,
                      grad_norm=grad_norm, loss=loss_val,
                      n_images=len(imgs), n_unique_images=len(img_counts),
                      n_dup_images=len(dup), dup_images=dict(list(dup.items())[:10]),
                      top_actions=dict(top_acts), sample_images=imgs[:8], sample_actions=acts[:8])

def _report_epoch_end(epoch: int, ep_loss: float, ep_time: float,
                       train_log: dict, ep_normal_cnt: int, ep_anom_cnt: int,
                       ep_action_cnt: dict, lr_dora: float, metric_writer):
    """Print the train statistics at the end of an epoch and append them to the jsonl."""
    ep_total = ep_normal_cnt + ep_anom_cnt
    ep_norm_r = ep_normal_cnt / max(1, ep_total)
    ep_top_act = sorted(ep_action_cnt.items(), key=lambda x: -x[1])[:3]
    avg_grad = train_log["grad_norm_total"] / max(1, train_log["count"])
    top_act_str = ", ".join(f"{a}:{c}" for a, c in ep_top_act)
    print(f"  Ep{epoch:02d}  loss={ep_loss:.4f}  "
          f"lr_dora={lr_dora:.2e}  "
          f"time={ep_time/60:.1f}min  "
          f"avg_grad_norm={avg_grad:.2f}  "
          f"normal_ratio={ep_norm_r:.3f}  top_actions=[{top_act_str}]")
    metric_writer.log(event="epoch_end",
                      epoch=epoch,
                      loss=ep_loss,
                      time_min=ep_time/60,
                      avg_grad_norm=avg_grad,
                      ep_normal_ratio=ep_norm_r,
                      ep_normal_cnt=ep_normal_cnt,
                      ep_anomaly_cnt=ep_anom_cnt,
                      ep_top3_actions={a: c for a, c in ep_top_act})

def _report_soft_label_diag(epoch: int, ep_soft_diag: dict, metric_writer):
    """Print the epoch-average soft label diagnostic and append it to the jsonl."""
    nd = ep_soft_diag["n_diag"]
    if not (USE_SOFT_LABEL and SOFT_LABEL_DIAG_ENABLED and nd > 0):
        return
    avg_p   = ep_soft_diag["avg_p_sum"]      / nd
    ratio_s = ep_soft_diag["ratio_soft_sum"] / nd
    lp = ep_soft_diag["loss_pos_sum"]  / nd
    ls = ep_soft_diag["loss_soft_sum"] / nd
    ln = ep_soft_diag["loss_neg_sum"]  / nd
    n_pos_avg  = ep_soft_diag["n_pos"]  / nd
    n_soft_avg = ep_soft_diag["n_soft"] / nd
    n_neg_avg  = ep_soft_diag["n_neg"]  / nd
    print(f"  [soft_label] avg_p={avg_p:.4f}  ratio_soft={100*ratio_s:.2f}% "
          f"(per-batch avg pairs: pos={n_pos_avg:.0f}, soft={n_soft_avg:.0f}, neg={n_neg_avg:.0f})")
    print(f"  [soft_label] loss_components: pos={lp:.3f}, soft={ls:.3f}, neg={ln:.3f}")
    metric_writer.log(event="soft_label_diag", epoch=epoch,
                      avg_p=avg_p, ratio_soft=ratio_s,
                      n_pos_avg=n_pos_avg, n_soft_avg=n_soft_avg, n_neg_avg=n_neg_avg,
                      loss_pos=lp, loss_soft=ls, loss_neg=ln)

def _report_multi_probe_diag(epoch: int, model, metric_writer):
    """Multi-probe diagnostic: probe divergence plus aggregator block weight. Skipped when there is no result."""
    diag = diagnose_multi_probe(model)
    if not diag:
        return
    metric_writer.log(event="multi_probe_diag", epoch=epoch, **diag)
    diag_str = ", ".join(f"{k}={v:.3f}" for k, v in diag.items())
    print(f"  [multi_probe_diag] {diag_str}")

_LR_GROUPS = ("position_emb", "dora", "logit", "multi_probe",
              "multi_probe_probe", "multi_probe_agg")

def _lr_snapshot(scheduler, lr_group_idx: dict) -> dict:
    """The current LR group values. A missing group reads 0.0."""
    last_lr = scheduler.get_last_lr()
    return {name: (last_lr[lr_group_idx[name]] if name in lr_group_idx else 0.0)
            for name in _LR_GROUPS}

def _log_train_step(*, metric_writer, tb,
                    epoch: int, step: int, global_step: int,
                    loss_val: float, loss_avg: float, lrs: dict,
                    grad_norm_total: float, grad_norm_nonlogit, grad_norm_logit,
                    samples_per_sec: float, batch_normal_ratio: float,
                    logit_scale, logit_bias, diag: dict, mem: dict):
    """Per-step jsonl and TB logging (only called on a will_log step)."""
    rec = {
        "event":              "train",
        "epoch":              epoch,
        "step":               step,
        "global_step":        global_step,
        "loss":               loss_val,
        "loss_avg_in_epoch":  loss_avg,
        "lr_dora":            lrs["dora"],
        "lr_logit":           lrs["logit"],
        "lr_position":        lrs["position_emb"],
        "lr_multi_probe":     lrs["multi_probe"],
        "lr_multi_probe_probe": lrs["multi_probe_probe"],
        "lr_multi_probe_agg":   lrs["multi_probe_agg"],
        "grad_norm_total":    grad_norm_total,
        "grad_norm_nonlogit": grad_norm_nonlogit,  # DoRA plus position
        "grad_norm_logit":    grad_norm_logit,
        "samples_per_sec":    samples_per_sec,
        "batch_normal_ratio": batch_normal_ratio,
        "logit_scale_exp":    float(logit_scale.detach().exp()) if logit_scale is not None else None,
        "logit_bias":         float(logit_bias.detach())        if logit_bias  is not None else None,
    }
    rec.update(diag)
    rec.update(mem)
    metric_writer.log(**rec)

    if tb is None:
        return
    tb.add_scalar("train/loss",     loss_val, global_step)
    tb.add_scalar("train/loss_avg", loss_avg, global_step)
    for tb_key, lr_key in [("lr_dora",        "dora"),
                            ("lr_logit",       "logit"),
                            ("lr_position",    "position_emb"),
                            ("lr_multi_probe", "multi_probe")]:
        tb.add_scalar(f"train/{tb_key}", lrs[lr_key], global_step)
    tb.add_scalar("train/grad_norm_total",    grad_norm_total,    global_step)
    tb.add_scalar("train/grad_norm_nonlogit", grad_norm_nonlogit, global_step)
    tb.add_scalar("train/grad_norm_logit",    grad_norm_logit,    global_step)
    tb.add_scalar("train/samples_per_sec",    samples_per_sec,    global_step)
    tb.add_scalar("data/batch_normal_ratio",  batch_normal_ratio, global_step)
    if logit_scale is not None:
        tb.add_scalar("train/logit_scale_exp",
                      float(logit_scale.detach().exp()), global_step)
    for k, val in diag.items():
        tb.add_scalar(f"diag/{k}", val, global_step)
    for k, val in mem.items():
        tb.add_scalar(f"mem/{k}", val, global_step)

def _print_dataloader_distribution(label_count, action_count, train_loader):
    """Print the sampler distribution and the batches per epoch at the start of training."""
    if label_count is not None:
        ratio_anom = label_count["anomaly"] / max(1, label_count["normal"])
        print(f"  raw distribution:  normal={label_count['normal']:,}  "
              f"anomaly={label_count['anomaly']:,}  ratio={ratio_anom:.2f}:1")
        top_actions = sorted(action_count.items(), key=lambda x: -x[1])[:5]
        print("  top-5 actions: " + ", ".join(f"{a}({c:,})" for a, c in top_actions))
        print(f"  sampler:  WeightedRandomSampler  "
              f"(label_balance={BALANCE_LABEL_TYPE}, "
              f"action_smooth={BALANCE_ACTION}, alpha={ACTION_SMOOTH_ALPHA})")
    else:
        print("  sampler:  random shuffle (no balance)")
    print(f"  train batches/epoch = {len(train_loader):,}  "
          f"(epoch_len_factor={EPOCH_LEN_FACTOR})")

def _diagnose_caption_lengths(train_rows, sample_size: int = 10000):
    """Report the sentence-count distribution of the captions (data characterisation)."""
    n = min(sample_size, len(train_rows))
    indices = random.Random(0).sample(range(len(train_rows)), n)
    def _cap_text(r):
        # image-level = captions (list); the old format used caption (str)
        caps = r.get("captions")
        if isinstance(caps, list):
            return caps[0] if caps else ""
        return r.get("caption", "")
    sent_counts = [len(_cap_text(train_rows[i]).split(". ")) for i in indices]
    c1  = sum(1 for x in sent_counts if x == 1)
    c2  = sum(1 for x in sent_counts if x == 2)
    c3p = sum(1 for x in sent_counts if x >= 3)
    pct1 = 100.0 * c1 / max(1, n)
    print(f"  caption sentence count (sample={n:,}): "
          f"1-sent={c1:,} ({pct1:.1f}%), 2-sent={c2:,}, ≥3-sent={c3p:,}")

def _verify_position_ids_device(model):
    """Check that text_model.embeddings.position_ids is on the training device."""
    try:
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        pe = base.text_model.embeddings.position_ids
    except AttributeError as e:
        print(f"  [verify] position_ids check skipped: {e}")
        return
    expected = torch.device(DEVICE)
    actual = pe.device
    ok = (expected.type == actual.type and
          (expected.index is None or expected.index == actual.index))
    print(f"  [verify] text position_ids: device={actual}, shape={tuple(pe.shape)} "
          f"(expected device={expected})")
    if not ok:
        print(f"  [warn] position_ids device mismatch — expected {expected}, got {actual}")

def _verify_tokenizer_pickleable(tokenizer):
    """tokenizer pickle round trip, exercising __getattr__/__setattr__ of _LowercaseTokenizerWrapper."""
    import pickle
    try:
        blob = pickle.dumps(tokenizer)
        restored = pickle.loads(blob)
        out = restored("Lowercase Verify")
        assert "input_ids" in out, "restored tokenizer missing input_ids"
        print(f"  [verify] tokenizer pickle round-trip ✓ ({len(blob):,} bytes, "
              f"lowercase wrapper preserved)")
    except Exception as e:
        raise RuntimeError(
            f"[verify] tokenizer pickle round-trip FAILED: {type(e).__name__}: {e}. "
            f"The DataLoader workers (NUM_WORKERS>0) cannot serialize the tokenizer and will crash. "
            f"Use NUM_WORKERS=0 or implement __reduce__ on _LowercaseTokenizerWrapper."
        )

class FineGrainedHead(nn.Module):
    """Project token/patch hidden states into the shared late-interaction space (L2-normalized)."""
    def __init__(self, tok_dim, patch_dim, fg_dim=256):
        super().__init__()
        self.tok_proj   = nn.Linear(tok_dim, fg_dim)
        self.patch_proj = nn.Linear(patch_dim, fg_dim)

    def forward(self, tok_hidden, patch_hidden):
        # project in fp32 then normalize, even under bf16 autocast
        tf = F.normalize(self.tok_proj(tok_hidden.float()),   dim=-1)  # (Nt,Lt,fg)
        pf = F.normalize(self.patch_proj(patch_hidden.float()), dim=-1)  # (Ni,Lp,fg)
        return tf, pf

def _fg_patch_pool(fg_patch, patch_pool):
    """2D avgpool downsample of the (Ni,Lp,d) patches, then renormalize. Shared by train and inference."""
    if not patch_pool or patch_pool <= 1:
        return fg_patch
    Ni, Lp, d = fg_patch.shape
    g = int(round(Lp ** 0.5))
    if g * g != Lp:
        # never return silently: skipping the pooling keeps Lp and blows the einsum up 16x
        # fail fast if the patch grid assumption is violated
        raise ValueError(
            f"_fg_patch_pool: Lp={Lp} is not a perfect square (g={g}) - patch grid assumption violated. "
            f"A silent pooling skip blows the FILIP einsum up 16x. Check the resolution, patch size and token layout.")
    fp = fg_patch.transpose(1, 2).reshape(Ni, d, g, g)
    fp = F.avg_pool2d(fp, kernel_size=patch_pool, ceil_mode=True)
    fp = fp.flatten(2).transpose(1, 2)
    return F.normalize(fp, dim=-1)

def _flair_patch_pool(v_loc, pool):
    """2D avgpool downsample of the (Ni,Lp,D) raw vision local tokens (no normalize: pre-projection raw)."""
    if not pool or pool <= 1:
        return v_loc
    Ni, Lp, D = v_loc.shape
    g = int(round(Lp ** 0.5))
    if g * g != Lp:
        raise ValueError(
            f"_flair_patch_pool: Lp={Lp} is not a perfect square (g={g}) - patch grid assumption violated "
            f"(register/CLS token, or a resolution/patch change).")
    x = v_loc.transpose(1, 2).reshape(Ni, D, g, g)
    x = F.avg_pool2d(x, kernel_size=pool, ceil_mode=True)
    return x.flatten(2).transpose(1, 2)  # (Ni, Lp', D)

class TextConditionedPooler(nn.Module):
    """FLAIR text-conditioned attention pooling (train only; a new trainable, not DoRA)."""
    def __init__(self, tok_dim, patch_dim, fdim=FLAIR_DIM, num_heads=FLAIR_NUM_HEADS,
                 patch_pool=FLAIR_PATCH_POOL, input_ln=FLAIR_INPUT_LN):
        super().__init__()
        assert fdim % num_heads == 0
        self.h, self.dh = num_heads, fdim // num_heads
        self.scale = self.dh ** -0.5
        self.patch_pool = patch_pool
        self.q_ln  = nn.LayerNorm(tok_dim,   eps=1e-6) if input_ln else nn.Identity()
        self.kv_ln = nn.LayerNorm(patch_dim, eps=1e-6) if input_ln else nn.Identity()
        self.q_proj   = nn.Linear(tok_dim,   fdim)
        self.k_proj   = nn.Linear(patch_dim, fdim)
        self.v_proj   = nn.Linear(patch_dim, fdim)
        self.out_proj = nn.Linear(fdim, fdim)
        self.t_proj   = nn.Linear(tok_dim, fdim)

    def forward(self, t_pooled, v_loc):
        """t_pooled: (Nt,Dt), v_loc: (Ni,Lp,Dv). Returns S: (Nt,Ni)."""
        v_loc = _flair_patch_pool(v_loc, self.patch_pool)  # (Ni,Lp',Dv)
        Nt = t_pooled.shape[0]
        Ni, Lp, _ = v_loc.shape
        H, Dh = self.h, self.dh
        # inside autocast(bf16); force fp32 for a stable softmax/normalize
        with torch.autocast(device_type="cuda", enabled=False):
            t_n = self.q_ln(t_pooled.float())  # (Nt,Dt)
            v_n = self.kv_ln(v_loc.float())  # (Ni,Lp,Dv)
            q = self.q_proj(t_n).view(Nt, H, Dh)  # (Nt,H,Dh)
            k = self.k_proj(v_n).view(Ni, Lp, H, Dh)  # (Ni,Lp,H,Dh)
            v = self.v_proj(v_n).view(Ni, Lp, H, Dh)
            scores = torch.einsum('ihd,jphd->ijhp', q, k) * self.scale  # (Nt,Ni,H,Lp)
            attn   = scores.softmax(dim=-1)  # over patches
            v_cond = torch.einsum('ijhp,jphd->ijhd', attn, v)  # (Nt,Ni,H,Dh)
            v_cond = self.out_proj(v_cond.reshape(Nt, Ni, H * Dh))  # (Nt,Ni,fdim)
            tq = self.t_proj(t_n)  # (Nt,fdim)
            v_cond = F.normalize(v_cond, dim=-1)
            tq     = F.normalize(tq, dim=-1)
            S = torch.einsum('ijd,id->ij', v_cond, tq)  # (Nt,Ni)
        return S

def finegrain_filip_loss(fg_tok, fg_patch, keypad, image_ids,
                         logit_scale=FINEGRAIN_LOGIT_SCALE,
                         patch_pool=FINEGRAIN_PATCH_POOL):
    """FILIP-style token/patch late-interaction text->image contrastive loss (local, in-batch)."""
    fg_patch = _fg_patch_pool(fg_patch, patch_pool)  # same downsample in training and inference
    # (Nt,Lt,Ni,Lp); memory = Nt*Lt*Ni*Lp, patch_pool shrinks Lp
    sim = torch.einsum('itd,jpd->itjp', fg_tok, fg_patch)
    # top-k mean over patch (k = round(Lp * ratio))
    _k = max(1, min(round(sim.size(-1) * FINEGRAIN_AGG_RATIO), sim.size(-1)))
    sim = sim.topk(_k, dim=-1).values.mean(dim=-1)  # (Nt,Lt,Ni)
    real = (~keypad).to(sim.dtype).unsqueeze(-1)  # (Nt,Lt,1)
    S = (sim * real).sum(dim=1) / real.sum(dim=1).clamp(min=1.0)  # (Nt,Ni)
    logits = logit_scale * S.float()
    with torch.no_grad():
        fg_acc = (logits.argmax(dim=1) == image_ids).float().mean()
    return F.cross_entropy(logits, image_ids), fg_acc.detach()

def flair_tcs_loss(S, image_ids, logit_scale=FLAIR_LOGIT_SCALE, symmetric=FLAIR_SYMMETRIC):
    """FLAIR text-conditioned similarity contrastive loss (local in-batch)."""
    logits = logit_scale * S.float()  # (Nt,Ni)
    loss_t2i = F.cross_entropy(logits, image_ids)
    with torch.no_grad():
        acc = (logits.argmax(dim=1) == image_ids).float().mean()
    if not symmetric:
        return loss_t2i, acc.detach()
    Ni = logits.shape[1]
    logits_i2t = logits.t()  # (Ni,Nt)
    pos = (image_ids.unsqueeze(0) ==
           torch.arange(Ni, device=S.device).unsqueeze(1)).float()  # (Ni,Nt)
    pos = pos / pos.sum(dim=1, keepdim=True).clamp(min=1.0)
    logp = F.log_softmax(logits_i2t, dim=1)
    loss_i2t = -(pos * logp).sum(dim=1).mean()
    return 0.5 * (loss_t2i + loss_i2t), acc.detach()

class JointEncoder(nn.Module):
    """Encode image and text with the SigLIP base model in a single forward."""

    def __init__(self, base_model: nn.Module, fg_enabled: bool = False,
                 flair_enabled: bool = False, pad_id: int = None):
        super().__init__()
        self.base = base_model
        _inner = base_model.get_base_model() if hasattr(base_model, "get_base_model") else base_model
        self._fg_head      = getattr(_inner, "finegrain_head", None)
        self._flair_pooler = getattr(_inner, "flair_pooler", None)
        self.fg_enabled    = bool(fg_enabled and self._fg_head is not None)
        self.flair_enabled = bool(flair_enabled and self._flair_pooler is not None)
        self.pad_id = pad_id

    def forward(self, pixel_values, input_ids, attention_mask):
        if self.flair_enabled:
            vo = self.base.vision_model(pixel_values=pixel_values)
            to = self.base.text_model(input_ids=input_ids, attention_mask=attention_mask)
            v, t = vo.pooler_output, to.pooler_output
            S = self._flair_pooler(t, vo.last_hidden_state)  # (Nt,Ni)
            return v, t, {"flair_S": S}
        if self.fg_enabled:
            vo = self.base.vision_model(pixel_values=pixel_values)
            to = self.base.text_model(input_ids=input_ids, attention_mask=attention_mask)
            v, t = vo.pooler_output, to.pooler_output
            fg_tok, fg_patch = self._fg_head(to.last_hidden_state, vo.last_hidden_state)
            keypad = (input_ids == self.pad_id) if self.pad_id is not None \
                     else torch.zeros_like(input_ids, dtype=torch.bool)
            return v, t, {"fg_tok": fg_tok, "fg_patch": fg_patch, "fg_keypad": keypad}
        v = self.base.get_image_features(pixel_values=pixel_values)
        t = self.base.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return v, t, {}

def unwrap_base(model: nn.Module):
    """Strip the JointEncoder wrapper and return the base model."""
    m = model
    if isinstance(m, JointEncoder):
        m = m.base
    return m

def train_single(physical_gpu: int,
                 extra_run_note: str = "", resume_path: str = ""):
    torch.cuda.set_device(physical_gpu)
    device = torch.device(f"cuda:{physical_gpu}")
    DEVICE = f"cuda:{physical_gpu}"

    set_seed(SEED)
    ensure_global_dirs()

    run_dir, run_name = make_run_dir(
        extra_run_note or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_NOTE}")
    paths = _setup_run_paths(run_dir)
    ckpt_dir = paths["ckpt"]
    tb_dir = paths["tb"]
    summary_path = paths["summary"]

    print_run_banner(run_name, run_dir)
    print(f"  [single-GPU] device={device}")
    tee, metric_writer = setup_run_logging(paths, run_name, run_dir)
    summary = {
        "run_name": run_name,
        "run_dir": run_dir,
        "started": datetime.now().isoformat(timespec="seconds"),
        "num_gpus": 1,
        "gpu": str(physical_gpu),
        "resumed_from": resume_path or None,
    }
    # inert, always -1; -1 means not selected
    best_score = -1.0
    best_epoch = -1
    best_metrics = {}
    global_step = 0
    early_stop_counter = 0
    last_reset_score = -1.0
    early_stopped = False
    start_epoch = 1  # load_resume_state overwrites this on resume
    resuming = bool(resume_path)

    try:
        print("\n[1/5] Loading manifest ...")
        rows = load_manifest()
        print(f"  manifest rows = {len(rows):,}")
        src_counter = {}
        for r in rows:
            s = r.get("source", "manifest")
            src_counter[s] = src_counter.get(s, 0) + 1
        print(f"  source distribution: {src_counter}")
        train_rows = list(rows)

        if SUBSET_MODE:
            rng = random.Random(SEED)
            rng.shuffle(train_rows)
            train_rows = train_rows[:SUBSET_TRAIN_N]
            print(f"  [SUBSET] train={len(train_rows):,}")

        metric_writer.log(event="data_ready", n_train=len(train_rows))

        print("\n[2/5] Model ...")
        model, tokenizer = build_model()
        model = model.to(device)
        _verify_position_ids_device(model)
        _verify_tokenizer_pickleable(tokenizer)

        # must attach to the peft model for build_optimizer to see it
        if FINEGRAIN_ENABLED:
            _fg_base = model.get_base_model() if hasattr(model, "get_base_model") else model
            _fg_dim = int(_fg_base.text_model.config.hidden_size)
            _pat_dim = int(_fg_base.vision_model.config.hidden_size)
            # attach to base so the optimizer picks up finegrain_head
            _fg_base.finegrain_head = FineGrainedHead(_fg_dim, _pat_dim, FINEGRAIN_DIM).to(device)
            _nfg = sum(p.numel() for p in _fg_base.finegrain_head.parameters())
            print(f"  [finegrain] FineGrainedHead attached (tok={_fg_dim}/pat={_pat_dim}→{FINEGRAIN_DIM}, "
                  f"{_nfg:,} params, λ={FINEGRAIN_WEIGHT}, pool={FINEGRAIN_PATCH_POOL}, "
                  f"logit_scale={FINEGRAIN_LOGIT_SCALE})")

        if FLAIR_ENABLED:
            _flb = model.get_base_model() if hasattr(model, "get_base_model") else model
            _ftdim = int(_flb.text_model.config.hidden_size)
            _fpdim = int(_flb.vision_model.config.hidden_size)
            _flb.flair_pooler = TextConditionedPooler(_ftdim, _fpdim).to(device)
            _nfl = sum(p.numel() for p in _flb.flair_pooler.parameters())
            print(f"  [flair] TextConditionedPooler attached "
                  f"(tok={_ftdim}/pat={_fpdim}→{FLAIR_DIM}, heads={FLAIR_NUM_HEADS}, "
                  f"{_nfl:,} params, λ={FLAIR_WEIGHT}, pool={FLAIR_PATCH_POOL}, "
                  f"input_ln={FLAIR_INPUT_LN}, train-only)")

        if resuming:
            print(f"\n  [resume] loading model weights from {resume_path}")
            load_model_weights_from_ckpt(model, resume_path)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        metric_writer.log(event="model_built",
                          n_trainable=n_trainable, n_total=n_total,
                          ratio=n_trainable / max(1, n_total))

        print("\n[3/5] DataLoader ...")
        train_tf = build_train_transform(IMAGE_SIZE)

        train_loader, label_count, action_count = build_train_loader(
            train_rows, tokenizer, train_tf)

        n_train = len(train_rows)
        _print_dataloader_distribution(label_count, action_count, train_loader)
        _diagnose_caption_lengths(train_rows)
        metric_writer.log(event="dataloader_ready",
                          n_train=n_train,
                          batches_per_epoch=len(train_loader),
                          n_normal=label_count["normal"] if label_count else None,
                          n_anomaly=label_count["anomaly"] if label_count else None,
                          balance_label=BALANCE_LABEL_TYPE,
                          balance_action=BALANCE_ACTION,
                          action_smooth_alpha=ACTION_SMOOTH_ALPHA,
                          num_gpus=1,
                          per_rank_batches_per_epoch=len(train_loader))

        print("\n[4/5] Optimizer / scheduler ...")
        steps_per_epoch = len(train_loader)
        total_steps = EPOCHS * steps_per_epoch
        schedule_total_steps = SCHEDULE_HORIZON_EPOCHS * steps_per_epoch
        warmup_steps = max(1, int(round(schedule_total_steps * WARMUP_RATIO)))

        batch_ratio = max(1.0, BATCH_SIZE / REF_BATCH)
        if LR_BATCH_SCALING == "sqrt":
            lr_scale = math.sqrt(batch_ratio)
        elif LR_BATCH_SCALING == "linear":
            lr_scale = float(batch_ratio)
        elif LR_BATCH_SCALING == "none":
            lr_scale = 1.0
        else:
            raise ValueError(f"LR_BATCH_SCALING={LR_BATCH_SCALING!r} (expected sqrt|linear|none)")

        optimizer = build_optimizer(model, lr_dora=LR, lr_scale=lr_scale)
        scheduler = build_scheduler(optimizer, schedule_total_steps, warmup_steps)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(USE_AMP and AMP_DTYPE == torch.float16))
        use_scaler = (AMP_DTYPE == torch.float16)
        lr_group_idx = {g.get("name", f"g{i}"): i for i, g in enumerate(optimizer.param_groups)}
        print(f"  LR_group_idx        = {lr_group_idx}")
        print(f"  batch (single GPU)  = {BATCH_SIZE}  (= in-batch negatives, gather off)")
        print(f"  total_steps (per rank) = {total_steps:,}  (EPOCHS={EPOCHS})")
        print(f"  schedule_horizon       = {schedule_total_steps:,}  "
              f"({SCHEDULE_HORIZON_EPOCHS} ep)")
        print(f"  warmup_steps           = {warmup_steps:,} "
              f"({100*WARMUP_RATIO:.1f}% of horizon)")
        print(f"  LR_SCALING             = {LR_BATCH_SCALING}  (ref batch={REF_BATCH})")
        print(f"  lr_scale               = {lr_scale:.4f}  "
              f"(batch/ref = {batch_ratio:g})")
        print(f"  LR_DoRA (base→eff)     = {LR:.2e} → {LR*lr_scale:.2e}")
        print(f"  LR_min_ratio           = {LR_MIN_RATIO}")
        summary["lr_scale"] = lr_scale
        summary["lr_batch_scaling"] = LR_BATCH_SCALING
        summary["lr_ref_batch"] = REF_BATCH
        summary["effective_lrs"] = {
            g.get("name", f"g{i}"): float(g["lr"])
            for i, g in enumerate(optimizer.param_groups)
        }

        _pad_id = getattr(tokenizer, "pad_token_id", None)
        # key padding needs pad_token_id; None makes pad tokens count as real
        assert (not FINEGRAIN_ENABLED) or (_pad_id is not None), \
            "FINEGRAIN_ENABLED but tokenizer.pad_token_id is None - the FILIP key padding cannot be built."
        joint = JointEncoder(
            model, fg_enabled=FINEGRAIN_ENABLED, flair_enabled=FLAIR_ENABLED,
            pad_id=_pad_id,
        ).to(device)
        train_model = joint

        tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(tb_dir)
        except Exception as e:
            print(f"  [warn] tensorboard unavailable: {e}")
            tb = None

        print("\n" + "-" * 64)
        print(f"  Training  (single-GPU)")
        print("-" * 64)

        unwrapped = unwrap_base(train_model)
        logit_scale = getattr(unwrapped, "logit_scale", None)
        logit_bias = getattr(unwrapped, "logit_bias", None)

        logit_params_set = {id(p) for p in get_logit_params(model)}
        nonlogit_params = [p for p in model.parameters()
                           if p.requires_grad and id(p) not in logit_params_set]
        logit_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) in logit_params_set]
        all_trainable = nonlogit_params + logit_params

        ema = None
        if EMA_ENABLED:
            half_life_steps = max(1, int(round(EMA_HALF_LIFE_EPOCHS * steps_per_epoch)))
            ema_decay = 0.5 ** (1.0 / half_life_steps)
            ema = ParamEMA(all_trainable, decay=ema_decay)
            print(f"  [EMA] enabled, decay={ema_decay:.6f} "
                  f"(half-life={EMA_HALF_LIFE_EPOCHS}ep = {half_life_steps:,} steps), "
                  f"start_epoch={EMA_START_EPOCH}")
            if POSITION_PRETRAINED_FREEZE:
                _emb = unwrapped.text_model.embeddings
                _snap = getattr(_emb, "_pretrained_pos_snapshot", None)
                _n_freeze = getattr(_emb, "_pretrained_pos_n_freeze", None)
                if _snap is None or _n_freeze is None:
                    raise RuntimeError(
                        "POSITION_PRETRAINED_FREEZE=True but the _pretrained_pos_snapshot / "
                        "_pretrained_pos_n_freeze buffers are missing.")
                _pos_param = _emb.position_embedding.weight
                ema.register_frozen_rows(_pos_param, _snap, _n_freeze)
                print(f"  [EMA] frozen rows [0..{_n_freeze-1}] registered")

        if resuming:
            print(f"\n  [resume] loading training state from {resume_path}")
            rs = load_resume_state(
                ckpt_path=resume_path,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                ema=ema, device=device,
            )
            start_epoch        = rs["start_epoch"]
            global_step        = rs["global_step"]
            best_score         = rs["best_score"]
            best_epoch         = rs["best_epoch"]
            best_metrics       = rs["best_metrics"]
            early_stop_counter = rs["early_stop_counter"]
            last_reset_score   = rs["last_reset_score"]
            print(f"  [resume] resumed: start_epoch={start_epoch} "
                  f"global_step={global_step} best={best_score:.3f}@ep{best_epoch} "
                  f"es_counter={early_stop_counter}")
            summary["resumed_at_start_epoch"] = start_epoch
            summary["resumed_global_step"]    = global_step
            summary["resumed_best_score"]     = best_score
            summary["resumed_best_epoch"]     = best_epoch

        train_log = defaultdict(float)
        t_step_prev = time.time()

        for epoch in range(start_epoch, EPOCHS + 1):
            train_model.train()
            t_ep = time.time()
            train_log.clear()
            t_step_prev = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # advances on every __iter__, so no reseed is needed

            if ema is not None and epoch == EMA_START_EPOCH:
                ema.reset_to_current()
                print(f"  [EMA] shadow ← current weight "
                      f"(EMA_START_EPOCH={EMA_START_EPOCH})")

            print(f"  epoch {epoch}: vision=frozen")

            ep_normal_cnt = 0
            ep_anom_cnt = 0
            ep_action_cnt = defaultdict(int)

            loop = tqdm(train_loader, desc=f"  Ep{epoch}/{EPOCHS}",
                        leave=False, dynamic_ncols=True,
                        disable=False)
            ep_soft_diag = _init_soft_diag_accumulator()

            for step, batch in enumerate(loop, 1):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                image_ids = batch["image_ids"].to(device, non_blocking=True)
                scene_ids_txt = batch["scene_ids_txt"].to(device, non_blocking=True)
                scene_ids_img = batch["scene_ids_img"].to(device, non_blocking=True)
                action_ids_txt = batch["action_ids_txt"].to(device, non_blocking=True)
                action_ids_img = batch["action_ids_img"].to(device, non_blocking=True)

                b_normal = sum(1 for x in batch["label_types"] if x == "normal")
                b_anom = len(batch["label_types"]) - b_normal
                ep_normal_cnt += b_normal
                ep_anom_cnt += b_anom
                for a in batch["actions"]:
                    ep_action_cnt[a] += 1

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                    v, t, aux = train_model(pixel_values, input_ids, attention_mask)
                    v = _to_tensor(v); t = _to_tensor(t)
                    fg_tok    = aux.get("fg_tok")
                    fg_patch  = aux.get("fg_patch")
                    fg_keypad = aux.get("fg_keypad")
                    flair_S   = aux.get("flair_S")
                    want_soft_diag = (
                        USE_SOFT_LABEL and SOFT_LABEL_DIAG_ENABLED
                        and ((step % LOG_EVERY_STEPS == 0) or (step == 1))
                    )

                    loss_main, soft_diag = compute_step_loss(
                        v, t, image_ids,
                        action_ids_txt, action_ids_img,
                        scene_ids_txt, scene_ids_img,
                        logit_scale=logit_scale, logit_bias=logit_bias,
                        want_soft_diag=want_soft_diag,
                        row_normalizer=None,
                    )
                    if soft_diag is not None:
                        _accumulate_soft_diag(ep_soft_diag, soft_diag)

                    if (ENABLE_PROBE_DIVERSITY_REG and NUM_PROBES > 1
                            and epoch > PROBE_DIVERSITY_WARMUP_EPOCHS):
                        loss_div = probe_diversity_loss(unwrapped)
                        loss = loss_main + PROBE_DIVERSITY_REG_LAMBDA * loss_div
                        loss_div_val = float(loss_div.detach().item())
                    else:
                        loss = loss_main
                        loss_div_val = 0.0

                    if UNIFORMITY_REG_ENABLED:
                        loss_uni = 0.5 * (uniformity_loss(v, UNIFORMITY_REG_T)
                                          + uniformity_loss(t, UNIFORMITY_REG_T))
                        loss = loss + UNIFORMITY_REG_LAMBDA * loss_uni
                        loss_uni_val = float(loss_uni.detach().item())
                    else:
                        loss_uni_val = 0.0

                    if FINEGRAIN_ENABLED and fg_tok is not None:
                        loss_fg, fg_acc = finegrain_filip_loss(fg_tok, fg_patch, fg_keypad, image_ids)
                        loss = loss + FINEGRAIN_WEIGHT * loss_fg
                        loss_fg_val = float(loss_fg.detach().item())
                        fg_acc_val = float(fg_acc.item())
                    else:
                        loss_fg_val = 0.0
                        fg_acc_val = 0.0

                    _lam = FLAIR_WEIGHT  # default; guards against a _lam NameError
                    if FLAIR_ENABLED and flair_S is not None:
                        loss_flair, fg_acc = flair_tcs_loss(flair_S, image_ids)
                        _W = FLAIR_WEIGHT_WARMUP_EPOCHS
                        if _W > 0 and epoch <= _W:  # epoch is 1-indexed (verified)
                            _frac = epoch / max(1, _W)
                            _lam = FLAIR_WEIGHT_WARMUP_START + (FLAIR_WEIGHT - FLAIR_WEIGHT_WARMUP_START) * _frac
                        else:
                            _lam = FLAIR_WEIGHT
                        loss = loss + _lam * loss_flair
                        loss_fg_val = float(loss_flair.detach().item())
                        fg_acc_val  = float(fg_acc.item())
                        if (not math.isfinite(loss_fg_val)) or loss_fg_val > FLAIR_LOSS_ABORT_THRESH:
                            raise HaltFlairLossExplosion(
                                f"FLAIR loss_tcs={loss_fg_val:.2f} > {FLAIR_LOSS_ABORT_THRESH} "
                                f"(step {global_step}) - pooler self-reinforcement suspected")

                if _FG_OR_FLAIR and (step == 1 or step % LOG_EVERY_STEPS == 0):
                    _tag = "flair" if FLAIR_ENABLED else "finegrain"
                    _lname = "loss_tcs" if FLAIR_ENABLED else "loss_filip"
                    print(f"    [{_tag}] step{step} loss_total={loss.item():.4f} "
                          f"loss_main={float(loss_main.detach()):.4f} "
                          f"{_lname}={loss_fg_val:.4f} fg_acc(local)={fg_acc_val:.3f} "
                          f"(λ={_lam if FLAIR_ENABLED else FINEGRAIN_WEIGHT})", flush=True)

                if FINEGRAIN_ENABLED and step == 1:
                    _Nt, _Lt, _Ni = fg_tok.shape[0], fg_tok.shape[1], fg_patch.shape[0]
                    _Lp = _fg_patch_pool(fg_patch[:1].detach(), FINEGRAIN_PATCH_POOL).shape[1]
                    _elem = _Nt * _Lt * _Ni * _Lp
                    _peak = torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0.0
                    print(f"    [fg-mem] sim=(Nt{_Nt},Lt{_Lt},Ni{_Ni},Lp{_Lp})={_elem/1e6:.0f}M "
                          f"(~{_elem*4/1024**3:.2f}GB fp32)  fwd-peak={_peak:.1f}GB (backward excluded)", flush=True)

                if FLAIR_ENABLED and step == 1 and torch.cuda.is_available():
                    print(f"    [flair-mem] S=(Nt{flair_S.shape[0]},Ni{flair_S.shape[1]}) "
                          f"fwd-peak={torch.cuda.max_memory_allocated()/1024**3:.1f}GB", flush=True)

                loss_val = loss.item()
                # tolerant in ep1 only, so a fresh-init loss_main (~8-9) does not false-abort
                # from ep2 tighten to 1.0 (eff 8.0) so a collapse (~37) fails fast
                _loss_thresh_scale = LOSS_EXPLOSION_EP1_SCALE if epoch == 1 else 1.0
                # collapse detection uses loss_main, not combined
                _guard_loss = float(loss_main.detach().item()) if _FG_OR_FLAIR else loss_val
                _check_loss_safety(_guard_loss, global_step, threshold_scale=_loss_thresh_scale, nan_val=loss_val)

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                # peak right after backward (a forward-only reading under-reports)
                if FINEGRAIN_ENABLED and step == 1 and torch.cuda.is_available():
                    print(f"    [fg-mem] post-backward peak="
                          f"{torch.cuda.max_memory_allocated()/1024**3:.1f}GB (exact OOM margin)", flush=True)

                if FLAIR_ENABLED and step == 1 and torch.cuda.is_available():
                    print(f"    [flair-mem] post-backward peak="
                          f"{torch.cuda.max_memory_allocated()/1024**3:.1f}GB (exact OOM margin)", flush=True)

                # verify after backward so GC recompute does not distort the peak reading
                if _FG_OR_FLAIR and step == 1:
                    # checks pooler_output == get_*_features (a warning, not an assert)
                    # compared in eval mode (dropout off)
                    _was_train = unwrapped.training
                    unwrapped.eval()
                    try:
                        with torch.no_grad(), torch.amp.autocast("cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                            _vp = _to_tensor(unwrapped.vision_model(pixel_values=pixel_values).pooler_output)
                            _tp = _to_tensor(unwrapped.text_model(
                                input_ids=input_ids, attention_mask=attention_mask).pooler_output)
                            _vref = _to_tensor(unwrapped.get_image_features(pixel_values=pixel_values))
                            _tref = _to_tensor(unwrapped.get_text_features(
                                input_ids=input_ids, attention_mask=attention_mask))
                        _dv = (_vp.float() - _vref.float()).abs().max().item()
                        _dt = (_tp.float() - _tref.float()).abs().max().item()
                        _tag = "OK" if (_dv < 5e-2 and _dt < 5e-2) else "WARN-MISMATCH"
                        print(f"    [verify] pooler_output vs get_*_features (eval) max|diff| "
                              f"v={_dv:.2e} t={_dt:.2e} [{_tag}]", flush=True)
                    except Exception as _e:
                        print(f"    [verify] skipped: {type(_e).__name__}: {_e}", flush=True)
                    finally:
                        unwrapped.train(_was_train)

                will_log = (step % LOG_EVERY_STEPS == 0) or (step == len(train_loader))
                if will_log:
                    grad_norm_nonlogit = compute_grad_norm(nonlogit_params)
                    grad_norm_logit = compute_grad_norm(logit_params)
                else:
                    grad_norm_nonlogit = None
                    grad_norm_logit = None

                if GRAD_CLIP > 0:
                    total_norm_tensor = torch.nn.utils.clip_grad_norm_(
                        all_trainable, GRAD_CLIP)
                    grad_norm_total = float(total_norm_tensor)
                else:
                    grad_norm_total = compute_grad_norm(all_trainable)

                _spike = (GRAD_SPIKE_SKIP and epoch > 1 and grad_norm_total > GRAD_SPIKE_THRESH)

                if _spike:
                    if use_scaler:
                        scaler.update()  # keeps fp16 consistent (a no-op under bf16)
                    optimizer.zero_grad(set_to_none=True)  # drop the destructive step; the weights are not updated
                    _dump_grad_spike(metric_writer, epoch, global_step,
                                     grad_norm_total, loss_val, batch, image_ids)
                else:
                    if use_scaler:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    if (LOGIT_SCALE_CLAMP_ENABLED and not LOGIT_SCALE_FREEZE
                            and logit_scale is not None):
                        with torch.no_grad():
                            logit_scale.clamp_(max=math.log(LOGIT_SCALE_MAX))
                    scheduler.step()
                    if ema is not None and epoch >= EMA_START_EPOCH:
                        ema.update()
                    # spike-skipped steps are excluded from the fg_acc accumulator
                    if _FG_OR_FLAIR:
                        train_log["fg_acc_sum"] += fg_acc_val
                        train_log["fg_acc_cnt"] += 1
                global_step += 1

                l = loss_val
                train_log["loss"] += l
                train_log["grad_norm_total"] += grad_norm_total
                train_log["count"] += 1

                if will_log:
                    now = time.time()
                    samples_per_sec = (LOG_EVERY_STEPS * BATCH_SIZE) \
                                      / max(1e-9, now - t_step_prev)
                    t_step_prev = now
                    avg = train_log["loss"] / step
                    lrs = _lr_snapshot(scheduler, lr_group_idx)

                    diag = (compute_sim_diagnostics(v.detach(), t.detach(), image_ids=image_ids)
                            if (step % DIAG_EVERY_STEPS == 0) else {})
                    diag["loss_div"] = loss_div_val
                    diag["loss_main"] = float(loss_main.detach().item())
                    diag["loss_uni"] = loss_uni_val
                    if diag.get("pos_sim_mean") is not None:
                        train_log["pos_sim_sum"] += diag["pos_sim_mean"]
                        train_log["neg_sim_max_sum"] += diag["neg_sim_max"]
                        train_log["margin_sum"] += diag.get(
                            "margin", diag["pos_sim_mean"] - diag.get("neg_sim_mean", 0.0))
                        train_log["diag_count"] += 1
                    mem = gpu_mem_stats()
                    batch_normal_ratio = b_normal / max(1, len(batch["label_types"]))

                    _log_train_step(
                        metric_writer=metric_writer, tb=tb,
                        epoch=epoch, step=step, global_step=global_step,
                        loss_val=l, loss_avg=avg, lrs=lrs,
                        grad_norm_total=grad_norm_total,
                        grad_norm_nonlogit=grad_norm_nonlogit,
                        grad_norm_logit=grad_norm_logit,
                        samples_per_sec=samples_per_sec,
                        batch_normal_ratio=batch_normal_ratio,
                        logit_scale=logit_scale, logit_bias=logit_bias,
                        diag=diag, mem=mem,
                    )
                    if grad_norm_total > 50.0:
                        print(f"  [warn] step {global_step}: grad_norm_total="
                              f"{grad_norm_total:.2f} (>50)")
                    if diag.get("margin", 1.0) < 0:
                        print(f"  [warn] step {global_step}: pos<neg margin="
                              f"{diag['margin']:.4f}")
                    loop.set_postfix(
                        loss=f"{avg:.4f}",
                        lr=f"{lrs['dora']:.2e}",
                        gn=f"{grad_norm_total:.2f}",
                        sps=f"{samples_per_sec:.0f}",
                    )
                elif will_log:
                    t_step_prev = time.time()

            ep_loss = train_log["loss"] / max(1, len(train_loader))
            ep_time = time.time() - t_ep
            lr_dora_now = scheduler.get_last_lr()[lr_group_idx.get("dora", 0)]
            _report_epoch_end(epoch, ep_loss, ep_time, train_log,
                                   ep_normal_cnt, ep_anom_cnt, ep_action_cnt,
                                   lr_dora_now, metric_writer)
            _report_soft_label_diag(epoch, ep_soft_diag, metric_writer)
            _report_multi_probe_diag(epoch, unwrapped, metric_writer)

            should_break_early = False
            _dcnt = max(1, int(train_log.get("diag_count", 0)))
            _pos = train_log.get("pos_sim_sum", 0.0) / _dcnt
            _negmax = train_log.get("neg_sim_max_sum", 0.0) / _dcnt
            _hardm = _pos - _negmax
            _meanm = train_log.get("margin_sum", 0.0) / _dcnt
            _gnorm = train_log["grad_norm_total"] / max(1, train_log["count"])
            _ls_exp = float(logit_scale.detach().exp()) if logit_scale is not None else None
            _fgacc = train_log.get("fg_acc_sum", 0.0) / max(1, int(train_log.get("fg_acc_cnt", 0)))
            # FLAIR only: early detection of pooled v,t corruption (trend only)
            _flair_nn = {}
            if FLAIR_ENABLED:
                with torch.no_grad():
                    _vn = F.normalize(v.detach().float(), dim=-1)
                    _tn = F.normalize(t.detach().float(), dim=-1)
                    _vs = (_vn @ _vn.t()).fill_diagonal_(-1).max(dim=1).values.mean().item()
                    _ts = (_tn @ _tn.t()).fill_diagonal_(-1).max(dim=1).values.mean().item()
                    _flair_nn = {"flair_pooled_v_nn_sim": _vs, "flair_pooled_t_nn_sim": _ts}
                    print(f"  [flair] pooled nn_sim  v={_vs:.4f}  t={_ts:.4f}  "
                          f"(a jump against the FILIP run means pooler self-reinforcement)")
            metric_writer.log(
                event="train_progress", epoch=epoch,
                train_loss=ep_loss, grad_norm=_gnorm,
                pos_sim_mean=_pos, neg_sim_max=_negmax,
                hard_margin=_hardm, mean_margin=_meanm,
                fg_inbatch_acc_local=_fgacc,  # FILIP local in-batch acc (no gather, so trend only)
                logit_scale_exp=_ls_exp, diag_samples=_dcnt,
                **_flair_nn,
            )
            print(f"  [train-progress] ep{epoch}: loss={ep_loss:.4f} "
                  f"hard_margin={_hardm:.4f} (pos={_pos:.4f} neg_max={_negmax:.4f}) "
                  f"fg_acc(local)={_fgacc:.3f} grad_norm={_gnorm:.3f}")

            if KEEP_LAST_N_CKPT != 0:
                _ema_active = (ema is not None and epoch >= EMA_START_EPOCH)
                if _ema_active:
                    ema.apply_shadow()  # the ckpt stores the EMA shadow weights
                save_checkpoint(
                    unwrapped, ckpt_dir, f"ep{epoch:02d}",
                    extra=build_ckpt_extra(epoch, num_gpus=1,
                                           ema_used=_ema_active),
                )
                if _ema_active:
                    ema.restore()
                try:
                    ep_dirs = sorted(
                        d for d in os.listdir(ckpt_dir)
                        if d.startswith("ep") and len(d) == 4 and d[2:].isdigit()
                           and os.path.isdir(f"{ckpt_dir}/{d}")
                    )
                    if KEEP_LAST_N_CKPT > 0 and len(ep_dirs) > KEEP_LAST_N_CKPT:
                        for stale in ep_dirs[:-KEEP_LAST_N_CKPT]:
                            try:
                                shutil.rmtree(f"{ckpt_dir}/{stale}")
                                print(f"  [ckpt-rolling] removed stale ckpt: {stale} "
                                      f"(keep last {KEEP_LAST_N_CKPT})")
                            except Exception as _rm_e:
                                print(f"  [ckpt-rolling] warn: failed to remove "
                                      f"{stale}: {_rm_e}")
                except Exception as _ls_e:
                    print(f"  [ckpt-rolling] warn: cleanup skipped: {_ls_e}")

            try:
                save_checkpoint(
                    unwrapped, ckpt_dir, "last",
                    extra=build_ckpt_extra(
                        epoch, num_gpus=1,
                        resume_enabled=True,
                    ),
                )
                save_resume_state(
                    ckpt_dir=ckpt_dir, epoch=epoch, global_step=global_step,
                    best_score=best_score, best_epoch=best_epoch,
                    best_metrics=best_metrics,
                    early_stop_counter=early_stop_counter,
                    last_reset_score=last_reset_score,
                    optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, ema=ema,
                )
            except Exception as _rs_e:
                print(f"  [resume] warn: save_resume_state failed at ep{epoch}: "
                      f"{type(_rs_e).__name__}: {_rs_e}")

            if should_break_early:
                print(f"\n  🛑 [EARLY STOP] best @ ep{best_epoch} = "
                      f"{best_score:.3f}, no improvement for "
                      f"{EARLY_STOP_PATIENCE} epochs → stopping early")
                early_stopped = True
                break

        final_epoch_meta = locals().get("epoch", EPOCHS)
        save_checkpoint(
            unwrap_base(train_model), ckpt_dir, "last",
            extra=build_ckpt_extra(
                final_epoch_meta,
                early_stopped=early_stopped,
                num_gpus=1,
            ),
        )
        if POSITION_PRETRAINED_FREEZE:
            print(f"  [pos-freeze-verify] ✅ pretrained position "
                  f"[0..{POSITION_PRETRAINED_SIZE-1}] unchanged for the whole run")
        print(f"\n  Done  (total {global_step} steps)")
        if tb:
            tb.close()

        summary.update({
            "ended": datetime.now().isoformat(timespec="seconds"),
            "total_steps": global_step,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "status": "completed",
        })

    except KeyboardInterrupt:
        print("\n  [interrupt] KeyboardInterrupt — partial run saved")
        metric_writer.log(event="run_interrupted")
        summary.update({
            "ended": datetime.now().isoformat(timespec="seconds"),
            "status": "interrupted",
        })

    except TrainHalt as e:
        tb_str = traceback.format_exc()
        print(f"\n  🛑 [HALT:{e.reason}] {e}\n{tb_str}")
        try:
            with open(f"{run_dir}/crash.txt", "w") as f:
                f.write(tb_str)
        except Exception:
            pass
        metric_writer.log(event="run_halted", halt_reason=e.reason, error=str(e))
        summary.update({
            "ended": datetime.now().isoformat(timespec="seconds"),
            "status": "halted",
            "halt_reason": e.reason,
            "error": str(e),
        })

    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"\n  [ERROR] {e}\n{tb_str}")
        try:
            with open(f"{run_dir}/crash.txt", "w") as f:
                f.write(tb_str)
        except Exception:
            pass
        metric_writer.log(event="run_crashed", error=str(e))
        summary.update({
            "ended": datetime.now().isoformat(timespec="seconds"),
            "status": "crashed",
            "error": str(e),
        })

    finally:
        if summary.get("status") != "completed":
            try:
                _locals = locals()
                if "train_model" in _locals and "ckpt_dir" in _locals:
                    save_checkpoint(
                        unwrap_base(_locals["train_model"]), ckpt_dir, "last",
                        extra=build_ckpt_extra(
                            _locals.get("epoch", 0),
                            early_stopped=_locals.get("early_stopped", False),
                            status=summary.get("status", "unknown"),
                            num_gpus=1,
                            resume_enabled=True,
                        ),
                    )
                    if all(k in _locals for k in ("optimizer", "scheduler")):
                        try:
                            save_resume_state(
                                ckpt_dir=ckpt_dir,
                                epoch=_locals.get("epoch", 0),
                                global_step=_locals.get("global_step", 0),
                                best_score=_locals.get("best_score", -1.0),
                                best_epoch=_locals.get("best_epoch", -1),
                                best_metrics=_locals.get("best_metrics", {}),
                                early_stop_counter=_locals.get("early_stop_counter", 0),
                                last_reset_score=_locals.get("last_reset_score", -1.0),
                                optimizer=_locals["optimizer"],
                                scheduler=_locals["scheduler"],
                                scaler=_locals.get("scaler"),
                                ema=_locals.get("ema"),
                            )
                        except Exception as _rs_e:
                            print(f"  [resume] warn: finally save_resume_state failed: "
                                  f"{type(_rs_e).__name__}: {_rs_e}")
            except Exception:
                pass
        try:
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception:
            pass
        try:
            metric_writer.close()
        except Exception:
            pass
        try:
            if tee is not None:
                tee.restore()
        except Exception:
            pass

def _run_single(physical_gpu: int, extra_run_note: str, resume_path: str = ""):
    """Single-GPU training entry point; torch.distributed is not used (no DDP, spawn or gather)."""
    torch.cuda.set_device(physical_gpu)
    train_single(physical_gpu=physical_gpu,
                 extra_run_note=extra_run_note, resume_path=resume_path)


def check_inputs() -> None:
    """Check that every training input is present. The manifest is not built here."""
    manifest = MANIFEST_PATH
    problems = []

    if not os.path.isdir(IMG_ROOT):
        problems.append(f"image root not found: {IMG_ROOT}")
    if not os.path.exists(manifest):
        problems.append(f"manifest not found: {manifest}\n"
                        "      build: python train/gen/gen_manifest.py")
    model_dir = os.path.join(os.environ.get("HF_HOME", HF_CACHE_DIR), "hub",
                             "models--" + MODEL_NAME.replace("/", "--"))
    if not os.path.isdir(model_dir):
        problems.append(f"HF model cache not found: {model_dir}")

    if problems:
        raise SystemExit(
            "[input check failed] fix the following and run again.\n"
            + "\n".join(f"  - {p}" for p in problems)
            + f"\n  current working directory: {os.getcwd()}\n"
              f"  (paths are relative to the repository root - run from there)")

    if os.path.exists(RECAP_CSV_PATH) and \
       os.path.getmtime(RECAP_CSV_PATH) > os.path.getmtime(manifest):
        print("  [warn] the caption CSV is newer than the manifest. Rebuild recommended:\n"
              "         python train/gen/gen_manifest.py --force")
    print(f"[check] inputs ok - manifest {os.path.basename(manifest)} "
          f"({os.path.getsize(manifest) / 2 ** 30:.2f} GiB), images {IMG_ROOT}")


def main():
    parser = argparse.ArgumentParser(
        description="SigLIP2-L/16-512 + DoRA + FILIP - full split (single GPU)")
    parser.add_argument("--gpu", type=int, default=0, help="single GPU id")
    parser.add_argument("--run-note", type=str, default="",
                        help="run directory name (default {timestamp}_{directory name})")
    parser.add_argument("--resume", type=str, default="",
                        help="path to a previous run's last ckpt (a run dir or .../checkpoints/last). "
                             "Restores optimizer/scheduler/EMA/RNG/counters and resumes from the next epoch.")
    args = parser.parse_args()
    check_inputs()



    resume_path = ""
    if args.resume:
        resume_path = _resolve_resume_ckpt_dir(args.resume)
        print(f"[launcher] --resume active, last ckpt = {resume_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    avail = torch.cuda.device_count()
    if args.gpu < 0 or args.gpu >= avail:
        raise ValueError(f"--gpu {args.gpu} invalid (available: 0..{avail-1})")

    print(f"[launcher] single GPU={args.gpu}  (no DDP/spawn, gather off, "
          f"BATCH_SIZE={BATCH_SIZE})")
    _run_single(args.gpu, args.run_note, resume_path)

if __name__ == "__main__":
    main()
