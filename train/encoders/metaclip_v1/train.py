"""
MetaCLIP v1 ViT-L-14-worldwide-xlmv fine-tuning on PAB (open_clip 3.3.x)
Trained with the CLIP contrastive loss (InfoNCE)

Two-run protocol (same as beit3 and metaclip2), epoch budget fixed at 4:
  run A (heldout)  EXCLUDE_HELDOUT=1 (default) -> per-epoch ckpts, pick e* with eval_heldout_openclip.py
  run B (all)      EXCLUDE_HELDOUT=0           -> adopt e* as-is, no scoring
  Both runs share the epoch budget so an epoch number means the same thing in each, which is what
  makes e* transferable. SAVE_DIR / LOG_FILE also default per mode, otherwise run B would overwrite
  run A's `epoch_{N}.pt` under the same name.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train.py                                   # run A
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py            # run A (DDP)
  EXCLUDE_HELDOUT=0 torchrun --nproc_per_node=2 train.py                   # run B (all)
  python train.py --train-csv /path/to/recap_or_merged.csv --epochs 4
"""
from __future__ import annotations
import argparse
import csv
import sys
import math
import os
import time

# Required for the xlm-v-base tokenizer conversion (sentencepiece -> fast, protobuf); avoids
# "Descriptors cannot be created directly". Must be set before open_clip/transformers is imported.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import open_clip

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "ViT-L-14-worldwide-xlmv"       # custom config matching the checkpoint vocab of 901,629 (xlm-v-base)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Register the custom open_clip config (ViT-L-14-worldwide-xlmv) from the repository copy rather
# than relying on the venv's model_configs. See register.py for the config itself.
sys.path.insert(0, os.path.join(_REPO, "assets", "model", "vlm_models", "MetaCLIP-L14-worldwide"))
import register as _oc_register  # noqa: E402,F401
PRETRAINED_WEIGHT = f"{_REPO}/assets/model/vlm_models/MetaCLIP-L14-worldwide/l14_worldwide.pt"
# ViT-L-14-worldwide-xlmv is not an open_clip built-in, so add the bundled config JSON (vocab 901,629 xlm-v)
_MC_CONFIG = f"{_REPO}/assets/model/vlm_models/MetaCLIP-L14-worldwide/ViT-L-14-worldwide-xlmv.json"
if os.path.exists(_MC_CONFIG):
    open_clip.add_model_config(_MC_CONFIG)
# Tokenizer: the custom config points at facebook/xlm-v-base (vocab 901,629, matching the model's
# token_embedding; mt5-base with 250K would be wrong).
TRAIN_CSV = os.environ.get("TRAIN_CSV", f"{_REPO}/assets/data/manifest/train.csv")
# ---- held-out bench exclusion (heldout_v1; md5 and gates are enforced) ----
HELDOUT_DIR     = os.environ.get("HELDOUT_DIR",
                                 os.environ.get("PAB_DATA_INFRA",
                                                f"{_REPO}/assets/data") + "/heldout_v1")
EXCLUDE_HELDOUT = os.environ.get("EXCLUDE_HELDOUT", "1").lower() not in ("0", "false", "no")

EPOCHS      = int(os.environ.get("EPOCHS", 4))
# The output paths depend on the mode; a fixed path would let run B overwrite run A's epoch_{N}.pt.
_RUN        = "heldout" if EXCLUDE_HELDOUT else "all"
SAVE_DIR    = os.environ.get("SAVE_DIR",
                             f"{_REPO}/assets/runs/metaclip_v1_{_RUN}/checkpoints")
LOG_FILE    = os.environ.get("LOG_FILE", f"{_REPO}/assets/runs/metaclip_v1_{_RUN}/logs/train_loss.log")


# heldout_bench (train/encoders/eval) is the single source for the bench definition, md5 and gates
def _heldout_bench_dir(start):
    """Locate the directory holding `heldout_bench.py`, so a moved file still resolves."""
    import os as _o
    d = start
    for _ in range(5):
        for cand in (_o.path.join(d, "eval"), d):
            if _o.path.exists(_o.path.join(cand, "heldout_bench.py")):
                return cand
        d = _o.path.dirname(d)
    raise ImportError("cannot find heldout_bench.py (check train/encoders/eval/)")


sys.path.insert(0, _heldout_bench_dir(os.path.dirname(os.path.abspath(__file__))))
import heldout_bench as HB                                           # noqa: E402


# ── Dataset ───────────────────────────────────────────────────────────────────
class PABDataset(Dataset):
    def __init__(self, csv_path: str, preprocess, tokenizer, max_text_len: int = 77):
        self.preprocess  = preprocess
        self.tokenizer   = tokenizer
        self.max_text_len = max_text_len
        self.samples: list[tuple[str, str]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filepath"], row["title"]))
        print(f"Dataset: {len(self.samples):,} samples (before exclusion)")
        if EXCLUDE_HELDOUT:
            # Drop the held-out bench images. A list md5 or gate mismatch aborts.
            self.samples = HB.filter_csv_rows(self.samples, path_key=0,
                                              heldout_dir=HELDOUT_DIR, require=True)
        else:
            print("  [heldout] EXCLUDE_HELDOUT=0 → no exclusion (bench numbers will be optimistically biased)")
        print(f"Dataset: {len(self.samples):,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        try:
            image = self.preprocess(Image.open(img_path).convert("RGB"))
        except Exception:
            image = torch.zeros(3, 224, 224)  # ViT-L-14 input: 224x224
        tokens = self.tokenizer([caption])[0]
        return image, tokens


# ── Loss ──────────────────────────────────────────────────────────────────────
def clip_loss(image_feats: torch.Tensor, text_feats: torch.Tensor,
              logit_scale: torch.Tensor) -> torch.Tensor:
    logits = logit_scale.exp() * image_feats @ text_feats.t()
    labels = torch.arange(len(logits), device=logits.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return (loss_i + loss_t) / 2


# ── Input check ─────────────────────────────────────────────────────────────
def check_inputs(args) -> None:
    """Validate every input before a GPU is touched: report all missing items at once, no fallback."""
    missing = []
    if not os.path.exists(args.train_csv):
        missing.append(f"training CSV: {args.train_csv}\n      build: python train/gen/gen_metaclip_v1_csv.py")
    if not os.path.exists(PRETRAINED_WEIGHT):
        missing.append(f"pretrained weights: {PRETRAINED_WEIGHT}")
    if not os.path.exists(_MC_CONFIG):
        missing.append(f"open_clip config: {_MC_CONFIG}")
    if EXCLUDE_HELDOUT and not os.path.isdir(HELDOUT_DIR):
        missing.append(f"heldout bench: {HELDOUT_DIR}\n      build: python train/gen/gen_heldout_v1.py")
    if missing:
        raise SystemExit("[check_inputs] stopping, the following inputs are missing:\n  - " + "\n  - ".join(missing))


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1

    check_inputs(args)

    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_master = (local_rank == 0)

    # Build the model empty, then load the local weights directly (avoids weights_only=True)
    if is_master:
        print(f"Loading {MODEL_NAME} (no pretrained yet) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=None, device=device
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    if is_master:
        print(f"Loading weights: {PRETRAINED_WEIGHT}")
    state = torch.load(PRETRAINED_WEIGHT, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if is_master:
        print(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:   print(f"  missing[:3]:    {missing[:3]}")
        if unexpected:print(f"  unexpected[:3]: {unexpected[:3]}")
    model = model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    dataset = PABDataset(args.train_csv, preprocess, tokenizer)
    sampler = DistributedSampler(dataset) if distributed else None
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=int(os.environ.get("NUM_WORKERS", 8)),   # fork-after-CUDA can deadlock on some boxes; drop to 0-2 if the first batch never arrives
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )
    total_steps  = len(loader) * args.epochs
    warmup_steps = args.warmup
    scaler = GradScaler("cuda")

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if is_master:      # same schema as beit3 and metaclip2, so run A/B is identifiable from the artifacts alone
        import json as _json
        with open(os.path.join(os.path.dirname(SAVE_DIR), "heldout_exclusion.json"), "w",
                  encoding="utf-8") as _f:
            _json.dump({"run": "A" if EXCLUDE_HELDOUT else "B",
                        "exclude_heldout": EXCLUDE_HELDOUT, "heldout_dir": HELDOUT_DIR,
                        "epochs": args.epochs, "train_csv": args.train_csv,
                        "n_samples": len(dataset)}, _f, indent=2, ensure_ascii=False)
        print(f"[run] {'A (heldout excluded)' if EXCLUDE_HELDOUT else 'B (all)'} · "
              f"epochs={args.epochs} · save={SAVE_DIR}")

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        m = model.module if distributed else model
        m.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        steps_done = (start_epoch - 1) * len(loader)
        for _ in range(steps_done):
            scheduler.step()
        if is_master:
            print(f"Resumed from {args.resume} → start_epoch={start_epoch}, scheduler advanced {steps_done} steps")

    log_f = open(LOG_FILE, "a" if args.resume else "w") if is_master else None

    if is_master:
        print(f"Train CSV: {args.train_csv}")
        print(f"Steps/epoch: {len(loader):,}  Total: {total_steps:,}")
        print(f"Batch: {args.batch_size}  LR: {args.lr}  Warmup: {warmup_steps}")

    step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        if distributed:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        epoch_loss = 0.0

        for i, (images, tokens) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)

            with autocast("cuda"):
                m = model.module if distributed else model
                img_f = F.normalize(m.encode_image(images), dim=-1)
                txt_f = F.normalize(m.encode_text(tokens),  dim=-1)
                loss  = clip_loss(img_f, txt_f, m.logit_scale)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            step += 1

            epoch_loss += loss.item()
            if is_master and (i + 1) % 100 == 0:
                avg = epoch_loss / (i + 1)
                lr  = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                msg = (f"Epoch {epoch}/{args.epochs}  "
                       f"Step {i+1}/{len(loader)}  "
                       f"Loss={avg:.4f}  LR={lr:.2e}  "
                       f"Time={elapsed:.0f}s")
                print(msg)
                log_f.write(msg + "\n")
                log_f.flush()

        if is_master:
            m = model.module if distributed else model
            ckpt = {
                "epoch": epoch,
                "state_dict": m.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            path = os.path.join(SAVE_DIR, f"epoch_{epoch}.pt")
            torch.save(ckpt, path)
            avg_loss = epoch_loss / len(loader)
            print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}  saved: {path}")
            if log_f:
                log_f.write(f"EPOCH {epoch} avg_loss={avg_loss:.4f}\n")
                log_f.flush()

    if log_f:
        log_f.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv",  type=str, default=TRAIN_CSV)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--lr",         type=float, default=5e-6)
    parser.add_argument("--wd",         type=float, default=0.2)
    parser.add_argument("--warmup",     type=int, default=1000)
    parser.add_argument("--resume",     type=str, default="", help="checkpoint .pt to resume from")
    args = parser.parse_args()
    main(args)
