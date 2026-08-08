#!/usr/bin/env python3
"""LLM2CLIP vision-LoRA trainer (stage 1 of the anchor5 adapter stack).

Trains a LoRA over every Linear in the vision tower against frozen text features
from `build_text_cache.py`, with hard negatives on both directions:
  i2t : image vs (batch text + a random slice of the cached train-text bank)
  t2i : text  vs (batch image + a detached image memory bank)

Checkpoint selection is train-internal: the cache's tail is split off as a
val gallery (10k) with 2k self-retrieval queries, and the best val score is
saved to OUT_DIR. Nothing outside the training pool is scored.

Output: OUT_DIR (default assets/runs/llm2clip_vision_lora) — the shipped stage-1
adapter is `assets/model/encoder/llm2clip_lora_v3_best`.

env: track4_llm2clip. Run from the repository root:
  LLM2CLIP_DEV=cuda:6 python train/encoders/llm2clip_anchor5/train_vision.py
"""
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"))

DEV = os.environ.get("LLM2CLIP_DEV", "cuda:0")
VIS = os.environ.get("LLM2CLIP_BASE", "microsoft/LLM2CLIP-Openai-L-14-336")
CACHE = os.environ.get("TEXT_CACHE", f"{_REPO}/assets/data/mining/llm2clip_text_cache.pt")
IMG_ROOT = os.environ.get("PAB_TRAIN", f"{_REPO}/assets/data/raw/pab_train")
OUT_DIR = os.environ.get("OUT_DIR", f"{_REPO}/assets/runs/llm2clip_vision_lora")
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BS = int(os.environ.get("BS", "64"))
LR = float(os.environ.get("LR", "1e-4"))
N_TXT_BANK = int(os.environ.get("N_TXT_BANK", "16384"))
M_IMG_BANK = int(os.environ.get("M_IMG_BANK", "8192"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "400"))
LORA_R = int(os.environ.get("LORA_R", "32"))   # the shipped stage-1 adapter was trained at r=32
N_TRAIN_POOL = int(os.environ.get("N_TRAIN_POOL", "140000"))
N_VALGAL = int(os.environ.get("N_VALGAL", "10000"))
N_VALQ = int(os.environ.get("N_VALQ", "2000"))


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


if not os.path.exists(CACHE):
    raise SystemExit(f"[train_vision] text cache not found: {CACHE}\n"
                     "  build it first: python train/encoders/llm2clip_anchor5/build_text_cache.py")

# FA3 probing in the remote code breaks without the wheel; keep FA2/sdpa.
import transformers.utils as _u                                   # noqa: E402
import transformers.utils.import_utils as _iu                     # noqa: E402
for _m in (_iu, _u):
    if hasattr(_m, "is_flash_attn_3_available"):
        setattr(_m, "is_flash_attn_3_available", lambda *a, **k: False)
if hasattr(_iu, "_flash_attn_3_available"):
    _iu._flash_attn_3_available = False
from transformers import AutoModel                                # noqa: E402
from peft import LoraConfig, get_peft_model                       # noqa: E402

# cache split: train pool / val gallery / val self-retrieval queries
c = torch.load(CACHE, map_location="cpu", weights_only=False)
paths_all = c["train_paths"]
text_all = F.normalize(c["train_text"].float(), dim=-1)
tr_paths = paths_all[:N_TRAIN_POOL]
tr_text = text_all[:N_TRAIN_POOL]
val_paths = paths_all[N_TRAIN_POOL:N_TRAIN_POOL + N_VALGAL]
val_qtext = text_all[N_TRAIN_POOL:N_TRAIN_POOL + N_VALQ]
log(f"train pool {len(tr_paths):,} | val gallery {len(val_paths):,} / queries {val_qtext.shape[0]:,}")
tr_text_dev = tr_text.to(DEV)

model = AutoModel.from_pretrained(VIS, torch_dtype=torch.bfloat16, trust_remote_code=True,
                                  attn_implementation="sdpa").to(DEV)
vis_linears = [n for n, m in model.named_modules()
               if isinstance(m, nn.Linear) and any(k in n.lower() for k in ("visual", "vision"))]
for p in model.parameters():
    p.requires_grad_(False)
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                                         bias="none", target_modules=vis_linears))
ls = next(p for n, p in model.named_parameters() if n.endswith("logit_scale"))
ls.requires_grad_(True)
log(f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
    f"| vision LoRA over {len(vis_linears)} Linear")

_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)
tfm = T.Compose([T.Resize(336, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(336),
                 T.ToTensor(), T.Normalize(_MEAN, _STD)])


def _resolve(p):
    """Manifest paths are 'train/imgs_X/...'; on disk that is train_jpg_512/Part {X//8+1}/imgs_X/..."""
    if os.path.isabs(p):
        return p
    m = re.match(r"train/imgs_(\d+)/(.+)", p)
    if m:
        x = int(m.group(1))
        return f"{IMG_ROOT}/train_jpg_512/Part {x // 8 + 1}/imgs_{x}/{m.group(2)}"
    return os.path.join(IMG_ROOT, p)


_MISS = {"n": 0}


def loadimg(p):
    try:
        return tfm(Image.open(_resolve(p)).convert("RGB"))
    except Exception:
        _MISS["n"] += 1
        if _MISS["n"] in (1, 100) or _MISS["n"] % 10000 == 0:
            log(f"warning: unreadable image #{_MISS['n']}: {p}")
        return torch.zeros(3, 336, 336)


class DS(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return loadimg(self.paths[i]), i


@torch.no_grad()
def enc_imgs(paths, bs=128):
    model.eval()
    out = []
    for px, _ in torch.utils.data.DataLoader(DS(paths), batch_size=bs, num_workers=8):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(F.normalize(model.get_image_features(px.to(DEV)).float(), dim=-1).cpu())
    model.train()
    return torch.cat(out)


@torch.no_grad()
def trainval():
    """Self-retrieval over the held-back cache tail (GT = own index)."""
    G = enc_imgs(val_paths)
    sim = val_qtext.to(DEV) @ G.to(DEV).T
    _, tk = sim.topk(10, 1)
    tk = tk.cpu()
    n = val_qtext.size(0)
    ms = r1 = 0.0
    for i in range(n):
        rank = next((r for r, j in enumerate(tk[i].tolist()) if j == i), -1)
        if rank >= 0:
            ms += 1 / (rank + 1)
            r1 += rank < 1
    return 100 * ms / n, 100 * r1 / n


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
dl = torch.utils.data.DataLoader(DS(tr_paths), batch_size=BS, shuffle=True,
                                 num_workers=10, drop_last=True, pin_memory=True)
img_bank = torch.zeros(M_IMG_BANK, tr_text.size(1), device=DEV)
bank_n = bank_ptr = 0
best, best_ep, step = -1.0, -1, 0
log(f"start: {len(dl)} steps/ep x {EPOCHS}  (txt_bank={N_TXT_BANK}, img_bank={M_IMG_BANK})")
m, a = trainval()
log(f"[val pre] score={m:.2f}/{a:.2f}")
for ep in range(EPOCHS):
    for px, bidx in dl:
        px = px.to(DEV)
        bidx = bidx.to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            imf = F.normalize(model.get_image_features(px).float(), dim=-1)
        scale = ls.exp().clamp(max=100).float()
        B = imf.size(0)
        pos_t = tr_text_dev[bidx]
        neg_ix = torch.randint(0, tr_text_dev.size(0), (N_TXT_BANK,), device=DEV)
        cand_t = torch.cat([pos_t, tr_text_dev[neg_ix]], 0)
        ce_i2t = F.cross_entropy(scale * (imf @ cand_t.t()), torch.arange(B, device=DEV))
        cand_i = torch.cat([imf, img_bank[:bank_n]], 0) if bank_n > 0 else imf
        ce_t2i = F.cross_entropy(scale * (pos_t @ cand_i.t()), torch.arange(B, device=DEV))
        loss = 0.5 * (ce_i2t + ce_t2i)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():                                   # detached image memory bank
            k = imf.detach()
            e = min(bank_ptr + B, M_IMG_BANK)
            n1 = e - bank_ptr
            img_bank[bank_ptr:e] = k[:n1]
            if n1 < B:
                img_bank[:B - n1] = k[n1:]
            bank_ptr = (bank_ptr + B) % M_IMG_BANK
            bank_n = min(bank_n + B, M_IMG_BANK)
        step += 1
        if step % 50 == 0:
            log(f"ep{ep} s{step} loss={loss.item():.4f} "
                f"(i2t={ce_i2t.item():.3f} t2i={ce_t2i.item():.3f}) scale={scale.item():.1f}")
        if step % EVAL_EVERY == 0:
            m, a = trainval()
            log(f"  [val] s{step} score={m:.2f}/{a:.2f}")
            if m > best:
                best, best_ep = m, ep
                model.save_pretrained(OUT_DIR)
                log("    new best -> saved")
    m, a = trainval()
    log(f"== ep{ep} val score={m:.2f}/{a:.2f} (best {best:.2f}) ==")
    if m > best:
        best, best_ep = m, ep
        model.save_pretrained(OUT_DIR)
        log("    new best -> saved")
log(f"done. best val score={best:.2f} (ep{best_ep}) -> {OUT_DIR}")
