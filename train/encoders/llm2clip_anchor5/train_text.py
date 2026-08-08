#!/usr/bin/env python3
"""LLM2CLIP text-adapter LoRA trainer (stage 2 — this produces the anchor5 adapter).

The stage-1 vision LoRA is merged into the tower first; a fresh LoRA is then
trained over `text_adapter.adaptor.*` only, from the multi-style hidden cache
(shipped: `assets/data/mining/llm2clip_hidden_cache_ms.pt`, 148k images x 5
presets; `build_hidden_cache.py` rebuilds it). MULTISTYLE=1 samples a random
preset per step and adds a consistency term that pulls paraphrases of the same
image together; MULTISTYLE=0 always uses the anchor preset (style 0). The
shipped adapter was trained on the 5-preset cache with multi-style sampling;
ANCHOR_P>0 additionally weights sampling toward the anchor preset.

This trainer only trains and saves: every epoch goes to OUT_DIR/ep{N} and the
final state to OUT_DIR/last. Checkpoint adoption happens outside the trainer.
The shipped stage-2 adapter is `assets/model/encoder/llm2clip_anchor5`.

env: track4_llm2clip. Run from the repository root:
  LLM2CLIP_DEV=cuda:6 MULTISTYLE=0 python train/encoders/llm2clip_anchor5/train_text.py
"""
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"))

DEV = os.environ.get("LLM2CLIP_DEV", "cuda:0")
VIS = os.environ.get("LLM2CLIP_BASE", "microsoft/LLM2CLIP-Openai-L-14-336")
HCACHE = os.environ.get("HIDDEN_CACHE", f"{_REPO}/assets/data/mining/llm2clip_hidden_cache_ms.pt")
V3_ADAPTER = os.environ.get("LLM2CLIP_V3_ADAPTER",
                            f"{_REPO}/assets/model/encoder/llm2clip_lora_v3_best")
IMG_ROOT = os.environ.get("PAB_TRAIN", f"{_REPO}/assets/data/raw/pab_train")
OUT_DIR = os.environ.get("OUT_DIR", f"{_REPO}/assets/runs/llm2clip_text_lora")
MULTISTYLE = int(os.environ.get("MULTISTYLE", "1"))
CONS_W = float(os.environ.get("CONS_W", "0.2"))
ANCHOR_P = float(os.environ.get("ANCHOR_P", "0.0"))   # >0: sample the anchor preset with this probability
EPOCHS = int(os.environ.get("EPOCHS", "4"))
BS = int(os.environ.get("BS", "64"))
LR = float(os.environ.get("LR", "5e-5"))
N_TXT_BANK = int(os.environ.get("N_TXT_BANK", "8192"))
M_IMG_BANK = 8192
LORA_R = 32


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


for p, what in ((HCACHE, "hidden cache (build_hidden_cache.py)"),
                (V3_ADAPTER, "stage-1 vision LoRA")):
    if not os.path.exists(p):
        raise SystemExit(f"[train_text] {what} not found: {p}")

# FA3 probing in the remote code breaks without the wheel; keep FA2/sdpa.
import transformers.utils as _u                                   # noqa: E402
import transformers.utils.import_utils as _iu                     # noqa: E402
for _m in (_iu, _u):
    if hasattr(_m, "is_flash_attn_3_available"):
        setattr(_m, "is_flash_attn_3_available", lambda *a, **k: False)
if hasattr(_iu, "_flash_attn_3_available"):
    _iu._flash_attn_3_available = False
from transformers import AutoModel                                # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model            # noqa: E402
from PIL import Image                                             # noqa: E402
import torchvision.transforms as T                                # noqa: E402

c = torch.load(HCACHE, map_location="cpu", weights_only=False)
tr_paths = c["train_paths"]
tr_hid_ms = c["train_hidden_ms"].float()                          # (N,K,4096), style 0 = anchor
N, K, _ = tr_hid_ms.shape
log(f"MULTISTYLE={MULTISTYLE} CONS_W={CONS_W} | train {N:,} imgs x {K} presets {c['styles']}")
tr_hid_dev = tr_hid_ms.to(DEV)
flat_hid = tr_hid_dev.view(N * K, -1)

base = AutoModel.from_pretrained(VIS, torch_dtype=torch.float32, trust_remote_code=True,
                                 attn_implementation="sdpa").to(DEV)
model = PeftModel.from_pretrained(base, V3_ADAPTER).merge_and_unload().float()   # fold stage 1 in
lora_targets = [n for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and "adaptor" in n.lower()]
for p in model.parameters():
    p.requires_grad_(False)
model = get_peft_model(model, LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                                         bias="none", target_modules=lora_targets)).float()
ls = next(p for n, p in model.named_parameters() if n.endswith("logit_scale"))
ls.requires_grad_(True)
log(f"trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
    f"| text LoRA over {len(lora_targets)} Linear")


def text_feat(hid):
    return F.normalize(model.get_text_features(hid).float(), dim=-1)


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


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
dl = torch.utils.data.DataLoader(DS(tr_paths), batch_size=BS, shuffle=True,
                                 num_workers=12, drop_last=True, pin_memory=True)
img_bank = torch.zeros(M_IMG_BANK, 1280, device=DEV)
bank_n = bank_ptr = step = 0
log(f"steps/ep {len(dl)} x {EPOCHS}")
for ep in range(EPOCHS):
    for px, bidx in dl:
        px = px.to(DEV)
        bidx = bidx.to(DEV)
        B = bidx.size(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            imf = F.normalize(model.get_image_features(px).float(), dim=-1)
        if MULTISTYLE:
            if ANCHOR_P > 0:
                ua = torch.rand(B, device=DEV) < ANCHOR_P
                ki = torch.where(ua, torch.zeros(B, dtype=torch.long, device=DEV),
                                 torch.randint(1, K, (B,), device=DEV))
            else:
                ki = torch.randint(0, K, (B,), device=DEV)
            pos_hid = tr_hid_dev[bidx, ki]
        else:
            pos_hid = tr_hid_dev[bidx, 0]                        # anchor preset only
        pos_t = text_feat(pos_hid)
        neg_t = text_feat(flat_hid[torch.randint(0, N * K, (N_TXT_BANK,), device=DEV)])
        scale = ls.exp().clamp(max=100).float()
        cand_t = torch.cat([pos_t, neg_t], 0)
        ce_i2t = F.cross_entropy(scale * (imf @ cand_t.t()), torch.arange(B, device=DEV))
        cand_i = torch.cat([imf, img_bank[:bank_n]], 0) if bank_n > 0 else imf
        ce_t2i = F.cross_entropy(scale * (pos_t @ cand_i.t()), torch.arange(B, device=DEV))
        loss = 0.5 * (ce_i2t + ce_t2i)
        if MULTISTYLE and CONS_W > 0:
            if ANCHOR_P > 0:                                     # pull weak presets toward the anchor
                kj = torch.randint(1, K, (B,), device=DEV)
                pa = text_feat(tr_hid_dev[bidx, 0])
                pw = text_feat(tr_hid_dev[bidx, kj])
                loss = loss + CONS_W * (1 - (pa * pw).sum(-1)).mean()
            else:                                                # pull any two presets together
                kj = (ki + 1 + torch.randint(0, K - 1, (B,), device=DEV)) % K
                pos_t2 = text_feat(tr_hid_dev[bidx, kj])
                loss = loss + CONS_W * (1 - (pos_t * pos_t2).sum(-1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():                                    # detached image memory bank
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
            log(f"ep{ep} s{step} loss={loss.item():.4f}")
    model.save_pretrained(f"{OUT_DIR}/ep{ep:02d}")
    log(f"== ep{ep} saved -> {OUT_DIR}/ep{ep:02d} ==")
model.save_pretrained(f"{OUT_DIR}/last")
log(f"done MULTISTYLE={MULTISTYLE} -> {OUT_DIR}")
