#!/usr/bin/env python3
"""anchor encoder inference adapter — takes an injected train module
(train/encoders/anchor_*/train.py) and exposes only the inference entry points.

The train module holds what training needs (model surgery, preprocessing helpers,
hyperparameters); the inference-time assembly happens here. The train module is thus the single
source for architecture constants, while the inference procedure is owned by the pipeline.

  init(module_path, **overrides)   import the train module, apply EVAL_* overrides, return it
  build_eval_transform(image_size)
  load_test_data(query_json, query_index, gallery_dir, _unused)
  load_for_inference(model_name, ckpt_path, device)
  encode_test_gallery(model, gallery_paths, eval_transform, ...)
  encode_test_queries(model, queries, tokenizer, ...)

Usage:
    import anchor_infer as AI
    mod = AI.init(f"{REPO}/train/encoders/anchor_tcap_all/train.py", EVAL_BATCH_SIZE=64)
    model, tok = AI.load_for_inference(mod.MODEL_NAME, ckpt, device="cuda:0")
    G, G_base = AI.encode_test_gallery(model, gallery_paths, AI.build_eval_transform(mod.IMAGE_SIZE))
"""
import importlib.util
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as Tv2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_MOD = None


def init(module_path, **overrides):
    """Import the train module by path and override EVAL_* and similar. Returns the module."""
    global _MOD
    spec = importlib.util.spec_from_file_location("_anchor_train_mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in overrides.items():
        setattr(mod, k, v)
    _MOD = mod
    return mod


def _g(name, default=None):
    """Required lookup of a train-module global; the train module owns the architecture constants."""
    if _MOD is None:
        raise RuntimeError("init(module_path) must be called first.")
    if default is None and not hasattr(_MOD, name):
        raise AttributeError(f"train module is missing {name} ({getattr(_MOD, '__file__', '?')})")
    return getattr(_MOD, name, default)


def _opt(name):
    """Optional lookup: returns the hook only if the train module defines it, else None so the
    caller can fall back. `_g(name, None)` cannot be used, because a None default marks the
    lookup as required.
    """
    if _MOD is None:
        raise RuntimeError("init(module_path) must be called first.")
    return getattr(_MOD, name, None)


# ── preprocessing ─────────────────────────────────────────────────────────
def build_eval_transform(image_size=None):
    """Resizing is already done by _cv2_load_image(target_size) via cv2 + INTER_AREA.
    This only applies uint8 (3,H,W) -> float[0,1] -> Normalize([-1,1]), with no augmentation.
    It must match the training tail so that train/eval preprocessing does not skew."""
    return Tv2.Compose([
        Tv2.ToDtype(torch.float32, scale=True),
        Tv2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])


class TestGalleryDataset(Dataset):
    """test gallery (.jpg, flat) → (pixel_values, basename)."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
        self._load = _g("_cv2_load_image")
        self._size = _g("IMAGE_SIZE")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img_u8 = self._load(p, target_size=self._size)   # cv2 BGR->RGB + INTER_AREA -> uint8 (3,H,W)
        except Exception as e:
            print(f"  [warn] gallery read fail: {p} ({e})")
            img_u8 = torch.zeros((3, self._size, self._size), dtype=torch.uint8)
        return {"pixel_values": self.transform(img_u8), "basename": Path(p).stem}


def _test_gallery_collate(batch):
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "basenames": [b["basename"] for b in batch]}


def load_test_data(query_json, query_index, gallery_dir, _unused=None):
    """Returns (queries, qorder, gallery_paths, None); qorder follows query_index.txt."""
    rows = {}
    with open(query_json, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows[str(r["query_index"])] = {"query_index": str(r["query_index"]),
                                           "caption": r["caption"],
                                           "change": r.get("change", "")}
    qorder = [l.strip() for l in open(query_index, encoding="utf-8") if l.strip()]
    queries = [rows[q] for q in qorder]
    gallery_paths = sorted(str(p) for p in Path(gallery_dir).glob("*.jpg"))
    return queries, qorder, gallery_paths, None


# ── encoding ──────────────────────────────────────────────────────────────
def encode_test_gallery(model, gallery_paths, eval_transform, device=None, batch_size=None):
    """test gallery -> (G, G_basenames). Inference only, so the batch size is large."""
    device = device or _g("DEVICE", "cuda:0")
    batch_size = batch_size or _g("EVAL_BATCH_SIZE", 256)
    amp_dtype, use_amp = _g("AMP_DTYPE", torch.bfloat16), _g("USE_AMP", True)
    enc_img = _opt("encode_image_features") or (
        lambda m, pv: _g("_to_tensor")(m.get_image_features(pixel_values=pv)))

    model.eval()
    loader = DataLoader(TestGalleryDataset(gallery_paths, eval_transform),
                        batch_size=batch_size, shuffle=False,
                        num_workers=_g("EVAL_NUM_WORKERS", 4),
                        collate_fn=_test_gallery_collate,
                        pin_memory=True, drop_last=False)
    G_emb, G_base = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Test gallery", leave=False):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                v = enc_img(model, pv)
            G_emb.append(F.normalize(v.float(), dim=-1).cpu())
            G_base.extend(batch["basenames"])
    return torch.cat(G_emb, dim=0), G_base


def encode_test_queries(model, queries, tokenizer, device=None, batch_size=None,
                        caption_override=None):
    """test queries -> (Q, Q_indices, Q_changes, Q_lengths).

    Q_lengths counts against pad_token_id rather than attention_mask, because the SigLIP2
    tokenizer returns no attention_mask and the mask sum would always equal max_length.
    """
    device = device or _g("DEVICE", "cuda:0")
    batch_size = batch_size or _g("EVAL_BATCH_SIZE", 256)
    amp_dtype, use_amp = _g("AMP_DTYPE", torch.bfloat16), _g("USE_AMP", True)
    max_len = _g("MAX_TEXT_LENGTH")
    norm_text = _g("_siglip2_normalize_text", None)
    assert_lower = _opt("_assert_lowercase")
    enc_txt = _opt("encode_text_features") or (
        lambda m, ids, mask: _g("_to_tensor")(
            m.get_text_features(input_ids=ids, attention_mask=mask)))

    model.eval()
    captions = caption_override if caption_override is not None else [q["caption"] for q in queries]
    Q_emb, Q_idx, Q_change, Q_len = [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), desc="  Test query", leave=False):
            caps = captions[i:i + batch_size]
            qs = queries[i:i + batch_size]
            if norm_text:                       # SigLIP2 was trained on lowercase input
                caps = [norm_text(c) for c in caps]
            if assert_lower:
                assert_lower(caps, where="eval_query")
            tok = tokenizer(caps, padding="max_length", truncation=True,
                            max_length=max_len, return_tensors="pt")
            ids = tok["input_ids"].to(device, non_blocking=True)
            mask = tok.get("attention_mask", torch.ones_like(tok["input_ids"])).to(device, non_blocking=True)
            pad_id = getattr(tokenizer, "pad_token_id", None)
            pad_id = 0 if pad_id is None else pad_id
            Q_len.extend(int(x) for x in (tok["input_ids"] != pad_id).sum(dim=-1).cpu().tolist())
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                t = enc_txt(model, ids, mask)
            Q_emb.append(F.normalize(t.float(), dim=-1).cpu())
            Q_idx.extend(q["query_index"] for q in qs)
            Q_change.extend(q.get("change", "") for q in qs)
    return torch.cat(Q_emb, dim=0), Q_idx, Q_change, Q_len


# ── checkpoint loading ────────────────────────────────────────────────────
def load_for_inference(model_name, ckpt_path, device=None):
    """Load a trained checkpoint (multi-probe + DoRA + position interpolation).

    Changing the surgery order makes the state_dict keys mismatch:
      AutoModel -> position interpolation -> pooler MHA split -> multi-probe head
      -> PEFT adapter -> restore extras_state.pt (position/logit/multi-probe)
    """
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    NUM_PROBES = _g("NUM_PROBES")
    PI_ENABLED = _g("POSITION_PI_ENABLED")
    PI_TARGET = _g("POSITION_PI_TARGET_LEN")
    PI_PRETRAIN = _g("POSITION_PI_PRETRAIN_LEN")
    PE_NEW_SIZE = _g("POSITION_EMBED_NEW_SIZE")

    meta_path = os.path.join(ckpt_path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta.get("pooler_mha_split", False), \
            "meta.json pooler_mha_split=False — checkpoint saved without the SplitMHA surgery."
        meta_K = int(meta.get("multi_probe_count", 1))
        if meta_K != NUM_PROBES:
            raise AssertionError(f"meta.json multi_probe_count={meta_K} != NUM_PROBES={NUM_PROBES}.")
        if meta.get("position_rope_enabled", False):
            raise AssertionError("meta.json position_rope_enabled=True — a RoPE checkpoint is incompatible with the PI code.")
        if meta.get("position_pi_enabled", False):
            meta_target = int(meta.get("position_pi_target_len", PI_TARGET))
            if meta_target != PI_TARGET:
                raise AssertionError(
                    f"meta.json position_pi_target_len={meta_target} != current {PI_TARGET}.")
        else:
            print("  [load] warn: position_pi_enabled missing from meta — proceeding with the current PI mode")
    else:
        print(f"  [load] warn: meta.json not found ({meta_path}) — skipping the compatibility check")

    print(f"  [load] loading {model_name} ...")
    base_model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"  [load] position interpolation (rows={PI_PRETRAIN} -> {PI_TARGET})")
    _g("apply_position_interpolation")(base_model, pretrain_len=PI_PRETRAIN, target_len=PI_TARGET)

    print("  [load] pooler MHA -> split Q/K/V/O Linear")
    _g("_replace_pooler_mha_with_split")(base_model)

    print(f"  [load] vision head -> MultiProbePoolingHead (K={NUM_PROBES})")
    _g("_replace_head_with_multi_probe")(
        base_model, num_probes=NUM_PROBES,
        perturb_std=_g("MULTI_PROBE_PERTURB_STD"),
        aggregator_init=_g("MULTI_PROBE_AGG_INIT"),
        perturb_mode=_g("MULTI_PROBE_PERTURB_MODE"),
        block_scale=_g("MULTI_PROBE_BLOCK_SCALE"))

    print(f"  [load] PEFT adapter <- {ckpt_path}")
    peft_model = PeftModel.from_pretrained(base_model, ckpt_path)

    extras_path = next((os.path.join(ckpt_path, c)
                        for c in ("extras_state.pt", "logit_scalars.pt")
                        if os.path.exists(os.path.join(ckpt_path, c))), None)
    if extras_path is None:
        raise FileNotFoundError(
            f"neither extras_state.pt nor logit_scalars.pt found in {ckpt_path} — the base "
            "parameters PEFT does not save (position/logit/multi-probe) are missing.")
    print(f"  [load] restoring extras <- {extras_path}")
    extras = torch.load(extras_path, map_location="cpu", weights_only=False)
    base = peft_model.get_base_model()

    if "position_embedding" not in extras:
        raise KeyError(f"'position_embedding' missing from extras ({extras_path}).")
    new_pe = extras["position_embedding"]
    saved_size = extras.get("position_embedding_size", int(new_pe.shape[0]))
    expected = PI_PRETRAIN if PI_ENABLED else PE_NEW_SIZE
    if saved_size != expected:
        raise ValueError(
            f"position_embedding size mismatch: saved={saved_size}, expected={expected} "
            f"(PI={PI_ENABLED}). PI mode and a non-PI checkpoint are incompatible.")
    base.text_model.embeddings.position_embedding.weight.data.copy_(new_pe)
    print(f"         position_embedding {tuple(new_pe.shape)}")

    if "logit_scale" in extras and hasattr(base, "logit_scale"):
        base.logit_scale.data.copy_(extras["logit_scale"].to(base.logit_scale.data))
    if "logit_bias" in extras and hasattr(base, "logit_bias"):
        base.logit_bias.data.copy_(extras["logit_bias"].to(base.logit_bias.data))

    if NUM_PROBES > 1:
        head = base.vision_model.head
        if not hasattr(head, "probe"):
            raise RuntimeError("head is not a MultiProbePoolingHead — check the surgery call order.")
        agg_type = extras.get("multi_probe_agg_type", getattr(head, "agg_type", "linear"))
        req = (["multi_probe_probe", "multi_probe_aggregator"] if agg_type == "linear"
               else ["multi_probe_probe", "multi_probe_agg_query",
                     "multi_probe_agg_init_bias", "multi_probe_agg_proj"])
        missing = [k for k in req if k not in extras]
        if missing:
            raise KeyError(f"missing extras keys (agg_type={agg_type}): {missing} ({extras_path}).")
        saved_K = int(extras.get("multi_probe_count", extras["multi_probe_probe"].shape[1]))
        if saved_K != NUM_PROBES:
            raise ValueError(f"ckpt K mismatch: saved={saved_K}, NUM_PROBES={NUM_PROBES}.")
        head.probe.data.copy_(extras["multi_probe_probe"])
        if agg_type == "linear":
            head.aggregator.weight.data.copy_(extras["multi_probe_aggregator"])
        else:
            head.agg_query.data.copy_(extras["multi_probe_agg_query"])
            head.agg_init_bias.data.copy_(extras["multi_probe_agg_init_bias"])
            head.agg_proj.weight.data.copy_(extras["multi_probe_agg_proj"])
            if "multi_probe_attn_temp" in extras:
                head.attn_temp = float(extras["multi_probe_attn_temp"])
        print(f"         multi-probe K={NUM_PROBES} agg_type={agg_type}")

    peft_model = peft_model.to(device).eval()
    print(f"  [load] ready ({device})")
    return peft_model, tokenizer
