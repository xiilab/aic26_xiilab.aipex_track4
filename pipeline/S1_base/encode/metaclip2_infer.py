#!/usr/bin/env python3
"""metaclip2_infer — MetaCLIP2 L/14 inference backend (the counterpart of `anchor_infer.py`).

Reassembles the trained DoRA adapter in the same order used during training and encodes the
gallery and queries. A checkpoint holds only adapter deltas plus extras, so it cannot be
interpreted without this reassembly procedure.

The split between this module and `encode_*.py` is model side vs dataset side: this file knows
only how to turn a model into vectors and nothing about the dataset. That lets 2 models x
3 datasets = 6 consumers share one implementation:

                    PAB test                UCC                        RSTP
  metaclip2         encode_metaclip2.py     uca_dump_metaclip2.py      rstp_dump_metaclip2.py
  mc2h378_peft      encode_mc2h378.py       uca_dump_mc2h378_peft.py   rstp_dump_mc2h378_peft.py

It lives in the pipeline tree rather than the training tree (train/encoders/**) on the same
principle as `anchor_infer.py`: the train module is the single source for architecture constants,
while the inference procedure is owned by the pipeline.

Loading: AutoModel(MetaClip2) -> position interpolation (77->128) -> PeftModel(DoRA adapter)
         -> restore extras (logit_scale / position_embedding). The FLAIR pooler is train-only and
         is not restored.
         PI patches TextEmbeddings.forward, so it must be applied before the adapter is attached.
         A wrong order raises no exception and only changes the embeddings silently.
Encoding: model.get_image_features / get_text_features (projected embedding) -> L2 normalization.

Epoch selection is handled by the held-out scorer `train/encoders/eval/eval_heldout.py`.
"""
import importlib.util
import os
from pathlib import Path

HERE        = os.path.dirname(os.path.abspath(__file__))
_REPO       = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
# The default train module is the metaclip2 trainer. A consumer on another backbone passes its own
# trainer path to import_train_module() (e.g. encode_mc2h378.py -> mc2h378_peft_all/train.py).
# METACLIP2_TRAIN_MODULE overrides the default.
MODULE_PATH = os.environ.get("METACLIP2_TRAIN_MODULE",
                             f"{_REPO}/train/encoders/metaclip2/train.py")


def import_train_module(path):
    spec = importlib.util.spec_from_file_location("mc2train", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(
            f"train module not found: {path}\n"
            f"  -> set METACLIP2_TRAIN_MODULE to the correct path, or update "
            f"metaclip2_infer.MODULE_PATH.")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_for_inference(M, ckpt_dir, device):
    """AutoModel(MetaClip2) -> PI 77->128 -> PeftModel(DoRA adapter) -> restore extras (logit_scale/position)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from peft import PeftModel

    base = AutoModel.from_pretrained(M.MODEL_NAME)
    tok  = AutoTokenizer.from_pretrained(M.MODEL_NAME)
    # PI must follow the training order: patch emb.forward (77->128) before attaching the adapter.
    if getattr(M, "POSITION_PI_ENABLED", False):
        M.apply_position_interpolation(base,
                                       pretrain_len=M.POSITION_PI_PRETRAIN_LEN,
                                       target_len=M.POSITION_PI_TARGET_LEN)
    peft_model = PeftModel.from_pretrained(base, ckpt_dir)
    inner = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model

    # Restore extras, i.e. base params PEFT does not save. flair_pooler is train-only, so it is skipped.
    ep = os.path.join(ckpt_dir, "extras_state.pt")
    if os.path.isfile(ep):
        extras = torch.load(ep, map_location="cpu", weights_only=False)
        if "logit_scale" in extras and hasattr(inner, "logit_scale"):
            inner.logit_scale.data.copy_(extras["logit_scale"].to(inner.logit_scale.data))
        if "position_embedding" in extras:
            pe = inner.text_model.embeddings.position_embedding.weight
            src = extras["position_embedding"].to(pe.data)
            if tuple(src.shape) == tuple(pe.shape):
                pe.data.copy_(src)
            else:
                print(f"  [load] position_embedding shape {tuple(src.shape)}!={tuple(pe.shape)} — skipped")
    else:
        print(f"  [load] extras_state.pt not found ({ckpt_dir}) — skipping logit/position restore")
    peft_model = peft_model.to(device).eval()
    return peft_model, tok


def build_eval_transform(M):
    import torch
    import torchvision.transforms.v2 as Tv2
    return Tv2.Compose([
        Tv2.ToDtype(torch.float32, scale=True),
        Tv2.Normalize(mean=M.IMG_NORM_MEAN, std=M.IMG_NORM_STD),   # OpenAI CLIP normalization
    ])


def _gallery_collate(batch):
    import torch
    return {"pv": torch.stack([b["pv"] for b in batch]),
            "name": [b["name"] for b in batch]}


def encode_gallery(M, model, paths, tf, device, bs):
    import torch
    from torch.utils.data import Dataset, DataLoader

    class _DS(Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i):
            try:
                u8 = M._cv2_load_image(paths[i], target_size=M.IMAGE_SIZE)
            except Exception:
                u8 = torch.zeros((3, M.IMAGE_SIZE, M.IMAGE_SIZE), dtype=torch.uint8)
            return {"pv": tf(u8), "name": Path(paths[i]).stem}

    loader = DataLoader(_DS(), batch_size=bs, shuffle=False, num_workers=4,
                        collate_fn=_gallery_collate, pin_memory=True)
    from tqdm import tqdm
    G, names = [], []
    for b in tqdm(loader, desc="  gallery", leave=False):
        pv = b["pv"].to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=M.AMP_DTYPE, enabled=M.USE_AMP):
            v = model.get_image_features(pixel_values=pv)
        import torch.nn.functional as F
        G.append(F.normalize(_to_t(v).float(), dim=-1).cpu()); names += b["name"]
    return torch.cat(G, 0), names


def encode_queries(M, model, queries, tok, device, bs):
    import torch, torch.nn.functional as F
    pad = getattr(tok, "pad_token_id", None); pad = 0 if pad is None else pad
    caps = [M._siglip2_normalize_text(q["caption"]) for q in queries]
    Q, Qidx, Qchg, Qlen = [], [], [], []
    from tqdm import tqdm
    for i in tqdm(range(0, len(caps), bs), desc="  query", leave=False):
        chunk = caps[i:i+bs]; qs = queries[i:i+bs]
        enc = tok(chunk, padding="max_length", truncation=True,
                  max_length=M.MAX_TEXT_LENGTH, return_tensors="pt")
        ids = enc["input_ids"].to(device, non_blocking=True)
        am  = enc["attention_mask"].to(device, non_blocking=True)
        Qlen += [int(x) for x in (enc["input_ids"] != pad).sum(-1).tolist()]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=M.AMP_DTYPE, enabled=M.USE_AMP):
            t = model.get_text_features(input_ids=ids, attention_mask=am)
        Q.append(F.normalize(_to_t(t).float(), dim=-1).cpu())
        Qidx += [q["query_index"] for q in qs]; Qchg += [q.get("change", "") for q in qs]
    return torch.cat(Q, 0), Qidx, Qchg, Qlen


def _to_t(x):
    """get_*_features may return a tensor or an output object; normalize both to a tensor."""
    import torch
    if torch.is_tensor(x):
        return x
    for a in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(x, a, None)
        if torch.is_tensor(v):
            return v
    raise TypeError(f"could not extract a tensor from get_*_features output: {type(x)}")
