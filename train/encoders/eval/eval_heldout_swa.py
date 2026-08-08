#!/usr/bin/env python3
"""Per-epoch held-out scoring for SWA range search. Held-out benches only.

Benches come solely from the repo split (assets/data/heldout_v1/split.json).

  main : queries      x gallery_easy
  hard : queries_hard x gallery_hardset

Metrics per epoch (single GT -> mAP@10 == MRR@10): mAP@10, R@1/5/10,
pos_cos_mean, negmax_cos_mean, margin_agg, margin_p50.

Results are written into the run directory: <run>/heldout_eval/ep{NN}.json

usage (run from the repository root):
  python train/encoders/eval/eval_heldout_swa.py \
      --trainer train/encoders/anchor_tcap_heldout/train.py \
      --run assets/runs/20260728_095720_anchor_tcap_heldout \
      --benches hard --epochs all --gpu 6
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HELDOUT_DIR = os.environ.get("HELDOUT_DIR", os.path.join(_REPO, "assets/data/heldout_v1"))
BENCH_PLAN = {"main": ("gallery_easy", "queries"),
              "hard": ("gallery_hardset", "queries_hard")}
OUT_SUBDIR = "heldout_eval"

# structural keys - if the run config and the trainer constants disagree, the adapter lands in the wrong place.
ARCH_HARD = ("MODEL_NAME", "IMAGE_SIZE", "MAX_TEXT_LENGTH", "POSITION_EMBED_NEW_SIZE",
             "POSITION_PRETRAINED_SIZE", "NUM_PROBES", "MULTI_PROBE_AGG_TYPE",
             "DORA_RANK", "DORA_ALPHA", "DORA_RANK_TEXT", "DORA_ALPHA_TEXT", "DORA_TARGETS")


def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0]


def _abs(p):
    """Turn a repository-root-relative path into an absolute one (independent of cwd)."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def import_trainer(path):
    p = _abs(path)
    if not os.path.exists(p):
        raise SystemExit(f"[error] trainer not found: {p}")
    name = "tr_" + stem(p) + "_" + os.path.basename(os.path.dirname(p))
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def check_arch(M, run, force=False):
    cfg_p = os.path.join(run, "config.json")
    if not os.path.exists(cfg_p):
        print("  no config.json - skipping the architecture check")
        return
    cfg = json.load(open(cfg_p, encoding="utf-8"))
    bad = []
    for k in ARCH_HARD:
        if k not in cfg or not hasattr(M, k):
            continue
        want, got = cfg[k], getattr(M, k)
        if isinstance(want, list):
            want, got = list(want), list(got)
        if want != got:
            bad.append(f"{k}: run={want!r} != trainer={got!r}")
    if bad:
        msg = "run architecture does not match the trainer constants:\n    " + "\n    ".join(bad)
        if not force:
            raise SystemExit(f"  [error] {msg}\n  -> --force overrides this (not recommended)")
        print(f"  ⚠ --force: {msg}")
    else:
        print("  [arch] run config matches the trainer constants")


def load_plans(M, benches, split_path, img_root):
    """Build the held-out bench only. Image paths come from the trainer mapping function with root injected."""
    sp = json.load(open(split_path, encoding="utf-8"))
    out = {}
    for b in benches:
        if b not in BENCH_PLAN:
            raise SystemExit(f"[error] supported benches: {sorted(BENCH_PLAN)} (got {b})")
        gk, qk = BENCH_PLAN[b]
        gal = [M.map_annotation_path_to_local(x, root=img_root) for x in sp[gk]]
        qs = [{"caption": q["caption"], "qid": str(q.get("qid", i)),
               "style": q.get("style")} for i, q in enumerate(sp[qk])]
        gt = [M.map_annotation_path_to_local(q["image"], root=img_root) for q in sp[qk]]
        row = {p: i for i, p in enumerate(gal)}
        if len(row) != len(gal):
            raise SystemExit(f"[error] {b}: duplicate gallery path")
        miss = [g for g in gt if g not in row]
        if miss:
            raise SystemExit(f"[error] {b}: {len(miss)} GT missing from the gallery (e.g. {miss[:2]})")
        out[b] = {"gallery": gal, "queries": qs, "gt_pos": [row[g] for g in gt]}
    return out, sp.get("stats", {})


def _feat(M, x):
    import torch
    fn = getattr(M, "_extract_feature", None)
    if callable(fn):
        return fn(x)
    if isinstance(x, torch.Tensor):
        return x
    for a in ("image_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(x, a, None)
        if isinstance(v, torch.Tensor):
            return v
    raise RuntimeError(f"unexpected feature output type: {type(x).__name__}")


def _base(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def norm_stats(M):
    """Follows the trainer constants: 0.5 for SigLIP2, the CLIP statistics for MetaCLIP2."""
    mean = getattr(M, "IMG_NORM_MEAN", None)
    std = getattr(M, "IMG_NORM_STD", None)
    if mean is None or std is None:
        return (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "siglip(0.5)"
    return tuple(mean), tuple(std), "clip(openai)"


def encode_gallery(M, model, paths, batch, device, mean, std, workers=8):
    """Gallery -> L2-normalized embeddings. The preprocessing matches the non-augmented training path."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    size = M.IMAGE_SIZE

    class _DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            try:
                return M._cv2_load_image(paths[i], size)
            except Exception:                       # a decode failure becomes a black image
                return torch.zeros((3, size, size), dtype=torch.uint8)

    mt = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    sd = torch.tensor(std, device=device).view(1, 3, 1, 1)
    dl = DataLoader(_DS(), batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=True, drop_last=False)
    out, done, t0 = [], 0, time.time()
    with torch.no_grad():
        for px in dl:
            px = px.to(device, non_blocking=True).float().div_(255.0).sub_(mt).div_(sd)
            v = _feat(M, _base(model).get_image_features(pixel_values=px))
            out.append(F.normalize(v.float(), dim=-1).cpu())
            done += px.shape[0]
            if done % (batch * 40) < batch or done == len(paths):
                el = time.time() - t0
                print(f"    gallery {done:>6,}/{len(paths):,}  {done / max(el, 1e-9):.0f} img/s",
                      flush=True)
    return torch.cat(out)


def encode_queries(M, model, tok, queries, batch, device):
    import torch
    import torch.nn.functional as F
    out = []
    with torch.no_grad():
        for i in range(0, len(queries), batch):
            caps = [q["caption"] for q in queries[i:i + batch]]
            enc = tok(caps, padding="max_length", truncation=True,
                      max_length=M.MAX_TEXT_LENGTH, return_tensors="pt")
            ids = enc["input_ids"].to(device)
            am = enc.get("attention_mask")
            am = (am.to(device) if am is not None else torch.ones_like(ids))
            t = _feat(M, _base(model).get_text_features(input_ids=ids, attention_mask=am))
            out.append(F.normalize(t.float(), dim=-1).cpu())
    return torch.cat(out)


def score_single_gt(Q, G, gt_pos, chunk=256):
    """With a single GT, mAP@10 = MRR@10. Also reports the pos and negmax cosines."""
    import torch
    n = Q.shape[0]
    r1 = r5 = r10 = 0
    rr = 0.0
    pos_all, neg_all = [], []
    for s in range(0, n, chunk):
        q = Q[s:s + chunk]
        sims = q @ G.T
        rows = torch.arange(sims.shape[0])
        gts = torch.as_tensor(gt_pos[s:s + sims.shape[0]], dtype=torch.long)
        pos = sims[rows, gts].clone()
        pos_all.append(pos)
        tmp = sims.clone()
        tmp[rows, gts] = -2.0
        neg_all.append(tmp.max(dim=1).values)
        rank = (sims > pos.unsqueeze(1)).sum(dim=1)      # a count of higher values = the 0-based rank
        for rk in rank.tolist():
            if rk < 10:
                r1 += rk < 1
                r5 += rk < 5
                r10 += 1
                rr += 1.0 / (rk + 1)
    pos_all = torch.cat(pos_all)
    neg_all = torch.cat(neg_all)
    return {"mAP@10": 100 * rr / n, "R@1": 100 * r1 / n, "R@5": 100 * r5 / n,
            "R@10": 100 * r10 / n, "n_query": n, "gallery": int(G.shape[0]),
            "pos_cos_mean": float(pos_all.mean()), "negmax_cos_mean": float(neg_all.mean()),
            "margin_agg": float(pos_all.mean() - neg_all.mean()),
            "margin_p50": float((pos_all - neg_all).median())}


def parse_epochs(spec, run):
    have = sorted(int(os.path.basename(d)[2:])
                  for d in glob.glob(os.path.join(run, "checkpoints", "ep[0-9][0-9]")))
    if not have:
        raise SystemExit(f"[error] checkpoints/epNN not found: {run}")
    if not spec or spec == "all":
        return have
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    miss = [e for e in out if e not in have]
    if miss:
        raise SystemExit(f"[error] ckpt not found: ep{miss} (available {have})")
    return out


def main():
    ap = argparse.ArgumentParser(description="score each epoch on the held-out bench only (the test set is not used)")
    ap.add_argument("--trainer", required=True, help="trainer .py (a path relative to the repository root is accepted)")
    ap.add_argument("--run", required=True, help="run directory (must hold checkpoints/epNN)")
    ap.add_argument("--epochs", default="all", help="all | 1-8 | 1,3,5")
    ap.add_argument("--benches", default="hard", help="main,hard - comma separated")
    ap.add_argument("--heldout-dir", default=None, help=f"heldout directory (default {HELDOUT_DIR})")
    ap.add_argument("--img-root", default=None, help="gallery image root (default = the trainer's IMG_ROOT)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--gallery-batch", type=int, default=0, help="0 = the trainer's EVAL_BATCH_SIZE")
    ap.add_argument("--query-batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="ignore the architecture mismatch")
    ap.add_argument("--overwrite", action="store_true", help="recompute existing per-epoch JSON")
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    import torch

    run = _abs(a.run)
    heldout_dir = _abs(a.heldout_dir) if a.heldout_dir else HELDOUT_DIR
    split_path = os.path.join(heldout_dir, "split.json")
    if not os.path.exists(split_path):
        raise SystemExit(f"[error] split.json not found: {split_path}")

    M = import_trainer(a.trainer)
    img_root = _abs(a.img_root or M.IMG_ROOT)
    mean, std, nname = norm_stats(M)
    bs = a.gallery_batch or getattr(M, "EVAL_BATCH_SIZE", 128)
    benches = tuple(x.strip() for x in a.benches.split(",") if x.strip())

    out_dir = os.path.join(run, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[eval:heldout] run={os.path.basename(run)}")
    print(f"  trainer={a.trainer}  model={M.MODEL_NAME}  img={M.IMAGE_SIZE}  txt={M.MAX_TEXT_LENGTH}")
    print(f"  norm={nname}  gallery_batch={bs}  img_root={img_root}")
    print(f"  split={split_path}")
    print("  benches=" + ",".join(benches) + "   (the challenge test set is not used)")
    check_arch(M, run, a.force)

    plans, sp_stats = load_plans(M, benches, split_path, img_root)
    for b in benches:
        print(f"  {b}: queries {len(plans[b]['queries']):,} / gallery {len(plans[b]['gallery']):,}")
    hmd5 = sp_stats.get("heldout_md5")
    qmd5 = sp_stats.get("query_md5")
    print(f"  split heldout_md5={str(hmd5)[:8]} query_md5={str(qmd5)[:8]}", flush=True)

    uniq = sorted({p for pl in plans.values() for p in pl["gallery"]})
    print(f"  gallery union {len(uniq):,}", flush=True)

    device = "cuda"
    todo = []
    for ep in parse_epochs(a.epochs, run):
        p = os.path.join(out_dir, f"ep{ep:02d}.json")
        if os.path.exists(p) and not a.overwrite:
            try:
                j = json.load(open(p, encoding="utf-8"))
                if all(b in j for b in benches):
                    print(f"  [ep{ep:02d}] reusing cache -> {os.path.relpath(p, _REPO)}")
                    continue
            except Exception:
                pass
        todo.append(ep)
    if not todo:
        print("\nevery epoch is already scored (--overwrite recomputes)")
        return

    for ep in todo:
        ck = os.path.join(run, "checkpoints", f"ep{ep:02d}")
        t0 = time.time()
        print(f"\n[ep{ep:02d}] {ck}", flush=True)
        model, tok = M.build_model()
        M.load_model_weights_from_ckpt(model, ck)
        model = model.to(device).eval()
        G_all = encode_gallery(M, model, uniq, bs, device, mean, std, a.workers)
        row_all = {p: i for i, p in enumerate(uniq)}
        out = {"ep": ep, "run": os.path.relpath(run, _REPO), "trainer": a.trainer,
               "model": M.MODEL_NAME, "img": M.IMAGE_SIZE, "norm": nname,
               "split": os.path.relpath(split_path, _REPO),
               "heldout_md5": hmd5, "query_md5": qmd5}
        for b, pl in plans.items():
            Q = encode_queries(M, model, tok, pl["queries"], a.query_batch, device)
            idx = torch.tensor([row_all[p] for p in pl["gallery"]])
            res = score_single_gt(Q, G_all[idx], pl["gt_pos"])
            out[b] = res
            print(f"  [{b}] mAP@10 {res['mAP@10']:.2f} | R@1 {res['R@1']:.2f} | "
                  f"R@5 {res['R@5']:.2f} | R@10 {res['R@10']:.2f} | "
                  f"negmax {res['negmax_cos_mean']:.4f}", flush=True)
        out["secs"] = round(time.time() - t0, 1)
        p = os.path.join(out_dir, f"ep{ep:02d}.json")
        json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  → {os.path.relpath(p, _REPO)} ({out['secs']:.0f}s)", flush=True)
        del G_all, model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
