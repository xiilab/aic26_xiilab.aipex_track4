#!/usr/bin/env python3
"""eval_heldout — score encoder epochs on the held-out bench and pick the best one.

Nothing is scored during training: the trainer only writes one checkpoint per epoch. This script
scores those checkpoints offline against the bench in `heldout_v1/split.json`.

  heldout_images.txt (50,653 images)  ← read by the trainer to exclude them (exclude_heldout_rows)
  split.json                         ← the bench this script reads
    main : queries 2,000        / gallery_easy    36,773   representative -> selection metric
    hard : queries_hard 1,762   / gallery_hardset 36,772   siblings added -> diagnostic only

Metrics match the competition scorer (one ground truth per query): mAP@10 = mean(1/rank), R@1/5/10.

Usage:
  # score anchor_tcap (SigLIP2) epochs 1-12 on the main bench
  python eval_heldout.py --trainer anchor_tcap_all/train.py \\
         --run $RUNS_ROOT/<RUN_TAG> --epochs 1-12 --gpu 6

  # add the diagnostic hard bench (selection still uses main)
  python eval_heldout.py --trainer metaclip2/train_mc2_metaclip2_dora.py \\
         --run <RUN> --epochs 1-5 --bench both --gpu 7

The trainer module is imported so that `build_model()`, `load_model_weights_from_ckpt()`,
`map_annotation_path_to_local()` and `_cv2_load_image()` are reused: the adapter is loaded onto a
model that went through the same surgery as during training (position interpolation, pooler split,
multi-probe). Inference uses the global path (`get_image_features` / `get_text_features`); the
FLAIR/FILIP heads are training-only and take no part in scoring.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))



ap = argparse.ArgumentParser(description="pick an encoder's best epoch on the held-out bench")
ap.add_argument("--trainer", required=True,
                help="trainer .py (e.g. anchor_tcap_all/train.py)")
ap.add_argument("--run", required=True, help="run directory holding checkpoints/epNN")
ap.add_argument("--epochs", required=True, help='"1-12" · "3,5,9" · "1-4,8"')
ap.add_argument("--bench", default="main", choices=["main", "hard", "both"],
                help="main=selection metric (default) · hard=diagnostic only")
ap.add_argument("--heldout-dir", default=None, help="heldout directory (default = the trainer's HELDOUT_DIR)")
ap.add_argument("--gpu", default="6")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out", default=None, help="result JSON (default = <run>/heldout_eval.json)")
ap.add_argument("--watch", action="store_true",
                help="run alongside training on another GPU, scoring checkpoints/epNN as they appear")
ap.add_argument("--poll", type=int, default=180, help="--watch polling interval in seconds")
ap.add_argument("--settle", type=int, default=60,
                help="--watch: treat the adapter file as complete once its size holds for this many seconds")
ap.add_argument("--train-proc", default="train_", help="process pattern used to decide when --watch may stop")
ap.add_argument("--force", action="store_true",
                help="proceed even when the run's architecture config differs from the trainer constants")
ap.add_argument("--deploy-rep", default=None, metavar="NAME",
                help="deploy the selection to assets/model_rep/encoder/<NAME> (tools/promote.py --rep)")
args = ap.parse_args()

# CUDA_VISIBLE_DEVICES must be set before torch initialises CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402

sys.path.insert(0, HERE)                                              # heldout_bench
import heldout_bench as HB                                            # noqa: E402

DEVICE = "cuda:0"


def import_trainer(path: str):
    """Load a trainer .py as a module. Relative paths resolve against this file."""
    if not os.path.isabs(path):
        cand = os.path.join(HERE, path)
        path = cand if os.path.exists(cand) else os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"trainer not found: {path}")
    spec = importlib.util.spec_from_file_location("trainer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["trainer_mod"] = mod
    spec.loader.exec_module(mod)
    return mod




def wait_for_ckpt(run: str, ep: int) -> str | None:
    """Wait until the epNN adapter appears and its size stops changing. None if training died."""
    while True:
        d = ckpt_dir(run, ep)
        f = os.path.join(d, "adapter_model.safetensors") if d else None
        if d and f and os.path.exists(f):
            last, stable = -1, None
            while True:
                sz = os.path.getsize(f)
                if sz == last:
                    if stable is None:
                        stable = time.time()
                    elif time.time() - stable >= args.settle:
                        return d
                else:
                    last, stable = sz, None
                time.sleep(min(15, args.poll))
        if not HB.still_training(args.train_proc):
            print(f"  [watch] no training process → stop waiting for ep{ep:02d}")
            return None
        time.sleep(args.poll)


def ckpt_dir(run: str, ep: int) -> str | None:
    """runs/<tag>/checkpoints/ep07 (or ep7). None if absent."""
    base = os.path.join(run, "checkpoints")
    for tag in (f"ep{ep:02d}", f"ep{ep}"):
        d = os.path.join(base, tag)
        if os.path.isdir(d):
            return d
    return None



def _base(model):
    """Strip the PEFT wrapper and return the base model, which has get_image/text_features."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def _feat(M, x):
    """Convert a `get_*_features` output to a Tensor.

    The return type differs per model: SigLIP2 gives a Tensor while MetaCLIP2 gives a
    `BaseModelOutputWithPooling`. The trainer's `_to_tensor()` is used when present so the rule
    matches training, with the same attribute order as fallback.
    """
    fn = getattr(M, "_to_tensor", None)
    if fn is not None:
        return fn(x)
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("pooler_output", "image_embeds", "text_embeds", "last_hidden_state"):
        v = getattr(x, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    raise RuntimeError(f"unexpected feature output type: {type(x).__name__}")


# Keys that change the structure or tensor shapes built by build_model(): a mismatch makes the
# scores meaningless.
ARCH_HARD = ("MODEL_NAME", "IMAGE_SIZE", "MAX_TEXT_LENGTH", "POSITION_EMBED_NEW_SIZE",
             "POSITION_PRETRAINED_SIZE", "NUM_PROBES", "MULTI_PROBE_AGG_TYPE",
             "DORA_RANK", "DORA_ALPHA", "DORA_RANK_TEXT", "DORA_ALPHA_TEXT", "DORA_TARGETS")
# Training-only head/optimisation keys: they do not affect the global embedding, so warn only.
ARCH_SOFT = ("FLAIR_ENABLED", "FINEGRAIN_ENABLED", "FINEGRAIN_DIM", "FINEGRAIN_PATCH_POOL",
             "POSITION_PI_ENABLED", "FREEZE_VISION_DORA", "FREEZE_POOLER_DORA")


def check_arch(M, run: str, force: bool):
    """Compare the architecture config in `<run>/config.json` with the trainer constants.

    Building a different structure than training used would place the adapter on the wrong modules
    and make the metrics meaningless. `ABL_*` environment variables can align the trainer constants.
    """
    cfg_path = os.path.join(run, "config.json")
    if not os.path.exists(cfg_path):
        print(f"  ⚠ {cfg_path} not found → skipping the architecture comparison")
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    hard, soft = [], []
    for keys, bucket in ((ARCH_HARD, hard), (ARCH_SOFT, soft)):
        for k in keys:
            if k not in cfg or not hasattr(M, k):
                continue
            want, got = cfg[k], getattr(M, k)
            if isinstance(want, list):
                want, got = list(want), list(got)
            if want != got:
                bucket.append(f"{k}: run={want!r} ≠ trainer={got!r}")
    for m in soft:
        print(f"  ⚠ (training-only key) {m}")
    if hard:
        msg = "the run's architecture config differs from the current trainer constants:\n    " + "\n    ".join(hard)
        if not force:
            raise SystemExit(f"  ✗ {msg}\n"
                             f"  → align them with ABL_* environment variables, or pass --force")
        print(f"  ⚠ --force: {msg}")
    elif not soft:
        print("  [arch] run config matches the trainer constants ✓")


def load_ckpt_weights(M, model, ck: str):
    """Load either the adapter (DoRA) or a full-FT state_dict, depending on the ckpt layout."""
    if os.path.exists(os.path.join(ck, "adapter_model.safetensors")):
        M.load_model_weights_from_ckpt(model, ck)
        return "adapter"

    full = os.path.join(ck, "model.safetensors")
    if not os.path.exists(full):
        raise FileNotFoundError(
            f"neither adapter_model.safetensors nor model.safetensors is present: {ck}")
    from safetensors.torch import load_file
    state = load_file(full)
    base = _base(model)
    res = base.load_state_dict(state, strict=False)
    n_miss, n_unexp = len(res.missing_keys), len(res.unexpected_keys)
    print(f"  [full-FT] loaded {len(state)} tensors "
          f"(missing {n_miss} · unexpected {n_unexp})")
    if n_miss and n_unexp:
        print(f"    missing e.g.: {res.missing_keys[:3]}")
        print(f"    unexpected e.g.: {res.unexpected_keys[:3]}")
        raise SystemExit(
            "  ✗ the full-FT state_dict does not match the build_model() structure "
            "(missing and unexpected keys at once) — check the run config.")
    # extras (logit_scale, position_embedding, multi-probe) live in a separate file.
    ex = os.path.join(ck, "extras_state.pt")
    if os.path.exists(ex):
        extras = torch.load(ex, map_location="cpu", weights_only=False)
        if "position_embedding" in extras:
            pe = base.text_model.embeddings.position_embedding.weight
            pe.data.copy_(extras["position_embedding"].to(pe))
        for k in ("logit_scale", "logit_bias"):
            if k in extras and hasattr(base, k):
                getattr(base, k).data.copy_(extras[k].to(getattr(base, k).data))
        print(f"  [full-FT] restored extras ({', '.join(extras.keys())})")
    return "full-FT"


@torch.no_grad()
def encode_gallery(M, model, paths, batch: int):
    """Encode gallery images into L2-normalised embeddings [N, D].

    Preprocessing matches the trainer's non-augmented path: `_cv2_load_image(p, IMAGE_SIZE)` gives a
    square INTER_AREA resize as uint8, then float [0,1] and Normalize(0.5, 0.5) = [-1, 1].
    """
    feats = []
    for i in range(0, len(paths), batch):
        px = torch.stack([M._cv2_load_image(p, M.IMAGE_SIZE) for p in paths[i:i + batch]])
        px = px.to(DEVICE, non_blocking=True).float().div_(255.0).sub_(0.5).div_(0.5)
        v = _feat(M, _base(model).get_image_features(pixel_values=px))
        feats.append(F.normalize(v.float(), dim=-1).cpu())
        if (i // batch) % 100 == 0:
            print(f"    gallery {min(i + batch, len(paths)):>6,}/{len(paths):,}", flush=True)
    return torch.cat(feats)


@torch.no_grad()
def encode_queries(M, model, tok, queries, batch: int):
    """Encode query captions into L2-normalised embeddings [Q, D]."""
    feats = []
    for i in range(0, len(queries), batch):
        caps = [r["caption"] for r in queries[i:i + batch]]
        enc = tok(caps, padding="max_length", truncation=True,
                  max_length=M.MAX_TEXT_LENGTH, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        t = _feat(M, _base(model).get_text_features(**enc))
        feats.append(F.normalize(t.float(), dim=-1).cpu())
    return torch.cat(feats)



def main():
    M = import_trainer(args.trainer)
    heldout_dir = args.heldout_dir or M.HELDOUT_DIR
    split = os.path.join(heldout_dir, "split.json")
    eps = HB.parse_epochs(args.epochs)
    benches = ["main", "hard"] if args.bench == "both" else [args.bench]

    print(f"[eval_heldout] trainer={os.path.basename(args.trainer)} "
          f"model={M.MODEL_NAME} img={M.IMAGE_SIZE} txt={M.MAX_TEXT_LENGTH}")
    print(f"[eval_heldout] run={args.run}")
    print(f"[eval_heldout] split={split}")

    # ---- load the bench and check the gates --------------------------------------
    md5 = HB.verify_identity()                       # same heldout as the trainer excluded
    HB.require_gates(strict=True)                    # a failed gate means no scoring
    loaded, gates = {}, {}
    for b in benches:
        q, g, gates = HB.load_bench(b, path=split)
        loaded[b] = (q, g, [M.map_annotation_path_to_local(p) for p in g])
        print(f"[{b}] queries {len(q):,} · gallery {len(g):,}")
    HB.report_gates(gates)
    # The exclusion list normally comes from the trainer; when a trainer does not define
    # HELDOUT_LIST, fall back to the list in --heldout-dir.
    _hl = getattr(M, "HELDOUT_LIST", None)
    if _hl is None:
        _hl = HB.heldout_list_path(args.heldout_dir)
        print(f"  [heldout] trainer has no HELDOUT_LIST → using {_hl}", flush=True)
    for b in benches:
        q, g, _ = loaded[b]
        HB.check_leak(q, g, b, _hl)

    # ---- compare the run's architecture config -----------------------------------
    check_arch(M, args.run, args.force)

    # ---- build the model once, then swap weights per epoch ------------------------
    print("\n[build] building the model (same surgery as training) ...", flush=True)
    model, tok = M.build_model()
    model.to(DEVICE).eval()

    rows, missing = [], []
    for ep in eps:
        ck = ckpt_dir(args.run, ep)
        if ck is None and args.watch:
            print(f"\n[ep{ep:02d}] waiting … {args.run}/checkpoints/ep{ep:02d}", flush=True)
            ck = wait_for_ckpt(args.run, ep)
            if ck:
                print(f"[ep{ep:02d}] ckpt arrived", flush=True)
        if ck is None:
            missing.append(ep)
            print(f"\n[ep{ep:02d}] no ckpt → skip")
            continue
        t0 = time.time()
        print(f"\n[ep{ep:02d}] {ck}", flush=True)
        kind = load_ckpt_weights(M, model, ck)
        rec = {"epoch": ep, "ckpt": ck, "ckpt_kind": kind}
        for b in benches:
            q, g, paths = loaded[b]
            G = encode_gallery(M, model, paths, args.batch)
            Q = encode_queries(M, model, tok, q, args.batch)
            rec[b] = m = HB.score(Q, G, q, g)
            print(f"  [{b}] mAP@10={m['mAP@10']:.4f} R@1={m['R@1']:.4f} "
                  f"R@5={m['R@5']:.4f} R@10={m['R@10']:.4f}")
            del G, Q
        torch.cuda.empty_cache()
        rec["sec"] = round(time.time() - t0, 1)
        rows.append(rec)

    if not rows:
        raise SystemExit(f"no epoch was scored (no ckpt: {missing})")

    # ---- table and best epoch (selection uses main only) -------------------------
    best, sel = HB.print_table(rows, benches)
    if missing:
        print(f"  ⚠ epochs skipped for want of a ckpt: {missing}")

    out = args.out or os.path.join(args.run, "heldout_eval.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"trainer": args.trainer, "run": args.run, "split": split, "heldout_md5": md5,
                   "select_bench": sel, "best_epoch": best["epoch"],
                   "missing_epochs": missing, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"saved → {out}")

    if args.deploy_rep:                              # deploy the selection, only if the inputs are intact
        if not best or best.get("epoch") is None:
            raise SystemExit("[deploy-rep] ✗ no epoch was selected — not deploying")
        src = ckpt_dir(args.run, best["epoch"])
        if not src or not os.path.exists(src):
            raise SystemExit(f"[deploy-rep] ✗ selected ckpt not found → not deploying: {src}")
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        print(f"[deploy-rep] ep{best['epoch']} → assets/model_rep/encoder/{args.deploy_rep}")
        rc = subprocess.run([sys.executable, os.path.join(_repo, "tools", "promote.py"),
                             "--rep", "--kind", "encoder", "--name", args.deploy_rep, "--src", src]).returncode
        if rc:
            raise SystemExit(f"[deploy-rep] ✗ promote failed (rc={rc})")
        print(f"[deploy-rep] ✓ cache_rep must be built per model with pipeline/S1_base/encode/encode_*.py --rep")


if __name__ == "__main__":
    main()
