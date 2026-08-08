#!/usr/bin/env python3
"""eval_heldout_openclip — score open_clip full-FT encoders per epoch on the held-out bench.

Target: `metaclip_v1/` (MetaCLIP v1 `ViT-L-14-worldwide-xlmv`, 224px). Because these are full
fine-tunes with checkpoints at `SAVE_DIR/epoch_{N}.pt`, the paths differ from the adapter evaluator
([`eval_heldout.py`](eval_heldout.py)) and the BEiT3 evaluator. Bench, metrics, md5 and gates come
from [`heldout_bench.py`](heldout_bench.py).

Encoding matches the scoring code
([`../../S1_base/encode/encode_{eva02,metaclip}_ft.py`](../../S1_base/encode)):
`model.encode_image` / `model.encode_text` followed by L2 normalisation.

Usage:
  # eva02 (base weights = an open_clip pretrained tag)
  python eval_heldout_openclip.py --model ViT-L-14-worldwide-xlmv \\
         --ckpt-dir $RUNS_ROOT/eva02_heldout/checkpoints --epochs 1-4 --gpu 6

  # metaclip v1 (base weights = local .pt with a custom config)
  python eval_heldout_openclip.py --model ViT-L-14-worldwide-xlmv \\
         --pretrained $VLM_MODELS/MetaCLIP-L14-worldwide/l14_worldwide.pt \\
         --ckpt-dir $RUNS_ROOT/metaclip_v1_heldout/checkpoints --epochs 1-4 --gpu 7

  # alongside training, scoring each epoch_{N}.pt as it appears
  python eval_heldout_openclip.py --model ViT-L-14-worldwide-xlmv \\
         --ckpt-dir <dir> --epochs 1-4 --watch --gpu 2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser(description="pick the best epoch of an open_clip full-FT run on the held-out bench")
ap.add_argument("--model", required=True, help="open_clip model name (e.g. ViT-L-14-worldwide-xlmv)")
ap.add_argument("--pretrained", default=None,
                help="base weights — open_clip tag or local .pt, loaded before the epoch ckpt")
ap.add_argument("--ckpt-dir", required=True, help="directory holding epoch_{N}.pt")
ap.add_argument("--epochs", required=True, help='"1-4" · "4,8"')
ap.add_argument("--bench", default="main", choices=["main", "hard", "both"])
ap.add_argument("--heldout-dir", default=None)
ap.add_argument("--gpu", default="6")
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--out", default=None, help="result JSON (default = <ckpt-dir>/heldout_eval.json)")
ap.add_argument("--watch", action="store_true", help="score checkpoints as they appear")
ap.add_argument("--poll", type=int, default=180)
ap.add_argument("--settle", type=int, default=60)
ap.add_argument("--train-proc", default="train.py",
                help="training process pattern used to decide when --watch may stop")
ap.add_argument("--deploy-rep", default=None, metavar="NAME",
                help="deploy the selection to assets/model_rep/encoder/<NAME> (tools/promote.py --rep)")
args = ap.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

sys.path.insert(0, HERE)

import open_clip                                                      # noqa: E402

# Register the custom open_clip config (ViT-L-14-worldwide-xlmv) from the repository copy rather
# than relying on the venv's model_configs. See register.py for the config itself.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(_REPO, "assets", "model", "vlm_models", "MetaCLIP-L14-worldwide"))
import register as _oc_register  # noqa: E402,F401
import torch                                                          # noqa: E402
import torch.nn.functional as F                                       # noqa: E402
from PIL import Image                                                 # noqa: E402

import heldout_bench as HB                                            # noqa: E402

DEVICE = "cuda:0"




def wait_for(path: str) -> bool:
    """Wait until the ckpt appears and its size stops changing. False if training died first."""
    while not os.path.exists(path):
        if not HB.still_training(args.train_proc):
            print(f"  [watch] no training process → stop waiting for {os.path.basename(path)}")
            return False
        time.sleep(args.poll)
    last, stable = -1, None
    while True:
        sz = os.path.getsize(path)
        if sz == last:
            if stable is None:
                stable = time.time()
            elif time.time() - stable >= args.settle:
                return True
        else:
            last, stable = sz, None
        time.sleep(min(15, args.poll))


def build(model_name: str, pretrained: str | None):
    """Build the open_clip model, preprocess and tokenizer. A local .pt is loaded directly."""
    is_local = bool(pretrained) and pretrained.endswith(".pt") and os.path.exists(pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=None if is_local else pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    if is_local:
        sd = torch.load(pretrained, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd.get("model", sd))
        res = model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()},
                                    strict=False)
        print(f"  [base] loaded {os.path.basename(pretrained)} "
              f"(missing {len(res.missing_keys)} · unexpected {len(res.unexpected_keys)})")
    return model.eval().to(DEVICE), preprocess, tokenizer


def load_epoch(model, path: str):
    """Load epoch_{N}.pt into the model in place, stripping the DDP `module.` prefix."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=False)
    n_miss, n_unexp = len(res.missing_keys), len(res.unexpected_keys)
    if n_miss or n_unexp:
        print(f"  [load] missing {n_miss} · unexpected {n_unexp}")
        if n_miss and n_unexp:
            print(f"    missing e.g.: {res.missing_keys[:3]}")
            print(f"    unexpected e.g.: {res.unexpected_keys[:3]}")
            raise SystemExit("  ✗ ckpt does not match the --model architecture.")
    return ck.get("epoch")


@torch.no_grad()
def encode_images(model, preprocess, paths):
    feats = []
    for i in range(0, len(paths), args.batch):
        px = torch.stack([preprocess(Image.open(p).convert("RGB"))
                          for p in paths[i:i + args.batch]]).to(DEVICE)
        feats.append(F.normalize(model.encode_image(px).float(), dim=-1).cpu())
        if (i // args.batch) % 50 == 0:
            print(f"    gallery {min(i + args.batch, len(paths)):>6,}/{len(paths):,}", flush=True)
    return torch.cat(feats)


@torch.no_grad()
def encode_texts(model, tokenizer, queries):
    feats = []
    for i in range(0, len(queries), args.batch):
        tok = tokenizer([r["caption"] for r in queries[i:i + args.batch]]).to(DEVICE)
        feats.append(F.normalize(model.encode_text(tok).float(), dim=-1).cpu())
    return torch.cat(feats)


def main():
    eps = HB.parse_epochs(args.epochs)
    benches = ["main", "hard"] if args.bench == "both" else [args.bench]
    out = args.out or os.path.join(args.ckpt_dir, "heldout_eval.json")

    print(f"[eval_heldout_openclip] model={args.model} pretrained={args.pretrained}")
    print(f"[eval_heldout_openclip] ckpt_dir={args.ckpt_dir}")
    md5 = HB.verify_identity(args.heldout_dir)
    HB.require_gates(args.heldout_dir, strict=True)

    loaded = {}
    for b in benches:
        q, g, _ = HB.load_bench(b, args.heldout_dir)
        loaded[b] = (q, g, [HB.map_to_local(p) for p in g])
        print(f"[{b}] queries {len(q):,} · gallery {len(g):,}")
        HB.check_leak(q, g, b, HB.heldout_list_path(args.heldout_dir))

    print("\n[build] building the open_clip model …", flush=True)
    model, preprocess, tokenizer = build(args.model, args.pretrained)

    rows, missing = [], []
    for ep in eps:
        ck = os.path.join(args.ckpt_dir, f"epoch_{ep}.pt")
        if not os.path.exists(ck):
            if args.watch:
                print(f"\n[ep{ep}] waiting … {ck}", flush=True)
                if not wait_for(ck):
                    missing.append(ep)
                    continue
                print(f"[ep{ep}] ckpt arrived", flush=True)
            else:
                missing.append(ep)
                print(f"\n[ep{ep}] no ckpt → skip")
                continue
        t0 = time.time()
        print(f"\n[ep{ep}] {ck}", flush=True)
        load_epoch(model, ck)
        rec = {"epoch": ep, "ckpt": ck}
        for b in benches:
            q, g, paths = loaded[b]
            G = encode_images(model, preprocess, paths)
            Q = encode_texts(model, tokenizer, q)
            rec[b] = m = HB.score(Q, G, q, g)
            print(f"  [{b}] mAP@10={m['mAP@10']:.4f} R@1={m['R@1']:.4f} "
                  f"R@5={m['R@5']:.4f} R@10={m['R@10']:.4f} margin={m['margin_agg']:.4f}")
            del G, Q
        torch.cuda.empty_cache()
        rec["sec"] = round(time.time() - t0, 1)
        rows.append(rec)
        with open(out, "w", encoding="utf-8") as f:                   # rewritten after every epoch
            json.dump({"model": args.model, "pretrained": args.pretrained,
                       "ckpt_dir": args.ckpt_dir, "heldout_md5": md5,
                       "split": HB.split_path(args.heldout_dir),
                       "rows": rows, "missing": missing}, f, indent=2, ensure_ascii=False)

    if not rows:
        raise SystemExit(f"no epoch was scored (missing: {missing})")

    best, sel = HB.print_table(rows, benches)
    if missing:
        print(f"  ⚠ epochs skipped for want of a ckpt: {missing}")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "pretrained": args.pretrained,
                   "ckpt_dir": args.ckpt_dir, "heldout_md5": md5,
                   "split": HB.split_path(args.heldout_dir), "select_bench": sel,
                   "best_epoch": best["epoch"], "rows": rows, "missing": missing},
                  f, indent=2, ensure_ascii=False)
    print(f"saved → {out}")

    if args.deploy_rep:                              # deploy the selection, only if the inputs are intact
        if not best or best.get("epoch") is None:
            raise SystemExit("[deploy-rep] ✗ no epoch was selected — not deploying")
        src = os.path.join(args.ckpt_dir, f"epoch_{best['epoch']}.pt")
        if not os.path.exists(src):
            raise SystemExit(f"[deploy-rep] ✗ selected ckpt not found → not deploying: {src}")
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        print(f"[deploy-rep] epoch_{best['epoch']}.pt → assets/model_rep/encoder/{args.deploy_rep}")
        rc = subprocess.run([sys.executable, os.path.join(_repo, "tools", "promote.py"),
                             "--rep", "--kind", "encoder", "--name", args.deploy_rep, "--src", src]).returncode
        if rc:
            raise SystemExit(f"[deploy-rep] ✗ promote failed (rc={rc})")
        print(f"[deploy-rep] ✓ cache_rep must be built per model with pipeline/S1_base/encode/encode_*.py --rep")


if __name__ == "__main__":
    main()
