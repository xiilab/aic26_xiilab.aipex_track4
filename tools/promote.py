#!/usr/bin/env python3
"""promote — place a training checkpoint at the reproduction deployment path.

  runs/<run>/checkpoints/<epoch>  ->  assets/model_rep/{encoder,reranker}/<name>

Only `_rep` is written; the adopted artifacts under `assets/model/` are never touched, so
md5 comparison against them stays meaningful. An existing reproduction deployment is moved
to `<name>.bak.<timestamp>` before being replaced. The source run is read-only.
The source layout differs per model family and is auto-detected: adapter dir (epNN),
open_clip epoch_N.pt, reranker exNNNN / step_N, beit3 checkpoint-*.pth. `--src` overrides it.

Usage (everything is deployed under assets/model_rep):
  python tools/promote.py --kind encoder  --name metaclip2      --run assets/runs/<run> --epoch 2
  python tools/promote.py --kind reranker --name jina_m0        --run assets/runs/<run> --epoch 8000
  python tools/promote.py --kind encoder  --name metaclip_v1    --src assets/runs/<run>/checkpoints/epoch_4.pt
  DRY_RUN=1 python tools/promote.py ...    # print the plan without copying
"""
import argparse
import os
import shutil
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_src(run: str, epoch: str) -> str | None:
    """Locate the checkpoint (directory or file) matching `epoch` under run/checkpoints."""
    ck = os.path.join(run, "checkpoints")
    cands = [
        os.path.join(run, f"step{epoch}"),                 # reranker (InternVL) step dir
        os.path.join(ck, f"ex{epoch}"),                    # reranker exNNNN dir
        os.path.join(ck, f"ep{int(epoch):02d}") if str(epoch).isdigit() else "",   # adapter ep09
        os.path.join(ck, f"ep{epoch}"),                    # adapter ep9
        os.path.join(ck, f"epoch_{epoch}.pt"),             # open_clip single file
        os.path.join(ck, f"checkpoint-{epoch}.pth"),       # beit3 (under checkpoints/)
        os.path.join(ck, "checkpoint-best.pth"),           # beit3 best
        # The BEiT3 vendor trainer writes checkpoints to the run root, not to checkpoints/.
        os.path.join(run, f"checkpoint-{epoch}.pth"),
        os.path.join(run, "checkpoint-best.pth"),
        # Literal names: last, swa_2_4, ex008000, step2500, ...
        os.path.join(ck, str(epoch)),
        os.path.join(run, str(epoch)),
    ]
    for c in cands:
        if c and os.path.exists(c):
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Promote a training checkpoint to the deployment path (safe copy)")
    ap.add_argument("--kind", required=True, choices=["encoder", "reranker"])
    ap.add_argument("--name", required=True, help="deployment name (e.g. metaclip2, metaclip_v1, jina_m0)")
    ap.add_argument("--run", help="run directory (required unless --src is given)")
    ap.add_argument("--epoch", help="epoch/step (required unless --src is given)")
    ap.add_argument("--src", help="source checkpoint path, bypassing auto-detection")
    ap.add_argument("--rep", action="store_true",
                    help="(compatibility flag, already the default) deploy to assets/model_rep/<kind>/<name>")
    a = ap.parse_args()
    dry = os.environ.get("DRY_RUN", "0") == "1"

    src = a.src or (find_src(a.run, a.epoch) if a.run and a.epoch else None)
    if not src:
        sys.exit(f"✗ source checkpoint not found (run={a.run} epoch={a.epoch}). Specify it with --src.")
    if not os.path.exists(src):
        sys.exit(f"✗ source does not exist: {src}")

    root = "model_rep"                       # reproduction target only; assets/model is never written
    dst = os.path.join(_REPO, "assets", root, a.kind, a.name)
    print(f"[promote] assets/{root}/{a.kind}/{a.name}")
    print(f"  src = {src}")
    print(f"  dst = {dst}")

    if os.path.exists(dst):
        bak = f"{dst}.bak.{time.strftime('%Y%m%d_%H%M%S')}"
        print(f"  backing up the existing deployment -> {bak}")
        if not dry:
            shutil.move(dst, bak)

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        print("  copy (dir): cp -a")
        if not dry:
            shutil.copytree(src, dst)
    else:
        print("  copy (file -> dir/): cp -a")
        if not dry:
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, os.path.join(dst, os.path.basename(src)))
    print("  ✓ promoted" + (" (DRY_RUN — nothing was copied)" if dry else ""))


if __name__ == "__main__":
    main()
