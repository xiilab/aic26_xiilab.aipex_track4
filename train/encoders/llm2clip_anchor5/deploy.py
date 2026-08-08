#!/usr/bin/env python3
"""Copy the trained LoRAs to the reproducible deployment location.

The member is a two-adapter stack, so each stage deploys to its own directory:

  train_vision.py : <vision run>          -> assets/model_rep/encoder/llm2clip_lora_v3_best
  train_text.py   : <text run>/ep{N}|last -> assets/model_rep/encoder/llm2clip_anchor5

`encode_llm2clip_anchor5.py` merges the vision LoRA into the tower and then attaches the text
adapter, so both are required. The text stage also needs `query_hidden.pt` (the precomputed
[1978, 4096] LLM hidden states) next to its adapter; it is copied from the source deployment
unless --query-hidden points elsewhere, because the 8B LLM that produces it is not bundled.

usage (run from the repository root):
  python train/encoders/llm2clip_anchor5/deploy.py --vision assets/runs/llm2clip_vision_lora
  python train/encoders/llm2clip_anchor5/deploy.py --text assets/runs/llm2clip_text_lora/ep03
"""
from __future__ import annotations

import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

DST_ROOT = os.path.join(_REPO, "assets/model_rep/encoder")
VISION_NAME = "llm2clip_lora_v3_best"
TEXT_NAME = "llm2clip_anchor5"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def _abs(p: str) -> str:
    """Resolve a relative path against the repository root."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def deploy_adapter(src: str, name: str, extra: dict[str, str] | None = None) -> str:
    """Copy one PEFT adapter directory to assets/model_rep/encoder/<name>."""
    missing = [f for f in ADAPTER_FILES if not os.path.exists(os.path.join(src, f))]
    if missing:
        have = sorted(os.listdir(src)) if os.path.isdir(src) else []
        raise SystemExit(f"[input check failed]\n  - not a PEFT adapter directory: {src}\n"
                         f"      missing {missing}\n      directory contents: {have[:12]}")

    dst = os.path.join(DST_ROOT, name)
    print(f"[deploy] {os.path.relpath(src, _REPO)}  ->  {os.path.relpath(dst, _REPO)}")
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    n = 0
    for f in ADAPTER_FILES:
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
        n += 1
    for f, p in (extra or {}).items():
        if not os.path.exists(p):
            raise SystemExit(f"[input check failed]\n  - {f} not found: {p}\n"
                             f"      it is produced by build_hidden_cache.py and shipped with the "
                             f"adopted adapter; pass --query-hidden to point at another copy")
        shutil.copy2(p, os.path.join(dst, f))
        n += 1
    size = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst))
    print(f"         {n} files · {size / 2 ** 20:.0f} MiB")
    return dst


def main():
    ap = argparse.ArgumentParser(
        description="deploy the llm2clip vision/text LoRAs to assets/model_rep/encoder/")
    ap.add_argument("--vision", help=f"vision-LoRA run directory -> {VISION_NAME}")
    ap.add_argument("--text", help=f"text-adapter checkpoint directory (ep{{N}} or last) -> {TEXT_NAME}")
    ap.add_argument("--query-hidden", default=None,
                    help="query_hidden.pt to ship with the text adapter "
                         "(default = the adopted assets/model/encoder/llm2clip_anchor5 copy)")
    args = ap.parse_args()
    if not (args.vision or args.text):
        raise SystemExit("nothing to do — pass --vision and/or --text")

    if args.vision:
        deploy_adapter(_abs(args.vision), VISION_NAME)
    if args.text:
        qh = _abs(args.query_hidden) if args.query_hidden else os.path.join(
            _REPO, "assets/model/encoder", TEXT_NAME, "query_hidden.pt")
        deploy_adapter(_abs(args.text), TEXT_NAME, extra={"query_hidden.pt": qh})


if __name__ == "__main__":
    main()
