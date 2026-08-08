#!/usr/bin/env python3
"""heldout_split_v1 - supplies the pre-built held-out split to the training code.

The split is built by train/gen/gen_heldout_v1.py. This module only reads it, and
aborts instead of starting training when the cache is missing.

CACHE_DIR must contain:
  split.json           stats including the gates, checked at training start
  heldout_images.txt   image keys to exclude from training (same format as manifest "image")

split v1: connected components of DINOv2-base(224px) embeddings at cos >= 0.95 form the
  groups, and exclusion is per group. 50,653 held-out images (5.00%), train 1,013,605 ->
  962,952. split md5 dbdf151e.

train.py uses only these two functions:
  hs.ensure_heldout_split(verbose=...)              -> stats (gate check)
  hs.load_heldout_images(build_if_missing=False)    -> set[str] (exclusion list)
"""
from __future__ import annotations

import json
import os

VERSION = "v1"
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# gen_heldout_v1.py writes the split here; resolved independently of cwd
CACHE_DIR = os.environ.get("HELDOUT_DIR", os.path.join(_REPO, f"assets/data/heldout_{VERSION}"))

_ARTIFACTS = ("split.json", "heldout_images.txt")


def _p(name: str) -> str:
    return os.path.join(CACHE_DIR, name)


def _require_cache() -> None:
    missing = [_p(n) for n in _ARTIFACTS if not os.path.exists(_p(n))]
    if missing:
        raise FileNotFoundError(
            "heldout cache missing - training will not start.\n"
            f"  missing file: {', '.join(missing)}\n"
            f"  CACHE_DIR={CACHE_DIR}\n"
            "  Build it with train/gen/gen_heldout_v1.py and put it at the path above.")


def ensure_heldout_split(verbose: bool = True) -> dict:
    """Return the stats from split.json. Does not build: aborts when the cache is missing."""
    _require_cache()
    with open(_p("split.json"), encoding="utf-8") as f:
        stats = json.load(f).get("stats", {})
    if verbose:
        gates = stats.get("gates", {})
        print(f"[heldout {VERSION}] cache dir = {CACHE_DIR} (loading the prebuilt split - not building)")
        print(f"[heldout {VERSION}] ready  heldout={stats.get('n_heldout'):,}  "
              f"md5={str(stats.get('heldout_md5'))[:8]}  "
              f"gates.all_pass={gates.get('all_pass')}")
    return stats


def load_heldout_images(build_if_missing: bool = False, workers: int | None = None) -> set:
    """The set of image keys to exclude from training. build_if_missing/workers are ignored (caller compatibility)."""
    _require_cache()
    with open(_p("heldout_images.txt"), encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}
