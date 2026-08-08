#!/usr/bin/env python3
"""Build the per-image training manifest (msr v1 / v2) from the caption CSV.

It uses the same filters, record shape and ordering as the trainer's manifest loader, so the
output is reproducible and the trainer only ever reads this file.

  v1  assets/data/raw/recaption/train_msr_v1.csv -> assets/data/manifest/pab_manifest_msr_v1.jsonl
      11 preset styles   used by: anchor_filip_all, anchor_filip_heldout
  v2  assets/data/raw/recaption/train_msr_v2.csv -> assets/data/manifest/pab_manifest_msr_v2.jsonl
      12 preset styles   used by: anchor_tcap_*, mc2h378_peft_*
  v1_scene  train_msr_v1.csv -> pab_manifest_msr_v1_scene.jsonl
      the v1 captions plus a scene field   used by: siglip_maxsim_*

Record (one JSON object per line, images in first-seen order):
  {"image": ..., "label_type": ..., "action_label": ..., "captions": [ ... ]}
label_type and action_label are taken from the row where the image first appears; later rows
only append captions.

Row filters (identical to the trainer):
  empty caption            -> dropped
  fewer than MIN_WORDS(5)  -> dropped

An existing manifest is skipped. The challenge test set is never read.

usage (run from the repository root):
  python train/gen/gen_manifest.py            # build what is missing
  python train/gen/gen_manifest.py --force    # ignore existing files and rebuild
RECAP_DIR / MANIFEST_DIR override the paths.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _abs(p: str) -> str:
    """Resolve a relative path against the repository root."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


CSV_DIR = _abs(os.environ.get("RECAP_DIR", "assets/data/raw/recaption"))
OUT_DIR = _abs(os.environ.get("MANIFEST_DIR", "assets/data/manifest"))

MIN_WORDS = 5                       # captions shorter than this many words are dropped

SPECS = {
    "v1": {"csv": "train_msr_v1.csv", "manifest": "pab_manifest_msr_v1.jsonl",
           "n_style": 11, "users": "anchor_filip_all · anchor_filip_heldout"},
    "v2": {"csv": "train_msr_v2.csv", "manifest": "pab_manifest_msr_v2.jsonl",
           "n_style": 12, "users": "anchor_tcap_* · mc2h378_peft_*"},
    "v1_scene": {"csv": "train_msr_v1.csv", "manifest": "pab_manifest_msr_v1_scene.jsonl",
                 "n_style": 11, "extra": ("scene",),
                 "users": "siglip_maxsim_all · siglip_maxsim_heldout"},
}


def label_from_row(r: dict) -> tuple:
    """CSV normal/anomaly columns -> (label_type, action_label). Identical to the trainer."""
    normal_val = (r.get("normal") or "").strip()
    anomaly_val = (r.get("anomaly") or "").strip()
    if anomaly_val:
        return "anomaly", anomaly_val.lower()
    if normal_val:
        return "normal", normal_val.lower()
    return "normal", ""


def paths_of(tag: str) -> tuple:
    spec = SPECS[tag]
    return os.path.join(CSV_DIR, spec["csv"]), os.path.join(OUT_DIR, spec["manifest"])


def report_existing(tag: str) -> None:
    """Report the state of an existing manifest (without re-reading all of it)."""
    csv_path, out_path = paths_of(tag)
    n = sum(1 for _ in open(out_path, "rb"))
    fresh = os.path.getmtime(out_path) >= os.path.getmtime(csv_path)
    print(f"[manifest {tag}] already present - skipped")
    print(f"                {os.path.relpath(out_path, _REPO)}  "
          f"{os.path.getsize(out_path) / 2 ** 30:.2f} GiB · {n:,} images")
    print(f"                against the CSV {'up to date' if fresh else 'stale - rebuild with --force'}")


def build(tag: str) -> None:
    """Write the per-image manifest in a single pass over the caption CSV."""
    spec = SPECS[tag]
    csv_path, out_path = paths_of(tag)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[manifest {tag}] {os.path.relpath(csv_path, _REPO)}  →  "
          f"{os.path.relpath(out_path, _REPO)}")
    print(f"                used by: {spec['users']}")

    extra = spec.get("extra", ())
    csv.field_size_limit(min(sys.maxsize, 1 << 31) - 1)
    stats = collections.Counter()
    styles = collections.Counter()
    out: dict[str, dict] = {}
    t0 = time.time()

    with open(csv_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stats["total"] += 1
            cap = (r.get("caption") or "").strip()
            if not cap:
                stats["empty_caption"] += 1
                continue
            if len(cap.split()) < MIN_WORDS:
                stats["short_caption"] += 1
                continue
            styles[r.get("style", "").strip()] += 1
            img = r["image_path"].strip()
            rec = out.get(img)
            if rec is None:
                lt, al = label_from_row(r)
                rec = {"image": img, "label_type": lt,
                       "action_label": al, "captions": [cap]}
                for fn in extra:                    # raw fields for the soft labels (the trainer normalises them)
                    rec[fn] = (r.get(fn) or "").strip().lower()
                out[img] = rec
            else:
                rec["captions"].append(cap)
            if stats["total"] % 2_000_000 == 0:
                print(f"                {stats['total']:>12,} rows  {len(out):>9,} images  "
                      f"{time.time() - t0:.0f}s", flush=True)

    n_cap = sum(len(v["captions"]) for v in out.values())
    print(f"                CSV total {stats['total']:,}  "
          f"drop(empty {stats['empty_caption']:,} / <{MIN_WORDS} words {stats['short_caption']:,})")
    print(f"                {len(styles)} styles " +
          ("even" if len(set(styles.values())) == 1 else f"uneven {dict(styles.most_common(3))}"))
    if len(styles) != spec["n_style"]:
        print(f"                [warn] {len(styles)} styles - expected {spec['n_style']}")

    # atomic write: an interrupted run leaves no half-written manifest
    tmp = out_path + ".partial"
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in out.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    print(f"                saved {len(out):,} images ({n_cap:,} captions) "
          f"in {time.time() - t0:.0f}s · {os.path.getsize(out_path) / 2 ** 30:.2f} GiB")


def main():
    ap = argparse.ArgumentParser(
        description="build the per-image training manifest (msr v1 / v2; the test set is not used)")
    ap.add_argument("--force", action="store_true", help="rebuild even if it already exists")
    args = ap.parse_args()

    todo = [t for t in SPECS if args.force or not os.path.exists(paths_of(t)[1])]
    problems = [f"caption CSV not found: {os.path.relpath(paths_of(t)[0], _REPO)}"
                for t in todo if not os.path.exists(paths_of(t)[0])]
    if problems:
        raise SystemExit("[input check failed] fix the following and run again.\n"
                         + "\n".join(f"  - {p}" for p in problems))

    for tag in SPECS:
        if tag in todo:
            build(tag)
        else:
            report_existing(tag)
        print()


if __name__ == "__main__":
    main()
