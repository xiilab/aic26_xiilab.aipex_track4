"""PAB 1M annotations → open_clip training CSV, the input to the metaclip_v1 trainer.

Output: assets/data/manifest/train.csv (filepath,title). It lives under data/manifest rather than
runs/ because it is a training input.

Image path rule:
  annotation "train/imgs_X/category/N.jpg"
  → {PAB_TRAIN}/train_webp/Part {X//8+1}/imgs_X/category/N.webp

Usage:
  python train/gen/gen_metaclip_v1_csv.py                  # → assets/data/manifest/train.csv
  OUT_CSV=<path> python train/gen/gen_metaclip_v1_csv.py   # override the output location
"""
from __future__ import annotations
import csv
import glob
import json
import os
import sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root — all default paths are relative to it

DATA_ROOT  = os.environ.get("PAB_TRAIN", f"{_REPO}/assets/data/raw/pab_train")
ANN_DIR    = f"{DATA_ROOT}/annotation/train"
OUT_CSV    = os.environ.get("OUT_CSV", f"{_REPO}/assets/data/manifest/train.csv")   # training input = data/manifest


def resolve_path(rel_path: str) -> str | None:
    parts = rel_path.split("/")
    # parts: ["train", "imgs_X", "category", "N.jpg"]
    if len(parts) < 4 or not parts[1].startswith("imgs_"):
        return None
    imgs_group = int(parts[1].split("_")[1])
    part_name  = f"Part {imgs_group // 8 + 1}"
    webp_name  = os.path.splitext(parts[-1])[0] + ".webp"
    return os.path.join(DATA_ROOT, "train_webp", part_name,
                        parts[1], parts[2], webp_name)


def main() -> None:
    if os.path.exists(OUT_CSV):           # reuse an existing CSV instead of rebuilding it
        print(f"[gen_csv] CSV already exists → reusing it, skipping the rebuild: {OUT_CSV}")
        return
    ann_files = sorted(glob.glob(f"{ANN_DIR}/imgs_*.json"))
    if not ann_files:
        sys.exit(f"No annotation files found in {ANN_DIR}")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    written = skipped = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["filepath", "title"])

        for ann_file in ann_files:
            with open(ann_file, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    img_path = resolve_path(item["image"])
                    if img_path is None or not os.path.exists(img_path):
                        skipped += 1
                        continue
                    caption = item["caption"].replace("\n", " ").strip()
                    writer.writerow([img_path, caption])
                    written += 1

            if written % 100_000 == 0 and written > 0:
                print(f"  {written:,} written, {skipped:,} skipped")

    print(f"\nDone. CSV: {OUT_CSV}")
    print(f"  Written : {written:,}")
    print(f"  Skipped : {skipped:,}")


if __name__ == "__main__":
    main()
