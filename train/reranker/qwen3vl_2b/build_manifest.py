#!/usr/bin/env python3
"""build_manifest — reranker training manifest (recap + hardneg → one jsonl row per image).

The first step, shared by all three rerankers (`r32`, `DoRA`, `jina`). The original-style caption
from the recap CSV becomes the positive anchor and the attribute-flipped captions from the hardneg
CSV become the flip negatives, written as one row per image.
Flip axes: `color`, `clothing_type`, `action_state`.

Output  `$OUTPUT_DIR/pab_manifest_rerank_msrv2_hardneg.jsonl`  (1,013,606 rows, 4.4 GB)
Used by `build_negcache.py`, shared between qwen3vl_2b (`N_IMG_NEG=5`) and jina (`N_IMG_NEG=8`)

The bundled manifest under `assets/data/mining/` (md5 `fd7e429ecfab`) makes running this optional;
it is only needed to rebuild from different recap/hardneg CSVs. A change to the input paths or the
axis set is detected through the provenance record and forces a rebuild.

Usage:
    RECAP_CSV=<recap.csv> HARDNEG_CSV=<hardneg.csv> OUTPUT_DIR=<out> \
      PAB_TRAIN=<PAB_Track4 root> python build_manifest.py
"""
import csv, json, os, sys
from collections import Counter
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repository root — all default paths are relative to it

csv.field_size_limit(sys.maxsize)

# ---- paths (train_webp holds the 1024-pixel originals; recap .jpg paths are rewritten to .webp) ----
DATA_ROOT       = os.environ.get("PAB_TRAIN", f"{_REPO}/assets/data/raw/pab_train")
IMG_ROOT        = f"{DATA_ROOT}/train_webp"     # 1024-pixel originals (4x the detail of jpg_512); every recap image is present
IMG_EXT         = ".webp"                        # rewrite the recap .jpg extension to the .webp in train_webp
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", f"{_REPO}/assets/runs/rerank_qw2b")

# ---- data (recap positives + hardneg flip negatives) ----
RECAP_CSV_PATH       = os.environ.get("RECAP_CSV", f"{_REPO}/assets/data/raw/recaption/train_msr_v2.csv")
HARDNEG_CSV_PATH     = os.environ.get("HARDNEG_CSV", f"{_REPO}/assets/data/mining/hardneg_flip.csv")
RERANK_MANIFEST      = f"{OUTPUT_DIR}/pab_manifest_rerank_msrv2_hardneg.jsonl"
RECAP_EXCLUDE_STYLES = ()                   # excluding the original style would remove the anchor — keep this empty
RECAP_MIN_CAPTION_WORDS = 5                 # skip captions shorter than 5 words
FLIP_AXES            = ["color", "clothing_type", "action_state"]   # the three person-attribute axes
FLIP_ALLOWED_AXES    = set(FLIP_AXES)
ORIG_STYLE           = "p00_original"   # caption preset used as the positive anchor

def map_annotation_path_to_local(ann_path, root=IMG_ROOT, ext=IMG_EXT):
    """ "train/imgs_8/full/10.jpg" → ".../<root>/Part 2/imgs_8/full/10<ext>".
    Reinterprets a recap .jpg path against IMG_ROOT and IMG_EXT. ext="" keeps the original extension."""
    parts = ann_path.split("/")
    assert parts[0] == "train" and len(parts) == 4, ann_path
    n = int(parts[1].replace("imgs_", ""))
    part_no = n // 8 + 1
    fname = (os.path.splitext(parts[3])[0] + ext) if ext else parts[3]
    return f"{root}/Part {part_no}/{parts[1]}/{parts[2]}/{fname}"



def parse_recap_csv():
    """recap CSV → [{image, caption, style}]. Empty and under-length captions are skipped."""
    if not os.path.exists(RECAP_CSV_PATH):
        raise FileNotFoundError(RECAP_CSV_PATH)
    rows, stats = [], {"total": 0, "excluded_style": 0, "empty": 0, "short": 0}
    with open(RECAP_CSV_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stats["total"] += 1
            style = (r.get("style") or "").strip()
            if style in RECAP_EXCLUDE_STYLES:
                stats["excluded_style"] += 1; continue
            cap = (r.get("caption") or "").strip()
            if not cap:
                stats["empty"] += 1; continue
            if len(cap.split()) < RECAP_MIN_CAPTION_WORDS:
                stats["short"] += 1; continue
            rows.append({"image": r["image_path"].strip(), "caption": cap, "style": style})
    return rows, stats


def parse_hardneg_csv():
    """hardneg long CSV (image_id,image_path,caption,axis) → {image (annotation): [(axis,caption),...]}.
    Skips rows whose axis is not in FLIP_ALLOWED_AXES, empty or under-length captions, and duplicate (image,axis)."""
    if not HARDNEG_CSV_PATH or not os.path.exists(HARDNEG_CSV_PATH):
        return {}
    hn, stats, axis_dist = {}, {"total": 0, "bad_axis": 0, "short": 0, "dup": 0, "kept": 0}, Counter()
    with open(HARDNEG_CSV_PATH, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stats["total"] += 1
            axis = (r.get("axis") or "").strip()
            cap = (r.get("caption") or "").strip()
            img = (r.get("image_path") or "").strip()
            if axis not in FLIP_ALLOWED_AXES:
                stats["bad_axis"] += 1; continue
            if not cap or len(cap.split()) < RECAP_MIN_CAPTION_WORDS:
                stats["short"] += 1; continue
            pairs = hn.setdefault(img, [])
            if any(a == axis for a, _ in pairs):
                stats["dup"] += 1; continue
            pairs.append((axis, cap)); axis_dist[axis] += 1; stats["kept"] += 1
    print(f"  [hardneg] rows={stats['total']:,} kept={stats['kept']:,} "
          f"(skip axis∉active={stats['bad_axis']:,}, short={stats['short']:,}, dup={stats['dup']:,}) "
          f"images={len(hn):,} axes={sorted(FLIP_ALLOWED_AXES)} dist={dict(axis_dist)}", flush=True)
    return hn


def _manifest_meta():
    """Provenance record: a change to the input paths or the axis set forces a rebuild."""
    return {"_meta": {"hardneg_csv_path": HARDNEG_CSV_PATH, "recap_csv_path": RECAP_CSV_PATH,
                      "flip_axes_active": sorted(FLIP_ALLOWED_AXES)}}


def build_manifest():
    """recap + hardneg → one manifest jsonl row per image.
    Record = {image, captions:[...], orig_caption, flip_captions:[...], flip_axes:[...]}.
    Cached on mtime plus provenance: the rebuild is skipped when both are current."""
    out_path = RERANK_MANIFEST
    if globals().get("FORCE"):
        print("  [manifest] --force → rebuilding", flush=True)
    srcs = [RECAP_CSV_PATH] + ([HARDNEG_CSV_PATH] if (HARDNEG_CSV_PATH and os.path.exists(HARDNEG_CSV_PATH)) else [])
    if os.path.exists(out_path) and not globals().get("FORCE"):
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                meta_ok = json.loads(f.readline()).get("_meta") == _manifest_meta()["_meta"]
        except Exception:
            meta_ok = False
        src_mtime = max(os.path.getmtime(s) for s in srcs)
        if meta_ok and os.path.getmtime(out_path) >= src_mtime:
            print(f"  [manifest] up-to-date (cache+provenance) -> {out_path}", flush=True)
            return
        print(f"  [manifest] rebuilding ({'provenance mismatch' if not meta_ok else 'input CSV is newer'})", flush=True)
    print(f"[manifest] building (recap={RECAP_CSV_PATH}, hardneg={HARDNEG_CSV_PATH}) ...", flush=True)
    recap_rows, st = parse_recap_csv()
    print(f"  [recap] total={st['total']:,} added={len(recap_rows):,} "
          f"(skip style={st['excluded_style']:,}, empty={st['empty']:,}, short={st['short']:,})", flush=True)
    hn = parse_hardneg_csv()
    out = {}
    for r in recap_rows:
        rec = out.get(r["image"])
        if rec is None:
            rec = out[r["image"]] = {"image": r["image"], "captions": [r["caption"]],
                                     "orig_caption": "", "flip_captions": [], "flip_axes": []}
        else:
            rec["captions"].append(r["caption"])
        if r["style"] == ORIG_STYLE and not rec["orig_caption"]:
            rec["orig_caption"] = r["caption"]
    if not any(rec["orig_caption"] for rec in out.values()):
        raise SystemExit(
            f"✗ 0 {ORIG_STYLE} captions — without an anchor every flip negative is discarded.\n"
            f"  The style column in CSV={RECAP_CSV_PATH} must hold preset keys.\n"
            f"  styles observed: {sorted({r['style'] for r in recap_rows})[:12]}")
    n_flip_img = n_flip = n_no_orig = 0
    for img, rec in out.items():
        pairs = hn.get(img, [])
        if pairs and not rec["orig_caption"]:
            n_no_orig += 1; continue
        if pairs:
            rec["flip_captions"] = [c for _, c in pairs]
            rec["flip_axes"] = [a for a, _ in pairs]
            n_flip_img += 1; n_flip += len(pairs)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_manifest_meta(), ensure_ascii=False) + "\n")
        for rec in out.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  [manifest] saved {len(out):,} images -> {out_path} | {n_flip_img:,} img with flips "
          f"({n_flip:,} flips, avg {n_flip/max(1,n_flip_img):.2f}/img)"
          + (f" | flips dropped for want of an orig caption: {n_no_orig:,}" if n_no_orig else ""), flush=True)



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="build the reranker training manifest (recap + hardneg)")
    ap.add_argument("--out", default=OUTPUT_DIR, help="output directory")
    ap.add_argument("--force", action="store_true", help="rebuild even when the cache is current")
    a = ap.parse_args()
    OUTPUT_DIR = a.out
    RERANK_MANIFEST = f"{OUTPUT_DIR}/pab_manifest_rerank_msrv2_hardneg.jsonl"
    FORCE = a.force
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    build_manifest()
