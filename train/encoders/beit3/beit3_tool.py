#!/usr/bin/env python3
"""beit3_tool — single entry point for BEiT3 training-data and held-out tooling.

BEiT3 takes a whole task-356 pair index via `--data_path <DIR>`, so several steps around training
operate on that index. All four live here as subcommands.

  build-index   recap (train_msr, presets p00-p11) / annotation -> task-356 pair index
  exclude       copy of the index with the held-out images removed  (required before training)
  remap-helip   remap HELIP hard pairs onto the filtered index's row numbers
  eval          score per-epoch ckpts on the held-out bench -> best epoch

Usual order:
  1. build-index (only when the index does not exist yet)
  2. exclude      -> training takes this output directory as --data_path
  3. remap-helip  -> only for the helip recipe, which passes `--use_helip`
  4. eval         -> during training (--watch) or after it

Bench definition, md5, gates and metrics all come from [`../heldout_bench.py`](../heldout_bench.py).

Examples:
  python beit3_tool.py exclude --src $BEIT3_DATA/pab_v2_multi_webp \\
         --dst $BEIT3_DATA/pab_v2_multi_webp_heldout
  python beit3_tool.py remap-helip --src $BEIT3_DATA/pab_full_webp \\
         --dst $BEIT3_DATA/pab_full_webp_heldout
  python beit3_tool.py eval --run $RUNS_ROOT/beit3_v2_heldout \\
         --epochs 0-3 --watch --gpu 2
  # build-index — pair index from recap (train_msr) captions; --recap '' uses the base annotation
  python beit3_tool.py build-index --style-mode multi        --out $BEIT3_DATA/pab_v2_multi_webp    # v2: captions-dict over every preset · goal/wentwrong/full
  python beit3_tool.py build-index --style-mode all          --out $BEIT3_DATA/pab_recap_full_webp  # one example per style
  python beit3_tool.py build-index --style-mode p00_original --out $BEIT3_DATA/pab_full_webp        # original captions for stage1/helip

Note: `eval` needs torch/timm/torchscale — env = `.venv_beit3eval`. The other subcommands use only
      the standard library and run under any python.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODERS = os.path.dirname(HERE)
EVAL_DIR = os.path.join(ENCODERS, "eval")
PKG = os.path.dirname(os.path.dirname(ENCODERS))
BEIT3_SRC = os.environ.get("BEIT3_SRC", os.path.join(PKG, "third_party", "beit3"))
BEIT3_MODELS = os.environ.get("BEIT3_MODELS", os.path.join(PKG, "assets", "model", "encoder", "beit3_pre"))  # spm + COCO init, same default as train.py

sys.path.insert(0, EVAL_DIR)
import heldout_bench as HB                                           # noqa: E402


# ════════════════════════════════════════════════════════════ common
def read_index_rows(ann_dir: str, n_files: int) -> list[str]:
    """Collect images in the same order as BEiT3's `BaseDataset`: pair_0 … pair_{n-1}, **numerically**.

    With `sorted()` string ordering, pair_10 would come before pair_2 and every row number would
    shift.
    """
    imgs = []
    for i in range(n_files):
        p = os.path.join(ann_dir, f"pair_{i}.json")
        if not os.path.exists(p):
            raise SystemExit(f"✗ {p} not found — check --n-files.")
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    imgs.append(json.loads(line)["image"])
    return imgs


def _load_recap(csv_path, exclude=(), min_words=5):
    """recap CSV (train_msr) → {(imgs_X, goal|wentwrong, num): {style: caption}}.

    image_path = "train/imgs_X/goal|wentwrong/N.jpg", the same rule the encoders' recap uses.
    Empty or short (<min_words) captions and excluded styles are skipped; p00_original acts as the
    original caption. The style list is not hardcoded — the CSV's style values become the keys, so a
    larger preset set flows through without a code change.
    """
    import csv
    rec, n = {}, 0
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n += 1
            st = (r.get("style") or "").strip()
            if st in exclude:
                continue
            cap = (r.get("caption") or "").strip()
            if not cap or len(cap.split()) < min_words:
                continue
            parts = (r.get("image_path") or "").split("/")
            if len(parts) < 4:
                continue
            key = (parts[1], parts[2], os.path.splitext(parts[3])[0])   # (imgs_X, goal/wentwrong, num)
            rec.setdefault(key, {})[st] = cap
    print(f"[build-index] recap CSV {csv_path} · {n:,} rows → {len(rec):,} images · captions per style")
    return rec


# ════════════════════════════════════════════════════════════ build-index
def cmd_build_index(a):
    """PAB train_webp goal/wentwrong pairs → BEiT3 pair_K.json (one file per group: pair_K = imgs_K).

    The two rows of a given (imgs_X, N) pair are written **back to back** so that with a large batch
    and no shuffling they land in the same batch and act as hard negatives. The vendor loader
    (datasets.py) reads pair_0..pair_{num_files-1} as one file per group, so exactly one file is
    written per group and empty groups become empty files (--num-files, PAB=75).
    """
    from pathlib import Path
    pab = Path(a.pab_root)
    ann_dir, webp_root = pab / "annotation" / "train", pab / "train_webp"
    out_ann = Path(a.out) / "annotation" / "train"
    out_ann.mkdir(parents=True, exist_ok=True)

    import random as _random
    REC = _load_recap(a.recap, tuple(a.style_exclude or ()), a.min_words) if a.recap else None
    STYLE_MODE = a.style_mode                     # all | multi | random | <style name, e.g. p00_original>
    RNG = _random.Random(a.seed)

    if STYLE_MODE == "multi":                                 # beit3_v2 layout — one captions-dict per image (all styles) · goal/wentwrong/full
        if REC is None:
            raise SystemExit("✗ multi mode requires a --recap CSV.")
        from collections import defaultdict
        by_group = defaultdict(list)                          # group_int → [(imgs_name, seg, num, caps), ...]
        for (imgs_name, seg, num), caps in REC.items():
            if caps:
                by_group[int(imgs_name.split("_")[1])].append((imgs_name, seg, num, caps))
        # The vendor loader (datasets.py) hardcodes pair_0..pair_{num_files-1} = imgs_0..imgs_{num_files-1},
        # one file per group. So write exactly one file per group (pair_{group}.json) and fill the range
        # with empty files for missing groups (per_file is ignored).
        over = sorted(g for g in by_group if g >= a.num_files)
        if over:
            print(f"  ⚠ imgs groups {over} are >= --num-files({a.num_files}) — the loader will not read them. Raise --num-files.")
        total, segc = 0, {}
        for grp in range(a.num_files):                        # pair_{grp}.json = imgs_{grp} (empty file if absent). image_id={grp}_{running} is a label the loader reassigns
            rows = []
            for run, (imgs_name, seg, num, caps) in enumerate(sorted(
                    by_group.get(grp, []), key=lambda t: (t[1], int(t[2]) if t[2].isdigit() else 10 ** 9))):
                rows.append({"image": f"train/{imgs_name}/{seg}/{num}.jpg",
                             "captions": caps, "image_id": f"{grp}_{run}"})
                segc[seg] = segc.get(seg, 0) + 1
            _write_chunk(out_ann, grp, rows)
            total += len(rows)
        print(f"[build-index] multi (v2 layout) · pair_0..pair_{a.num_files - 1} ({a.num_files} files) · {total:,} image rows · segments {segc}")
        return

    def find_part(imgs_name):
        for d in sorted(webp_root.iterdir()):
            if d.is_dir() and d.name.startswith("Part") and (d / imgs_name).is_dir():
                return d.name
        return None

    def load_caps(imgs_json):
        out = {}
        with open(imgs_json, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                parts = d.get("image", "").split("/")
                if len(parts) < 4:
                    continue
                out[(parts[2], os.path.splitext(parts[3])[0])] = d.get("caption", "")
        return out

    files = sorted(ann_dir.glob("imgs_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"[build-index] {len(files)} annotation files · output {out_ann}")

    from collections import defaultdict
    by_group, n_pairs = defaultdict(list), 0                  # group_int(imgs_K) → that group's entries. goal/wentwrong are consecutive rows (hard-neg).
    for ann in files:
        imgs_name = ann.stem
        grp = int(imgs_name.split("_")[1])
        part = find_part(imgs_name)
        if part is None:
            continue
        caps = None if REC is not None else load_caps(ann)
        gdir, wdir = webp_root / part / imgs_name / "goal", webp_root / part / imgs_name / "wentwrong"
        if not (gdir.is_dir() and wdir.is_dir()):
            continue
        g = {p.stem for p in gdir.glob("*.webp")}
        w = {p.stem for p in wdir.glob("*.webp")}
        for num in sorted(g & w, key=lambda x: int(x) if x.isdigit() else 10 ** 9):
            if REC is not None:                                   # recap mode — train_msr style captions
                gm = REC.get((imgs_name, "goal", num), {})
                wm = REC.get((imgs_name, "wentwrong", num), {})
                sts = sorted(set(gm) & set(wm))                   # styles present for both goal and wentwrong
                if not sts:
                    continue
                if STYLE_MODE == "random":
                    sts = [RNG.choice(sts)]                       # one random style per image
                elif STYLE_MODE != "all":
                    sts = [STYLE_MODE] if STYLE_MODE in gm and STYLE_MODE in wm else []
                for st in sts:
                    sfx = "" if STYLE_MODE == "random" else f"_{st}"
                    by_group[grp].append({"image": f"train/{imgs_name}/goal/{num}.jpg", "caption": gm[st],
                                          "image_id": f"{imgs_name}_{num}_g{sfx}"})
                    by_group[grp].append({"image": f"train/{imgs_name}/wentwrong/{num}.jpg", "caption": wm[st],
                                          "image_id": f"{imgs_name}_{num}_w{sfx}"})
                    n_pairs += 1
            else:                                                 # base annotation
                gc, wc = caps.get(("goal", num)), caps.get(("wentwrong", num))
                if not (gc and wc):
                    continue
                by_group[grp].append({"image": f"train/{imgs_name}/goal/{num}.jpg", "caption": gc,
                                      "image_id": f"{imgs_name}_{num}_g"})
                by_group[grp].append({"image": f"train/{imgs_name}/wentwrong/{num}.jpg", "caption": wc,
                                      "image_id": f"{imgs_name}_{num}_w"})
                n_pairs += 1

    # The vendor loader (datasets.py) hardcodes pair_0..pair_{num_files-1} = imgs_0..imgs_{num_files-1},
    # one file per group. So write one file per group (pair_{grp}.json) and fill the range with empty
    # files for missing groups (per_file is ignored — a group is never split).
    over = sorted(gp for gp in by_group if gp >= a.num_files)
    if over:
        print(f"  ⚠ imgs groups {over} are >= --num-files({a.num_files}) — the loader will not read them. Raise --num-files.")
    total = 0
    for grp in range(a.num_files):
        rows = by_group.get(grp, [])
        _write_chunk(out_ann, grp, rows)
        total += len(rows)
    print(f"[build-index] {n_pairs:,} pairs · pair_0..pair_{a.num_files - 1} ({a.num_files} files) · {total:,} rows")


def _write_chunk(out_ann, idx, chunk):
    p = out_ann / f"pair_{idx}.json"
    with open(p, "w", encoding="utf-8") as f:
        for d in chunk:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  [write] {p}  lines={len(chunk)}")


# ════════════════════════════════════════════════════════════ exclude
def cmd_exclude(a):
    """Copy the index with the held-out image rows removed. Images are carried over as symlinks."""
    src_ann = os.path.join(a.src, "annotation", "train")
    if not os.path.isdir(src_ann):
        raise SystemExit(f"✗ {src_ann} not found — --src must be a directory containing annotation/train/.")
    files = sorted(glob.glob(os.path.join(src_ann, "*.json")))
    if not files:
        raise SystemExit(f"✗ no *.json in {src_ann}.")

    hl = a.heldout_list or HB.heldout_list_path(a.heldout_dir)
    if not os.path.exists(hl):
        raise SystemExit(f"✗ exclusion list not found: {hl}")
    HB.verify_identity(a.heldout_dir)
    HB.require_gates(a.heldout_dir, strict=True)
    hold = HB.load_heldout_set(hl)
    print(f"[exclude] list {len(hold):,} images ← {hl}")
    print(f"[exclude] src {a.src} ({len(files)} json)")

    dst_ann = os.path.join(a.dst, "annotation", "train")
    os.makedirs(dst_ann, exist_ok=True)

    tot_in = tot_out = 0
    dropped, kept_imgs = set(), set()
    for fp in files:
        keep = []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                tot_in += 1
                key = HB._as_jpg(json.loads(line)["image"])
                if key in hold:
                    dropped.add(key)
                else:
                    kept_imgs.add(key)
                    keep.append(line)
        tot_out += len(keep)
        with open(os.path.join(dst_ann, os.path.basename(fp)), "w", encoding="utf-8") as f:
            f.write("\n".join(keep) + ("\n" if keep else ""))

    print(f"[exclude] rows {tot_in:,} → {tot_out:,} (-{tot_in - tot_out:,}, "
          f"{100 * (tot_in - tot_out) / max(1, tot_in):.2f}%)")
    print(f"[exclude] {len(dropped):,} images excluded · {len(kept_imgs):,} images kept")
    if not dropped:
        raise SystemExit("✗ 0 rows dropped — the index's image format may differ from the list. "
                         f"example: {json.loads(open(files[0]).readline())['image']!r}")
    leak = kept_imgs & hold
    print(f"[verify] remaining index ∩ exclusion list = {len(leak):,} {'✓' if not leak else '✗ leak'}")

    for name in sorted(os.listdir(a.src)):                  # symlink the images and hardpairs
        if name == "annotation":
            continue
        s, d = os.path.join(a.src, name), os.path.join(a.dst, name)
        if os.path.lexists(d):
            continue
        os.symlink(os.path.realpath(s), d)
        print(f"[link] {name} → {os.path.realpath(s)}")

    _save(os.path.join(a.dst, "heldout_exclusion.json"),
          {"src": a.src, "dst": a.dst, "heldout_list": hl,
           "heldout_md5": HB.md5_of_list(sorted(hold)), "heldout_size": len(hold),
           "lines_in": tot_in, "lines_out": tot_out, "images_dropped": len(dropped),
           "images_kept": len(kept_imgs), "leak": len(leak), "json_files": len(files)})


# ════════════════════════════════════════════════════════════ remap-helip
def cmd_remap_helip(a):
    """Move mined HELIP hard pairs onto the filtered index's row numbers (nothing is re-mined).

    `hard_pairs` is (N, k) where the values are row numbers in the training index, and
    `HelipBatchSampler` asserts `shape[0] == len(dataset)`. Once exclusion shrinks N, that assert
    fires.

      · excluded seed rows       → removed
      · excluded neighbours      → slot removed, then the surviving neighbours are tiled to refill k
      · seeds with no neighbours → marked as outliers (excluded as seeds) and filled with their own index
    """
    import numpy as np

    hp_src = os.path.join(a.src, a.name)
    if not os.path.exists(hp_src):
        raise SystemExit(f"✗ hardpairs not found: {hp_src}")
    HB.verify_identity(a.heldout_dir)
    hold = HB.load_heldout_set(a.heldout_list or HB.heldout_list_path(a.heldout_dir))

    src_imgs = read_index_rows(os.path.join(a.src, "annotation", "train"), a.n_files)
    dst_imgs = read_index_rows(os.path.join(a.dst, "annotation", "train"), a.n_files)
    print(f"[rows] src {len(src_imgs):,} → dst {len(dst_imgs):,}")

    hp = np.load(hp_src)
    print(f"[hardpairs] {a.name} shape={hp.shape} dtype={hp.dtype}")
    if hp.shape[0] != len(src_imgs):
        raise SystemExit(f"✗ hardpairs N({hp.shape[0]:,}) ≠ src row count({len(src_imgs):,}) — "
                         f"--src is not the data these hardpairs were mined against.")

    keep = [i for i, im in enumerate(src_imgs) if HB._as_jpg(im) not in hold]
    if [src_imgs[i] for i in keep] != dst_imgs:
        raise SystemExit("✗ the dst index is not 'src minus the excluded rows' — "
                         "check that it was produced by `exclude`.")
    print(f"[map] kept {len(keep):,} · removed {len(src_imgs) - len(keep):,}")

    old2new = np.full(len(src_imgs), -1, dtype=np.int64)
    old2new[np.asarray(keep, dtype=np.int64)] = np.arange(len(keep), dtype=np.int64)

    k = hp.shape[1]
    out = np.zeros((len(keep), k), dtype=np.int32)
    n_full = n_pad = n_empty = dropped_slots = 0
    empty_rows = []
    for new_i, old_i in enumerate(keep):
        surv = old2new[hp[old_i]]
        surv = surv[surv >= 0]
        dropped_slots += k - len(surv)
        if len(surv) == k:
            out[new_i] = surv; n_full += 1
        elif len(surv):
            out[new_i] = np.resize(surv, k); n_pad += 1        # tile the survivors
        else:
            out[new_i] = new_i; n_empty += 1; empty_rows.append(new_i)

    print(f"[remap] all neighbours survived {n_full:,} · some survived, tiled {n_pad:,} · none survived {n_empty:,}")
    print(f"[remap] neighbour slots removed {dropped_slots:,}/{len(keep) * k:,} "
          f"({100 * dropped_slots / (len(keep) * k):.2f}%)")

    om_src = os.path.splitext(hp_src)[0] + "_outlier.npy"
    if os.path.exists(om_src):
        om = np.load(om_src).astype(bool)
        if om.shape != (len(src_imgs),):
            raise SystemExit(f"✗ outlier mask shape {om.shape} ≠ ({len(src_imgs)},)")
        om_new = om[np.asarray(keep)]
        print(f"[outlier] source {int(om.sum()):,} → kept {int(om_new.sum()):,}")
    else:
        om_new = np.zeros(len(keep), dtype=bool)
        print("[outlier] no source mask → creating one")
    if empty_rows:
        om_new[np.asarray(empty_rows)] = True
    n_seeds = int((~om_new).sum())
    print(f"[outlier] final {int(om_new.sum()):,} excluded → {n_seeds:,} seed candidates")

    assert out.shape == (len(keep), k) and om_new.shape == (len(keep),)
    assert out.min() >= 0 and out.max() < len(keep), (out.min(), out.max())
    if n_seeds == 0:
        raise SystemExit("✗ 0 seed candidates — the remap is wrong.")

    hp_dst = os.path.join(a.dst, a.name)
    om_dst = os.path.splitext(hp_dst)[0] + "_outlier.npy"
    for p in (hp_dst, om_dst):
        if os.path.islink(p):
            os.unlink(p)
    np.save(hp_dst, out); np.save(om_dst, om_new)
    print(f"saved → {hp_dst}\nsaved → {om_dst}")
    _save(os.path.join(a.dst, "helip_remap.json"),
          {"src": a.src, "dst": a.dst, "name": a.name,
           "heldout_md5": HB.md5_of_list(sorted(hold)),
           "n_src": len(src_imgs), "n_dst": len(keep), "k": int(k),
           "neighbors_full": n_full, "neighbors_padded": n_pad, "neighbors_empty": n_empty,
           "dropped_slots": int(dropped_slots), "outliers": int(om_new.sum()),
           "seed_candidates": n_seeds})


# ════════════════════════════════════════════════════════════ eval
def cmd_eval(a):
    """Score per-epoch ckpts on the held-out bench and pick the best epoch (needs torch/timm/torchscale).

    Preprocessing and tokenisation match `third_party/beit3/inference.py`:
    Resize((384,384), BICUBIC) · IMAGENET_INCEPTION mean/std · XLMRoberta(beit3.spm) ·
    `model(image=…, only_infer=True)` / `model(text_description=…, padding_mask=…, only_infer=True)`.
    The token length is detected from the ckpt's `args.num_max_bpe_tokens`.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    sys.path.insert(0, BEIT3_SRC)

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from timm import create_model
    from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
    from torchvision import transforms
    from transformers import XLMRobertaTokenizer
    import modeling_finetune                                          # noqa: F401  registers the model with create_model

    DEV = "cuda:0"
    TF = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])

    spm = a.spm or os.path.join(BEIT3_MODELS, "beit3.spm")
    if not os.path.exists(spm):
        raise SystemExit(f"✗ beit3.spm not found: {spm} (pass --spm)")
    eps = HB.parse_epochs(a.epochs) if a.epochs else []
    if not eps and not a.with_best:
        raise SystemExit("either --epochs or --with-best is required.")
    benches = ["main", "hard"] if a.bench == "both" else [a.bench]
    out = a.out or os.path.join(a.run, "heldout_eval.json")

    def ck_path(label):
        return os.path.join(a.run, f"checkpoint-{label}.pth")

    def still_training():
        try:
            return bool(subprocess.run(["pgrep", "-f", a.train_proc],
                                       capture_output=True, text=True).stdout.strip())
        except Exception:
            return True

    def wait_for(path):
        while not os.path.exists(path):
            if not still_training():
                print(f"  [watch] no training process → stop waiting for {os.path.basename(path)}")
                return False
            time.sleep(a.poll)
        last, stable = -1, None
        while True:
            sz = os.path.getsize(path)
            if sz == last:
                if stable is None:
                    stable = time.time()
                elif time.time() - stable >= a.settle:
                    return True
            else:
                last, stable = sz, None
            time.sleep(min(15, a.poll))

    def ckpt_args(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        arg = ck.get("args")
        if arg is None:
            return {}
        return dict(vars(arg)) if hasattr(arg, "__dict__") else dict(arg)

    def load_model(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = create_model(a.model)
        res = model.load_state_dict(ck.get("model", ck), strict=False)
        nm, nu = len(res.missing_keys), len(res.unexpected_keys)
        if nm or nu:
            print(f"  [load] missing {nm} · unexpected {nu}")
            if nm and nu:
                print(f"    missing e.g.: {res.missing_keys[:3]}\n    unexpected e.g.: {res.unexpected_keys[:3]}")
                raise SystemExit("  ✗ ckpt does not match the model architecture — check --model.")
        return model.eval().to(DEV)

    def seg(tok, text, max_len):
        ids = tok.convert_tokens_to_ids(tok.tokenize(text))[:max_len - 2]
        ids = [tok.bos_token_id] + ids + [tok.eos_token_id]
        n = len(ids)
        return ids + [tok.pad_token_id] * (max_len - n), [0] * n + [1] * (max_len - n)

    @torch.no_grad()
    def enc_gallery(model, paths):
        feats = []
        for i in range(0, len(paths), a.batch):
            px = torch.stack([TF(Image.open(p).convert("RGB")) for p in paths[i:i + a.batch]])
            v, _ = model(image=px.to(DEV), only_infer=True)
            feats.append(F.normalize(v.float(), dim=-1).cpu())
            if (i // a.batch) % 100 == 0:
                print(f"    gallery {min(i + a.batch, len(paths)):>6,}/{len(paths):,}", flush=True)
        return torch.cat(feats)

    @torch.no_grad()
    def enc_queries(model, tok, queries, max_len):
        feats = []
        for i in range(0, len(queries), a.batch):
            s = [seg(tok, r["caption"], max_len) for r in queries[i:i + a.batch]]
            ids = torch.tensor([x[0] for x in s], device=DEV)
            pad = torch.tensor([x[1] for x in s], device=DEV)
            _, t = model(text_description=ids, padding_mask=pad, only_infer=True)
            feats.append(F.normalize(t.float(), dim=-1).cpu())
        return torch.cat(feats)

    print(f"[eval] run={a.run}\n[eval] model={a.model} spm={spm}")
    md5 = HB.verify_identity(a.heldout_dir)
    HB.require_gates(a.heldout_dir, strict=True)

    loaded = {}
    for b in benches:
        q, g, _ = HB.load_bench(b, a.heldout_dir)
        loaded[b] = (q, g, [HB.map_to_local(p) for p in g])
        print(f"[{b}] queries {len(q):,} · gallery {len(g):,}")
        HB.check_leak(q, g, b, HB.heldout_list_path(a.heldout_dir))

    labels = [str(e) for e in eps] + (["best"] if a.with_best else [])
    have = [ck_path(l) for l in labels if os.path.exists(ck_path(l))]
    if not have and not a.watch:
        raise SystemExit(f"no ckpt to score: {a.run}")
    if not have:
        first = ck_path(labels[0])
        print(f"[watch] waiting for the first ckpt … {first}", flush=True)
        if not wait_for(first):
            raise SystemExit("stopped waiting: no training process.")
        have = [first]

    ca = ckpt_args(have[0])
    trained = ca.get("num_max_bpe_tokens")
    max_len = a.max_len or trained or 64
    src = "--max-len" if a.max_len else ("ckpt args" if trained else "default")
    print(f"[eval] max_len={max_len} ({src})" + (f" · trained={trained}" if trained else ""))
    if a.max_len and trained and a.max_len != trained:
        print(f"  ⚠ --max-len {a.max_len} ≠ trained {trained} — text embeddings will differ from training")
    if ca.get("input_size") not in (None, 384):
        print(f"  ⚠ ckpt input_size={ca['input_size']} but preprocessing is fixed at 384")
    if ca.get("data_path"):
        print(f"[eval] training data = {ca['data_path']}")

    tok = XLMRobertaTokenizer(spm)
    rows, missing = [], []
    for label in labels:
        ck = ck_path(label)
        if not os.path.exists(ck):
            if a.watch:
                print(f"\n[ep{label}] waiting … {ck}", flush=True)
                if not wait_for(ck):
                    missing.append(label); continue
                print(f"[ep{label}] ckpt arrived", flush=True)
            else:
                missing.append(label); print(f"\n[ep{label}] no ckpt → skip"); continue
        t0 = time.time()
        print(f"\n[ep{label}] {ck}", flush=True)
        model = load_model(ck)
        rec = {"epoch": label, "ckpt": ck}
        for b in benches:
            q, g, paths = loaded[b]
            G, Q = enc_gallery(model, paths), enc_queries(model, tok, q, max_len)
            rec[b] = m = HB.score(Q, G, q, g)
            print(f"  [{b}] mAP@10={m['mAP@10']:.4f} R@1={m['R@1']:.4f} "
                  f"R@5={m['R@5']:.4f} R@10={m['R@10']:.4f} margin={m['margin_agg']:.4f}")
            if a.dump_pool:
                # Build reranker evaluation inputs from this encoder's held-out embeddings.
                HB.dump_bench_inputs(b, q, g, Q, G,
                                     os.path.join(a.dump_pool, b),
                                     topk=a.pool_topk, heldout_dir=a.heldout_dir)
            del G, Q
        del model
        torch.cuda.empty_cache()
        rec["sec"] = round(time.time() - t0, 1)
        rows.append(rec)
        _save(out, {"model": a.model, "run": a.run, "max_len": max_len,
                    "trained_max_len": trained, "train_data": ca.get("data_path"),
                    "heldout_md5": md5, "split": HB.split_path(a.heldout_dir),
                    "rows": rows, "missing": missing}, quiet=True)

    if not rows:
        raise SystemExit(f"no ckpt was scored (missing: {missing})")
    best, sel = HB.print_table(rows, benches)
    if missing:
        print(f"  ⚠ skipped for want of a ckpt: {missing}")
    _save(out, {"model": a.model, "run": a.run, "max_len": max_len,
                "trained_max_len": trained, "train_data": ca.get("data_path"),
                "heldout_md5": md5, "split": HB.split_path(a.heldout_dir),
                "select_bench": sel, "best_epoch": best["epoch"],
                "rows": rows, "missing": missing})

    if getattr(a, "deploy_rep", None):               # deploy the selection, only if the inputs are intact
        if not best or best.get("epoch") is None:
            raise SystemExit("[deploy-rep] ✗ no epoch was selected — not deploying")
        src = os.path.join(a.run, f"checkpoint-{best['epoch']}.pth")
        if not os.path.exists(src):
            raise SystemExit(f"[deploy-rep] ✗ selected ckpt not found → not deploying: {src}")
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        print(f"[deploy-rep] checkpoint-{best['epoch']}.pth → assets/model_rep/encoder/{a.deploy_rep}")
        rc = subprocess.run([sys.executable, os.path.join(_repo, "tools", "promote.py"),
                             "--rep", "--kind", "encoder", "--name", a.deploy_rep, "--src", src]).returncode
        if rc:
            raise SystemExit(f"[deploy-rep] ✗ promote failed (rc={rc})")
        print(f"[deploy-rep] ✓ build the cache_rep cache with pipeline/S1_base/encode/encode_beit3.py "
              f"--recipe {{v2|helip}} --rep")


def _save(path, doc, quiet=False):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    if not quiet:
        print(f"saved → {path}")


# ════════════════════════════════════════════════════════════ CLI
def main():
    ap = argparse.ArgumentParser(
        description="BEiT3 training-data and held-out tooling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="order: build-index → exclude → (remap-helip) → eval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-index", help="recap(train_msr)/annotation → task-356 pair index")
    p.add_argument("--pab-root", default=HB.PAB_TRAIN, help="PAB_Track4 root (annotation/ + train_webp/)")
    p.add_argument("--out", required=True, help="output directory (annotation/train/pair_K.json is written there)")
    p.add_argument("--per-file", type=int, default=10000)
    p.add_argument("--num-files", type=int, default=75,
                   help="multi mode: number of pair_0..pair_{N-1} files, matching the vendor datasets.py num_files (PAB=75)")
    p.add_argument("--recap", default=os.path.join(os.path.dirname(HB.PAB_TRAIN), "recaption", "train_msr_v1.csv"),
                   help="build the index from a recap CSV (train_msr); styles are taken from the CSV's style column "
                        "(v2=p00-p10, 11 styles · msr_v2 adds p11_compound, 12). Use --recap '' for the base annotation.")
    p.add_argument("--style-mode", default="all",
                   help="recap styles: all (one example per style) · multi (captions-dict per image) · "
                        "random (one fixed style per image) · <style name, e.g. p00_original for stage1>")
    p.add_argument("--style-exclude", nargs="*", default=[], help="styles to exclude (none by default — p00_original is the original)")
    p.add_argument("--min-words", type=int, default=5, help="skip captions shorter than this")
    p.add_argument("--seed", type=int, default=16, help="seed for style-mode=random")
    p.set_defaults(fn=cmd_build_index)

    p = sub.add_parser("exclude", help="copy the index with the held-out images removed")
    p.add_argument("--src", required=True, help="source data directory (contains annotation/train/)")
    p.add_argument("--dst", required=True, help="directory to write the filtered index to")
    p.add_argument("--heldout-dir", default=None)
    p.add_argument("--heldout-list", default=None)
    p.set_defaults(fn=cmd_exclude)

    p = sub.add_parser("remap-helip", help="remap HELIP hard pairs onto the excluded index's row numbers")
    p.add_argument("--src", required=True, help="source data directory the hardpairs were mined against")
    p.add_argument("--dst", required=True, help="excluded index directory, written to")
    p.add_argument("--name", default="helip_hardpairs.npy")
    p.add_argument("--n-files", type=int, default=75)
    p.add_argument("--heldout-dir", default=None)
    p.add_argument("--heldout-list", default=None)
    p.set_defaults(fn=cmd_remap_helip)

    p = sub.add_parser("eval", help="score per-epoch ckpts on the held-out bench → best epoch")
    p.add_argument("--run", required=True, help="run directory holding checkpoint-N.pth")
    p.add_argument("--epochs", default="", help='"0-3" · "0,2"')
    p.add_argument("--with-best", action="store_true", help="also score checkpoint-best.pth")
    p.add_argument("--max-len", type=int, default=None,
                   help="the --num_max_bpe_tokens used in training; detected from the ckpt args by default")
    p.add_argument("--bench", default="main", choices=["main", "hard", "both"])
    p.add_argument("--model", default="beit3_large_patch16_384_retrieval")
    p.add_argument("--spm", default=None)
    p.add_argument("--heldout-dir", default=None)
    p.add_argument("--gpu", default="6")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--out", default=None, help="result JSON (default = <run>/heldout_eval.json)")
    p.add_argument("--deploy-rep", default=None, metavar="NAME",
                   help="deploy the selection to assets/model_rep/encoder/<NAME> (tools/promote.py --rep)")
    p.add_argument("--watch", action="store_true", help="score checkpoints as they appear")
    p.add_argument("--poll", type=int, default=180)
    p.add_argument("--settle", type=int, default=60)
    p.add_argument("--train-proc", default="run_beit3_finetuning")
    p.add_argument("--dump-pool", default=None,
                   help="for reranker evaluation: directory to dump the bench in eval-set shape "
                        "(gallery symlinks · query_text.json · union_pool.pt · base_score.pt)")
    p.add_argument("--pool-topk", type=int, default=10, help="candidate K for --dump-pool")
    p.set_defaults(fn=cmd_eval)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
