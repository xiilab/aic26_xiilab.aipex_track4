#!/usr/bin/env python3
"""train — single entry point for the three BEiT3 recipes (v2 · stage1 · helip).

The vendor tree's `run_beit3_finetuning.py` is **imported and called directly**, not spawned as a
subprocess. This replaces the shell launchers; only the per-recipe differences live in `RECIPES`.

  v2       beit3_v2 multi-style captions-dict, randomly sampled · t64  · lr 1e-5
  stage1   the init for helip (original captions)               · t128 · lr 1e-5
  helip    continues from the stage-1 best, HELIP hard pairs    · t128 · lr 2e-6

Held-out exclusion must already be done at the **index** stage: pass the directory produced by
`beit3_tool.py exclude` as `--data` (`helip` also needs `remap-helip`).

Two-run protocol — `--data` decides whether the bench is excluded, and the gate **enforces** it.
  run A (heldout)  EXCLUDE_HELDOUT=1 (default) · --data <excluded>  -> per-epoch ckpts, pick best epoch e*
  run B (all)      EXCLUDE_HELDOUT=0           · --data <original>  -> adopt e* as-is, no scoring
  Because exclusion is the default, passing an original index **stops before training starts**
  (same policy as metaclip2 `exclude_heldout_rows()`: never train on everything silently).

Usage:
  python train.py v2     --gpu 6 --data <excluded pab_v2_multi_webp> --out <run directory>
  python train.py stage1 --gpu 2 --data <excluded pab_full_webp>     --out <run>
  python train.py helip  --gpu 6 --data <excluded pab_full_webp>     --out <run> \\
                         --init <stage1 run>/checkpoint-{N}.pth
  EXCLUDE_HELDOUT=0 python train.py v2 --gpu 6 --data <original>     # run B (all)
  python train.py v2 --print-args        # print the final arguments without running

env: `.venv_beit3eval` (needs torchscale). `--print-args` works without torch: the gate runs
     **after** it returns, so `heldout_bench` (which loads torch) is imported late.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HERE = os.path.dirname(os.path.abspath(__file__))
ENCODERS = os.path.dirname(HERE)
EVAL_DIR = os.path.join(ENCODERS, "eval")
PKG = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
BEIT3_SRC = os.environ.get("BEIT3_SRC", os.path.join(PKG, "third_party", "beit3"))
BEIT3_DATA = os.environ.get("BEIT3_DATA", f"{_REPO}/assets/data/manifest")   # pair index root

# ---- held-out exclusion (same policy as metaclip2 `exclude_heldout_rows()`) ----
# BEiT3 receives a whole index, so the exclusion already happened in `beit3_tool.py exclude`.
# All that is checked here is whether --data really is that output (i.e. this is run A).
# Set EXCLUDE_HELDOUT=0 to opt out.
HELDOUT_DIR     = os.environ.get("HELDOUT_DIR",
                                 os.environ.get("PAB_DATA_INFRA", f"{_REPO}/assets/data") + "/heldout_v1")
EXCLUDE_HELDOUT = os.environ.get("EXCLUDE_HELDOUT", "1").lower() not in ("0", "false", "no")
HELDOUT_REQUIRE_GATES = True

# Init shared by all three runs (COCO retrieval pretraining); only helip continues from stage-1.
# BEiT3 training assets (spm tokenizer + COCO init) live in assets/model/encoder/beit3_pre/.
BEIT3_MODELS = os.environ.get("BEIT3_MODELS", f"{_REPO}/assets/model/encoder/beit3_pre")
COCO_INIT = os.environ.get(
    "BEIT3_INIT", f"{BEIT3_MODELS}/beit3_large_patch16_384_coco_retrieval.pth")

# Arguments shared by all three runs.
COMMON = [
    "--model", "beit3_large_patch16_384",
    "--task", "356",
    "--input_size", "384",
    "--drop_path", "0.16",
    "--checkpoint_activations",
    "--weight_decay", "0.05",
    "--layer_decay", "0.85",
    "--batch_size", "184",
    "--update_freq", "1",
    "--save_ckpt", "--save_ckpt_freq", "1",
    "--seed", "16",
]

# Epoch budget, identical across the three recipes and both runs, so that an epoch number means the
# same thing in run A and run B (which is what makes the e* transfer valid).
# Override with the EPOCHS environment variable or by appending `--epochs N` on the CLI
# (passthrough arguments come last and therefore win).
EPOCHS = os.environ.get("EPOCHS", "4")

# Per-recipe differences only. `data` comes from the CLI; `default_data` below is the location of
# the index before exclusion, for reference.
RECIPES = {
    "v2": dict(
        desc="beit3_v2 — recap multi-style captions-dict · resampled per epoch",
        default_data=f"{BEIT3_DATA}/pab_v2_multi_webp",
        args=["--lr", "1e-5", "--warmup_steps", "440", "--epochs", EPOCHS],
    ),
    "stage1": dict(
        desc="stage-1 — original captions · t128 (produces the init for helip)",
        default_data=f"{BEIT3_DATA}/pab_full_webp",
        args=["--num_max_bpe_tokens", "128", "--lr", "1e-5", "--warmup_steps", "440",
              "--epochs", EPOCHS, "--view_aug_mode", "global",
              "--num_workers", "4", "--no_pin_mem"],
    ),
    "helip": dict(
        desc="helip — HELIP hard pairs, continuing from the stage-1 best",
        default_data=f"{BEIT3_DATA}/pab_full_webp",
        needs_init=True,
        args=["--num_max_bpe_tokens", "128", "--lr", "2e-6", "--warmup_steps", "200",
              "--epochs", EPOCHS, "--view_aug_mode", "global",
              "--num_workers", "4", "--no_pin_mem",
              "--use_helip", "--helip_p", "1", "--helip_gamma", "0.3"],
    ),
}


def check_heldout_exclusion(data: str) -> dict | None:
    """Verify that `--data` is a `beit3_tool.py exclude` output, which is what run A requires.

    Reads the `heldout_exclusion.json` marker that `cmd_exclude` left in dst and checks three things:
      1. the marker exists       — otherwise this is the original (all-data) index
      2. heldout_md5 matches     — blocks a stale copy built from a different heldout version
      3. leak == 0               — the exclusion actually took effect (exclude already checks this)
    Bench definition, md5 and gates come from the single source `eval/heldout_bench.py`.
    Returns the marker contents for logging. Run B opts out explicitly with `EXCLUDE_HELDOUT=0`.
    """
    if not EXCLUDE_HELDOUT:
        print("  [heldout] EXCLUDE_HELDOUT=0 → training on everything (run B). "
              "Do not score this run's ckpts on the held-out bench: it trained on the bench images.")
        return None

    sys.path.insert(0, EVAL_DIR)
    import heldout_bench as HB                                       # noqa: E402  (loads torch — only after print-args)

    mark = os.path.join(data, "heldout_exclusion.json")
    if not os.path.exists(mark):
        raise SystemExit(
            f"✗ no held-out exclusion marker: {mark}\n"
            f"  --data looks like the original index — training on it would silently become run B.\n"
            f"  For run A, build the excluded copy first:\n"
            f"    python beit3_tool.py exclude --src {data} --dst {data.rstrip('/')}_heldout\n"
            f"    (helip also needs remap-helip)\n"
            f"  If training on everything is intended, say so with EXCLUDE_HELDOUT=0.")

    with open(mark, encoding="utf-8") as f:
        doc = json.load(f)
    want = HB.verify_identity(HELDOUT_DIR)                           # list vs split.json md5
    if HELDOUT_REQUIRE_GATES:
        HB.require_gates(HELDOUT_DIR, strict=True)
    got = doc.get("heldout_md5")
    if got != want:
        raise SystemExit(
            f"✗ the excluded copy was built from a different held-out split.\n"
            f"    {mark} → {got}\n"
            f"    current {HELDOUT_DIR} → {want}\n"
            f"  Re-run `exclude` against the same heldout.")
    if doc.get("leak"):
        raise SystemExit(f"✗ {doc['leak']} leaked images remain in the excluded copy: {mark}")
    print(f"  [heldout] excluded copy verified ✓ {data}")
    print(f"  [heldout] {doc.get('images_dropped'):,} images excluded · "
          f"{doc.get('images_kept'):,} kept · lines {doc.get('lines_in'):,}→{doc.get('lines_out'):,} · leak 0")
    return doc


def build_argv(a) -> list[str]:
    """Recipe + CLI → the argument list for run_beit3_finetuning."""
    r = RECIPES[a.recipe]
    data = a.data or r["default_data"]
    mode = "heldout" if EXCLUDE_HELDOUT else "all"                   # let the run name show run A/B
    out = a.out or f"{_REPO}/assets/runs/beit3_{a.recipe}_{mode}"
    init = a.init or COCO_INIT

    if r.get("needs_init") and not a.init:
        raise SystemExit(
            f"✗ '{a.recipe}' requires --init (a stage-1 checkpoint-N.pth).\n"
            f"  Train stage-1 first, then pick its best epoch on the held-out bench:\n"
            f"    python beit3_tool.py eval --run <stage1 run> --epochs 0-{int(EPOCHS) - 1}")
    if not os.path.exists(init):
        raise SystemExit(f"✗ init ckpt not found: {init}  (--init / BEIT3_INIT)")
    if not os.path.isdir(os.path.join(data, "annotation", "train")):
        raise SystemExit(f"✗ {data}/annotation/train not found — pass a `beit3_tool.py exclude` output as --data.")

    spm = a.spm or os.path.join(BEIT3_MODELS, "beit3.spm")
    if not os.path.exists(spm):
        raise SystemExit(f"✗ beit3.spm not found: {spm}")

    argv = list(COMMON) + list(r["args"]) + [
        "--sentencepiece_model", spm,
        "--finetune", init,
        "--data_path", data,
        "--output_dir", out,
        "--log_dir", os.path.join(out, "log"),
    ]
    if a.recipe == "helip":
        hp = a.hardpairs or os.path.join(data, "helip_hardpairs.npy")
        if not os.path.exists(hp):
            raise SystemExit(
                f"✗ hard pairs not found: {hp}\n"
                f"  They must be remapped onto the excluded index:\n"
                f"    python beit3_tool.py remap-helip --src <original> --dst {data}")
        argv += ["--helip_hardpairs", hp]
    argv += a.extra
    return argv, out


def main():
    ap = argparse.ArgumentParser(
        description="BEiT3 training (imports and calls run_beit3_finetuning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {k:7s} {v['desc']}" for k, v in RECIPES.items()))
    ap.add_argument("recipe", choices=list(RECIPES))
    ap.add_argument("--gpu", default="6", help="CUDA_VISIBLE_DEVICES (default 6)")
    ap.add_argument("--data", default=None, help="training data directory (the excluded index)")
    ap.add_argument("--out", default=None,
                    help="run directory (default assets/runs/beit3_<recipe>_{heldout|all}, per EXCLUDE_HELDOUT)")
    ap.add_argument("--init", default=None, help="--finetune init ckpt (required for helip)")
    ap.add_argument("--hardpairs", default=None, help="helip: remapped hardpairs (default <data>/helip_hardpairs.npy)")
    ap.add_argument("--spm", default=None)
    ap.add_argument("--print-args", action="store_true", help="print the final arguments and exit")
    ap.add_argument("extra", nargs="*", help="arguments appended verbatim to run_beit3_finetuning")
    a, _passthrough = ap.parse_known_args()          # unrecognised options (--epochs, --batch_size, …) pass straight through
    a.extra = list(a.extra) + _passthrough

    argv, out = build_argv(a)
    print(f"[train] recipe={a.recipe} — {RECIPES[a.recipe]['desc']}")
    print(f"[train] run {'A (heldout excluded)' if EXCLUDE_HELDOUT else 'B (all)'} · epochs={EPOCHS}")
    print(f"[train] gpu={a.gpu} · out={out}")
    if a.print_args:                                                 # this path runs without torch, ahead of the gate
        print("[train] argv:\n  " + " ".join(argv))
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu                       # before heldout_bench imports torch
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")

    doc = check_heldout_exclusion(a.data or RECIPES[a.recipe]["default_data"])

    os.makedirs(os.path.join(out, "log"), exist_ok=True)
    with open(os.path.join(out, "heldout_provenance.json"), "w", encoding="utf-8") as f:
        json.dump({"recipe": a.recipe, "run": "A" if EXCLUDE_HELDOUT else "B",
                   "exclude_heldout": EXCLUDE_HELDOUT, "epochs": EPOCHS,
                   "data": a.data or RECIPES[a.recipe]["default_data"],
                   "heldout_dir": HELDOUT_DIR, "exclusion": doc}, f, indent=2, ensure_ascii=False)

    # Put the vendor tree on the import path and make it cwd (sibling imports and relative defaults)
    sys.path.insert(0, BEIT3_SRC)
    os.chdir(BEIT3_SRC)
    import run_beit3_finetuning as R                                 # noqa: E402

    sys.argv = ["run_beit3_finetuning.py"] + argv                    # this is what get_args() reads
    opts, ds_init = R.get_args()
    R.main(opts, ds_init)


if __name__ == "__main__":
    main()
