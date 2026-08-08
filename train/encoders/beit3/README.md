# `beit3`

BEiT3-large/384 full fine-tuning on PAB — three recipes from one entry point.

| recipe | | |
|---|---|---|
| `v2` | recap multi-style captions-dict, resampled each epoch | 64 text tokens · lr 1e-5 |
| `stage1` | original captions — builds the init for `helip` | 128 text tokens · lr 1e-5 |
| `helip` | continues from a `stage1` checkpoint, HELIP hard pairs | 128 text tokens · lr 2e-6 |

`stage1` is not deployed; it only exists to produce `helip`'s `--init`.

Training calls the vendor tree's `run_beit3_finetuning.py` by **import**, not as a subprocess.
Environment: `track4_beit3` (needs `torchscale` — see [`requirements/README.md`](../../../requirements/README.md)).

## Configuration

Shared by all three recipes (`COMMON` in `train.py`):

| | |
|---|---|
| model | `beit3_large_patch16_384` · task 356 · 384px |
| batch | 184 · `update_freq` 1 |
| optim | weight decay 0.05 · layer decay 0.85 · drop path 0.16 |
| epochs | 4 (override with `EPOCHS=n`) |
| seed | 16 |
| init | `assets/model/encoder/beit3_pre/beit3_large_patch16_384_coco_retrieval.pth` |

Unrecognised flags are passed straight through to `run_beit3_finetuning.py`, and they win over the
recipe (e.g. appending `--epochs 7`).

## Two-run protocol

`--data` decides whether the held-out bench is excluded, and a gate **enforces** it.

| | | |
|---|---|---|
| run A | `EXCLUDE_HELDOUT=1` (default) · `--data <excluded index>` | per-epoch ckpts → pick best epoch `e*` |
| run B | `EXCLUDE_HELDOUT=0` · `--data <original index>` | adopt `e*` as-is, never scored |

Exclusion happens at the **index** stage, so run A needs a directory built by `beit3_tool.py
exclude`. Passing the original index without opting out stops the launch instead of silently
training on the bench. The run directory name reflects the mode (`beit3_<recipe>_{heldout|all}`).

## Inputs

| path | | how to build |
|---|---|---|
| `assets/data/manifest/pab_v2_multi_webp` | `v2` index | shipped ([`ARTIFACTS.md`](../../../ARTIFACTS.md)) |
| `assets/data/manifest/pab_full_webp` | `stage1`/`helip` index | shipped |
| `assets/model/encoder/beit3_pre/beit3.spm` | tokenizer | shipped |
| `assets/model/encoder/beit3_pre/beit3_large_patch16_384_coco_retrieval.pth` | init | shipped |
| `assets/data/heldout_v1` | held-out bench | `python train/gen/gen_heldout_v1.py` |
| `third_party/beit3` | vendor tree | shipped (`BEIT3_SRC`) |

## The index

Four directories in `assets/data/manifest/` — two caption layouts × full/excluded:

| directory | rows | |
|---|---:|---|
| `pab_v2_multi_webp` | 1,013,605 | full · multi-style captions — source for `v2` |
| `pab_v2_multi_webp_heldout` | 962,952 | excluded — **what `v2` run A trains on** |
| `pab_full_webp` | 1,013,606 | full · single caption + HELIP hard pairs — source for `stage1`/`helip` |
| `pab_full_webp_heldout` | 962,952 | excluded — **what `stage1`/`helip` run A train on** |

`pab_v2_multi_webp` rows carry a `captions` dict keyed by the recaption presets
(`p00_original … p10_formal`) and the loader samples one per epoch, which is the multi-style
augmentation of the `v2` recipe. `pab_full_webp` rows carry a single `caption`, and only that layout
ships the HELIP hard pairs (`helip_hardpairs{,_outlier,_r2,_r2_outlier}.npy`). Both store the `goal`
and `wentwrong` rows of a pair back to back, so with a large batch and no shuffling they land in the
same batch as hard negatives.

Images are not copied into an index — each one carries a `train_webp` symlink, and the loader
resolves `train/imgs_X/{goal,wentwrong}/N.jpg` to
`<index>/train_webp/Part {X//8+1}/imgs_X/…/N.webp`. Repoint it after moving the tree.

## Prepare the index

The `_heldout` pair ships too, so normally there is nothing to prepare. To rebuild it:

```bash
# run A trains on this copy — drops the 50,653 held-out images (5.00%), leaving 962,952 rows
python beit3_tool.py exclude --src assets/data/manifest/pab_v2_multi_webp \
                             --dst assets/data/manifest/pab_v2_multi_webp_heldout

# helip only — exclude shifts the row numbers, so the hard pairs are remapped onto them
python beit3_tool.py remap-helip --src assets/data/manifest/pab_full_webp \
                                 --dst assets/data/manifest/pab_full_webp_heldout
```

`exclude` leaves a `heldout_exclusion.json` marker in `--dst`; `train.py` reads it to verify the
md5 and that no bench image leaked. `build-index` / `exclude` / `remap-helip` need only the
standard library — any python works.

`build-index` is for training on a **new** recaption set, not for reproducing the shipped index —
its captions come from whatever `--recap` CSV it is given:

```bash
python beit3_tool.py build-index --style-mode multi        --out <dir>   # v2 layout
python beit3_tool.py build-index --style-mode p00_original --out <dir>   # stage1 · helip layout
```

It writes `annotation/` only, so add the `train_webp` symlink before training on the result.

## Train

```bash
# run A — held-out excluded
python train.py v2     --gpu 6 --data <excluded index> --out <run>
python train.py stage1 --gpu 2 --data <excluded index> --out <run>
python train.py helip  --gpu 6 --data <excluded index> --out <run> \
                       --init <stage1 run>/checkpoint-{N}.pth

# run B — full split
EXCLUDE_HELDOUT=0 python train.py v2 --gpu 6 --data <original index> --out <run>

# print the final argv without running
python train.py v2 --print-args
```

| flag | |
|---|---|
| `--gpu` | `CUDA_VISIBLE_DEVICES` (default `6`) |
| `--data` | training index directory |
| `--out` | run directory (default `assets/runs/beit3_<recipe>_{heldout\|all}`) |
| `--init` | `--finetune` init ckpt (required for `helip`) |
| `--hardpairs` | `helip`: remapped hard pairs (default `<data>/helip_hardpairs.npy`) |
| `--print-args` | print the argv and exit — works without torch |

Checkpoints land in `<out>/checkpoint-{N}.pth`, plus `<out>/log/` and
`<out>/heldout_provenance.json`.

## Pick the best epoch

Score the **run A** checkpoints against the held-out bench:

```bash
python beit3_tool.py eval --run <heldout run> --epochs 0-3 --gpu 2 --with-best
```

Add `--watch` to score checkpoints as training writes them. Results go to
`<run>/heldout_eval.json`. `eval` is the one subcommand that needs torch/timm/torchscale.

## Deploy

```bash
python train/encoders/beit3/deploy.py <all run> --recipe v2 --epoch 3
```

Copies `checkpoint-{epoch}.pth` from the **run B** directory to
`assets/model_rep/encoder/{beit3_v2|beit3_helip}/checkpoint-best.pth`, renamed so consumers do not
need the epoch number. The destination is wiped first.

## Files

| | |
|---|---|
| `train.py` | training entry point for all three recipes |
| `beit3_tool.py` | `build-index` · `exclude` · `remap-helip` · `eval` |
| `deploy.py` | copies the selected epoch to `assets/model_rep/encoder/` |

## Evaluation data

The Track 4 test set is never opened by this code. Epoch selection uses only the held-out split
carved out of PAB train; bench definition, md5 and gates come from the single source
[`../eval/heldout_bench.py`](../eval/heldout_bench.py).
