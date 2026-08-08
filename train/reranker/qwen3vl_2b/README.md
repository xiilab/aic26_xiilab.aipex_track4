# `qwen3vl_2b`

Qwen3-VL-Reranker-2B + DoRA, trained in both directions so the score function `s(text, image)` is
calibrated from either side.

| path | anchor | positive | negative |
|---|---|---|---|
| A — attribute grounding | image X | `orig_cap` | `flip_cap` (colour, clothing, action) |
| B — cross-image ranking | caption C | image X | the anchor encoder's top-K similar images |

`loss = CE_A + λB·CE_B`. Path B's negatives come from the same distribution as deployment inference.
The shared positive `s(C, X)` is forwarded once and reused by both paths.

## Configuration

Constants live in the `1. Config` block of `train.py`; there are no hyper-parameter flags.

| | |
|---|---|
| base | `Qwen/Qwen3-VL-Reranker-2B` |
| adapter | DoRA r=16 α=32 dropout 0.10, text decoder only |
| targets | `q_proj` `k_proj` `v_proj` `o_proj` `gate_proj` `up_proj` `down_proj` |
| score head | frozen — only the adapter trains |
| image | smart-resize, 4·32² to 1280·32² pixels (~1024 vision tokens at webp 1024) |
| lr | 1e-4 · AdamW β(0.9, 0.95) · 80 warmup steps, then cosine |
| grad accum | 8 · grad clip 1.0 |
| λB | 1.0 |
| epochs | 1 over the whole cache |
| checkpoints | `<run>/checkpoints/ex{NNNNNN}` every 1000 examples, plus `last` |
| seed | 42 |

Checkpoints are periodic in **examples seen**, not epochs: one epoch takes roughly 25 h, so epoch
granularity would leave no steps to choose between.

## Data flow

```bash
# 1. manifest — recap positives + hardneg flip negatives, one row per image
python build_manifest.py

# 2. negcache — hard image negatives from the anchor encoder's top-1 failures
python build_negcache.py --gpu 6

# 3. train
python -u train.py --gpu 6
```

Step 1 is optional when the bundled manifest under `assets/data/mining/` is present; it is only
needed to rebuild from different recap/hardneg CSVs. Step 2 keeps only the queries whose base top-1
is wrong, drops near-duplicate distractors (image-image cos ≥ 0.85), and discards a "false failure"
whose distractors are all near-duplicates. Both steps rebuild automatically when their provenance or
`meta` no longer matches the current settings.

The `NEG_CACHE` path is composed from `ANCHOR_CKPT`, `NEG_POOL_SIZE`, `N_IMG_NEG` and `NEARDUP_TAU`,
so those constants must match between `build_negcache.py` and `train.py` or training aborts at an
assert.

## Inputs

| path | | how to build |
|---|---|---|
| `assets/data/raw/recaption/train_msr_v2.csv` | recap captions | shipped (`RECAP_CSV`) |
| `assets/data/mining/hardneg_flip.csv` | flip negatives | shipped (`HARDNEG_CSV`) |
| `assets/data/raw/pab_train/train_jpg_512` | anchor encoding input | dataset |
| `assets/data/raw/pab_train/train_webp` | training images (1024px) | dataset |
| `assets/model/encoder/siglip_mining` | anchor adapter | shipped (`ENCODER_CKPT`) |
| `Qwen/Qwen3-VL-Reranker-2B` | base model | HF cache |

Writes go only to `OUTPUT_DIR` (default `assets/runs/rerank_qw2b`); the shared tree is read-only.

## Pick a step, then deploy

```bash
python train/reranker/eval/eval_step.py --member dora --run <run> --steps ex006000,ex007000,ex008000
python train/reranker/qwen3vl_2b/deploy.py <run> --step ex007000
```

`deploy.py` copies `checkpoints/ex{NNNNNN}/` to `assets/model_rep/reranker/qwen3vl_2b/`, wiping the
destination first. The reproduction cache is built separately:
`pipeline/S2_rerank/score_union_qwen_4b.py --rep --name qwen3vl_2b`.

## Files

| | |
|---|---|
| `build_manifest.py` | recap + hardneg → manifest jsonl (shared by all three rerankers) |
| `build_negcache.py` | hard image negatives from anchor top-1 failures |
| `train.py` | bidirectional DoRA training |
| `deploy.py` | copies the selected step to `assets/model_rep/reranker/` |
| `adopted_config.json` · `adopted_env.json` | the deployed run's config and environment |

## Evaluation data

The Track 4 test set is never opened by this code. Path B queries use train captions only, and step
selection uses the pair bench built by [`../eval/eval_step.py`](../eval/eval_step.py).
