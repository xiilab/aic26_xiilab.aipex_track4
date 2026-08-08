# `metaclip_v1`

MetaCLIP v1 `ViT-L-14-worldwide-xlmv` full fine-tuning on PAB with CLIP contrastive loss (InfoNCE),
built on `open_clip` 3.3.x.

## Configuration

| | |
|---|---|
| model | `ViT-L-14-worldwide-xlmv` (custom config, vocab 901,629 · xlm-v-base) |
| input | 224px · 77 text tokens |
| tuning | full fine-tune (no adapter) |
| batch | 256 per rank |
| epochs | 4 (override with `EPOCHS=n` or `--epochs`) |
| lr | 5e-06 · weight decay 0.2 · 1000 warmup steps, then cosine |
| loss | symmetric InfoNCE, AMP + grad clip 1.0 |
| workers | 8 |

The `open_clip` config and tokenizer are registered from the repository copy under
`assets/model/vlm_models/MetaCLIP-L14-worldwide/`, not from the venv — the checkpoint's vocab only
matches `facebook/xlm-v-base`.

## Two-run protocol

| | | |
|---|---|---|
| run A | `EXCLUDE_HELDOUT=1` (default) | per-epoch ckpts → pick best epoch `e*` |
| run B | `EXCLUDE_HELDOUT=0` | adopt `e*` as-is, never scored |

Held-out rows are dropped from the CSV at load time; the list md5 and gates must match or the run
aborts. The output paths follow the mode (`assets/runs/metaclip_v1_{heldout|all}/`) so run B cannot
overwrite run A's `epoch_{N}.pt`. Both runs use the same 4-epoch budget, which is what makes `e*`
transferable.

## Inputs

`check_inputs()` validates all of these before any GPU is touched, and reports every missing item at once.

| path | | how to build |
|---|---|---|
| `assets/data/manifest/train.csv` | `filepath`,`title` pairs | `python train/gen/gen_metaclip_v1_csv.py` |
| `assets/model/vlm_models/MetaCLIP-L14-worldwide/l14_worldwide.pt` | pretrained weights | shipped |
| `assets/model/vlm_models/MetaCLIP-L14-worldwide/ViT-L-14-worldwide-xlmv.json` | `open_clip` config | shipped |
| `assets/data/heldout_v1` | held-out bench | `python train/gen/gen_heldout_v1.py` |

## Train

```bash
# run A — single GPU
CUDA_VISIBLE_DEVICES=0 python train.py

# run A — DDP
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py

# run B — full split
EXCLUDE_HELDOUT=0 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
```

| flag | |
|---|---|
| `--train-csv` | training CSV (default `assets/data/manifest/train.csv`, or `TRAIN_CSV`) |
| `--batch-size` | per-rank batch (default 256) |
| `--epochs` | default 4 |
| `--lr` `--wd` `--warmup` | 5e-06 · 0.2 · 1000 |
| `--resume` | `epoch_{N}.pt` to resume from (restores optimizer and re-advances the scheduler) |

Outputs, under `assets/runs/metaclip_v1_{heldout|all}/`:
`checkpoints/epoch_{N}.pt` · `checkpoints/../heldout_exclusion.json` · `logs/train_loss.log`.
`SAVE_DIR` and `LOG_FILE` override the locations.

## Pick the best epoch

Score the **run A** checkpoints against the held-out bench:

```bash
python train/encoders/eval/eval_heldout_openclip.py \
  --model ViT-L-14-worldwide-xlmv \
  --ckpt-dir assets/runs/metaclip_v1_heldout/checkpoints \
  --epochs 1-4 --gpu 6
```

Add `--watch` to score checkpoints as training writes them. Results go to
`<ckpt-dir>/heldout_eval.json`.

## Deploy

```bash
python train/encoders/metaclip_v1/deploy.py assets/runs/metaclip_v1_all/checkpoints --epoch 4
```

Copies `epoch_{epoch}.pt` from the **run B** directory to
`assets/model_rep/encoder/metaclip_v1/`, keeping the filename — encoding scripts take it directly
via `--checkpoint <path>`. The destination is wiped first.

## Files

| | |
|---|---|
| `train.py` | training entry point |
| `deploy.py` | copies the selected epoch to `assets/model_rep/encoder/metaclip_v1/` |

## Evaluation data

The Track 4 test set is never opened by this code. Epoch selection uses only the held-out split
carved out of PAB train; bench definition, md5 and gates come from the single source
[`../eval/heldout_bench.py`](../eval/heldout_bench.py). The trainer itself computes no retrieval
metrics.
