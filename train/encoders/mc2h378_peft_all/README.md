# `mc2h378_peft_all`

MetaCLIP2-huge-378 + DoRA — FLAIR text-conditioned pooling, self-residual batching.
Trained on the full split.

## Run

From the repository root.

```bash
python train/encoders/mc2h378_peft_all/train.py --gpu 0 --run-note mc2h378_peft_all
```

| `train.py` flag | |
|---|---|
| `--gpu` | single GPU id |
| `--run-note` | run directory name; an existing name aborts the launch |
| `--resume` | path to a previous run's `checkpoints/last` |

Outputs land in `assets/runs/mc2h378_peft_all/`.

## Next step

```bash
python train/encoders/mc2h378_peft_all/build_swa.py assets/runs/mc2h378_peft_all 2 4
python train/encoders/mc2h378_peft_all/deploy.py    assets/runs/mc2h378_peft_all
```

The range comes from [`mc2h378_peft_heldout/search_swa_range.py`](../mc2h378_peft_heldout/search_swa_range.py); omitting it uses the built-in `2 4`.

`deploy.py` writes to `assets/model_rep/encoder/mc2h378_peft/`.

## Inputs

`check_inputs()` runs before the GPU is used and reports everything missing at once:

| | how to build |
|---|---|
| `assets/data/manifest/pab_manifest_msr_v2.jsonl` | `python train/gen/gen_manifest.py` |
| `assets/data/raw/pab_train/train_jpg_512` | dataset symlink |
| HF cache for `facebook/metaclip-2-worldwide-huge-378` | `huggingface-cli download` |
| `assets/data/manifest/mc2h378_selfresidual_neighbors.pt` | shipped with the repository |

## Files

| | |
|---|---|
| `train.py` | training entry point |
| `build_swa.py` | averages late-epoch weights into `checkpoints/swa` |
| `deploy.py` | copies the SWA checkpoint to the deployment path |
