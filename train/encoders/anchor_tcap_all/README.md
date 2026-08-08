# `anchor_tcap_all`

SigLIP2-large-512 + DoRA — FLAIR text-conditioned pooling.
Trained on the full split.

## Run

From the repository root.

```bash
python train/encoders/anchor_tcap_all/train.py --gpu 0 --run-note anchor_tcap_all
```

| `train.py` flag | |
|---|---|
| `--gpu` | single GPU id |
| `--run-note` | run directory name; an existing name aborts the launch |
| `--resume` | path to a previous run's `checkpoints/last` |

Outputs land in `assets/runs/anchor_tcap_all/`.

## Next step

```bash
python train/encoders/anchor_tcap_all/build_swa.py assets/runs/anchor_tcap_all 8 10
python train/encoders/anchor_tcap_all/deploy.py    assets/runs/anchor_tcap_all
```

The range comes from [`anchor_tcap_heldout/search_swa_range.py`](../anchor_tcap_heldout/search_swa_range.py); omitting it uses the built-in `8 10`.

`deploy.py` writes to `assets/model_rep/encoder/anchor_tcap/`.

## Inputs

`check_inputs()` runs before the GPU is used and reports everything missing at once:

| | how to build |
|---|---|
| `assets/data/manifest/pab_manifest_msr_v2.jsonl` | `python train/gen/gen_manifest.py` |
| `assets/data/raw/pab_train/train_jpg_512` | dataset symlink |
| HF cache for `google/siglip2-large-patch16-512` | `huggingface-cli download` |

## Files

| | |
|---|---|
| `train.py` | training entry point |
| `build_swa.py` | averages late-epoch weights into `checkpoints/swa` |
| `deploy.py` | copies the SWA checkpoint to the deployment path |
