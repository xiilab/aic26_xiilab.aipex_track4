# `siglip_maxsim_all`

SigLIP2-large-384 + DoRA — multi-probe pooling, soft labels.
Trained on the full split.

## Run

From the repository root.

```bash
python train/encoders/siglip_maxsim_all/train.py --gpu 0 --run-note siglip_maxsim_all
```

| `train.py` flag | |
|---|---|
| `--gpu` | single GPU id |
| `--run-note` | run directory name; an existing name aborts the launch |
| `--resume` | path to a previous run's `checkpoints/last` |

Outputs land in `assets/runs/siglip_maxsim_all/`.

## Next step

```bash
python train/encoders/siglip_maxsim_all/build_swa.py assets/runs/siglip_maxsim_all <lo> <hi>
python train/encoders/siglip_maxsim_all/deploy.py    assets/runs/siglip_maxsim_all
```

The range comes from [`siglip_maxsim_heldout/search_swa_range.py`](../siglip_maxsim_heldout/search_swa_range.py) and must be given.

`deploy.py` writes to `assets/model_rep/encoder/siglip_maxsim/`.

## Inputs

`check_inputs()` runs before the GPU is used and reports everything missing at once:

| | how to build |
|---|---|
| `assets/data/manifest/pab_manifest_msr_v1_scene.jsonl` | `python train/gen/gen_manifest.py` |
| `assets/data/raw/pab_train/train_jpg_512` | dataset symlink |
| HF cache for `google/siglip2-large-patch16-384` | `huggingface-cli download` |

## Files

| | |
|---|---|
| `train.py` | training entry point |
| `build_swa.py` | averages late-epoch weights into `checkpoints/swa` |
| `deploy.py` | copies the SWA checkpoint to the deployment path |
