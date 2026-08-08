# `metaclip2`

MetaCLIP2-L/14 + DoRA — FLAIR text-conditioned pooling.
Trained on the full split with DDP. There is no `_heldout` pair, so the SWA range is fixed
rather than searched.

## Run

From the repository root.

```bash
python train/encoders/metaclip2/train.py --gpus 0,1,2,3,4 --run-note metaclip2_all
```

| `train.py` flag | |
|---|---|
| `--gpus` | comma-separated GPU ids; the count is the DDP world size |
| `--run-note` | run directory name; an existing name aborts the launch |
| `--master-port` | 0 picks a free port |
| `--master-addr` | default `127.0.0.1` |
| `--resume` | path to a previous run's `checkpoints/last` |

The world size is part of the recipe — every LR is scaled by `sqrt(world_size)`, so changing
the GPU count changes the result. It is recorded in `summary.json` and in each checkpoint's
`meta.json`.

Outputs land in `assets/runs/metaclip2_all/`.

## Next step

```bash
python train/encoders/metaclip2/build_swa.py assets/runs/metaclip2_all
python train/encoders/metaclip2/deploy.py    assets/runs/metaclip2_all
```

`build_swa.py` averages ep02–ep04 and takes no range argument.

`deploy.py` writes to `assets/model_rep/encoder/metaclip2/`.

## Inputs

`check_inputs()` runs before the GPU is used and reports everything missing at once:

| | how to build |
|---|---|
| `assets/data/manifest/pab_manifest_msr_v1.jsonl` | `python train/gen/gen_manifest.py` |
| `assets/data/raw/pab_train/train_jpg_512` | dataset symlink |
| HF cache for `facebook/metaclip-2-worldwide-l14` | `huggingface-cli download` |

## Files

| | |
|---|---|
| `train.py` | training entry point |
| `build_swa.py` | averages late-epoch weights into `checkpoints/swa` |
| `deploy.py` | copies the SWA checkpoint to the deployment path |
