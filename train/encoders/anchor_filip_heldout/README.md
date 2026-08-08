# `anchor_filip_heldout`

SigLIP2-large-512 + DoRA — FILIP late interaction.
Trained with the held-out split excluded.

## Run

From the repository root.

```bash
python train/encoders/anchor_filip_heldout/train.py --gpu 0 --run-note anchor_filip_heldout
```

| `train.py` flag | |
|---|---|
| `--gpu` | single GPU id |
| `--run-note` | run directory name; an existing name aborts the launch |
| `--resume` | path to a previous run's `checkpoints/last` |

Outputs land in `assets/runs/anchor_filip_heldout/`.

## Next step

```bash
python train/encoders/anchor_filip_heldout/search_swa_range.py assets/runs/anchor_filip_heldout --gpu 0
```

The printed range goes to [`anchor_filip_all/build_swa.py`](../anchor_filip_all/build_swa.py).

## Inputs

`check_inputs()` runs before the GPU is used and reports everything missing at once:

| | how to build |
|---|---|
| `assets/data/manifest/pab_manifest_msr_v1.jsonl` | `python train/gen/gen_manifest.py` |
| `assets/data/raw/pab_train/train_jpg_512` | dataset symlink |
| HF cache for `google/siglip2-large-patch16-512` | `huggingface-cli download` |
| `assets/data/heldout_v1` | `python train/gen/gen_heldout_v1.py --gpu 0` |

## Files

| | |
|---|---|
| `train.py` | training entry point |
| `search_swa_range.py` | finds the SWA epoch range from the held-out metrics |
| `heldout_split_v1.py` | supplies the held-out image list (imported by `train.py`) |
