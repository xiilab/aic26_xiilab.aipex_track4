# train — encoders · rerankers · data generation

Everything here produces the checkpoints that `pipeline/` encodes and scores.
The full walkthrough (data → training → deploy → encode) is in
[`pipeline/README.md`](../pipeline/README.md); this page is just the map.

```
gen/        one-time inputs: recaption CSVs, manifests, held-out split
encoders/   ten base-ensemble members
reranker/   three fine-tuned rerankers (the other four are zero-shot)
```

## Two-run protocol

Checkpoint selection never touches the test set:

1. **run A** (`*_heldout`) trains with the held-out images excluded and picks the
   best epoch/step e\* on the held-out bench (`encoders/eval/`, `reranker/eval/`).
2. **run B** (`*_all`) trains on the full data with the same budget and adopts e\*
   without scoring anything.
3. `deploy.py` copies run B's e\* checkpoint to `assets/model_rep/`.

Trainers read the exclusion list from `assets/data/heldout_v1/` and stop if its
md5 or gates do not match.

## Encoders

| directory | member(s) | backbone |
|---|---|---|
| `anchor_tcap_{heldout,all}` | `anchor_tcap` | SigLIP2-L512 + DoRA |
| `anchor_filip_{heldout,all}` | `anchor_filip` | SigLIP2-L512 + DoRA |
| `siglip_maxsim_{heldout,all}` | `siglip_maxsim` | SigLIP2-L384 + DoRA |
| `mc2h378_peft_{heldout,all}` | `mc2h378_peft` | MetaCLIP2 huge-378 + DoRA |
| `metaclip2` | `metaclip2` | MetaCLIP2 l14 + DoRA (single trainer, `EXCLUDE_HELDOUT` switches run A/B) |
| `metaclip_v1` | `metaclip_v1` | MetaCLIP v1 l14 (open_clip full FT) |
| `beit3` | `beit3_v2` · `beit3_helip` | BEiT3-large full FT (`beit3_tool.py` builds the webp index) |
| `llm2clip_anchor5` | *(S4b tail-NN encoder, not a base member)* | LLM2CLIP L-14-336, two stacked LoRAs: `train_vision.py` → `llm2clip_lora_v3_best`, then `train_text.py` on top → `llm2clip_anchor5`. Both trainer caches ship in `assets/data/mining/llm2clip_*.pt`; `build_{text,hidden}_cache.py` rebuild them (only they need the unbundled 8B LLM) |

`gme`, `eva02_pre` are zero-shot — no trainer, encode directly.

## Rerankers

`internvl_r32` (LoRA) · `qwen3vl_2b` (DoRA) · `jina_m0` (DoRA). Each directory has
`train.py` and `deploy.py`; mining inputs are prebuilt in `assets/data/mining/`
(`build_manifest.py` / `build_negcache.py` regenerate them). Step selection:
`reranker/eval/eval_step.py` (accuracy gate + calibration).

## Environment

Run every trainer from the repository root. **Training does not all happen in one env** — it uses
three of the five, and the mapping is held in code by
[`ops/06_train.sh`](../ops/06_train.sh), which launches `train.py` under the right interpreter:

```bash
bash ops/06_train.sh --list                  # targets and the env each one needs
GPU=3 bash ops/06_train.sh internvl_r32 -- --lora-r 32 --lr 1e-4
DRY=1 bash ops/06_train.sh --all             # print the plan; needs no env or data
```

| target | env | wrong env gives |
|---|---|---|
| `anchor_{tcap,filip}_*` · `mc2h378_peft_*` · `siglip_maxsim_*` · `metaclip2` | `track4_train` | — |
| `qwen3vl_2b` | `track4_train` | 4.30.2: `ImportError: cannot import name 'Cache'` |
| `jina_m0` | `track4_train` | 5.9.0: fails to load · other 5.x: **silently random-initialised** |
| `metaclip_v1` | `track4_beit3` | `ModuleNotFoundError: open_clip` |
| `beit3` | `track4_beit3` | vendored `run_beit3_finetuning` needs `torchscale` |
| **`internvl_r32`** | **`track4_vllm`** | torch 2.8: **0 optimiser steps, checkpoint still written** |

### Two of those failures are silent

**`internvl_r32` on torch 2.8 trains on nothing.** Its MoE calls `torch._grouped_mm`, which torch
2.8 accepts on compute capability **9.0 only** — Hopper — so on a B300 (sm_103) every forward
raises `RuntimeError`. `train.py` catches it per step and continues, so the run finishes early,
prints no error, and writes a checkpoint. Always smoke it first and check the step count actually
moves:

```bash
GPU=3 bash ops/06_train.sh internvl_r32 -- --max-steps 20 --heldout 500
```

**`jina_m0` can train against noise.** `train.py` loads `JinaVLForRanking` with a bare
`AutoModel.from_pretrained(..., trust_remote_code=True)` — no `key_mapping`, no `missing_keys`
check, unlike `pipeline/S2_rerank/score_union_jina.py`. transformers 5.x registers the Qwen2-VL key
rename under the *class* name, and this subclass does not match it, so a mismatched version
initialises both towers randomly without raising. Measured `missing_keys`: 4.51.3 → 0,
**5.4.0 → 0**, 4.30.2 → `ImportError`, 5.9.0 → load fails. `track4_train` (5.4.0) is therefore safe
as shipped; moving jina training to a newer transformers needs the `KEY_MAP` guard copied over
first.

### GPU selection: `--gpu` or `CUDA_VISIBLE_DEVICES`, never both

Twelve trainers take `--gpu` and index the **physical** device (`cuda:{physical_gpu}`);
`metaclip2` and `metaclip_v1` take none and read `CUDA_VISIBLE_DEVICES`. Setting both breaks the
first group — the process then sees one device numbered 0 while the script asks for `cuda:$GPU`.
Exporting alone is not enough either: `beit3`, `internvl_r32`, `jina_m0` and `qwen3vl_2b` assign
`os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu` themselves before importing torch, so their own
default (`0`, or `6` for beit3) wins and the job quietly lands elsewhere. `06_train.sh` picks the
right one per target; pass `--gpu` after `--` to override `GPU=`.

### What re-training does and does not reproduce

It reproduces the **environment**, not the weights. Seeds are uneven — the encoder trainers and
`jina_m0`/`qwen3vl_2b` set `manual_seed` + `cudnn.deterministic`, while `beit3`, `metaclip_v1` and
`internvl_r32` set no seed at all, and no trainer calls `torch.use_deterministic_algorithms`. A
re-trained checkpoint will not match the adopted one bit for bit.

None of this is needed to reproduce the submission: the adopted adapters ship under
`assets/model/{encoder,reranker}/` and `ops/04`/`05` read them directly. Version-pin rationale is
in [`requirements/README.md`](../requirements/README.md); the runbook is in
[`ops/README.md`](../ops/README.md) §8.
