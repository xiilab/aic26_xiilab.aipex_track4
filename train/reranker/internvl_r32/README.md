# `internvl_r32`

InternVL3.5-30B-A3B + LoRA r32 — the main reranker, trained with a pairwise Bradley-Terry loss.

The score is `logit(yes) - logit(no)`, the same head the zero-shot scorer uses, so fine-tuning does
not change the scale and the result drops straight into the deployment cache. Training minimises
`-logsigmoid(score(chosen) - score(rejected))`.

## Configuration

| | |
|---|---|
| base | `assets/model/vlm_models/InternVL3_5-30B-A3B-HF` |
| adapter | LoRA r=16 (`--lora-r 32` for the deployed setting), α=2r, dropout 0.05 |
| targets | `q_proj` `k_proj` `v_proj` `o_proj` on the language side; vision untouched |
| loss | pairwise Bradley-Terry over (chosen, rejected) |
| lr | 1e-4 · AdamW · grad clip 1.0 |
| grad accum | 8 |
| checkpoints | `<out>/step{N}` every `--save-every` steps (default 500) |

**Environment: the `track4_vllm` conda env**, created by
`bash requirements/setup_conda_envs.sh --only vllm`. On torch 2.8 the MoE layers dispatch to
`torch._grouped_mm` (Hopper only), so every forward raises `RuntimeError`; `train.py` catches
per-example exceptions, so the run finishes silently with **0 steps**. If the step counter stays at
0, this is why.

## Data flow

`$PY_VLLM` below is that env's interpreter; the steps without it run in `track4_train`.

```bash
export PY_VLLM=$(conda info --base)/envs/track4_vllm/bin/python

# 1. mine hard negatives with the SigLIP2 anchor (150k subsample)
python mine_hardneg.py --gpu 6 --bs 48

# 2a. A_rescue — failure-driven pairs, reuses the embedding cache from step 1 (no re-encoding)
python build_rescue_pairs.py

# 2b. B_antibreak — pairs where the zero-shot reranker breaks a query the base got right
$PY_VLLM build_antibreak_pairs.py --gpu 6 --max-q 6000

# 3. merge A_rescue + B_antibreak into assets/data/mining/dpo_train.jsonl, then train
CUDA_VISIBLE_DEVICES=6 $PY_VLLM -u train.py \
    --data assets/data/mining/dpo_train.jsonl --lora-r 32 --lr 1e-4 --grad-accum 8 --save-every 500
```

A_rescue and B_antibreak pull in opposite directions — one recovers failures, the other prevents
regressions — so both are needed in the merged training set.

Every builder refuses to overwrite an existing output; pass `--force` to rebuild.

## Inputs

| path | | how to build |
|---|---|---|
| `assets/data/mining/rerank_ft_subsample_150k.jsonl` | mining pool | shipped |
| `assets/model/encoder/siglip_mining` | anchor adapter | shipped |
| `assets/data/mining/hardneg_anchor_mined_emb.pt` | embedding cache | `mine_hardneg.py` |
| `assets/data/mining/hardneg_pool_orig.jsonl` | image paths | shipped |
| `assets/data/mining/dpo_train.jsonl` | training pairs | A_rescue + B_antibreak |
| `assets/model/vlm_models/InternVL3_5-30B-A3B-HF` | base model | HF cache |

## Smoke test

```bash
$PY_VLLM -u train.py --max-steps 20 --heldout 500
```

Holds out 500 pairs and reports pair-acc and mean margin before and after training, so a
non-functioning setup is visible in minutes rather than hours.

## Pick a step, then deploy

```bash
python train/reranker/eval/eval_step.py --member r32 --run <run> --steps step1000,step2000,step2500
python train/reranker/internvl_r32/deploy.py <run> --step 2500
```

`deploy.py` copies `step{N}/` to `assets/model_rep/reranker/internvl_r32/`, wiping the destination
first. `eval_step.py --deploy-rep` does the same automatically at selection time.

The reproduction cache is built separately:
`pipeline/S2_rerank/score_union_hf_4b.py --rep --name internvl_r32`.

## Files

| | |
|---|---|
| `mine_hardneg.py` | anchor-based hard-negative mining, writes the shared embedding cache |
| `build_rescue_pairs.py` | A_rescue pairs from retrieval failures |
| `build_antibreak_pairs.py` | B_antibreak pairs from zero-shot regressions |
| `train.py` | LoRA training |
| `deploy.py` | copies the selected step to `assets/model_rep/reranker/` |

## Evaluation data

The Track 4 test set is never opened by this code. Step selection uses the pair bench built by
[`../eval/eval_step.py`](../eval/eval_step.py).
