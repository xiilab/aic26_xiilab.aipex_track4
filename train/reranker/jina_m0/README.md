# `jina_m0`

jina-reranker-m0 + DoRA, trained in both directions. Same structure as
[`../qwen3vl_2b`](../qwen3vl_2b), with an extra action-negative path.

Its fusion weight is 0, but the S4 tail (near-duplicate promotion) uses it, so it remains a required
deployment member.

| path | anchor | positive | negative |
|---|---|---|---|
| A — attribute grounding | image X | `orig_cap` | `flip_caps` |
| B — cross-image ranking | styled recap query | image X | `img_negs` (appearance) + `action_negs` (behaviour) |

`loss = CE_A + λB·CE_B`, with λB = 2.0. Path B queries are sampled uniformly across the 12 recap
styles, from train captions only.

## jina specifics

`JinaVLForRanking` (Qwen2-VL-2B, `trust_remote_code`) has `lm_head=Identity` plus an MLP score head.
The deployment score is `sigmoid(MLP(hidden[-1]) - LOGIT_BIAS)`, but cross-entropy needs the **raw
logit before the sigmoid**, so `score_logit()` replicates the forward pass:
`Qwen2VLForConditionalGeneration.forward(base, …)` → `base.score(hidden)`.

`LOGIT_BIAS = 2.65` and `SCORE_TOKEN_ID = 100` are copied from jina's `modeling.py` and must be
updated together with upstream. The score MLP head stays frozen; only the DoRA adapter trains.

## Configuration

Constants live in the `1. Config` block of `train.py`; there are no hyper-parameter flags.

| | |
|---|---|
| base | `jinaai/jina-reranker-m0` |
| adapter | DoRA r=16 α=32 dropout 0.10, text decoder only |
| targets | `q_proj` `k_proj` `v_proj` `o_proj` `gate_proj` `up_proj` `down_proj` |
| image | 4·28² to 1024·28² pixels (Qwen2-VL uses a 28×28 effective patch) |
| max length | 2048 (caption + ~1024 image tokens + overhead) |
| lr | 1e-4 · AdamW β(0.9, 0.95) · 80 warmup steps, then cosine |
| grad accum | 16 · grad clip 1.0 |
| negatives | 4 image + 4 action per example, 3 flip captions |
| λB | 2.0 |
| epochs | 1 over the whole cache |
| checkpoints | `<run>/checkpoints/ex{NNNNNN}` every 1000 examples, plus `last` |
| seed | 42 |

## Data flow

```bash
# 0. once — fetch the base model
./download_model.sh

# 1. negcache — take the qwen3vl_2b cache and append action hard negatives
python build_negcache.py --gpu 6

# 2. train
/opt/conda/bin/python3.11 -u train.py --gpu 6
```

`build_negcache.py` starts from the qwen3vl_2b negcache (top-8 variant), embeds captions with MiniLM,
and keeps up to 6 of the top-80 neighbours whose action label differs (cosine floor 0.55). That gives
"same scene and appearance, different action" negatives, which push the model to attend to the action.

`download_model.sh` fetches the structure and small files with `snapshot_download`, then resumes the
large blobs with `curl -C -`, because HF intermittently stalls at 0 B/s and `snapshot_download` alone
does not finish. `HF_CACHE` must match the `HF_HOME` the training code uses.

## Inputs

| path | | how to build |
|---|---|---|
| `assets/data/mining/negcache_hardimg_ep06_poolfull_top8_tau0.85.pt` | source cache | `../qwen3vl_2b/build_negcache.py` with `N_IMG_NEG=8` |
| `assets/data/mining/negcache_action_top8a6.pt` | training cache | `build_negcache.py` |
| `assets/data/raw/recaption/train_msr_v2.csv` | recap styles | shipped (`RECAP_CSV`) |
| `assets/data/raw/pab_train/train_webp` | images | dataset |
| `jinaai/jina-reranker-m0` | base model | `download_model.sh` |

Writes go only to `OUTPUT_DIR` (default `assets/runs/rerank_jina`).

## Pick a step, then deploy

```bash
python train/reranker/eval/eval_step.py --member jina --run <run> --steps ex007000,ex008000,ex009000
python train/reranker/jina_m0/deploy.py <run> --step ex008000
```

`deploy.py` copies `checkpoints/ex{NNNNNN}/` to `assets/model_rep/reranker/jina_m0/`, wiping the
destination first. The reproduction cache is built separately:
`pipeline/S2_rerank/score_union_jina.py --rep --name jina_m0`.

## Files

| | |
|---|---|
| `download_model.sh` | fetches the base model into the HF cache |
| `build_negcache.py` | appends action hard negatives to the shared negcache |
| `train.py` | bidirectional DoRA training |
| `deploy.py` | copies the selected step to `assets/model_rep/reranker/` |
| `adopted_config.json` · `adopted_env.json` | the deployed run's config and environment |

## Evaluation data

The Track 4 test set is never opened by this code. Path B queries use train captions only, and step
selection uses the pair bench built by [`../eval/eval_step.py`](../eval/eval_step.py).
