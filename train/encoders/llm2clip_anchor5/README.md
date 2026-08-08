# `llm2clip_anchor5`

LLM2CLIP (L-14-336) with two stacked LoRAs. Not a base member — it produces `anchor5_feats.pt`,
one of the six encoders S4b tail-NN matches near-duplicates with.

| stage | trains | output |
|---|---|---|
| 1 vision | LoRA over every Linear in the vision tower | `llm2clip_lora_v3_best` |
| 2 text | LoRA over `text_adapter.adaptor.*`, on top of stage 1 merged in | `llm2clip_anchor5` |

Inference merges stage 1 into the tower and then attaches stage 2, so both are required
(`../../../pipeline/S1_base/encode/encode_llm2clip_anchor5.py`).

Environment: `track4_llm2clip` (transformers 4.x — LLM2CLIP's remote code imports
`transformers.onnx`, removed in 5.x).

## The 8B LLM is only needed for the caches

LLM2CLIP turns a caption into a pooled Llama-8B hidden state (4096d) before `text_adapter`. Running
that model during training or inference would be wasteful, so both stages read a cache built once:

| cache | built by | holds |
|---|---|---|
| `llm2clip_text_cache.pt` | `build_text_cache.py` | frozen text features (1280d) for stage 1 |
| `llm2clip_hidden_cache_ms.pt` | `build_hidden_cache.py` | multi-style pooled hiddens (4096d) for stage 2 |

Both caches **ship** in `assets/data/mining/` (training-split rows only — no training script
reads the test set), so training runs without the LLM. The LLM itself is **not bundled** —
to rebuild a cache, download `LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned` into
`assets/model/vlm_models/`, or point `LLM2CLIP_LLM` at it.

The query-side counterpart — `assets/model/encoder/llm2clip_anchor5/query_hidden.pt`, the
pooled 8B hiddens of the 1,978 query captions — is an inference input, so its rebuild script
lives with the encode scripts: `pipeline/S1_base/encode/build_llm2clip_query_hidden.py`.
It is why inference never loads the 8B model either.

## Pipeline

```bash
# 1. stage-1 cache — frozen text features for the vision trainer
LLM2CLIP_DEV=cuda:6 python train/encoders/llm2clip_anchor5/build_text_cache.py

# 2. stage 1 — vision LoRA
LLM2CLIP_DEV=cuda:6 python train/encoders/llm2clip_anchor5/train_vision.py

# 3. stage-2 cache — multi-style hiddens (K recaption presets per image)
LLM2CLIP_DEV=cuda:6 python train/encoders/llm2clip_anchor5/build_hidden_cache.py

# 4. stage 2 — text adapter, stage 1 merged underneath
LLM2CLIP_DEV=cuda:6 python train/encoders/llm2clip_anchor5/train_text.py

# 5. deploy both stages
python train/encoders/llm2clip_anchor5/deploy.py \
    --vision assets/runs/llm2clip_vision_lora --text assets/runs/llm2clip_text_lora/ep03
```

`MULTISTYLE=1` (default) samples a random preset per step and adds a consistency term pulling
paraphrases of the same image together — the shipped adapter was trained this way on the 5-preset
cache. `MULTISTYLE=0` is the anchor-preset-only baseline; `ANCHOR_P>0` weights sampling toward
the anchor preset.

## Checkpoint selection

Neither stage reads the evaluation set.

- **stage 1** splits the tail of its own cache into a 10k val gallery and 2k self-retrieval queries
  (ground truth = the image's own index) and keeps the best-scoring epoch. Nothing outside the
  training pool is scored.
- **stage 2** does not select at all — every epoch goes to `OUT_DIR/ep{N}` and the final state to
  `OUT_DIR/last`. Adoption happens outside the trainer.

Both cache builders drop the held-out images by default (`EXCLUDE_HELDOUT=1`, list from
`assets/data/heldout_v1/`), so the encoder never trains on the bench the other encoders are
selected against.

## Files

| | |
|---|---|
| `build_text_cache.py` | frozen text features -> `llm2clip_text_cache.pt` |
| `build_hidden_cache.py` | multi-style pooled hiddens -> `llm2clip_hidden_cache_ms.pt` |
| `train_vision.py` | stage 1 — vision-tower LoRA |
| `train_text.py` | stage 2 — text-adapter LoRA |
| `deploy.py` | copies either stage to `assets/model_rep/encoder/` |

Key environment variables: `LLM2CLIP_DEV` · `LLM2CLIP_BASE` · `LLM2CLIP_LLM` ·
`LLM2CLIP_V3_ADAPTER` · `TEXT_CACHE` · `HIDDEN_CACHE` · `OUT_DIR` · `EXCLUDE_HELDOUT` ·
`HELDOUT_DIR` · `MULTISTYLE`.
