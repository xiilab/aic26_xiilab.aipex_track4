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

Run every trainer from the repository root, in the conda env listed in
[`requirements/README.md`](../requirements/README.md).
