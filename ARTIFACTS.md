# Artifacts

**Download → [`assets/` bundle on Google Drive](https://drive.google.com/drive/folders/11j_Oe6Ha8Ji4WOf2hCMP51v3HKlhOu7p?usp=drive_link)**

`assets/` is distributed separately from the repository — past GitHub's limits. It carries what was
produced here (trained weights, stage caches, derived data) plus the `hf_cache` of third-party base
weights. The large `vlm_models` bases are not redistributed; fetch those from their official sources
(see below). Everything in the bundle is offered for research use, under the terms each artifact
carries — [Licensing](#licensing).

The download mirrors the repository layout, so the top level can be copied or symlinked in place:

```bash
ln -sfn /path/to/download/assets/cache assets/cache
ln -sfn /path/to/download/assets/model assets/model
```

`assets/` itself ships empty with the clone; everything under it is ignored by git —
`bash setup_assets.sh` creates the skeleton, and `--check` reports which trees are still empty.

## Licensing

Summarised from each model card and dataset page at the revisions this work used, with the licence
tag quoted verbatim. It is a pointer, not legal advice — read the upstream terms before you rely on
any of it, and re-check them, since a licence can change between revisions.

**Two constraints cut across everything here.**

1. **PAB is research-only.** The benchmark states that it *"can only be used for research, any
   commercial usage is forbidden"* ([Shuyu-XJTU/CMP](https://github.com/Shuyu-XJTU/CMP)). Every
   checkpoint below was trained on it, and every tree under `data/` is derived from it, so that
   restriction reaches the whole distribution regardless of what the base licence permits.
2. **A non-commercial base makes a non-commercial adapter.** A DoRA/LoRA adapter is a derivative of
   the weights it was trained against and is useless without them, so it inherits their terms. The
   repository's own MIT `LICENSE` covers **code only** — no model weight is relicensed by it.

### Adopted checkpoints — `model/encoder` · `model/reranker`

| checkpoint | form | base | base licence | terms carried |
|---|---|---|---|---|
| `anchor_filip` · `anchor_tcap` | DoRA | `google/siglip2-large-patch16-512` | Apache-2.0 | Apache-2.0 |
| `siglip_maxsim` · `siglip_mining` | DoRA | `google/siglip2-large-patch16-384` | Apache-2.0 | Apache-2.0 |
| `metaclip2` | DoRA | `facebook/metaclip-2-worldwide-l14` | **CC-BY-NC-4.0** | **non-commercial**, attribution required |
| `mc2h378_peft` | DoRA | `facebook/metaclip-2-worldwide-huge-378` | **CC-BY-NC-4.0** | **non-commercial**, attribution required |
| `metaclip_v1` | full FT | MetaCLIP v1 L/14-worldwide-xlmv | **CC-BY-NC** | **non-commercial** — the file contains the base weights themselves |
| `beit3_v2` · `beit3_helip` · `beit3_pre` | full FT | BEiT3-large-384 ([microsoft/unilm](https://github.com/microsoft/unilm)) | MIT | MIT |
| `llm2clip_lora_v3_best` · `llm2clip_anchor5` | LoRA ×2 | `microsoft/LLM2CLIP-Openai-L-14-336` | Apache-2.0 | Apache-2.0 |
| `internvl_r32` | LoRA r32 | `InternVL3_5-30B-A3B-HF` | Apache-2.0 | Apache-2.0 |
| `qwen3vl_2b` | DoRA | `Qwen/Qwen3-VL-Reranker-2B` | Apache-2.0 | Apache-2.0 |
| `jina_m0` | DoRA | `jinaai/jina-reranker-m0` | **CC-BY-NC-4.0** | **non-commercial**, attribution required |

The four non-commercial rows — `metaclip2`, `mc2h378_peft`, `metaclip_v1`, `jina_m0` — are S1/S2
members of the submitted pipeline, so **the pipeline as a whole is non-commercial**. Commercial use
of any of them needs a separate licence from the upstream authors (Jina AI sells one for
`jina-reranker-m0`; Meta grants no commercial licence for CC-BY-NC MetaCLIP). CC-BY-NC-4.0 does
permit redistribution for non-commercial purposes with attribution, which is what the Drive bundle
relies on for these four adapters.

`llm2clip_anchor5/query_hidden.pt` holds hidden states produced by
`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` (tagged MIT, itself a Llama-3 derivative). It is
model output, not weights, and the model is never loaded at inference.

### Third-party bases

`hf_cache` is mirrored in the bundle, and so are two `vlm_models` entries (`DFN5B-CLIP-ViT-H-14-378`
and `CLIP-convnext_xxlarge-laion2B`); the rest of `vlm_models` is fetched from the official sources.
Mirroring is a redistribution rather than a fetch, so the terms below are what it rests on: the
CC-BY-NC rows permit it for non-commercial purposes with attribution, and the Apple licence permits
it for research purposes as long as the copy carries the agreement and attribution notice — which
is why `DFN5B-CLIP-ViT-H-14-378/LICENSE` travels with that directory.
`Llama-3.2-11B-Vision-Instruct` and `Pixtral-12B-2409` are gated on the Hub and are not mirrored —
fetch those in your own name rather than around the gate.

| repository | licence tag | note |
|---|---|---|
| `google/siglip2-large-patch16-{384,512}` | apache-2.0 | |
| `facebook/metaclip-2-worldwide-{l14,huge-378}` | **cc-by-nc-4.0** | non-commercial |
| MetaCLIP v1 L/14-worldwide-xlmv | **CC-BY-NC** | non-commercial; from `facebookresearch/MetaCLIP`, not the Hub |
| `jinaai/jina-reranker-m0` | **cc-by-nc-4.0** | non-commercial; commercial licence sold separately |
| `microsoft/LLM2CLIP-Openai-L-14-336` | apache-2.0 | |
| `openai/clip-vit-large-patch14-336` | no tag on the model card | upstream `openai/CLIP` is MIT |
| `Qwen/Qwen3-VL-Reranker-{2B,8B}` · `Qwen3-VL-Embedding-8B` · `Qwen3.6-35B-A3B` | apache-2.0 | |
| `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct` | apache-2.0 | |
| `InternVL3_5-30B-A3B-HF` | apache-2.0 | |
| `Ovis2.5-9B` | apache-2.0 | `NOTICE` records its Qwen3-8B + SigLIP2 lineage |
| `Pixtral-12B-2409` | apache-2.0 | **gated** on the Hub |
| `Llama-3.2-11B-Vision-Instruct` | **llama3.2 community licence** | **gated**, and the card sets `extra_gated_eu_disallowed` — not licensed to EU-domiciled entities. Accept it yourself; ships `LICENSE.txt` + `USE_POLICY.md` |
| `DFN5B-CLIP-ViT-H-14-378` | **apple-amlr** | Apple ML Research Model licence: **research purposes only**. Redistribution is allowed but must carry a copy of the agreement and the required attribution notice |
| `CLIP-convnext_xxlarge-laion2B` | mit | |
| `timm/eva02_large_patch14_clip_336…` | mit | |
| `facebook/dinov2-base` · `google/mt5-base` · `sentence-transformers/all-MiniLM-L6-v2` | apache-2.0 | |
| `facebook/xlm-v-base` · `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` | mit | the latter is a Llama-3 derivative |

### Data

| tree | derived from | terms |
|---|---|---|
| `raw/pab_{train,test}` | PAB | **not redistributed** — get it from [Shuyu-XJTU/CMP](https://github.com/Shuyu-XJTU/CMP) |
| `raw/{ucc,uca,rstp}` | UCF-Crime-Caption · UCA · RSTPReid | **not redistributed** — original sources, each under its own terms |
| `raw/recaption` | PAB train captions, rewritten by Qwen3-VL-30B (Apache-2.0, no restriction on outputs) | inherits the source terms |
| `manifest` · `mining` · `benches` · `heldout_v1` | PAB train — the BEiT3 pair index carries its captions verbatim | inherits the source terms |
| `cache/**` | scores over the test gallery | inherits the source terms |

PAB grants no explicit redistribution right for its annotations, so the derived trees are offered
for research reproduction only. Rebuild them locally instead with `train/gen/*` and the
`build_*` scripts if that matters for your use — every one of them is regenerable, and
[Training](README.md#training) gives the order.

## Required by goal

| goal | trees | size |
|---|---|---:|
| Reproduce the submission (`run_reproduce.sh best`) | `cache/` subset (20 files) + evaluation data | 1.7 G |
| Rebuild the S1 base score | + `cache/s1_base/members/` | 3.3 G |
| Run the pipeline from images | + `model/` (`encoder` `reranker` `hf_cache`) · fetch `vlm_models` | 84 G + 230 G fetched |
| Reproduce training | + `data/` | 27 G |
| Compare against the reproduction run | + `model_rep/` `cache_rep/` | 33 G |

`run_reproduce.sh` loads no model weights — it reads cached stage outputs only.

## Trees

| tree | size | contents |
|---|---:|---|
| `cache/` | 5.0 G | adopted-run stage caches: `s1_base` `s2_rerank` `s4_nn` `s4_tail` |
| `model/` | 84 G | `encoder` 29 G · `reranker` 262 M — trained here · `hf_cache` 55 G — third-party bases |
| `data/` | 27 G | `raw/recaption` 12 G · `manifest` 9.6 G · `mining` 5.9 G · `benches` 213 M · `heldout_v1` 8.2 M |
| `model_rep/` | 28 G | reproduction-run weights (`tools/promote.py --rep`) |
| `cache_rep/` | 4.8 G | caches re-encoded from `model_rep/` (`--rep`) |

`assets/runs/` is where training and reproduction runs write their output. It is not part of the
distribution and nothing in the pipeline reads it.

## Layout

Fully populated, `assets/` looks like this. `†` marks the one tree the download does not carry —
the large third-party bases you fetch yourself:

```
assets/
├── model/
│   ├── encoder/
│   ├── reranker/
│   ├── hf_cache/
│   │   ├── hub/models--<org>--<repo>/
│   │   ├── modules/
│   │   └── xet/
│   └── vlm_models/  †
│       ├── InternVL3_5-30B-A3B-HF/
│       ├── Qwen3.6-35B-A3B/
│       ├── Llama-3.2-11B-Vision-Instruct/
│       ├── Pixtral-12B-2409/
│       ├── Ovis2.5-9B/
│       ├── Qwen3-VL-Embedding-8B/
│       ├── DFN5B-CLIP-ViT-H-14-378/
│       ├── CLIP-convnext_xxlarge-laion2B/
│       └── MetaCLIP-L14-worldwide/
├── cache/
│   ├── s1_base/
│   │   ├── base_score.pt
│   │   ├── union_pool.pt
│   │   └── members/
│   ├── s2_rerank/
│   │   ├── <member>_union_cache.pt
│   │   ├── recs_8b_p3_k20.pt
│   │   ├── recs_2b_dora_k5_p3.pt
│   │   └── fuse_cache/{internvl_r32,llama32v,pixtral}/
│   ├── s4_nn/
│   │   ├── {dfn,convnext}_gallery_emb.pt
│   │   └── {gme,qwen3vl_embed8b,anchor5,metaclip2}_feats.pt
│   └── s4_tail/{internvl_r32,jina_m0,llama}_nntail_cache.pt
├── data/
│   ├── raw/{pab_test,pab_train,ucc,uca,rstp}/ · recaption/
│   ├── manifest/
│   ├── mining/
│   ├── benches/{ruleclean,ucc,uca,rstp,rerankstep}/
│   └── heldout_v1/
├── model_rep/
├── cache_rep/
└── runs/
```

## `cache/` — inference subset

```
s1_base/base_score.pt  union_pool.pt
s2_rerank/{internvl_r32,qwen3vl_2b,8b,pixtral,llama,ovis,jina_m0}_union_cache.pt
s2_rerank/recs_8b_p3_k20.pt  recs_2b_dora_k5_p3.pt
s4_nn/{dfn,convnext}_gallery_emb.pt  {gme,qwen3vl_embed8b,anchor5,metaclip2}_feats.pt
s4_tail/{internvl_r32,jina_m0,llama}_nntail_cache.pt
```

`s1_base/members/` (10 files, 3.3 G) holds the per-encoder scores `build_base.py` fuses; needed only
to rebuild `base_score.pt`, which `WEIGHTS=<json> run_reproduce.sh` forces.

## `model/`

**`encoder`** — 29 G. Adapters (`adapter_config.json` + `adapter_model.safetensors` +
`extras_state.pt` + `meta.json`): `anchor_filip` `anchor_tcap` `mc2h378_peft` `metaclip2`
`siglip_maxsim` `siglip_mining`, 30–62 M each. The two `llm2clip_*` directories carry the adapter
pair only, `anchor5` plus its `query_hidden.pt`.
Full fine-tunes: `beit3_v2` 7.6 G, `beit3_helip` 7.6 G, `metaclip_v1` 13 G.
`llm2clip_lora_v3_best` 57 M is the stage-1 vision LoRA of the `anchor5` stack:
`encode_llm2clip_anchor5.py` merges it into the vision tower before attaching `llm2clip_anchor5`,
which targets `text_adapter.*` only, so both directories are required.
`beit3_pre` 1.3 G is the third-party BEiT3 init (`beit3.spm` + COCO-retrieval checkpoint).
`siglip_mining` is the mining anchor for `train/reranker/*/build_negcache.py`, not a retrieval member.

`PEFT` below stands for those four files — `adapter_config.json` · `adapter_model.safetensors` ·
`extras_state.pt` · `meta.json`:

```
assets/model/encoder/
├── anchor_filip/            PEFT
├── anchor_tcap/             PEFT
├── mc2h378_peft/            PEFT
├── metaclip2/               PEFT
├── siglip_maxsim/           PEFT
├── siglip_mining/           PEFT
├── llm2clip_lora_v3_best/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── llm2clip_anchor5/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── query_hidden.pt
├── beit3_v2/checkpoint-best.pth
├── beit3_helip/checkpoint-best.pth
├── beit3_pre/
│   ├── beit3.spm
│   └── beit3_large_patch16_384_coco_retrieval.pth
└── metaclip_v1/epoch_4.pt
```

**`reranker`** — 262 M. Adapters: `internvl_r32` 121 M (LoRA r32), `jina_m0` 73 M, `qwen3vl_2b` 69 M
(both DoRA r16). Each needs its base model from `hf_cache` or `vlm_models`.

```
assets/model/reranker/
├── internvl_r32/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── jina_m0/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── qwen3vl_2b/
    ├── adapter_config.json
    └── adapter_model.safetensors
```

**`hf_cache`** — 55 G. Ships in the Drive folder as `hf_cache/`; drop it at
`assets/model/hf_cache` and skip the rest of this section. The scripts default to
`HF_HOME=assets/model/hf_cache` and run with `HF_HUB_OFFLINE=1`, so nothing is fetched at run time.
It carries third-party weights under their own terms, several of them non-commercial —
[Licensing](#licensing).

To build the cache from the Hub instead:

```bash
export HF_HOME=$PWD/assets/model/hf_cache

# trained encoders — base weights an adapter or full fine-tune is applied to
huggingface-cli download google/siglip2-large-patch16-512      # anchor_filip · anchor_tcap
huggingface-cli download google/siglip2-large-patch16-384      # siglip_maxsim
huggingface-cli download facebook/metaclip-2-worldwide-huge-378 # mc2h378_peft
huggingface-cli download facebook/metaclip-2-worldwide-l14     # metaclip2
huggingface-cli download facebook/xlm-v-base                   # metaclip_v1 tokenizer (vocab 901629)
huggingface-cli download microsoft/LLM2CLIP-Openai-L-14-336    # anchor5 (S4b) base — two LoRAs ship in-repo
huggingface-cli download openai/clip-vit-large-patch14-336     # anchor5 image processor

# trained rerankers
huggingface-cli download Qwen/Qwen3-VL-Reranker-2B             # qwen3vl_2b
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 # jina_m0 negcache

# zero-shot members — never trained, still required to run the pipeline
huggingface-cli download Alibaba-NLP/gme-Qwen2-VL-2B-Instruct  # gme (S1 member · S4b tail-NN)
huggingface-cli download timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k  # eva02_pre
huggingface-cli download Qwen/Qwen3-VL-Reranker-8B             # 8b

# data generation
huggingface-cli download facebook/dinov2-base                  # held-out split grouping
```

`eva02_pre` uses the **336** variant (`EVA02-L-14-336` / `merged2b_s6b_b61k` in `open_clip` terms);
the 224 checkpoint is a different model and yields a different member score.

`jinaai/jina-reranker-m0` stalls partway through `snapshot_download`, so it has its own resumable
fetcher:

```bash
bash train/reranker/jina_m0/download_model.sh
```

Pin a revision with `huggingface-cli download <repo> --revision <sha>`. The ones this work used:

```
Alibaba-NLP/gme-Qwen2-VL-2B-Instruct                 9cfa6413f704a7c1cf5064d240748e10c876b286
Qwen/Qwen3-VL-Reranker-8B                            b212dc8c91a8164aef1ea2de9c1a867611e75c04
Qwen/Qwen3-VL-Reranker-2B                            4bd860ac4f15ad1897a214615cccc700f8f71818
jinaai/jina-reranker-m0                              94bfe0aeb2d4dd7978362699cddd5893d4e0adc8
facebook/metaclip-2-worldwide-huge-378               2baff0da4b1f2fa6559db35137826bd7fba4b8e7
facebook/metaclip-2-worldwide-l14                    f48c6b1784e33f2211894af4d9f1f08381f04f12
google/siglip2-large-patch16-384                     1b426889ea62b5a72bf9839009a1b184bfc9c178
google/siglip2-large-patch16-512                     49488218e80259885f3be61d7a9455faf833b7a8
microsoft/LLM2CLIP-Openai-L-14-336                   92512331f393a003c3d98404677f991c188162c9
timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k   4f62907359c8506be7021582f360564693b22c15
facebook/dinov2-base                                 f9e44c814b77203eaa57a6bdbbd535f21ede1415
sentence-transformers/all-MiniLM-L6-v2               1110a243fdf4706b3f48f1d95db1a4f5529b4d41
facebook/xlm-v-base                                  68c75dd7733d2640b3a98114e3e94196dc543fe1
McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp      31474e395ada192e8ed1586db6be79fb3b70c9c0
openai/clip-vit-large-patch14-336                     ce19dc912ca5cd21c8a653c79e251e808ccabcd1
```

`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp` is a provenance record, not a runtime input: it is
the text encoder that produced `assets/model/encoder/llm2clip_anchor5/query_hidden.pt` (the
precomputed [1978, 4096] query hidden states). `encode_llm2clip_anchor5.py` reads that dump and never
loads a text model, so nothing in this repository fetches the repo — the cached 28 KB snapshot holds
only `modeling_llama_encoder.py`, no weights.

**`vlm_models`** — 230 G, **not distributed**. Plain directories under `VLM_MODELS`
(default `assets/model/vlm_models`). None of these are fine-tuned here: they are used zero-shot, or
they are the base a trained adapter loads on top of. Fetch them from their official sources.

| path | size | used by | role |
|---|---:|---|---|
| `Qwen3.6-35B-A3B` | 69 G | `build_antibreak_pairs.py` | zero-shot scorer |
| `InternVL3_5-30B-A3B-HF` | 58 G | `score_union_hf_4b.py` | base for the `internvl_r32` adapter |
| `Llama-3.2-11B-Vision-Instruct` | 32 G | S2 llama member | zero-shot |
| `Pixtral-12B-2409` | 24 G | S2 pixtral member | zero-shot |
| `Ovis2.5-9B` | 18 G | `score_union_ovis.py` | zero-shot |
| `Qwen3-VL-Embedding-8B` | 16 G | S1/S4 embedding member | zero-shot |
| `DFN5B-CLIP-ViT-H-14-378` | 7.4 G | S4 tail-NN member | zero-shot |
| `CLIP-convnext_xxlarge-laion2B` | 4.5 G | S4 tail-NN member | zero-shot |
| `MetaCLIP-L14-worldwide` | 4.1 G | `encode_metaclip.py` | base fallback only — see below |
| `LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned` | 16 G | `train/encoders/llm2clip_anchor5/build_*_cache.py` | cache build only — see below |

`MetaCLIP-L14-worldwide` is not an HF repository. Two small files from it **are** distributed, because
`encode_metaclip.py` and `train/encoders/metaclip_v1/train.py` import them to register the
`ViT-L-14-worldwide-xlmv` architecture (vocab 901,629, matching `facebook/xlm-v-base`):

```
assets/model/vlm_models/MetaCLIP-L14-worldwide/register.py
assets/model/vlm_models/MetaCLIP-L14-worldwide/ViT-L-14-worldwide-xlmv.json
```

Its 4.1 G `l14_worldwide.pt` is not needed for normal use: `encode_metaclip.py` defaults to the
deployed fine-tune `assets/model/encoder/metaclip_v1/epoch_4.pt`, a full fine-tune that carries every
weight. The base is read only when `--checkpoint ""` selects the untuned fallback.

`LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned` is needed only to rebuild the `anchor5` inputs —
the two trainer caches ship in `assets/data/mining/` (training-split rows only) and the
inference-side `query_hidden.pt` ships with the adapter
(rebuild: `pipeline/S1_base/encode/build_llm2clip_query_hidden.py`). LLM2CLIP
turns a caption into a pooled 8B hidden state before its text adapter, so the trainers read a cache
of those hiddens and inference reads `assets/model/encoder/llm2clip_anchor5/query_hidden.pt` — the
model itself is never loaded by the pipeline.

Fetch these from their official sources under their own terms — the revisions above are what this
work used. `Llama-3.2-11B-Vision-Instruct` and `Pixtral-12B-2409` are gated and must be accepted on
the Hub in your own name; `DFN5B-CLIP-ViT-H-14-378` is research-only under the Apple ML Research
Model licence. Per-repository terms: [Licensing](#licensing).

## `data/`

| path | size | contents |
|---|---:|---|
| `raw/recaption` | 12 G | `train_msr_v{1,2,3}.csv` multi-style LLM recaptions |
| `manifest` | 14 G | encoder manifests · `train.csv` · `mc2h378_selfresidual_neighbors.pt` · BEiT3 pair indexes |
| `mining` | 19 G | reranker manifests · negcaches · hard-negative and DPO pairs · `llm2clip_{text_cache,hidden_cache_ms}.pt` (anchor5 trainer inputs — 8B-derived, training-split rows only) |
| `benches` | 213 M | `ruleclean` `ucc` `uca` `rstp` `rerankstep` benches |
| `heldout_v1` | 8.2 M | `heldout_images.txt` (50,653) · `split.json` |

`manifest/` and `mining/` are derived from `raw/recaption` by `train/gen/gen_manifest.py`,
`train/reranker/qwen3vl_2b/build_manifest.py` and the `build_negcache.py` scripts.

### BEiT3 pair index

BEiT3 does not read the manifests above. It takes a **pair index** in the vendor's task-356 layout,
shipped in `manifest/` as four directories — two caption layouts × full/excluded:

| path | size | rows | what it is |
|---|---:|---:|---|
| `manifest/pab_v2_multi_webp` | 3.2 G | 1,013,605 | full index, multi-style captions — source for `beit3_v2` |
| `manifest/pab_v2_multi_webp_heldout` | 3.0 G | 962,952 | the same, held-out images removed — **what `beit3_v2` trains on** |
| `manifest/pab_full_webp` | 453 M | 1,013,606 | full index, single caption + HELIP hard pairs — source for `stage1`/`helip` |
| `manifest/pab_full_webp_heldout` | 369 M | 962,952 | the same, excluded — **what `stage1`/`helip` train on** |

```
<index>/annotation/train/pair_0.json … pair_74.json     one file per imgs_K group
<index>/train_webp -> $PAB_TRAIN/train_webp             symlink, images are not copied
pab_full_webp*/helip_hardpairs{,_outlier,_r2,_r2_outlier}.npy   HELIP hard pairs (row indices)
pab_*_heldout/heldout_exclusion.json                    exclusion record (counts, list md5, leak)
```

**Full vs `_heldout`.** The full indexes cover every PAB train image; the `_heldout` ones have the
50,653 held-out images dropped (5.00%), leaving 962,952 rows. Training reads the `_heldout` version
— `train.py` refuses a full index, so the held-out bench cannot leak into training — and the full
version exists to regenerate it and to run `remap-helip`.

**`v2_multi` vs `full`.** The caption layout differs:

```jsonc
// pab_v2_multi_webp — one row per image, every recaption preset in a dict
{"image": "train/imgs_0/goal/0.jpg", "image_id": "0_0",
 "captions": {"p00_original": "…", "p01_lexical": "…", …, "p10_formal": "…"},
 "scene": "…", "normal": "…", "anomaly": "…"}

// pab_full_webp — one row per image, the original caption only
{"image": "train/imgs_0/goal/3.jpg", "caption": "…", "image_id": "imgs_0_3_g"}
```

The loader samples one value out of `captions` per epoch, which is the multi-style augmentation
`beit3_v2` is trained with. `stage1` trains on single original captions and produces the init that
`helip` continues from, so `pab_full_webp` also carries the HELIP hard pairs (`_r2` is the second
mining round). Both layouts store the `goal` and `wentwrong` rows of a pair back to back, so with a
large batch and no shuffling they land in the same batch as hard negatives.

`pair_K.json` is JSONL despite the extension. The vendor loader reads `pair_0 … pair_{num_files-1}`
in order and renumbers `image_id` to the running row index, so every file in the range must exist,
empty ones included.

The `train_webp` symlink is what makes an index usable — the loader resolves
`train/imgs_X/{goal,wentwrong}/N.jpg` to `<index>/train_webp/Part {X//8+1}/imgs_X/…/N.webp`. Repoint
it after moving the tree:

```bash
for d in assets/data/manifest/pab_v2_multi_webp* assets/data/manifest/pab_full_webp*; do
  ln -sfn "$PWD/assets/data/raw/pab_train/train_webp" "$d/train_webp"
done
```

The `_heldout` pair is reproducible from the full pair, so it can be rebuilt instead of downloaded.
`exclude` verifies `heldout_v1/` before writing (list md5 `dbdf151e4bcab04f66c0f7a260ca19cb`, 50,653
images, gates all_pass) and records the result in `heldout_exclusion.json`; both indexes report
962,952 rows kept and **0 leak**:

```bash
python train/encoders/beit3/beit3_tool.py exclude \
    --src assets/data/manifest/pab_v2_multi_webp \
    --dst assets/data/manifest/pab_v2_multi_webp_heldout
python train/encoders/beit3/beit3_tool.py exclude \
    --src assets/data/manifest/pab_full_webp \
    --dst assets/data/manifest/pab_full_webp_heldout
python train/encoders/beit3/beit3_tool.py remap-helip \
    --src assets/data/manifest/pab_full_webp \
    --dst assets/data/manifest/pab_full_webp_heldout      # helip only: exclude shifts the row numbers
```

`exclude` symlinks every non-`annotation` entry across, so `train_webp` and the HELIP `.npy` files
follow automatically.

### Datasets are not included

`assets/data/raw/{pab_test,pab_train,rstp,uca,ucc}` are site-local symlinks in this checkout, not
part of the distribution. Repoint them or set:

| variable | default | for |
|---|---|---|
| `PAB_TEST` | `assets/data/raw/pab_test` | evaluation gallery + `query_index.txt` |
| `PAB_TRAIN` | `assets/data/raw/pab_train` | training root |
| `PAB_JPG` | `$PAB_TRAIN/train_jpg_512` | 512px images |
| `PAB_WEBP` | `$PAB_TRAIN/train_webp` | 1024px images |
| `GALLERY` · `QUERY_INDEX` | under `$PAB_TEST` | `run_reproduce.sh` inputs |

Training image paths resolve by remapping `imgs_<M>` to `Part {M//8+1}`.

## Verification

```bash
du -sh assets/*            # compare with the sizes above
./run_reproduce.sh best    # stage 0 names every missing input and how to build it
```
