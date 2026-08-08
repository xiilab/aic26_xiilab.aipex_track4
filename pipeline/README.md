# pipeline — S1 → S4

The submitted chain. Inference replays pre-computed scores from `assets/cache/`, so
`bash run_reproduce.sh best` needs no GPU and finishes in ~16 s.

This document covers the other direction: **what produces those caches**, in the order you
would run it.

```
S1_base/    encoder members → base score → candidate union
S2_rerank/  cross-encoders re-score the union
S3_assign/  fuse + injective assignment
S4_tail/    NN completion · R@5 promotion · consensus · resolution demote
utils/      gallery_norm — column convention, shared by every stage
```

Column order is `sorted(os.listdir(GALLERY))`, row order is `query_index.txt`. Every member
follows it; `utils/gallery_norm.py` is the single source.

---

## 0. Recap data — download first

Encoder training reads **multi-style recaptions of the PAB train split**, not the raw
captions. They are large and slow to generate (LLM calls over 1M images), so get the
prebuilt CSVs before anything else.

```bash
# place under assets/data/raw/recaption/
assets/data/raw/recaption/
├─ train_msr_v1.csv   3.6 GB   11 presets × 1,013,606 images   → anchor_filip · beit3
└─ train_msr_v2.csv   3.9 GB   12 presets × 1,013,605 images   → anchor_tcap · mc2h378_peft · heldout
```

`assets/data/raw/` absorbs site paths as symlinks — point `recaption/` wherever the files
live, or copy them in. `RECAP_CSV` overrides the path per script.

### Regenerating instead of downloading

`train/gen/msr_axes_generate.py` rebuilds them from the annotation captions through a local
vLLM endpoint. The endpoint is **not** part of this repo — serve it first.

**1. Serve the rewriter.** The adopted rewriter is `Qwen3-VL-30B-A3B-Instruct`. It is the one
model *not* bundled — it is needed only to regenerate — so download it into
`assets/model/vlm_models/` (or point `VLM_MODELS` at wherever it lives). It must be
**vision-capable**: with `--img-max-side` the script attaches the source image as a base64
data URI and switches to the image-conditioned system prompt, so a text-only server silently
yields different captions.

```bash
# env: track4_vllm
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
  --model assets/model/vlm_models/Qwen3-VL-30B-A3B-Instruct \
  --served-model-name qwen3vl-30b --port 8000 --tensor-parallel-size 2
```

**2. Generate.** `--endpoint` takes the `/v1` base; the script reads `/v1/models` for the id.

```console
$ python train/gen/msr_axes_generate.py --endpoint http://127.0.0.1:8000
  [recap] assets/data/raw/recaption/train_msr_v1.csv  (3.6 GB)
```

Inputs beyond the endpoint:

| what | default | environment variable |
|---|---|---|
| source captions (annotation JSONL) | `assets/data/raw/pab_train/annotation/train` | `PAB_ANNOT` |
| source images (with `--img-max-side`) | `assets/data/raw/pab_train/train_jpg_512` | `PAB_JPG` |
| output | `assets/data/raw/recaption/train_msr_v1.csv` | `OUT_DIR` · `RECAP_NAME` |

Image paths are resolved by remapping `imgs_<M>` to `Part {M//8+1}`.

An interrupted generation leaves `done_image_ids.txt` next to the CSV and resumes from it
instead. `--dry-run` prints the prompts without calling the endpoint. Budget days, not hours.

`--rewriter <tag>` records the source model in the `style` column as `@tag`, so several
endpoints can be mixed and the rewriter itself treated as an axis (LaCLIP-style source
diversification). Reproducing the shipped CSVs means one rewriter and no tag.

The script carries two preset tables, selected with `--preset-set`:

| table | keys | |
|---|---|---|
| `msr` (default) | `p01_lexical` … `p11_compound` | the styles of the shipped CSVs |
| `axes` | `p01_keyword` … `p11_recall_alt` | redesigned to cover the eight axes evenly |

The default therefore writes the same style keys the shipped CSVs use, and the downstream lists keep
matching. **Regeneration still reproduces the *method*, not the shipped bytes** — `TEMPERATURES_MSR`
mirrors the `axes` preset sharing the same coordinates, and sampling is not seeded, so captions
differ run to run. That is what the download is for.

Two practical consequences:

- the preset keys are hardcoded downstream — `gen_heldout_v1.py:STYLE_SET` (`p01_lexical` …
  `p10_formal`) and `train/reranker/jina_m0/train.py:STYLE_LIST` (`p00_original` …
  `p11_compound`). `--preset-set axes` or `--style-naming axes` writes keys neither list knows, so
  both need switching first, otherwise held-out query sampling matches nothing;
- `--presets` narrows the run to a subset, and `--style-naming axes` writes the axis slug
  (`x10.brd.sent.neut.asrt.spfr.lexi.obsv`) instead of the preset id.

---

## 1. Manifest and held-out split

Two derived inputs, both built once and reused by every trainer.

```bash
python train/gen/gen_manifest.py                 # → assets/data/manifest/pab_manifest_msr_v{1,2}.jsonl
python train/gen/gen_heldout_v1.py --gpu 6       # → assets/data/heldout_v1/{heldout_images.txt, split.json}
```

Both refuse to overwrite; pass `--force` to rebuild.

The held-out split is what makes checkpoint selection test-free: DINOv2 embeddings group
near-duplicates (cos ≥ 0.95 connected components) and whole groups are excluded, giving
50,653 images (5.00%) that no encoder ever trains on. Trainers read the exclusion list and
**stop** if its md5 or gates do not match.

---

## 2. Encoder members → `assets/cache/s1_base/members/`

Ten members. One name throughout — the member key in `final.json`, the checkpoint directory
under `assets/model/encoder/`, and the cache file stem are all the same string.

| member | cache file | encode |
|---|---|---|
| `anchor_tcap` | `anchor_tcap_tta_views.pt` | `encode_anchor_tcap.py --run A --epochs 9` |
| `anchor_filip` | `anchor_filip_tta_views.pt` | `encode_anchor_filip.py` |
| `metaclip2` | `metaclip2_feats.pt` | `encode_metaclip2.py` |
| `mc2h378_peft` | `mc2h378_peft_score.pt` | `encode_mc2h378.py` |
| `beit3_v2` | `beit3_v2_score.pt` | `encode_beit3.py --recipe v2` |
| `beit3_helip` | `beit3_helip_score.pt` | `encode_beit3.py --recipe helip` |
| `gme` | `gme_feats.pt` | `encode_gme.py` — *zero-shot* |
| `eva02_pre` | `eva02_pre_score.pt` | `encode_eva02.py` — *zero-shot* |
| `metaclip_v1` | `metaclip_v1_score.pt` | `encode_metaclip.py` |
| `siglip_maxsim` | `siglip_maxsim_score.pt` | `encode_siglip_maxsim.py` |

Three shapes of cache (`tta_views` · `feats` · `score`) — `build_base.py` has one loader each,
so an encode script may not change its output format.

Each line runs in the conda env listed in [`requirements/README.md`](../requirements/README.md);
they are not interchangeable (transformers is pinned differently in each).

```bash
G=6
E=pipeline/S1_base/encode
# ── track4_train
CUDA_VISIBLE_DEVICES=$G python $E/encode_anchor_tcap.py --run A --epochs 9
python $E/encode_anchor_filip.py   --gpu $G
python $E/encode_metaclip2.py            --gpu $G
python $E/encode_mc2h378.py        --gpu $G
# ── track4_beit3  (open_clip · torchscale)
python $E/encode_beit3.py          --gpu $G --recipe v2
python $E/encode_beit3.py          --gpu $G --recipe helip
CUDA_VISIBLE_DEVICES=$G python $E/encode_eva02.py             # no --checkpoint = zero-shot
CUDA_VISIBLE_DEVICES=$G python $E/encode_metaclip.py
# ── track4_gme / track4_llm2clip
python $E/encode_gme.py            --gpu $G                   # track4_gme
python $E/encode_siglip_maxsim.py  --stage base   --gpu $G    # track4_llm2clip
python $E/encode_siglip_maxsim.py  --stage maxsim --gpu $G    # track4_beit3
```

Most take `--out --limit`; `--limit` truncates gallery and queries for a smoke run and is
refused together with `--rep`. `encode_anchor_tcap` / `encode_eva02` / `encode_metaclip` take
the GPU from `CUDA_VISIBLE_DEVICES` instead of `--gpu`.

`siglip_maxsim` is one script run twice, in different envs: `--stage base` (track4_llm2clip)
dumps the pooled SigLIP2-L DoRA score, then `--stage maxsim` (track4_beit3) rescores each
query's own top-10 with BEiT3-helip token↔patch MaxSim and blends
`0.95·base + 0.05·rowminmax(MaxSim)` in place. Everything outside the top-10 stays at base —
the member is named after its *base*; the MaxSim tokens come from `beit3_helip`.

### tail-NN embeddings → `assets/cache/s4_nn/`

S4b and S4c match candidates by raw embedding, not by score, so six encoders dump features
to a second directory. `run_reproduce.sh` checks all six before it starts.

| file | producer | env |
|---|---|---|
| `gme_feats.pt` | `encode_gme.py` | track4_gme |
| `metaclip2_feats.pt` | `encode_metaclip2.py` | track4_train |
| `qwen3vl_embed8b_feats.pt` | `encode_qwen3vl_embed.py` | track4_train |
| `anchor5_feats.pt` | `encode_llm2clip_anchor5.py` | track4_llm2clip |
| `dfn_gallery_emb.pt` | `encode_gallery_emb.py --enc dfn` | track4_beit3 |
| `convnext_gallery_emb.pt` | `encode_gallery_emb.py --enc convnext` | track4_beit3 |

`encode_gme.py` and `encode_metaclip2.py` produce a member cache **and** a tail-NN copy, but they
only write the `s4_nn/` copy under `--rep`; in adopted mode copy it across yourself.
`encode_llm2clip_anchor5.py` stacks two bundled LoRAs (vision `llm2clip_lora_v3_best` merged
first, then the `llm2clip_anchor5` text adapter) and reads query features from the precomputed
`query_hidden.pt` — the 8B LLM behind them is not bundled (see `train/encoders/llm2clip_anchor5/`).

---

## 3. Base and union

```bash
ENS_DEV=cuda:6 python pipeline/S1_base/build_base.py     # → assets/cache/s1_base/base_score.pt
python pipeline/S1_base/build_union.py                   # → union_pool.pt + s2_rerank/*_union_cache.pt
```

`build_base.py` reads only `members/` and the fixed weights in
`tools/ensemble/weights/final.json` — no model is loaded, `ENS_DEV` just says where the
tensor math runs (CPU works, much slower). The union is `5-base union (champ ·
mapmax · r5 · r10 · r20) ∪ current base top-20`, median 29 candidates per query.

`build_union.py` **merges** existing reranker scores onto the richer union rather than
re-scoring — running the S2 scorers from scratch is the expensive path.

---

## 4. Rerankers → `assets/cache/s2_rerank/`

Seven members re-score the union. Three are fine-tuned, four are zero-shot. Cache files are
`<name>_union_cache.pt`; the name is also the comb key and the adapter directory.

| member | model | script |
|---|---|---|
| `internvl_r32` | InternVL3_5-30B-A3B + LoRA | `score_union_hf_4b.py --model … --adapter … --name internvl_r32` |
| `pixtral` | Pixtral-12B-2409 *(zs)* | `score_union_pixtral_4b.py` |
| `qwen3vl_2b` | Qwen3-VL-Reranker-2B + DoRA | `score_union_qwen_4b.py --qwen 2b --adapter … --name qwen3vl_2b` |
| `llama` | Llama-3.2-11B-Vision *(zs)* | `score_union_hf_4b.py --model … --name llama` |
| `8b` | Qwen3-VL-Reranker-8B *(zs)* | `score_union_qwen_4b.py --qwen 8b --name 8b` |
| `ovis` | Ovis2.5-9B *(zs)* | `score_union_ovis.py` |
| `jina_m0` | jina-reranker-m0 + DoRA | `score_union_jina.py` |

Some members feed S3 comb, others only the S4 tail stages, so all seven caches
are required. Adapters live in `assets/model/reranker/<name>/`, base weights in
`assets/model/vlm_models/`.

All seven run in **`track4_vllm`**. `internvl_r32` in particular **must**: its MoE needs
`torch._grouped_mm`, and on torch 2.8 every forward raises and the scorer ends at 0 steps
without an error — see [`requirements/README.md`](../requirements/README.md).

S3 also reads `s2_rerank/fuse_cache/{internvl_r32,pixtral,llama32v}/` — the same yes/no
logits laid out as a fixed top-20 array instead of a sparse dict. It needs **no GPU**: the
prompt and token sets are identical to the union scorers and the top-20 candidates are a
subset of the union pool, so `dump_fuse_cache.py --from-union` just re-lays the values
(verified against the shipped dumps: 39,560 pairs, max|Δ| = 0, for all three).

```bash
for n in internvl_r32 pixtral llama32v; do
  python pipeline/S2_rerank/dump_fuse_cache.py --name $n --from-union   # add --rep for cache_rep
done
```

The candidate columns come from `recs_8b_p3_k20.pt` (`build_recs.py`, also GPU-free), so
whenever the base changes, rebuild **recs and fuse_cache together** — mixing generations
leaves the two disagreeing on ~19% of candidates. Re-scoring with a live VLM is only needed
when no union cache exists.

---

## 5. S3 · S4

Deterministic, no GPU, no model loading — everything from here on is arithmetic over the
cached scores. `run_submission.py` (repo root) drives S3 and calls `tail_refinement.chain()`:

```
S3  comb = w_sim·z(sim) + Σ wₖ·z(rerankₖ)  →  injective assignment (one query ↔ one image)
S4a overlay      freeze ranks 1–7, refill 8–10 from the union pool; a newcomer must beat
                 the incumbent by a margin guard (m=0.5 z) to displace it
S4b NN complete  1-NN of a rank-8/9/10 candidate, agreed by ≥4 encoders at cos ≥ 0.80,
                 is inserted at rank 10
S4c R@5 promote  a near-dup outside comb-top5 (cos ≥ 0.80 to top1) that ovis/internvl_r32/
                 jina_m0 rank highly is promoted to rank 5; ranks 1–4 frozen
S4d cons6        6 rerankers unanimously prefer rank2 over rank1 → local augmenting path
                 re-assigns injectively; re-detect until convergence
```

Stop early with `POST=none` (S3 only) or `FINAL_PASS=off` (skip S4e).

Run this through `run_reproduce.sh`, not by calling `run_submission.py` bare. The script
exports `COMB_W` / `TAIL_W` / `TAU_PX` from `tools/ensemble/adopted.py`; without them S4a
falls back to `W0_DEFAULT` inside `tail_refinement.py`, which is **not** the adopted vector
(`8b` 0.3 and `llama` 0.2 there vs 0.0 / 0.1 in `final.json`) and yields a different answer.

---

## 6. Reproduction mode

Every producing script takes `--rep`, which swaps model and cache roots:

```
assets/model_rep/{encoder,reranker}/     checkpoints chosen on the held-out split
assets/cache_rep/{s1_base,s2_rerank,…}/  scores encoded from those checkpoints
```

```bash
python train/encoders/metaclip2/deploy.py <all_run> --epoch <e*>      # ckpt → model_rep
python pipeline/S1_base/encode/encode_metaclip2.py --gpu 6 --rep            # → cache_rep
python pipeline/S1_base/build_base.py  --rep
python pipeline/S1_base/build_union.py --rep
python pipeline/S2_rerank/score_union_jina.py --rep
REP=1 bash run_reproduce.sh best
```

`REP=1` points the whole chain at `assets/cache_rep/`. Deploy scripts, one per trained model
— the encoder ones live beside the training code, so the directory keeps its `_all` suffix:

```
train/encoders/{anchor_tcap_all, anchor_filip_all, mc2h378_peft_all, beit3, metaclip2, metaclip_v1}/deploy.py
train/reranker/{internvl_r32, jina_m0, qwen3vl_2b}/deploy.py
```

The eval scripts can do it in one step instead, with their `--deploy-rep` hook.

---

## Paths

Scripts resolve everything from the repository root; no absolute paths. Override with env:

| env | default |
|---|---|
| `PAB_TEST` · `GALLERY` · `QUERY_TEXT` · `QUERY_INDEX` | `assets/data/raw/pab_test/…` |
| `RECAP_CSV` · `MANIFEST_DIR` · `MINE_DIR` | `assets/data/{raw/recaption,manifest,mining}` |
| `HELDOUT_DIR` | `assets/data/heldout_v1` |
| `S1_CACHE` · `S1_MEMBERS` · `S4_NN` | `assets/cache/{s1_base,s1_base/members,s4_nn}` |
| `ENCODER_CKPT` · `ENCODER_CKPT_DIR` · `VLM_MODELS` · `HF_CACHE` | `assets/model/…` |
| `OUTPUT_DIR` · `RUNS_ROOT` | `assets/runs` |

At run time `run_reproduce.sh` adds a few more: `TRACK4` (artifact root), `REDUMP_DIR`,
`ENC_SRC`, `WORK`, `PY`, `G`. Weights come from `WEIGHTS` (a `final.json`-shaped file) —
`COMB_W` / `TAIL_W` / `TAU_PX` override single stages and are meant for experiments.
