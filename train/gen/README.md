# `gen`

Data generators. Everything here reads the PAB **train** split only — no script in this directory
opens the Track 4 test set.

| script | builds | consumed by |
|---|---|---|
| `msr_axes_generate.py` | `assets/data/raw/recaption/train_msr_v*.csv` — multi-style recaptions | `gen_manifest.py`, `gen_heldout_v1.py`, reranker manifests |
| `gen_manifest.py` | `assets/data/manifest/pab_manifest_msr_v{1,2}.jsonl` · `_v1_scene.jsonl` | encoder trainers |
| `gen_metaclip_v1_csv.py` | `assets/data/manifest/train.csv` (`filepath`,`title`) | `train/encoders/metaclip_v1` |
| `gen_heldout_v1.py` | `assets/data/heldout_v1/` — `heldout_images.txt` + `split.json` | every trainer (exclusion) and evaluator (bench) |

Run order: `msr_axes_generate.py` → `gen_manifest.py` / `gen_metaclip_v1_csv.py` / `gen_heldout_v1.py`.
All four skip work when their output already exists; `--force` rebuilds.

The recaption CSVs are shipped, so regeneration is only needed to change the caption set.

---

## `msr_axes_generate.py` — recaptioning

Rewrites each annotation caption into 11 styles, defined as coordinates over 8 control axes
(coverage, compression, syntax, register, epistemic, specificity, paraphrase, stance). Output is one
CSV row per (image, style), plus one row carrying the original caption.

It talks to an **OpenAI-compatible `/v1` endpoint**, which is not part of this repository. Serve the
rewriter first.

### 1. Serve the rewriter — Qwen3-VL-30B

Use **`Qwen3-VL-30B-A3B-Instruct`**. It is the one model not bundled here, since it is needed only for
generation: download it into `assets/model/vlm_models/`, or point `VLM_MODELS` at wherever it lives.

The model must be **vision-capable**. With `--img-max-side` the script attaches the source image as a
base64 data URI and switches to the image-conditioned system prompt; a text-only server accepts the
request and silently returns different captions.

```bash
# env: track4_vllm
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
  --model assets/model/vlm_models/Qwen3-VL-30B-A3B-Instruct \
  --served-model-name qwen3vl-30b --port 8000 --tensor-parallel-size 2
```

### 2. Generate

`--endpoint` takes the base up to `/v1`; the script reads `/v1/models` for the model id.

```bash
# text-only
python train/gen/msr_axes_generate.py --endpoint http://127.0.0.1:8000

# image-conditioned (448px long side)
python train/gen/msr_axes_generate.py --endpoint http://127.0.0.1:8000 --img-max-side 448
```

Budget days, not hours: one pass is ~1M images × 11 styles.

### Inputs

| what | default | environment variable |
|---|---|---|
| source captions (annotation JSONL) | `assets/data/raw/pab_train/annotation/train` | `PAB_ANNOT` |
| source images (with `--img-max-side`) | `assets/data/raw/pab_train/train_jpg_512` | `PAB_JPG` |
| output | `assets/data/raw/recaption/train_msr_v1.csv` | `OUT_DIR` · `RECAP_NAME` |

Image paths are resolved by remapping `imgs_<M>` to `Part {M//8+1}`.

### Output

`train_msr_v1.csv` with columns `image_id, image_path, style, caption, scene, normal, anomaly`.
Alongside it, `control_vectors.json` records the axes and presets actually used, and
`done_image_ids.txt` tracks progress.

### Flags

| flag | |
|---|---|
| `--dry-run` | print every preset's prompt and exit; no endpoint needed |
| `--limit N` | first N images only, for a smoke test |
| `--preset-set` | `msr` (default) = the 11 styles of the shipped CSVs · `axes` = 11 redesigned to cover the axes evenly |
| `--style-naming` | `name` (default) = `p01_lexical` · `axes` = 8-axis slug, which restores the coordinates on its own |
| `--presets` | comma-separated subset of the active table |
| `--img-max-side` | long-side pixels for image conditioning (0 = text only) |
| `--concurrency` · `--chunk` | in-flight requests (32) and images per write batch (64) |
| `--rewriter <tag>` | records the source as `@tag` in the `style` column, so several endpoints can be mixed |
| `--exemplars` · `--k-shot` | in-context exemplars; every source is verified to be a train caption, and the run aborts otherwise |
| `--force` | ignore `done_image_ids.txt` and regenerate from the start |

### Resuming

An interrupted run leaves `done_image_ids.txt` next to the CSV and resumes from it, appending to the
existing file. A CSV present **without** that marker is treated as a downloaded artifact and
generation is skipped — pass `--force` to override, or `--out-dir` to write elsewhere.

### Reproducibility

Regeneration reproduces the *method*, not the shipped bytes. `TEMPERATURES_MSR` mirrors the `axes`
preset at the same coordinates, and sampling is not seeded, so captions differ run to run. Use the
shipped CSVs to reproduce the submitted models.

---

## `gen_manifest.py`

Turns a recaption CSV into the per-image manifest the encoder trainers read. Filters, record shape
and ordering match the trainer's own loader, so the trainer never re-derives anything.

```bash
python train/gen/gen_manifest.py            # all three variants
python train/gen/gen_manifest.py --force
```

| variant | source | used by |
|---|---|---|
| v1 | `train_msr_v1.csv`, 11 styles | `anchor_filip_{all,heldout}` |
| v2 | `train_msr_v2.csv`, 12 styles | `anchor_tcap_*` · `mc2h378_peft_*` |
| v1_scene | `train_msr_v1.csv` plus a `scene` field | `siglip_maxsim_*` |

## `gen_metaclip_v1_csv.py`

Flattens the annotations into the two-column CSV `open_clip` expects. Rows whose image is missing
under `train_webp` are skipped and counted.

```bash
python train/gen/gen_metaclip_v1_csv.py
OUT_CSV=<path> python train/gen/gen_metaclip_v1_csv.py
```

## `gen_heldout_v1.py`

Builds the held-out bench used for every encoder selection decision. Groups are derived from DINOv2
near-duplicate components rather than from ids, because the annotations carry no video or clip id;
whole components are removed, so no image of a held-out group survives in train.

```bash
python train/gen/gen_heldout_v1.py --gpu 6
```

Pipeline: annotation → index → DINOv2 224px embeddings → all pairs at cos ≥ 0.95 → union-find
components → stratified sampling and repair → gates → two files. Intermediates stay in memory.

Output `assets/data/heldout_v1/`:

| file | read by |
|---|---|
| `heldout_images.txt` (50,653) | trainers, to exclude those images |
| `split.json` | evaluators — exclusion list, main/hard benches, gates |

`split.json` carries `stats.heldout_md5` and the gate results; `train/encoders/eval/heldout_bench.py`
verifies both before any training or scoring starts, so a changed list stops the caller.
