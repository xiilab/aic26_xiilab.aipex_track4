# ops — running the pipeline from scratch (runbook)

**select → deploy → encode**, in that order. Every script is run from the repository root and the
shared settings live in a single file, [`env.sh`](env.sh).

```
ops/env.sh              shared settings (GPU · env interpreters · adopted values · run roots)
ops/00_setup_envs.sh    create the five conda envs + doctor verification
ops/01_stage_assets.sh  stage data/model + check the required inputs
ops/02_select.sh        pick an epoch/step on the held-out bench
ops/03_deploy.sh        selected checkpoint -> assets/model_rep
ops/04_encode.sh        encode the 10 S1 members -> cache -> base·union
ops/05_rerank.sh        re-score with the 7 S2 rerankers + 3 fuse_cache dumps
ops/06_train.sh         (optional) re-train a member in the env its dependencies pin
```

`06` is not part of the reproduction path — the adopted weights already ship. It exists so that
re-training a member uses the same pinned env the adopted weights came from; see §8.

Full order (S3 and S4 are driven together by `run_reproduce.sh`):

```
[0] 00_setup_envs.sh                 five conda envs        <- the only real disk consumer
[1] 01_stage_assets.sh               symlink the assets
[2] 02_select.sh                     selection (skip it to use the adopted values)
[3] 03_deploy.sh --adopted           -> model_rep
[4] 04_encode.sh --all               -> s1_base/members (10 members)
[5] 04_encode.sh --tail              -> s4_nn (6 encoders)
[6] 04_encode.sh --build             -> base_score.pt + union_pool.pt
[7] 05_rerank.sh --all               -> s2_rerank/*_union_cache.pt (7)   <- heaviest GPU stage
[8] 05_rerank.sh --fuse              -> s2_rerank/fuse_cache/{3}
[9] run_reproduce.sh best            S3 fuse+assign -> S4 tail -> answer
```

Every script accepts `DRY=1` (or `--dry-run`) — **previewing the commands before running is
recommended.**

---

## 0. At a glance

```bash
bash ops/00_setup_envs.sh --dry-run      # see what would be installed
bash ops/00_setup_envs.sh                # the five envs
bash ops/01_stage_assets.sh --check      # see what is empty
bash ops/01_stage_assets.sh              # stage (symlinks by default)

bash ops/02_select.sh --list             # targets and the adopted baseline
bash ops/03_deploy.sh --adopted          # <- skip selection, deploy the adopted values

# Reproduction run: REP=1 everywhere. It reads model_rep and writes cache_rep, leaving the
# shipped assets/cache untouched. Dropping it on any one line mixes the two roots.
REP=1 bash ops/04_encode.sh --list
     bash ops/04_encode.sh --smoke metaclip2   # smoke forces REP=0 into ops/smoke/ — never a rep artifact
REP=1 bash ops/04_encode.sh --all              # 10 members
REP=1 bash ops/04_encode.sh --tail             # the 6 s4_nn encoders
REP=1 bash ops/04_encode.sh --build            # base + union_pool (from scratch)
     bash ops/05_rerank.sh --smoke jina_m0     # quick check of one S2 member
REP=1 bash ops/05_rerank.sh --all              # the 7 S2 rerankers (heaviest GPU stage)
REP=1 bash ops/05_rerank.sh --recs             # assemble recs_*.pt (no GPU)
REP=1 bash ops/05_rerank.sh --fuse             # the 3 fuse_cache dumps
REP=1 bash run_reproduce.sh best               # final answer
```

**`REP=1` is not optional here.** Every line above defaults to `REP=0`, which reads the adopted
`assets/model/` and **overwrites the shipped `assets/cache/`** — including the git-tracked
`s2_rerank/*_union_cache.pt` (05 warns about this at run time). Re-creating the adopted caches on
purpose is the one case where you leave it off; that path needs no `03_deploy.sh` at all, since
`assets/model/` already holds the adopted weights. See §6 for the roots themselves.

To choose everything yourself, replace `03_deploy.sh --adopted` with `02_select.sh <model>` then
`03_deploy.sh <model> --pick <value>`.

---

## 1. Environments (`00_setup_envs.sh`)

The five envs are **not interchangeable — each pins a different transformers version**. A
successful install does not mean a working env, so doctor runs the real imports inside each one.

| env | transformers | used by |
|---|---|---|
| `track4_train` | 5.4.0 | training · `encode_metaclip2` · `encode_mc2h378` · `encode_anchor_*` · `encode_qwen3vl_embed` |
| `track4_beit3` | 4.30.2 | `encode_beit3` · `encode_eva02` · `encode_metaclip` · `encode_gallery_emb` · `encode_siglip_maxsim --stage maxsim` (torchscale·timm·open_clip) |
| `track4_gme` | 4.51.3 | `encode_gme` |
| `track4_llm2clip` | 4.56.2 | `encode_siglip_maxsim --stage base` · `encode_llm2clip_anchor5` · `score_union_ovis` (needs transformers 4.x) |
| `track4_vllm` | 5.9.0 | the S2 reranker scorers except ovis · `dump_fuse_cache` (torch 2.11+cu130) |

`core/train/beit3/gme/llm2clip` use cu129 wheels; only `vllm` uses cu130. A CUDA 13.0 driver runs
both. Version-pin rationale is in [`../requirements/README.md`](../requirements/README.md).

```bash
bash ops/00_setup_envs.sh --only beit3        # one env
FORCE=1 bash ops/00_setup_envs.sh --only gme  # remove and recreate
bash ops/00_setup_envs.sh --doctor            # verify only
```

---

## 2. Assets (`01_stage_assets.sh`)

`assets/**` is gitignored, so a fresh clone has `assets/{data,model,cache}` holding only
`.gitkeep`. Encoding needs at least these three:

```
assets/data/raw/pab_test/{gallery, query_index.txt, query_text.json}   evaluation input
assets/model/{encoder, hf_cache, vlm_models}                           backbones and zero-shot weights
assets/data/heldout_v1/                                                selection bench
```

The sources are large (`model` 313G · `data` 27G), so **symlinks are the default** — the inputs are
read-only, so it is safe and costs no space. `MODE=copy` makes real copies:

```bash
bash ops/01_stage_assets.sh --check                  # check only
MODE=copy bash ops/01_stage_assets.sh --only heldout # copy just the small ones (8.2M)
```

Two paths are left alone, because replacing them wholesale drops whatever is already in place:

- **`assets/cache`** — downloaded from the Drive bundle, or produced by `04`/`05`. Its four subtrees
  fill at different times, so only its status is reported; `01` names which files are missing.
- **`assets/model`** — its four children (`encoder`·`hf_cache`·`reranker`·`vlm_models`) are staged
  individually rather than the parent.

> `MODE=copy` uses `cp -a`, so **an entry that is itself a symlink is copied as a link**. Use
> `cp -rL` when the real data is needed.
> Empty vs populated is decided by **the presence of a regular file** — a directory holding only
> `.gitkeep` counts as empty.

---

## 3. Selection (`02_select.sh`)

**Selection uses the held-out split only.** Picking an epoch by test mAP would make the result
test-derived, and open-data epoch selection is known to anti-correlate with Track 4. The selection
metric is the **main** bench of `heldout_v1/split.json` (2,000 queries / 36,773 gallery); the hard
bench is diagnostic only.

```bash
bash ops/02_select.sh --list                     # targets, adopted values, heldout runs
bash ops/02_select.sh anchor_tcap --epochs 1-12
bash ops/02_select.sh --all                      # ~16 min per eval × models
```

Each family uses a different tool; the script picks it:

| family | tool |
|---|---|
| anchor_tcap · anchor_filip · mc2h378_peft · metaclip2 | `train/encoders/eval/eval_heldout.py` (adapters) |
| siglip_maxsim | `train/encoders/siglip_maxsim_heldout/search_swa_range.py` — the only member with no built-in SWA range, so 02 searches the range itself instead of scoring epochs |
| beit3_v2 · beit3_helip | `train/encoders/beit3/beit3_tool.py eval` |
| metaclip_v1 | `train/encoders/eval/eval_heldout_openclip.py` (full FT) |
| the three rerankers | `train/reranker/eval/eval_step.py` (pair accuracy) |

`eval_heldout.py` and `eval_step.py` accept a `--deploy-rep <name>` hook that selects and deploys
in one step.

### There are two run roots (mixing them writes into the source tree)

| variable | default | used for |
|---|---|---|
| `RUNS_SRC` | `$SRC_REPO/assets/runs` | input for **selection (02)**. Per-epoch checkpoints exist only there. Read-only |
| `RUNS_LOCAL` | `$REPO/assets/runs` | input for **deployment (03)**. The adopted checkpoints copied locally |

`build_swa.py` **writes** to `<run>/checkpoints/swa`, so it must not point at the source tree.
`assert_writable_run` in `env.sh` blocks writes to a source path (`FORCE_SRC=1` overrides).

Run directory names differ between trees (`metaclip2_FULL` vs a timestamped name), so `RUN_CAND`
and `HELDOUT_CAND` hold a **candidate list** and the first one that exists is used. A new naming
scheme only needs another candidate added to those arrays in `env.sh`.

> Heldout runs are large — a 12-epoch encoder run reaches ~90G, a 10-epoch one ~120G.

---

## 4. Deployment (`03_deploy.sh`)

**`assets/model_rep/` is the only deployment target.** Overwriting the adopted `assets/model/`
would make the md5 comparison meaningless, and the `deploy.py` scripts only ever write to
`model_rep`.

```bash
bash ops/03_deploy.sh --adopted            # the ten adopted values (reproduce the baseline)
bash ops/03_deploy.sh metaclip2 --pick 2
bash ops/03_deploy.sh anchor_tcap --pick 8-10
bash ops/03_deploy.sh --verify             # md5 the deployment against the source tree
```

Adopted values:

| model | adopted | deployment |
|---|---|---|
| anchor_tcap | SWA 8–10 | `build_swa.py <run> 8 10` -> `deploy.py <run>` |
| anchor_filip | SWA 8–10 | same |
| mc2h378_peft | SWA 2–4 | same |
| metaclip2 | `ep02` | `deploy.py <run> --epoch 2` |
| beit3_v2 | `checkpoint-3` | `deploy.py <run> --recipe v2 --epoch 3` |
| beit3_helip | `checkpoint-2` | `deploy.py <run> --recipe helip --epoch 2` |
| metaclip_v1 | `epoch_4` | `deploy.py <ckpt_dir> --epoch 4` |
| internvl_r32 | `step2500` | `deploy.py <run> --step 2500` |
| jina_m0 | `ex008000` | `deploy.py <run> --step ex008000` |
| qwen3vl_2b | `ex007000` | `deploy.py <run> --step ex007000` |

> `MC2_RUN` overrides the run directory used for `metaclip2` when the candidate list does not
> resolve to the intended one.

`tools/promote.py` is the generic manual alternative; it detects the checkpoint layout per family.

---

## 5. Encoding (`04_encode.sh`)

`REP` selects the pair of roots: `REP=0` (the default) reads `assets/model` and writes
`assets/cache`; `REP=1` adds `--rep`, reading `model_rep` and writing `cache_rep`. **The
deployment root and the cache root have to match** for `REP=1 run_reproduce.sh` to be coherent.

```bash
bash ops/04_encode.sh --list             # per-member artifact state
bash ops/04_encode.sh --smoke metaclip2  # --limit 64, quick check (not canonical)
bash ops/04_encode.sh --all              # 10 members
bash ops/04_encode.sh --tail             # the 6 s4_nn encoders
bash ops/04_encode.sh --build            # base + union
```

`base` = the flat 10 members, `norm(Σ wₑ·norm(scoreₑ))`. There are three cache shapes
(`tta_views`/`feats`/`score`) with one loader each in `build_base.py`, so **an encode script must
not change its output format.**

### `--build` defaults to `--no-merge`

`build_union.py` **merges rather than regenerating** — it requires the existing `*_union_cache.pt`
(`REDUMP_SRC`·`REDUMP_SRC2`) as input. From scratch those do not exist, so `04 --build` passes
`--no-merge` to produce `union_pool.pt` only and leaves the re-scoring to `05`. To reuse existing
caches and skip S2, use `04 --build --merge`.

Easy to trip over:
- `encode_anchor_tcap` · `encode_eva02` · `encode_metaclip` have no `--gpu`; they read `CUDA_VISIBLE_DEVICES`.
- `--limit` cannot be combined with `--rep`.
- Without `--rep`, `gme` and `metaclip2` write no `s4_nn` copy — copy it across yourself.
- Gallery columns are `sorted(os.listdir(GALLERY))` and rows are `query_index.txt`; `utils/gallery_norm.py` is the single source.

### Smoke coverage

| stage | smoke |
|---|---|
| S1 encoding | 5/10 (`--limit` support: metaclip2·mc2h378_peft·beit3_v2·beit3_helip·siglip_maxsim) |
| S2 rerankers | 7/7 (`--limit` or `--q-end`) |
| S3/S4 | not needed — `run_reproduce.sh` runs in ~16 s without a GPU |

`anchor_tcap`·`anchor_filip`·`eva02_pre`·`metaclip_v1`·`gme` have no `--limit`, so a smoke run
would encode everything; `04 --smoke` skips them with a warning.

### Reference timings (full run, no GPU contention)

| member | time |
|---|---|
| `mc2h378_peft` | ~5 min |
| `beit3_v2`·`beit3_helip`·`eva02_pre`·`metaclip_v1` | ~10 min |
| `anchor_tcap`·`anchor_filip` (3 TTA views) | ~25 min |
| **`gme`** (Qwen2-VL-2B) | **~80 min** <- slowest |

---

## 6. S2 rerankers (`05_rerank.sh`)

Seven members map onto five scripts. All of them need `track4_vllm` and a GPU — this is **the
heaviest GPU stage of the pipeline**, so running it member by member or in slices is usually
better.

| member | script | model |
|---|---|---|
| `internvl_r32` | `score_union_hf_4b.py` | InternVL3_5-30B-A3B-HF + LoRA (MoE · required) |
| `llama` | `score_union_hf_4b.py` | Llama-3.2-11B-Vision-Instruct (zero-shot) |
| `8b` | `score_union_qwen_4b.py` | Qwen3-VL-Reranker-8B (reuses recs) |
| `qwen3vl_2b` | `score_union_qwen_4b.py` | Qwen3-VL 2B + DoRA (reuses recs) |
| `pixtral` | `score_union_pixtral_4b.py` | Pixtral-12B-2409 (zero-shot) |
| `ovis` | `score_union_ovis.py` | Ovis2.5-9B (zero-shot) |
| `jina_m0` | `score_union_jina.py` | jina-reranker-m0 + adapter |

```bash
bash ops/05_rerank.sh --list             # members, models, artifact state
bash ops/05_rerank.sh internvl_r32       # a selection
bash ops/05_rerank.sh 8b --slice 0:500   # query slice -> saved as a shard
bash ops/05_rerank.sh --merge 8b         # merge the shards
bash ops/05_rerank.sh --recs             # assemble recs_*.pt (no GPU)
bash ops/05_rerank.sh --fuse             # the 3 fuse_cache dumps (S3 contract)
```

### The recs path

`8b` and `qwen3vl_2b` can reuse a `recs_*.pt` dump so that **only new (q,c) pairs** reach the
model. The scorer looks for it under `$TRACK4/<file>`, but `TRACK4` defaults to
`assets/cache/work`, which does not exist, so `05` sets `TRACK4`·`WORKDIR`·`POOL_FILE` explicitly
and passes `RECS` to `dump_fuse_cache` for the same reason.

`RECS_DIR` follows the mode being run (`$CACHE/s2_rerank`, overridable): a `--rep` run must reuse
the scores produced by the `model_rep` adapters, not the adopted ones. Reusing an adopted
`qwen3vl_2b` dump inside a reproduction run would mix two adapter generations.

**Reuse is optional.** The dump is produced by `05 --recs` *from* the union caches, so on a
from-scratch run it does not exist yet — requiring it would deadlock (recs needs the union cache;
the union cache would need recs). `--reuse-recs` is therefore passed only when the file is present,
and a warning names the missing path otherwise. The order that avoids the extra work is:

```bash
bash ops/05_rerank.sh --all     # 8b · qwen3vl_2b score the full pool the first time
bash ops/05_rerank.sh --recs    # assemble recs_*.pt from the union caches (no GPU)
bash ops/05_rerank.sh --fuse    # needs the recs dump
```

`recs_*.pt` is assembled from the base score and the reranker union scores, so `--recs` needs no
GPU. Consumers hard-code the file names (`recs_8b_p3_k20.pt`, `recs_2b_dora_k5_p3.pt`).

**A rebuilt recs dump has different candidates than the distributed one — expected, and harmless.**
`cand` is the base top-K, and `build_recs.py` reads the base from `BASE_PT`, defaulting to
`assets/cache/s1_base/base_score.pt`. The distributed `recs_*.pt` was not built from that file: its
top-20 columns match `base_score.pt` for only 4.8% of queries as a set (0% in order), because it
predates it and came from an earlier base in the greedy sweep. So `05 --recs` on a from-scratch run
produces a dump whose candidate lists differ from the shipped artifact — do not read that as a
failed reproduction. Measured on the full test set, replacing the shipped dump with a rebuilt one
leaves the final answer **byte-identical**, in both the replay and the full-recompute
configurations: `cand`/`sim` feed S4a's fallback path only, and every candidate it reaches is
already covered by the union caches. Compare metrics and answer lines, never the recs md5.

### fuse_cache is a separate format

`score_union_*.py` re-scores the whole union pool, whereas `dump_fuse_cache.py` produces the
**fixed champion top-20 column** format. That is the contract `IV_CACHE` in S3 `fuse.py` reads
(internvl_r32 · pixtral · llama32v), so both are required.

---

## 7. After this runbook

The S2 rerankers run in `track4_vllm` and all seven are needed (`internvl_r32` is MoE and
mandatory). S3 and S4 are driven together by `run_reproduce.sh`.

```bash
# S3+S4 (no GPU, ~16 s)
bash run_reproduce.sh best
```

**The three post-S4b nntail caches** — S4b (NN completion) inserts near-duplicate candidates from
outside the union at ranks 8–10, so S4d (cons6 propagation) finds no reranker score for them.
These caches fill that gap, and they need `run_reproduce`'s `$WORK` to exist (a couple of hundred
pairs, a few minutes):

```bash
for n in internvl_r32 jina_m0 llama; do
  $PY_VLLM pipeline/S4_tail/dump_nntail_cache.py --name $n --work <WORK>
done
```

Do not call `run_submission.py` directly instead of `run_reproduce.sh`: without
`COMB_W`/`TAIL_W`/`TAU_PX`, S4a falls back to `W0_DEFAULT` rather than the adopted vector and
produces a different answer.

Reference md5: adopted weights through every stage = `f6290321`, stopping at S3 = `98471257`.

---

## 8. Training (`06_train.sh`, optional)

**Nothing above needs this.** Every adopted adapter ships under `assets/model/{encoder,reranker}/`,
and `04`/`05` read them directly — reproduction starts at encoding. `06` is for re-training a
member, and its only job is to launch `train.py` under the env that member's dependencies pin,
the same way `05` routes `ovis` to `track4_gme`.

```bash
bash ops/06_train.sh --list                  # targets and the env each one needs
bash ops/06_train.sh anchor_tcap_all
GPU=3 bash ops/06_train.sh internvl_r32 -- --lora-r 32 --lr 1e-4 --grad-accum 8
DRY=1 bash ops/06_train.sh --all             # print the plan; needs no env or data
```

Arguments after `--` are appended verbatim to `train.py`, and apply to a single target.

### `GPU=` works, but the two mechanisms must not be mixed

Twelve of the fourteen scripts take `--gpu` and index the **physical** device
(`cuda:{physical_gpu}`), so `06` passes `--gpu $GPU` to those and exports nothing. `metaclip2` and
`metaclip_v1` take no `--gpu` and read `CUDA_VISIBLE_DEVICES`, so those get the export instead.

Doing both would break them: with `CUDA_VISIBLE_DEVICES=3` the process sees a single device
numbered 0, while the script asks for `cuda:3`. And exporting alone is not enough either —
`beit3`, `internvl_r32`, `jina_m0` and `qwen3vl_2b` assign
`os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu` themselves before importing torch, so their own
default (`0`, or `6` for beit3) silently wins and the job lands on a device you did not pick. Pass
`--gpu` explicitly after `--` to override `GPU=`.

### The env is not interchangeable, and two failures are silent

| target | env | consequence of the wrong one |
|---|---|---|
| `anchor_{tcap,filip}_*` · `mc2h378_peft_*` · `siglip_maxsim_*` · `metaclip2` | `track4_train` | — |
| `qwen3vl_2b` | `track4_train` | 4.30.2: `ImportError: cannot import name 'Cache'` (loud) |
| `jina_m0` | `track4_train` | 5.x **randomly initialises both towers with no error** |
| `metaclip_v1` | `track4_beit3` | `ModuleNotFoundError: open_clip` (loud) |
| `beit3` | `track4_beit3` | vendored `run_beit3_finetuning` needs `torchscale` (loud) |
| **`internvl_r32`** | **`track4_vllm`** | torch 2.8: **0 optimiser steps, checkpoint still written** |

`internvl_r32` is the sharp edge. Its MoE calls `torch._grouped_mm`, which on torch 2.8 accepts
compute capability **9.0 only** — Hopper — so on this box (B300, sm_103) every forward raises
`RuntimeError`. `train.py` catches it per step and continues, so the run finishes quickly, reports
no error, and writes a checkpoint trained on nothing. torch 2.11 (`track4_vllm`) is the fix, and
the same call there returns normally.

`jina_m0` is the quiet one: `train.py` loads `JinaVLForRanking` with a bare
`AutoModel.from_pretrained(trust_remote_code=True)`, without the `key_mapping`/`missing_keys` guard
that `score_union_jina.py` carries. Measured, the checkpoint loads with `missing_keys=0` on both
4.51.3 and **5.4.0**, so `track4_train` is safe as-is; 5.9.0 fails to load at all. Rationale and the
full measurement table are in [`../requirements/README.md`](../requirements/README.md).

`CUBLAS_WORKSPACE_CONFIG` follows `env.sh` and is unset by default, matching the container the
adopted weights were trained in, so the GEMM kernels do not change between training and
encoding/scoring. `KEEP_CUBLAS_WORKSPACE=1` leaves an inherited value alone.

> Seeds are a separate matter and `06` does not touch them. `manual_seed` +
> `cudnn.deterministic` are set by the encoder trainers and by `jina_m0`/`qwen3vl_2b`;
> `beit3`, `metaclip_v1` and `internvl_r32` set no seed, and no trainer calls
> `torch.use_deterministic_algorithms`. Re-training therefore reproduces the *environment*, not the
> weights bit-for-bit.

---

## GPU

`GPU` and `GPU2` in `env.sh` select the devices (default 6 and 7). `ENS_DEV` must be a GPU — on
CPU the base build takes 51+ minutes and starves any training dataloader. Long encodes are better
launched with `nohup … &` and a log file (one eval takes ~16 min).
