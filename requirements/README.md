# Runtime environments (conda)

The `*.txt` files are pip requirements and `setup_conda_envs.sh` builds a conda env from each.
**The PyTorch wheels are pinned to `+cuXXX` local versions, so `--extra-index-url` is mandatory**
(the requirement files carry no index — the script adds it). The four cu129 files pull in
`core.txt` (torch 2.8.0+cu129); `vllm.txt` pins its own torch.

```bash
bash requirements/setup_conda_envs.sh              # all five
bash requirements/setup_conda_envs.sh --only gme   # one env
bash requirements/setup_conda_envs.sh --dry-run    # print the commands only
```

`core.txt` gets no env of its own — reproduction needs only torch and numpy, so install it into
whatever environment you already have. `core_cpu.txt` is the same set without the `+cu129` local
version, for a CPU-only replay (no GPU is touched either way):

```bash
pip install -r requirements/core_cpu.txt \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

`track4_beit3` needs one package on top of its requirements file:

```bash
conda run -n track4_beit3 pip install torchscale==0.2.0 --no-deps
```

`torchscale` cannot go in `beit3.txt`: its `setup.py` pins `fairscale==0.4.0` and `timm==0.4.12`,
which conflict with the `timm==1.0.26` / `fairscale==0.4.13` the rest of that env needs, and pip
rejects `--no-deps` as a per-requirement option. Those pins are stricter than what the package
actually uses — `fairscale.nn.{checkpoint_wrapper,wrap}` and `timm.models.layers.drop_path`, both
present in the pinned versions — so `--no-deps` is safe here.

| env | requirements | index | what it runs |
|---|---|---|---|
| `track4_train` | `train.txt` (transformers 5.4.0 · peft 0.18.1) | cu129 | most **training** (see [below](#training-uses-three-of-these-envs-not-one)) · `encode_metaclip2` · `encode_mc2h378` · `encode_anchor_{tcap,filip}` · `encode_qwen3vl_embed` |
| `track4_beit3` | `beit3.txt` (open_clip 3.3.0 · torchscale · transformers 4.30.2) | cu129 | `encode_beit3` · `encode_metaclip` · `encode_eva02` · `encode_gallery_emb` (dfn·convnext) · beit3 training/eval |
| `track4_gme` | `gme.txt` (**transformers 4.51.3**) | cu129 | `encode_gme` |
| `track4_llm2clip` | `llm2clip.txt` (**transformers 4.x**) | cu129 | `encode_llm2clip_anchor5` · `encode_siglip_maxsim` |
| `track4_vllm` | `vllm.txt` (torch 2.11+cu130 · vllm 0.22.0) | **cu130** | `score_union_*` · `dump_fuse_cache` (reranker VLMs) |

## Why the versions are pinned

Moving everything to "just the latest" breaks in specific places, so the families are kept apart.

- **`gme.txt` = transformers 4.51.3** — 5.x refactored configs to be nested, so GME's remote code
  cannot find `config.vocab_size` and dies
  (`AttributeError: 'GmeQwen2VLConfig' object has no attribute 'vocab_size'`).
- **`llm2clip.txt` = transformers 4.x** — LLM2CLIP's remote code does
  `from transformers.onnx import OnnxConfig`, and that module was removed in 5.x
  (`ModuleNotFoundError: transformers.onnx`). Verified working on 4.57.1; the pin is 4.56.2.
- **`beit3.txt` = transformers 4.30.2 + torchscale** — the vendored BEiT3 tree assumes this
  combination. The encoders that need `open_clip` (metaclip_v1 · eva02 · dfn/convnext) use this env
  as well.
- **`ovis` (score_union_ovis) = `gme.txt`, transformers 4.51.3** — Ovis2.5's remote code reads
  `self.llm.is_parallelizable`, an attribute removed in 5.x. Its auto_map also ships separate config
  and modeling copies, which makes `AutoModel.from_config(vit_config)` fail as Unrecognized, so the
  script re-registers the config instance's actual class to bridge them (shim at the top of
  `score_union_ovis.py`). Any 4.x loads, but only **4.51.3** reproduces the distributed
  `ovis_union_cache.pt` bit-exactly (max|Δ| = 0). On 4.56.2 (`llm2clip.txt`) the scores still
  correlate at 0.9995 but only 44.6% of pairs are identical (max|Δ| 0.75 · top-1 97.98%), which is
  enough to move the final answer. The scorer is deterministic within one env — two runs agree
  exactly — so the version is the whole difference. `ops/05_rerank.sh` sets `PY_OVIS=$PY_GME`.
- **`vllm.txt` = torch 2.11+cu130** — the MoE in `internvl_r32` (InternVL3.5-30B-A3B) requires
  `torch._grouped_mm`, and torch 2.8 accepts **compute capability 9.0 and nothing else**. That is
  Hopper only, so it also refuses Blackwell: on this box (B300, sm_103) a minimal bf16 call raises
  `RuntimeError: torch._grouped_mm is only supported on CUDA devices with compute capability = 9.0`,
  while the same call on torch 2.11 returns normally. Every forward raises, and both the scorer and
  `train/reranker/internvl_r32/train.py` swallow it — the run ends at 0 steps with no error.

## Training uses three of these envs, not one

Training is **not needed to reproduce the submission** — every adopted adapter ships under
`assets/model/{encoder,reranker}/`, and `ops/04`/`05` read those directly. It matters when a member
is re-trained, and then the env is not a preference: the same pins that make scoring reproducible
decide whether training runs at all. Two of the three failure modes are silent.

`ops/06_train.sh` holds this mapping in code, so `bash ops/06_train.sh <target>` picks the
interpreter rather than leaving it to whichever env is active.

| target | env | why |
|---|---|---|
| `anchor_{tcap,filip}_{all,heldout}` · `mc2h378_peft_{all,heldout}` · `siglip_maxsim_{all,heldout}` · `metaclip2` | `track4_train` | peft adapters, no `open_clip` |
| `jina_m0` | `track4_train` | see below |
| `qwen3vl_2b` | `track4_train` | Qwen3-VL needs 5.x; 4.30.2 raises `ImportError: cannot import name 'Cache'` |
| `metaclip_v1` | `track4_beit3` | needs `open_clip` 3.3.0, absent from `train` and `vllm` |
| `beit3` | `track4_beit3` | imports the vendored `run_beit3_finetuning` → `torchscale` |
| **`internvl_r32`** | **`track4_vllm`** | torch 2.11 for `_grouped_mm` (above). **`track4_train` gives 0 steps, silently** |

**`jina_m0` is the one to watch.** `train.py` loads `JinaVLForRanking` with a plain
`AutoModel.from_pretrained(..., trust_remote_code=True)` — no `key_mapping`, no `missing_keys`
check. transformers 5.x registers the Qwen2-VL key rename under the *class* name, which this
subclass does not match, so a mismatched version initialises the towers randomly and trains happily
against noise. That is the bug `score_union_jina.py`'s `KEY_MAP` guard exists for. Measured
`missing_keys` when loading the checkpoint:

| transformers | result |
|---|---|
| 4.51.3 (`gme`) | 0 — loads |
| **5.4.0 (`train`)** | **0 — loads; this is why `06_train.sh` sends jina here** |
| 4.30.2 (`beit3`) | `ImportError: cannot import name 'Cache'` |
| 5.9.0 (`vllm`) | load fails outright (`TypeError`) |

So the pin protects training as much as scoring, but only `track4_train` happens to be safe
*without* a guard — moving `jina_m0` training to a newer transformers needs the same `KEY_MAP`
treatment first.

## Model weights

Everything is inside the repository — no download is needed.

- `assets/model/hf_cache/hub/` — HF snapshots (`HF_HOME` points here):
  siglip2-large-512 · metaclip-2-worldwide-{l14,huge-378} · gme-Qwen2-VL-2B · jina-reranker-m0 ·
  Qwen3-VL-Reranker-{2B,8B} · **LLM2CLIP-Openai-L-14-336** · **clip-vit-large-patch14-336** ·
  **xlm-v-base** (metaclip_v1 tokenizer) · **eva02_large_patch14_clip_336**
- `assets/model/vlm_models/` — large models placed directly:
  InternVL3_5-30B-A3B-HF · Llama-3.2-11B-Vision · Pixtral-12B · Ovis2.5-9B · MetaCLIP-L14-worldwide ·
  **DFN5B-CLIP-ViT-H-14-378** · **CLIP-convnext_xxlarge-laion2B** · **Qwen3-VL-Embedding-8B**
- `assets/model/encoder/`, `assets/model/reranker/` — adopted adapters and checkpoints
  (`llm2clip_anchor5` ships the adapter plus `query_hidden.pt`, its precomputed LLM hidden states)

Reproduction deployments are written only to `assets/model_rep/` (see `tools/promote.py`); the
adopted weights are left untouched.
