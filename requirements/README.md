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
| `track4_train` | `train.txt` (transformers 5.4.0 · peft 0.18.1) | cu129 | encoder/reranker **training** · `encode_metaclip2` · `encode_mc2h378` · `encode_anchor_{tcap,filip}` · `encode_qwen3vl_embed` |
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
- **`ovis` (score_union_ovis) = transformers 4.x** — Ovis2.5's remote code reads
  `self.llm.is_parallelizable`, an attribute removed in 5.x. Its auto_map also ships separate config
  and modeling copies, which makes `AutoModel.from_config(vit_config)` fail as Unrecognized, so the
  script re-registers the config instance's actual class to bridge them (shim at the top of
  `score_union_ovis.py`). Verified working on 4.57.1.
- **`vllm.txt` = torch 2.11+cu130** — the MoE in `internvl_r32` (InternVL3.5-30B-A3B) requires
  `torch._grouped_mm` (Hopper/sm_90+). On torch 2.8 every forward raises and the scorer ends
  silently at 0 steps.

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
