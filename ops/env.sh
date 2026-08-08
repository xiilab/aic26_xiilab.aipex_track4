#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ops/env.sh — shared settings sourced by every ops/*.sh. Not meant to be run.
#
#   source ops/env.sh
#
# Override through the environment:
#   GPU=7 SRC_REPO=/other/path source ops/env.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$OPS_DIR/.." && pwd)"
cd "$REPO"                                  # every script assumes the repository root as CWD

# ── GPU ────────────────────────────────────────────────────────────────────
GPU="${GPU:-6}"
GPU2="${GPU2:-7}"

# ── Source repository (read-only) ──────────────────────────────────────────
# Holds the original training runs, data and adopted weights. Never written to.
SRC_REPO="${SRC_REPO:-/DATA/XIILAB.AIpex_track4}"

# Two run roots; mixing them up means writing into the source tree:
#   RUNS_SRC   input for selection (02). Per-epoch checkpoints of a heldout run exist only
#              in the source tree. Read-only.
#   RUNS_LOCAL input for deployment (03). The adopted checkpoints copied here, and the target
#              of tools that write, such as build_swa.
RUNS_SRC="${RUNS_SRC:-$SRC_REPO/assets/runs}"
RUNS_LOCAL="${RUNS_LOCAL:-$REPO/assets/runs}"
# What the child Python processes see. 02 overrides it to SRC, 03/04 to LOCAL.
RUNS_ROOT="${RUNS_ROOT:-$RUNS_LOCAL}"
export RUNS_ROOT

# Block writes into the source tree (build_swa.py creates <run>/checkpoints/swa).
assert_writable_run() {
  case "$1" in
    "$SRC_REPO"/*) [ "${FORCE_SRC:-0}" = 1 ] || die "refusing to write into the source tree: $1
      $SRC_REPO is read-only. Use a run under $RUNS_LOCAL, or copy the epochs you need
      there first. Pass FORCE_SRC=1 to write into the source tree anyway.";;
  esac
}

# ── conda env -> interpreter ───────────────────────────────────────────────
# The five envs pin different transformers versions and are not interchangeable:
#   train 5.4.0 · beit3 4.30.2 · gme 4.51.3 · llm2clip 4.56.2 · vllm 5.9.0
PREFIX="${PREFIX:-track4_}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /opt/conda)}"
PY_TRAIN="${PY_TRAIN:-$CONDA_BASE/envs/${PREFIX}train/bin/python}"
PY_BEIT3="${PY_BEIT3:-$CONDA_BASE/envs/${PREFIX}beit3/bin/python}"
PY_GME="${PY_GME:-$CONDA_BASE/envs/${PREFIX}gme/bin/python}"
PY_LLM2CLIP="${PY_LLM2CLIP:-$CONDA_BASE/envs/${PREFIX}llm2clip/bin/python}"
PY_VLLM="${PY_VLLM:-$CONDA_BASE/envs/${PREFIX}vllm/bin/python}"
PY_ENS="${PY_ENS:-/opt/conda/bin/python}"     # ensemble/score arithmetic (loads no model)

E="pipeline/S1_base/encode"

# ── Adopted selections ─────────────────────────────────────────────────────
# The baseline that 02_select.sh is compared against when choosing epochs/steps directly.
declare -A ADOPTED=(
  [anchor_tcap]="swa:8-10"      # build_swa.py <run> 8 10
  [anchor_filip]="swa:8-10"
  [mc2h378_peft]="swa:2-4"
  [metaclip2]="ep:2"
  [beit3_v2]="ep:3"             # checkpoint-3.pth
  [beit3_helip]="ep:2"          # checkpoint-2.pth
  [metaclip_v1]="ep:4"          # epoch_4.pt
  [internvl_r32]="step:2500"    # <run>/step2500
  [jina_m0]="step:ex008000"     # <run>/checkpoints/ex008000
  [qwen3vl_2b]="step:ex007000"
)

# ── Adopted run directories (relative to a run root) ───────────────────────
# Run directory names differ between trees, so each member carries a candidate list and the
# first one that exists is used. A new naming scheme only needs another candidate here.
declare -A RUN_CAND=(
  [anchor_tcap]="anchor_tcap_all"
  [anchor_filip]="anchor_filip_all"
  [mc2h378_peft]="mc2h378_peft_all"
  [metaclip2]="metaclip2_FULL 20260609_070733_mc2_l14_distill_ddp_w5 mc2_l14_all_w5_e2_FULL"
  [beit3_v2]="beit3_v2_FULL"
  [beit3_helip]="beit3_helip_FULL"
  [metaclip_v1]="metaclip_v1_FULL"
  [internvl_r32]="rerank_internvl_r32"
  [jina_m0]="rerank_jina_m0"
  [qwen3vl_2b]="rerank_qwen3vl_2b"
)

# Heldout runs used for selection. Rerankers have no separate heldout run — their step
# checkpoints are picked by pair accuracy on the same run. beit3_helip has none either, since
# stage1 is helip's init.
declare -A HELDOUT_CAND=(
  [anchor_tcap]="anchor_tcap_heldout"
  [anchor_filip]="anchor_filip_heldout"
  [mc2h378_peft]="mc2h378_peft_heldout"
  [metaclip2]="metaclip2_heldout 20260730_011922_mc2_l14_distill_heldout_w5"
  [beit3_v2]="beit3_v2_heldout"
  [beit3_helip]="beit3_stage1_heldout"
  [metaclip_v1]="metaclip_ft_heldout metaclip_v1_heldout"   # v1_heldout holds logs only; checkpoints are in ft_heldout
  [internvl_r32]="rerank_internvl_r32"
  [jina_m0]="rerank_jina_m0"
  [qwen3vl_2b]="rerank_qwen3vl_2b"
)

# resolve_run <root> "<cand1> <cand2> …" -> first path that exists; otherwise the first
# candidate's path plus a non-zero exit code.
resolve_run() {
  local root="$1" c
  for c in $2; do
    if [ -d "$root/$c" ]; then printf '%s\n' "$root/$c"; return 0; fi
  done
  printf '%s\n' "$root/${2%% *}"; return 1
}
# Pick a run that actually holds weights: the local tree first, then the source tree.
resolve_run_with_ckpt() {
  local cand="$1" p
  for root in "$RUNS_LOCAL" "$RUNS_SRC"; do
    p="$(resolve_run "$root" "$cand")" || continue
    if find "$p" \( -name '*.safetensors' -o -name '*.pth' -o -name 'epoch_*.pt' \) -print -quit 2>/dev/null | grep -q .; then
      printf '%s\n' "$p"; return 0
    fi
  done
  resolve_run "$RUNS_LOCAL" "$cand"; return 1
}

# ── Output ─────────────────────────────────────────────────────────────────
say()  { printf '\033[1;36m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m⚠\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# DRY=1 prints the commands instead of running them (shared by every script).
DRY="${DRY:-0}"

# Missing inputs abort, except under DRY=1: previewing the plan before the envs and data
# exist is what that mode is for, so it only warns.
need_py() {
  local py="$1" name="$2"
  [ -x "$py" ] && return 0
  if [ "$DRY" = 1 ]; then warn "(dry) $name env not present yet: $py"; return 0; fi
  die "$name env not found: $py
      -> bash ops/00_setup_envs.sh --only ${name}"
}
need_path() {
  [ -e "$1" ] && return 0
  if [ "$DRY" = 1 ]; then warn "(dry) input not present yet: $1"; return 0; fi
  die "input not found: $1
      -> $2"
}
run() {
  printf '  $ %s\n' "$*"
  [ "$DRY" = 1 ] || "$@"
}
