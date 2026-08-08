#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 00_setup_envs.sh — create the five conda envs and verify them (doctor)
#
# The repository already ships requirements/setup_conda_envs.sh. This script calls it and then
# checks by import that **each env is actually usable** — the transformers pin differs per env,
# so a successful install does not mean a working env.
#
# Usage:
#   bash ops/00_setup_envs.sh --dry-run        # print the commands only
#   bash ops/00_setup_envs.sh                  # create all five
#   bash ops/00_setup_envs.sh --only beit3     # one env
#   bash ops/00_setup_envs.sh --doctor         # skip creation, verify only
#   bash ops/00_setup_envs.sh --only gme --bg  # background + log (ops/logs/setup_gme.log)
#   bash ops/00_setup_envs.sh --only vllm      # foreground, live progress
#   bash ops/00_setup_envs.sh --delegate       # delegate to the repository script (buffered output)
#   FORCE=1 bash ops/00_setup_envs.sh --only beit3   # remove and recreate that env
#
# An existing env is not removed; `pip install -r` is re-applied on top. FORCE=1 recreates it.
# Do not launch several envs at once — they contend for the same ~/.cache/pip and end up slower.
# torch wheels stay in the cache, so the cu129 envs (train·beit3·gme·llm2clip) are fast from the
# second one on. Only vllm is cu130 (torch 2.11) and downloads its own wheels.
#
# env -> what it runs
#   track4_train     encoder/reranker training · encode_metaclip2 · encode_mc2h378
#                    · encode_anchor_{tcap,filip} · encode_qwen3vl_embed
#   track4_beit3     encode_beit3 · encode_eva02 · encode_metaclip · encode_gallery_emb
#   track4_gme       encode_gme
#   track4_llm2clip  encode_siglip_maxsim · encode_llm2clip_anchor5
#   track4_vllm      S2 reranker scorers (score_union_*) · dump_fuse_cache
# ─────────────────────────────────────────────────────────────────────────────
# ── Tolerating edits while running ─────────────────────────────────────────
# bash reads a script by byte offset, so editing the original during an install that takes tens
# of minutes makes it resume mid-word and run something else (requirements/... read as
# quirements/..., for example). The script therefore re-executes itself from a snapshot copy
# taken at startup, after which edits to the original are harmless. The copy lives in /tmp, so
# the original directory is passed as _OPS_DIR for env.sh to be found.
if [ -z "${_OPS_SNAPSHOT:-}" ]; then
  _snap="$(mktemp)"; cat "${BASH_SOURCE[0]}" > "$_snap"
  _OPS_SNAPSHOT=1 _OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
    bash "$_snap" "$@"; _rc=$?
  rm -f "$_snap"; exit "$_rc"
fi

source "${_OPS_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env.sh"

DOCTOR_ONLY=0; BG=0; DELEGATE=0; PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --doctor)  DOCTOR_ONLY=1 ;;
    --dry-run) DRY=1; PASS+=(--dry-run) ;;
    --only)    PASS+=(--only "$2"); ONLY="$2"; shift ;;
    --force)   PASS+=(--force) ;;
    --bg)      BG=1 ;;
    --delegate) DELEGATE=1 ;;   # delegate to the repository script (conda run, buffered output)
    -h|--help) sed -n '2,40p' "${_OPS_DIR:-.}/00_setup_envs.sh"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

# ── Background execution ───────────────────────────────────────────────────
# A pip install takes minutes to tens of minutes per env. --bg detaches with nohup and prints
# the log path. Do not launch several at once — they contend for ~/.cache/pip and its http
# cache. To run several in sequence, use --bg without --only (all five in one job).
if [ "$BG" = 1 ]; then
  LOGDIR="$REPO/ops/logs"; mkdir -p "$LOGDIR"
  tag="${ONLY:-all}"
  LOG="$LOGDIR/setup_${tag}.log"
  say "[00] background install — $tag"
  # DRY has to be passed through, otherwise a DRY=1 preview would install for real.
  if [ "$DRY" = 1 ]; then
    printf '  $ nohup bash %s %s > %s 2>&1 &\n' "$OPS_DIR/00_setup_envs.sh" "${PASS[*]-}" "$LOG"
    warn "(dry) nothing was launched. Run without DRY to launch it."
    exit 0
  fi
  nohup bash "$OPS_DIR/00_setup_envs.sh" ${PASS[@]+"${PASS[@]}"} > "$LOG" 2>&1 &
  pid=$!
  ok "PID $pid · log $LOG"
  echo "  follow   : tail -f $LOG"
  echo "  finished?: kill -0 $pid 2>/dev/null && echo running || echo done"
  echo "  verify   : bash ops/00_setup_envs.sh --doctor"
  exit 0
fi

# ── 1. Creation ────────────────────────────────────────────────────────────
# env -> requirements file · PyTorch index (the requirements pin +cuXXX, so an index is required)
CU129="https://download.pytorch.org/whl/cu129"
CU130="https://download.pytorch.org/whl/cu130"
declare -A NODEPS_OF=( [beit3]="torchscale==0.2.0" )
ENV_ROWS=(
  "train:train.txt:$CU129"
  "beit3:beit3.txt:$CU129"
  "gme:gme.txt:$CU129"
  "llm2clip:llm2clip.txt:$CU129"
  "vllm:vllm.txt:$CU130"
)

# Direct install, not through conda run: conda run buffers the child's output to the end, which
# hides progress and looks like a hang. Calling the env's python -m pip directly stays live.
setup_one() {
  local name="$1" req="$2" idx="$3"
  local env="${PREFIX}${name}" py="$CONDA_BASE/envs/${PREFIX}${name}/bin/python"
  local reqfile="$REPO/requirements/$req"
  [ -f "$reqfile" ] || die "requirements file not found: $reqfile"
  say "──── $env  ($req · $(basename "$idx"))"

  if [ -x "$py" ]; then
    if [ "${FORCE:-0}" = 1 ]; then
      warn "FORCE=1 -> removing the existing env"; run conda env remove -y -n "$env"
    else
      echo "  already present -> skipping creation, re-applying pip only"
    fi
  fi
  [ -x "$py" ] || run conda create -y -n "$env" "python=$PYVER"
  local P="$CONDA_BASE/envs/$env/bin/python"
  run "$P" -m pip install --upgrade pip --root-user-action=ignore

  run "$P" -m pip install -r "$reqfile" --extra-index-url "$idx" --root-user-action=ignore

  # Installed with --no-deps (a requirements file cannot express per-package --no-deps)
  #   beit3 : torchscale 0.2.0's metadata pins timm==0.4.12, which cannot be resolved together
  #           with beit3.txt's timm==1.0.26 (ResolutionImpossible). The code only uses
  #           torchscale.model.BEiT3 and architecture.config.EncoderConfig, so dependency
  #           resolution is not needed and timm works through the deprecation shim in 1.0.26.
  if [ -n "${NODEPS_OF[$name]:-}" ]; then
    warn "$name: installing with --no-deps -> ${NODEPS_OF[$name]}"
    # shellcheck disable=SC2086
    run "$P" -m pip install ${NODEPS_OF[$name]} --no-deps --root-user-action=ignore
  fi
  echo
}

if [ "$DOCTOR_ONLY" = 0 ]; then
  command -v conda >/dev/null || die "conda not found. Set CONDA_BASE or put conda on PATH."
  PYVER="${PYVER:-3.11}"
  if [ "$DELEGATE" = 1 ]; then
    say "[00] creating conda envs — delegated to requirements/setup_conda_envs.sh (output buffered)"
    [ -f "$REPO/requirements/setup_conda_envs.sh" ] || die "not found: $REPO/requirements/setup_conda_envs.sh"
    bash "$REPO/requirements/setup_conda_envs.sh" "${PASS[@]+"${PASS[@]}"}"
  else
    say "[00] creating conda envs (direct · live progress)"
    for row in "${ENV_ROWS[@]}"; do
      IFS=":" read -r n r i <<< "$row"
      [ -n "${ONLY:-}" ] && [ "$ONLY" != "$n" ] && continue
      setup_one "$n" "$r" "$i"
    done
  fi
  echo
fi

# ── 2. Verification (doctor) ───────────────────────────────────────────────
# Check that each env passes the imports it is used for. A failure here would otherwise only
# surface during encoding, so it is caught at this point.
say "[00] doctor — per-env import check"

check_env() {
  local name="$1" py="$2" code="$3"
  printf '  %-18s' "$name"
  if [ ! -x "$py" ]; then printf '\033[31m✗ missing\033[0m  %s\n' "$py"; return 1; fi
  if out=$("$py" -c "$code" 2>&1); then
    printf '\033[32m✓\033[0m %s\n' "$out"
  else
    printf '\033[31m✗\033[0m\n%s\n' "$(echo "$out" | tail -4 | sed 's/^/      /')"
    return 1
  fi
}

V='import torch,transformers as t; print(f"torch {torch.__version__} · transformers {t.__version__} · cuda {torch.cuda.is_available()} ({torch.cuda.device_count()}gpu)")'

# beit3 installs torchscale with --no-deps, so "does it import" is not enough. The symbols the
# BEiT3 code actually uses are checked directly, i.e. whether the timm 1.x shim still works.
BEIT3_CHK='import open_clip, timm
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.data import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.optim.lookahead import Lookahead
from timm.utils import get_state_dict
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig
print(f"timm {timm.__version__} · torchscale+shim OK", end=" · ")'

fails=0
[ -z "${ONLY:-}" ] || say "  (--only ${ONLY} -> verifying that env only)"
for row in \
  "track4_train:$PY_TRAIN:$V" \
  "track4_beit3:$PY_BEIT3:$BEIT3_CHK; $V" \
  "track4_gme:$PY_GME:$V" \
  "track4_llm2clip:$PY_LLM2CLIP:import open_clip,peft; $V" \
  "track4_vllm:$PY_VLLM:import vllm; print('vllm',vllm.__version__); $V" ; do
  n="${row%%:*}"; rest="${row#*:}"; p="${rest%%:*}"; c="${rest#*:}"
  [ -n "${ONLY:-}" ] && [ "$n" != "${PREFIX}${ONLY}" ] && continue
  check_env "$n" "$p" "$c" || fails=$((fails+1))
done

echo
if [ "$fails" -gt 0 ]; then
  warn "$fails env(s) failed. Reinstall with: FORCE=1 bash ops/00_setup_envs.sh --only <name>"
  exit 1
fi
ok "envs ready -> next: bash ops/01_stage_assets.sh"
