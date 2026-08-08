#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Create the five conda environments for PAB Track4. env-to-script mapping and pins: README.md.
# Also creates the empty asset/result directories, which git cannot carry.
# requirements/*.txt pin +cuXXX local-version wheels, so pip needs the PyTorch index added
# (the files themselves carry no --extra-index-url):
#   core / train / beit3 / gme / llm2clip → cu129
#   vllm                                  → cu130   (torch 2.11.0+cu130)
# A CUDA 13.0 driver is compatible with both wheels.
#
# Usage:
#   bash requirements/setup_conda_envs.sh                 # all five
#   bash requirements/setup_conda_envs.sh --dry-run       # print the commands only
#   bash requirements/setup_conda_envs.sh --only gme      # a single env
#   PREFIX=pab_ PYVER=3.11 FORCE=1 bash requirements/setup_conda_envs.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ="$REPO/requirements"
PREFIX="${PREFIX:-track4_}"
PYVER="${PYVER:-3.11}"
CU129="https://download.pytorch.org/whl/cu129"
CU130="https://download.pytorch.org/whl/cu130"

DRY=0; ONLY=""; FORCE="${FORCE:-0}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only) ONLY="$2"; shift ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# env name : requirements file : index
ENVS=(
  "train:train.txt:$CU129"
  "beit3:beit3.txt:$CU129"
  "gme:gme.txt:$CU129"
  "llm2clip:llm2clip.txt:$CU129"
  "vllm:vllm.txt:$CU130"
)

# Directories git cannot carry, since git tracks files and these start out empty. Created here rather
# than kept alive with placeholder files. Everything deeper is created by whichever script writes it.
DIRS=(
  assets assets/cache assets/cache_rep assets/data assets/data/raw
  assets/model assets/model_rep assets/runs
  results results/reproduced
)

# Packages that cannot live in a requirements file: their own pins conflict with this env's, and pip
# accepts --no-deps only as a command-line flag. torchscale pins fairscale==0.4.0 and timm==0.4.12
# but uses only fairscale.nn.{checkpoint_wrapper,wrap} and timm.models.layers.drop_path, both present
# in the versions beit3.txt pins.
declare -A NO_DEPS=( [beit3]="torchscale==0.2.0" )

command -v conda >/dev/null || { echo "✗ conda not found."; exit 1; }
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

run() { echo "  \$ $*"; [ "$DRY" = 1 ] || "$@"; }

post_install() {            # $1 = short env name, $2 = full env name
  local extra="${NO_DEPS[$1]:-}"
  [ -n "$extra" ] || return 0
  echo "  extra, installed with --no-deps: $extra"
  # shellcheck disable=SC2086
  run conda run -n "$2" pip install $extra --no-deps
}

echo "[setup] conda=$CONDA_BASE · python=$PYVER · prefix=$PREFIX · $([ "$DRY" = 1 ] && echo DRY-RUN || echo EXECUTE)"
df -h "$CONDA_BASE" | awk 'NR==2{printf "  disk: %s free (%s used) — the five envs need tens of GB\n",$4,$5}'
echo

echo "──────── directories ────────"
for d in "${DIRS[@]}"; do run mkdir -p "$REPO/$d"; done
echo

for row in "${ENVS[@]}"; do
  IFS=":" read -r name req idx <<< "$row"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && continue
  env="${PREFIX}${name}"
  reqfile="$REQ/$req"
  [ -f "$reqfile" ] || { echo "✗ requirements file not found: $reqfile"; exit 1; }
  cu="$([ "$idx" = "$CU130" ] && echo cu130 || echo cu129)"
  echo "──────── $env  ($req · $cu) ────────"

  if conda env list | awk '{print $1}' | grep -qx "$env"; then
    if [ "$FORCE" = 1 ]; then
      echo "  removing the existing env (FORCE=1)"; run conda env remove -y -n "$env"
    else
      echo "  ⚠ already exists → skipping creation (FORCE=1 recreates it). Updating pip only:"
      run conda run -n "$env" pip install -r "$reqfile" --extra-index-url "$idx"
      post_install "$name" "$env"
      echo; continue
    fi
  fi

  run conda create -y -n "$env" "python=$PYVER"
  run conda run -n "$env" python -m pip install --upgrade pip
  run conda run -n "$env" pip install -r "$reqfile" --extra-index-url "$idx"
  post_install "$name" "$env"
  [ "$DRY" = 1 ] || echo "  ✓ $env  →  $CONDA_BASE/envs/$env/bin/python"
  echo
done

cat <<EOF
[setup] done. Interpreters the pipeline uses (see the README.md mapping):
  ${PREFIX}train     $CONDA_BASE/envs/${PREFIX}train/bin/python
      encode_metaclip2 · encode_mc2h378 · encode_anchor_{tcap,filip} · encode_qwen3vl_embed · training
  ${PREFIX}beit3     $CONDA_BASE/envs/${PREFIX}beit3/bin/python
      encode_beit3 · encode_metaclip · encode_eva02 · encode_gallery_emb(dfn·convnext)
  ${PREFIX}gme       $CONDA_BASE/envs/${PREFIX}gme/bin/python        encode_gme
  ${PREFIX}llm2clip  $CONDA_BASE/envs/${PREFIX}llm2clip/bin/python   encode_llm2clip_anchor5
  ${PREFIX}vllm      $CONDA_BASE/envs/${PREFIX}vllm/bin/python       score_union_* · dump_fuse_cache
EOF
