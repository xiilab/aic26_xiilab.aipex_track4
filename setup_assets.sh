#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_assets.sh — create the empty assets/ and results/ tree
#
# git tracks files, not directories, so a clone carries assets/.gitkeep and nothing else. This
# script lays out the full skeleton of ARTIFACTS.md §Layout, so the Drive bundle can be unpacked
# or symlinked directly into place, tree by tree.
#
# It only ever calls `mkdir -p` — existing directories and files are left untouched, and running
# it twice changes nothing. No conda, no python, no network.
#
# Usage:
#   bash setup_assets.sh              # create the tree
#   bash setup_assets.sh --check      # report what is filled and what is empty, create nothing
#   bash setup_assets.sh --dry-run    # print the mkdir commands only
#   ROOT=/mnt/track4 bash setup_assets.sh     # build the tree somewhere else
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

DRY=0; CHECK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --check)   CHECK=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# Leaf directories — mkdir -p creates every parent, so only the deepest path is listed.
# `runs/` is training output rather than a distributed tree, but the trainers expect it to exist.
DIRS=(
  assets/cache/s1_base/members
  assets/cache/s2_rerank/fuse_cache/internvl_r32
  assets/cache/s2_rerank/fuse_cache/llama32v
  assets/cache/s2_rerank/fuse_cache/pixtral
  assets/cache/s4_nn
  assets/cache/s4_tail
  assets/cache_rep
  assets/data/raw/pab_test
  assets/data/raw/pab_train
  assets/data/raw/recaption
  assets/data/raw/ucc
  assets/data/raw/uca
  assets/data/raw/rstp
  assets/data/manifest
  assets/data/mining
  assets/data/benches/ruleclean
  assets/data/benches/ucc
  assets/data/benches/uca
  assets/data/benches/rstp
  assets/data/benches/rerankstep
  assets/data/heldout_v1
  assets/model/encoder
  assets/model/reranker
  assets/model/hf_cache
  assets/model/vlm_models
  assets/model_rep
  assets/runs
  results/reproduced
)

# The trees --check reports on: path | needed for | what fills it.
# Only the first two columns are padded, and both are ASCII — bash printf pads by byte, so a
# multi-byte `·` in a padded field would shift the row.
CHECKS=(
  "assets/data/raw/pab_test:reproduction:Track 4 test set — gallery · query_index.txt · query_text.json"
  "assets/cache/s1_base:reproduction:base_score.pt · union_pool.pt (+ members/ to rebuild them)"
  "assets/cache/s2_rerank:reproduction:7 union caches · recs_*.pt · fuse_cache/"
  "assets/cache/s4_nn:reproduction:6 tail-NN embeddings"
  "assets/cache/s4_tail:reproduction:3 nntail caches"
  "assets/model/encoder:re-scoring:adopted encoder checkpoints"
  "assets/model/reranker:re-scoring:adopted reranker adapters"
  "assets/model/hf_cache:training:third-party bases, read through HF_HOME"
  "assets/model/vlm_models:re-scoring:zero-shot bases — fetched yourself, not in the bundle"
  "assets/data/raw/pab_train:training:PAB train split"
  "assets/data/raw/recaption:training:12-style recaption CSVs"
  "assets/data/manifest:training:caption manifests · BEiT3 pair index"
  "assets/data/mining:training:negative caches · preference pairs"
  "assets/data/benches:evaluation:ruleclean · ucc · uca · rstp · rerankstep"
  "assets/data/heldout_v1:selection:heldout_images.txt · split.json"
)

# A tree counts as filled once it holds a real file at any depth — the subdirectories this script
# creates (members/ · fuse_cache/ · benches/*) must not read as content. A staged tree is usually a
# symlink to the read-only source, which find would not descend into, so that short-circuits first.
filled() {
  [ -e "$1" ] || return 1
  [ -L "$1" ] && return 0
  [ -n "$(find "$1" -mindepth 1 \( -type f -o -type l \) -print -quit 2>/dev/null)" ]
}

if [ "$CHECK" = 1 ]; then
  echo "[assets] $ROOT"
  echo
  missing=0
  for row in "${CHECKS[@]}"; do
    IFS=":" read -r path need what <<< "$row"
    if filled "$ROOT/$path"; then
      mark="✓"
    elif [ -d "$ROOT/$path" ]; then
      mark="·"; missing=$((missing + 1))
    else
      mark="✗"; missing=$((missing + 1))
    fi
    printf '  %s %-26s %-13s %s\n' "$mark" "$path" "$need" "$what"
  done
  echo
  echo "  ✓ filled   · empty (directory exists)   ✗ not created yet"
  if [ "$missing" -gt 0 ]; then
    echo "  $missing tree(s) still to fill — see ARTIFACTS.md for the bundle and its checksums."
  else
    echo "  every tree is populated."
  fi
  exit 0
fi

echo "[assets] creating the tree under $ROOT$([ "$DRY" = 1 ] && echo '  (DRY-RUN)')"
for d in "${DIRS[@]}"; do
  if [ -d "$ROOT/$d" ]; then
    continue
  elif [ "$DRY" = 1 ]; then
    echo "  \$ mkdir -p $ROOT/$d"
  else
    mkdir -p "$ROOT/$d"
    echo "  + $d"
  fi
done
echo
echo "Nothing was overwritten. Unpack or symlink the Drive bundle into these trees, then:"
echo "  bash setup_assets.sh --check"
