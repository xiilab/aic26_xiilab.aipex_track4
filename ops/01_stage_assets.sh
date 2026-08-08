#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 01_stage_assets.sh — populate the destination tree with the encoding inputs, then check them
#
# In a fresh destination tree assets/{data,model,cache} hold only .gitkeep. Encoding needs these
# three:
#   assets/data/raw/pab_test/{gallery,query_index.txt,query_text.json}   evaluation input
#   assets/model/encoder/*, assets/model/hf_cache/                       zero-shot backbones
#   assets/data/heldout_v1/                                             selection bench (02)
#
# They are large, so the default is a **symlink** (the inputs are read-only, so this is safe):
#   assets/model 313G · assets/data 27G  -> 340G if copied for real.
# Set MODE=copy for real copies; --only narrows it to a single item.
#
# Usage:
#   bash ops/01_stage_assets.sh --check          # check only, create nothing
#   bash ops/01_stage_assets.sh                  # stage with symlinks
#   MODE=copy bash ops/01_stage_assets.sh --only heldout   # copy just the small ones
#   bash ops/01_stage_assets.sh --dry-run
# ─────────────────────────────────────────────────────────────────────────────
# ── Tolerating edits while running ─────────────────────────────────────────
# bash reads a script by byte offset, so editing the original during a job that runs for tens of
# minutes makes it resume mid-word and either run something else or die quietly. The script
# therefore re-executes itself from a snapshot taken at startup. The copy lives in /tmp, so the
# original directory is passed as _OPS_DIR for env.sh to be found.
if [ -z "${_OPS_SNAPSHOT:-}" ]; then
  _snap="$(mktemp)"; cat "${BASH_SOURCE[0]}" > "$_snap"
  _OPS_SNAPSHOT=1 _OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" \
    bash "$_snap" "$@"; _rc=$?
  rm -f "$_snap"; exit "$_rc"
fi

source "${_OPS_DIR:-$(dirname "${BASH_SOURCE[0]}")}/env.sh"

MODE="${MODE:-symlink}"      # symlink | copy
CHECK_ONLY=0; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check)   CHECK_ONLY=1 ;;
    --copy)    MODE=copy ;;
    --only)    ONLY="$2"; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

# Item: name | source (relative to SRC_REPO) | destination
#
# Two paths are deliberately not handled here:
#   assets/cache  — the repository ships 22 real files (union_pool.pt · *_union_cache.pt ·
#                   fuse_cache/*.npy); replacing it wholesale would delete those tracked files.
#   assets/model  — its four children (encoder·hf_cache·reranker·vlm_models) are staged
#                   individually; symlinking the parent would drop the tracked .gitkeep files.
# Adopted weights plus the selection and encoding inputs only; caches are produced, not staged.
# Training data (recaption·manifest·mining) is excluded — stage it with --only when needed.
ITEMS=(
  "test:assets/data/raw/pab_test:assets/data/raw/pab_test"
  "heldout:assets/data/heldout_v1:assets/data/heldout_v1"
  "benches:assets/data/benches:assets/data/benches"
  "encoder:assets/model/encoder:assets/model/encoder"
  "reranker:assets/model/reranker:assets/model/reranker"
  "hf_cache:assets/model/hf_cache:assets/model/hf_cache"
  "vlm_models:assets/model/vlm_models:assets/model/vlm_models"
)
# Training data (excluded by default — stage individually with e.g. `--only recap`)
EXTRA_ITEMS=(
  "recap:assets/data/raw/recaption:assets/data/raw/recaption"
  "manifest:assets/data/manifest:assets/data/manifest"
  "mining:assets/data/mining:assets/data/mining"
)
[ -n "${ONLY:-}" ] && ITEMS+=("${EXTRA_ITEMS[@]}")

say "[01] staging assets — MODE=$MODE · source $SRC_REPO"
[ -d "$SRC_REPO" ] || die "source tree not found: $SRC_REPO  (set SRC_REPO)"
df -k /DATA | tail -1 | awk '{printf "  free disk: %.1f GiB\n", $4/1048576}'
echo

stage() {
  local name="$1" src="$SRC_REPO/$2" dst="$REPO/$3"
  [ -n "$ONLY" ] && [ "$ONLY" != "$name" ] && return 0
  printf '  %-12s' "$name"

  if [ ! -e "$src" ]; then printf '\033[33m⚠ source missing\033[0m %s\n' "$src"; return 0; fi

  # Leave anything already populated alone. The test looks for a **regular file**, so the empty
  # subdirectories a git clone creates (encoder/ · hf_cache/ …) are not mistaken for content.
  # find|head would trip pipefail through SIGPIPE, so -quit ends it instead.
  local first; first=$(find "$dst" -type f ! -name '.gitkeep' -print -quit 2>/dev/null || true)
  if [ -n "$first" ] || [ -L "$dst" ]; then
    printf '\033[32m✓ already present\033[0m (%s)\n' "$([ -L "$dst" ] && echo symlink || echo files)"
    return 0
  fi

  if [ "$CHECK_ONLY" = 1 ]; then printf '\033[33mempty (needs staging)\033[0m\n'; return 0; fi
  local sz; sz=$(du -sh "$src" 2>/dev/null | cut -f1 || true)   # measured only for a real staging run
  printf '%s -> ' "$sz"

  if [ "$MODE" = copy ]; then
    printf 'copy\n'
    run mkdir -p "$(dirname "$dst")"
    run rm -rf "$dst"
    run cp -a "$src" "$dst"          # symlinks inside the source are preserved; use cp -rL for real data
  else
    printf 'symlink\n'
    run rm -rf "$dst"
    run ln -s "$src" "$dst"
  fi
}

for row in "${ITEMS[@]}"; do
  IFS=":" read -r n s d <<< "$row"
  stage "$n" "$s" "$d"
done

# ── assets/cache status (not staged — it is repository content) ────────────
echo
say "[01] assets/cache — shipped with the repository (never replaced)"
for f in s1_base/union_pool.pt s1_base/base_score.pt s2_rerank/8b_union_cache.pt s4_nn/gme_feats.pt; do
  if [ -e "assets/cache/$f" ]; then ok "$f"
  else printf '  \033[33m·\033[0m %s — missing (build it by encoding, or copy it in individually)\n' "$f"; fi
done

# ── Required encoding inputs ───────────────────────────────────────────────
echo
say "[01] checking the required inputs"
miss=0
# Existence alone is not enough for a directory — one holding only .gitkeep counts as missing.
chk() {
  local p="$1" label="$2" have=0
  if [ -f "$p" ]; then have=1
  elif [ -d "$p" ] && [ -n "$(find -L "$p" -type f ! -name '.gitkeep' -print -quit 2>/dev/null || true)" ]; then have=1
  fi
  if [ "$have" = 1 ]; then ok "$label"
  else printf '  \033[31m✗\033[0m %s — %s\n' "$label" "$p"; miss=$((miss+1)); fi
}
chk assets/data/raw/pab_test/gallery          "evaluation gallery (36,773 images)"
chk assets/data/raw/pab_test/query_index.txt  "query order (row convention)"
chk assets/data/raw/pab_test/query_text.json  "query captions (1,978)"
chk assets/model/hf_cache                     "HF cache (zero-shot backbones)"
chk assets/model/encoder                      "adopted encoder weights"
chk assets/data/heldout_v1                    "held-out bench (for selection)"
echo
if [ "$miss" -gt 0 ]; then
  warn "$miss missing — populate the paths above or point at them with:"
  echo "      PAB_TEST · GALLERY · QUERY_INDEX · QUERY_TEXT · HF_CACHE · HELDOUT_DIR"
  exit 1
fi
ok "staging complete -> next: bash ops/02_select.sh --list"
