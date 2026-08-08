set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANT="${1:-best}"
COMB_W_ENV="${COMB_W:-}"        # the script fills COMB_W itself, so remember any externally supplied value
# Interpreter: whatever has requirements/core.txt installed. PY=<path> overrides; otherwise take the
# first of the active env, python3, python. The replay imports only torch and numpy.
if [ -z "${PY:-}" ]; then
  for _c in "${CONDA_PREFIX:-}/bin/python" python3 python; do
    [ -n "$_c" ] && command -v "$_c" >/dev/null 2>&1 && { PY="$_c"; break; }
  done
fi
[ -n "${PY:-}" ] || { echo "✗ no python found. Install requirements/core.txt and set PY=<interpreter>."; exit 1; }
command -v "$PY" >/dev/null 2>&1 || { echo "✗ PY=$PY is not executable."; exit 1; }
"$PY" -c "import torch, numpy" 2>/dev/null || {
  echo "✗ $PY cannot import torch/numpy."
  echo "  pip install -r requirements/core.txt --extra-index-url https://download.pytorch.org/whl/cu129"
  echo "  or point PY= at an interpreter that already has them."; exit 1; }
G="${G:-0}"
WORK="${WORK:-$HERE/.work}"
# cache root: assets/cache by default, assets/cache_rep when REP=1
CACHE="$HERE/assets/cache"; [ "${REP:-0}" = "1" ] && CACHE="$HERE/assets/cache_rep"
PAB_TEST="${PAB_TEST:-$HERE/assets/data/raw/pab_test}"
GALLERY="${GALLERY:-$PAB_TEST/gallery}"
QUERY_INDEX="${QUERY_INDEX:-$PAB_TEST/query_index.txt}"
case "$VARIANT" in best) ;; *) echo "the only variant is best"; exit 2;; esac

# ── 0. Input check: stop immediately and name whatever is missing ─────────────
miss=0
chk() { [ -e "$1" ] || { echo "  ✗ missing: $1   ($2)"; miss=1; }; }
echo "[0/4] input check"
chk "$GALLERY"     "evaluation gallery — set GALLERY"
chk "$QUERY_INDEX" "evaluation query order — set QUERY_INDEX"
# distributed artifacts
[ "${REP:-0}" = "1" ] && echo "  REP=1 → cache root $CACHE (reproduction check mode)"
for f in s1_base/union_pool.pt s2_rerank/recs_8b_p3_k20.pt s2_rerank/recs_2b_dora_k5_p3.pt; do chk "$CACHE/$f" "distributed artifact"; done
for n in internvl_r32 qwen3vl_2b 8b pixtral llama ovis jina_m0; do chk "${REDUMP_DIR:-$CACHE/s2_rerank}/${n}_union_cache.pt" "distributed artifact"; done
for n in internvl_r32 jina_m0 llama; do chk "$CACHE/s4_tail/${n}_nntail_cache.pt" "distributed artifact"; done
BASE_PT="${BASE_PT:-$CACHE/s1_base/base_score.pt}"
# A different WEIGHTS set re-fuses S1 with those values; the distributed base_score.pt is built from the deployed weights
REBUILD_BASE=0
if [ -n "${WEIGHTS:-}" ] && [ -z "${BASE_PT_FIXED:-}" ]; then REBUILD_BASE=1; fi
chk "$BASE_PT" "S1 base score — build with pipeline/S1_base/build_base.py"
ENC_FILES="dfn_gallery_emb.pt convnext_gallery_emb.pt gme_feats.pt qwen3vl_embed8b_feats.pt anchor5_feats.pt metaclip2_feats.pt"
ENC_SRC="${ENC_SRC:-$CACHE/s4_nn}"                  # S4b tail-NN encoder embeddings
for f in $ENC_FILES; do chk "$ENC_SRC/$f" "encoder embeddings — build with pipeline/S1_base/encode/"; done
[ $miss -eq 0 ] || { echo; echo "→ inputs are missing."; exit 1; }
echo "  ✓ all present"

# ── 1. Load the fixed weights ─────────────────────────────────────────────
# Single source is tools/ensemble/adopted.py, defaulting to tools/ensemble/weights/final.json.
#   WEIGHTS=<json> substitutes a different weight set (see tools/ensemble/weights/)
WQ="$PY $HERE/tools/ensemble/adopted.py"
if [ -n "${COMB_W:-}" ]; then
  echo "[1/4] comb weights from the COMB_W environment variable (tools/ensemble ignored)"
else
  COMB_W=$($WQ --get comb --variant "$VARIANT")
fi
TAIL_W=$($WQ --get tail_w)
TAU_PX=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['tau_px'])" "$($WQ --get s4e)")
echo "[1/4] weights loaded (tools/ensemble · ${WEIGHTS:-deployed weights}) · variant=$VARIANT · comb=$COMB_W · tau_px=$TAU_PX"

# ── 2. Build the work directory: symlink only what is needed, no global mirror ─
echo "[2/4] building the work directory $WORK"
rm -rf "$WORK"; mkdir -p "$WORK/outputs/fuse_internvl"
if [ "$REBUILD_BASE" = 1 ]; then
  echo "  [S1] re-fusing base with the WEIGHTS set → $WORK/base_score.pt"
  ENS_DEV="cuda:$G" $PY "$HERE/pipeline/S1_base/build_base.py" --out "$WORK/base_score.pt" | sed 's/^/    /'
  BASE_PT="$WORK/base_score.pt"
fi
ln -s "$BASE_PT"                      "$WORK/greedy_R10_base_score.pt"
ln -s "$CACHE/s1_base/union_pool.pt"  "$WORK/union_pool.pt"
RD="${REDUMP_DIR:-$CACHE/s2_rerank}"
for n in internvl_r32 qwen3vl_2b 8b pixtral llama ovis jina_m0; do ln -s "$RD/${n}_union_cache.pt" "$WORK/${n}_union_cache.pt"; done
for n in internvl_r32 jina_m0 llama;                          do ln -s "$CACHE/s4_tail/${n}_nntail_cache.pt"   "$WORK/${n}_nntail_cache.pt"; done
for f in recs_8b_p3_k20.pt recs_2b_dora_k5_p3.pt; do ln -s "$CACHE/s2_rerank/$f" "$WORK/$f"; done
for d in internvl_r32 pixtral llama32v;              do ln -s "$CACHE/s2_rerank/fuse_cache/$d" "$WORK/outputs/fuse_internvl/$d"; done
for f in $ENC_FILES;                              do ln -s "$ENC_SRC/$f" "$WORK/$f"; done

# ── 3. Run the inference entrypoint ───────────────────────────────────────
echo "[3/4] running the pipeline → $WORK/run.log"
COMB_W="$COMB_W" TAIL_W="$TAIL_W" TAU_PX="$TAU_PX" \
TRACK4="$WORK" TRACK4_CODE="$HERE" \
PAB_TEST="$PAB_TEST" GALLERY="$GALLERY" TRACK4_GALLERY="$GALLERY" QUERY_INDEX="$QUERY_INDEX" \
BASE=r10 POST="${POST:-full}" IMPUTE="${IMPUTE:-zero}" FINAL_PASS="${FINAL_PASS:-on}" ENS_DEV="cuda:$G" \
OUT="$WORK/answer_final.txt" \
  $PY "$HERE/run_submission.py" > "$WORK/run.log" 2>&1 || { tail -30 "$WORK/run.log"; exit 1; }

# ── 4. Finalise the answer ────────────────────────────────────────────────
STAGE=""
case "${POST:-full}" in
  full) [ "${FINAL_PASS:-on}" = on ] || STAGE="_S4c" ;;   # without the second S4d pass
  none) STAGE="_S3" ;;                                     # no tail refinement
  *)    STAGE="_POST${POST}" ;;
esac
OUTFILE="$HERE/results/reproduced/answer_reproduced_${TAG:-$VARIANT}${STAGE}_noext.txt"
if [ "${POST:-full}" = "none" ]; then SRC="$WORK/answer_final.txt"; else
case "${FINAL_PASS:-on}" in
  off|none|0|false|no) SRC="$WORK/answer_tailoverlay_r5promote_top1prop_noext.txt" ;;   # stops after the first S4d pass
  *)                   SRC="$WORK/answer_final_noext.txt" ;;                             # through the second S4d pass
esac
fi
mkdir -p "$(dirname "$OUTFILE")"
cp "$SRC" "$OUTFILE"
MD5=$(md5sum "$OUTFILE" | cut -c1-8)
# The reference md5 only means anything for the deployed weights run through every stage.
# A run with different weights or an earlier stop has nothing to compare against, so it is skipped.
if [ -z "${TAG:-}${WEIGHTS:-}${COMB_W_ENV}" ] && [ "${POST:-full}" = full ]; then
  EXPECT=f6290321
  echo "[4/4] answer → results/reproduced/$(basename "$OUTFILE")  md5=$MD5  (expected $EXPECT)"
  [ "$MD5" = "$EXPECT" ] && echo "  ✓ reproduced the reference answer" \
                         || echo "  ⚠ md5 mismatch — check the versions of the input artifacts"
else
  echo "[4/4] answer → results/reproduced/$(basename "$OUTFILE")  md5=$MD5  (experimental run — not compared against the reference md5)"
  [ "${POST:-full}" = none ] && echo "  note: with the deployed weights, the S3-only answer has md5 98471257"
fi
echo "  intermediates: $WORK (S3 fusion → S4a overlay → S4b NN → S4c R@5 → S4d cons6 + final pass)"
echo "  metrics are in the README.md results table (nothing is scored at runtime)."
