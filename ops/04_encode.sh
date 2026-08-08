#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 04_encode.sh — encode the S1 members with the deployed weights -> assets/cache_rep/
#
# base = the flat 10 members. With --rep the weights come from model_rep and the output goes to
# cache_rep (the deployment and cache roots must match for REP=1 run_reproduce.sh to be coherent).
#
#   member               artifact                        env
#   anchor_tcap          anchor_tcap_tta_views.pt        track4_train
#   anchor_filip         anchor_filip_tta_views.pt       track4_train
#   metaclip2            metaclip2_feats.pt              track4_train
#   mc2h378_peft         mc2h378_peft_score.pt           track4_train
#   beit3_v2             beit3_v2_score.pt               track4_beit3
#   beit3_helip          beit3_helip_score.pt            track4_beit3
#   eva02_pre            eva02_pre_score.pt              track4_beit3   (zero-shot)
#   metaclip_v1          metaclip_v1_score.pt            track4_beit3
#   gme                  gme_feats.pt                    track4_gme     (zero-shot)
#   siglip_maxsim        siglip_maxsim_score.pt          track4_llm2clip
#
#   The six tail-NN encoders (used by S4b/S4c) are built separately with --tail.
#
# Envs are not interchangeable — the transformers pins differ (4.30.2 / 4.51.3 / 4.56.2 / 5.4.0).
# encode_anchor_tcap · encode_eva02 · encode_metaclip have no --gpu flag and take the GPU
#    from CUDA_VISIBLE_DEVICES.
# --limit cannot be combined with --rep (smoke runs only).
#
# Usage:
#   bash ops/04_encode.sh --list                 # members, commands, artifact state
#   bash ops/04_encode.sh --all                  # all ten members (slow)
#   bash ops/04_encode.sh metaclip2 beit3_v2
#   bash ops/04_encode.sh --smoke metaclip2      # quick check with --limit 64 (not a rep run)
#   bash ops/04_encode.sh --tail                 # the six s4_nn encoders
#   bash ops/04_encode.sh --build                # base + union (union_pool only — from scratch)
#   bash ops/04_encode.sh --build --merge        # merge existing *_union_cache.pt (S2 can be skipped)
#   GPU=7 bash ops/04_encode.sh --all
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

REP="${REP:-0}"           # 0 = adopted path (model->cache) · 1 = --rep (model_rep->cache_rep)
                          #   this repository reproduces the adopted run, so 0 is the default
LIST=0; ALL=0; TAIL=0; BUILD=0; SMOKE=0; MERGE=0; TARGETS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --list)    LIST=1 ;;
    --all)     ALL=1 ;;
    --tail)    TAIL=1 ;;
    --build)   BUILD=1 ;;
    --merge)   MERGE=1 ;;    # build_union merge mode (needs the existing caches)
    --smoke)   SMOKE=1; REP=0 ;;
    --adopted) REP=0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) die "unknown argument: $1" ;;
    *)  TARGETS+=("$1") ;;
  esac
  shift
done

RF=(); [ "$REP" = 1 ] && RF=(--rep)
LF=(); [ "$SMOKE" = 1 ] && LF=(--limit 64)
CACHE="assets/cache"; [ "$REP" = 1 ] && CACHE="assets/cache_rep"
MEM="$CACHE/s1_base/members"

# A smoke run isolates its artifacts. --limit cannot be used with --rep, so REP becomes 0 and the
# default output would be the **adopted cache** assets/cache/s1_base/members. A truncated 64-image
# file left there would later be picked up by build_base silently, so the path is redirected.
if [ "$SMOKE" = 1 ]; then
  MEM="ops/smoke/members"
  export S1_MEMBERS="$REPO/$MEM"
  export S4_NN="$REPO/ops/smoke/s4_nn"
  mkdir -p "$S1_MEMBERS" "$S4_NN"
fi

# Only members that support --limit are a real smoke test; the rest encode everything (hours).
LIMIT_OK=" metaclip2 mc2h378_peft beit3_v2 beit3_helip siglip_maxsim "

MODELS=(anchor_tcap anchor_filip metaclip2 mc2h378_peft beit3_v2 beit3_helip
        eva02_pre metaclip_v1 gme siglip_maxsim)
declare -A OUT_OF=(
  [anchor_tcap]=anchor_tcap_tta_views.pt   [anchor_filip]=anchor_filip_tta_views.pt
  [metaclip2]=metaclip2_feats.pt           [mc2h378_peft]=mc2h378_peft_score.pt
  [beit3_v2]=beit3_v2_score.pt             [beit3_helip]=beit3_helip_score.pt
  [eva02_pre]=eva02_pre_score.pt           [metaclip_v1]=metaclip_v1_score.pt
  [gme]=gme_feats.pt                       [siglip_maxsim]=siglip_maxsim_score.pt
)
declare -A ENV_OF=(
  [anchor_tcap]=train  [anchor_filip]=train  [metaclip2]=train  [mc2h378_peft]=train
  [beit3_v2]=beit3     [beit3_helip]=beit3   [eva02_pre]=beit3  [metaclip_v1]=beit3
  [gme]=gme            [siglip_maxsim]=llm2clip
)

# ── Per-member encoding ────────────────────────────────────────────────────
enc_anchor_tcap()  { need_py "$PY_TRAIN" train
  run env CUDA_VISIBLE_DEVICES="$GPU" "$PY_TRAIN" $E/encode_anchor_tcap.py "${RF[@]+"${RF[@]}"}"; }
enc_anchor_filip() { need_py "$PY_TRAIN" train
  run "$PY_TRAIN" $E/encode_anchor_filip.py --gpu "$GPU" "${RF[@]+"${RF[@]}"}"; }
enc_metaclip2()    { need_py "$PY_TRAIN" train
  run "$PY_TRAIN" $E/encode_metaclip2.py --gpu "$GPU" "${RF[@]+"${RF[@]}"}" "${LF[@]+"${LF[@]}"}"; }
enc_mc2h378_peft() { need_py "$PY_TRAIN" train
  run "$PY_TRAIN" $E/encode_mc2h378.py --gpu "$GPU" "${RF[@]+"${RF[@]}"}" "${LF[@]+"${LF[@]}"}"; }
enc_beit3_v2()     { need_py "$PY_BEIT3" beit3
  run "$PY_BEIT3" $E/encode_beit3.py --recipe v2 --gpu "$GPU" "${RF[@]+"${RF[@]}"}" "${LF[@]+"${LF[@]}"}"; }
enc_beit3_helip()  { need_py "$PY_BEIT3" beit3
  run "$PY_BEIT3" $E/encode_beit3.py --recipe helip --gpu "$GPU" "${RF[@]+"${RF[@]}"}" "${LF[@]+"${LF[@]}"}"; }
enc_eva02_pre()    { need_py "$PY_BEIT3" beit3      # zero-shot — no --checkpoint
  run env CUDA_VISIBLE_DEVICES="$GPU" "$PY_BEIT3" $E/encode_eva02.py "${RF[@]+"${RF[@]}"}"; }
enc_metaclip_v1()  { need_py "$PY_BEIT3" beit3
  run env CUDA_VISIBLE_DEVICES="$GPU" "$PY_BEIT3" $E/encode_metaclip.py "${RF[@]+"${RF[@]}"}"; }
enc_gme()          { need_py "$PY_GME" gme         # zero-shot
  run "$PY_GME" $E/encode_gme.py --gpu "$GPU" "${RF[@]+"${RF[@]}"}"; }
enc_siglip_maxsim(){ need_py "$PY_LLM2CLIP" llm2clip
  run "$PY_LLM2CLIP" $E/encode_siglip_maxsim.py --gpu "$GPU" "${RF[@]+"${RF[@]}"}" "${LF[@]+"${LF[@]}"}"; }

# ── Listing ────────────────────────────────────────────────────────────────
if [ "$LIST" = 1 ]; then
  say "[04] 10 S1 members · REP=$REP -> $MEM"
  printf '  %-15s %-30s %-9s %s\n' "member" "artifact" "env" "state"
  printf '  %-15s %-30s %-9s %s\n' "--------------" "-----------------------------" "--------" "-----"
  for m in "${MODELS[@]}"; do
    f="$MEM/${OUT_OF[$m]}"
    if [ -e "$f" ]; then s="✓ $(du -h "$f" 2>/dev/null | cut -f1)"; else s="·"; fi
    printf '  %-15s %-30s %-9s %s\n' "$m" "${OUT_OF[$m]}" "${ENV_OF[$m]}" "$s"
  done
  echo; echo "  print the commands only: DRY=1 bash ops/04_encode.sh --all"
  exit 0
fi

# ── The six tail-NN encoders ───────────────────────────────────────────────
if [ "$TAIL" = 1 ]; then
  say "[04] tail-NN embeddings -> $CACHE/s4_nn  (S4b·S4c match on raw embeddings, not scores)"
  need_py "$PY_TRAIN" train; need_py "$PY_BEIT3" beit3
  need_py "$PY_GME" gme;     need_py "$PY_LLM2CLIP" llm2clip
  run "$PY_GME"      $E/encode_gme.py             --gpu "$GPU" "${RF[@]+"${RF[@]}"}"
  run "$PY_TRAIN"    $E/encode_metaclip2.py             --gpu "$GPU" "${RF[@]+"${RF[@]}"}"
  run "$PY_TRAIN"    $E/encode_qwen3vl_embed.py   --gpu "$GPU" "${RF[@]+"${RF[@]}"}"
  run "$PY_LLM2CLIP" $E/encode_llm2clip_anchor5.py --gpu "$GPU" "${RF[@]+"${RF[@]}"}"
  run "$PY_BEIT3"    $E/encode_gallery_emb.py --enc dfn      --gpu "$GPU" "${RF[@]+"${RF[@]}"}"
  run "$PY_BEIT3"    $E/encode_gallery_emb.py --enc convnext --gpu "$GPU" "${RF[@]+"${RF[@]}"}"
  echo
  warn "in adopted mode (--adopted) gme and metaclip2 write no s4_nn copy — copy it across yourself."
  ok "tail-NN done"
  exit 0
fi

# ── base + union ───────────────────────────────────────────────────────────
if [ "$BUILD" = 1 ]; then
  say "[04] building base + the candidate union"
  # No model is loaded. ENS_DEV only says where the tensor math runs — on CPU it takes 51+ min.
  run env ENS_DEV="cuda:$GPU" "$PY_ENS" pipeline/S1_base/build_base.py  "${RF[@]+"${RF[@]}"}"

  # build_union **merges** by default, so it requires the existing *_union_cache.pt as input.
  # From scratch those do not exist, so --no-merge builds union_pool only and S2 (05) scores it.
  # Use --merge to reuse existing caches instead.
  if [ "$MERGE" = 1 ]; then
    warn "merge mode — the existing *_union_cache.pt (REDUMP_SRC·REDUMP_SRC2) must be present"
    run "$PY_ENS" pipeline/S1_base/build_union.py "${RF[@]+"${RF[@]}"}"
  else
    run "$PY_ENS" pipeline/S1_base/build_union.py --no-merge "${RF[@]+"${RF[@]}"}"
  fi
  echo
  ok "-> $CACHE/s1_base/{base_score.pt,union_pool.pt}"
  if [ "$MERGE" = 1 ]; then
    echo "  -> $CACHE/s2_rerank/*_union_cache.pt (merged) · next: bash ops/05_rerank.sh --fuse"
  else
    echo "  next: bash ops/05_rerank.sh --all      # the seven S2 rerankers (track4_vllm · GPU)"
  fi
  echo "  final reproduction:  REP=1 bash run_reproduce.sh best"
  exit 0
fi

# ── Run ────────────────────────────────────────────────────────────────────
[ "$ALL" = 1 ] && TARGETS=("${MODELS[@]}")
[ "${#TARGETS[@]}" -gt 0 ] || die "name a target. List them with: bash ops/04_encode.sh --list"
[ "$SMOKE" = 1 ] && warn "smoke mode — --limit 64, REP disabled. Do not use the artifacts as canonical."

need_path assets/data/raw/pab_test/gallery "bash ops/01_stage_assets.sh"
if [ "$REP" = 1 ]; then
  need_path assets/model_rep/encoder "bash ops/03_deploy.sh --adopted"
fi

FAILED=""
for m in "${TARGETS[@]}"; do
  case " ${MODELS[*]} " in *" $m "*) ;; *) die "unknown member: $m  (list them with --list)";; esac
  if [ "$SMOKE" = 1 ] && [[ "$LIMIT_OK" != *" $m "* ]]; then
    warn "$m does not support --limit -> skipped in a smoke run (it would encode everything)"
    continue
  fi
  echo; say "[04] $m  (env=track4_${ENV_OF[$m]} · GPU=$GPU$([ "$SMOKE" = 1 ] && printf " · smoke"))"
  # One failing member does not stop the rest. Under set -e the whole run used to die, so the
  # later members were never attempted (a missing siglip_maxsim blocked both beit3 members).
  if ! "enc_$m"; then
    warn "$m failed — continuing"
    FAILED="$FAILED $m"
  fi
done

echo
say "[04] artifacts"
for m in "${TARGETS[@]}"; do
  f="$MEM/${OUT_OF[$m]}"
  [ -e "$f" ] && ok "$m -> ${OUT_OF[$m]} ($(du -h "$f" | cut -f1))" || warn "$m -> no artifact ($f)"
done
if [ -n "$FAILED" ]; then
  echo; warn "failed members:$FAILED"
fi
echo
echo "  next: bash ops/04_encode.sh --tail  then  bash ops/04_encode.sh --build"
