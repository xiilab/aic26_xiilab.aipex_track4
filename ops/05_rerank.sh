#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 05_rerank.sh — the seven S2 rerankers re-score the union pool -> cache_rep/s2_rerank/
#                plus the three fuse_cache dumps S3 reads
#
# They run in **track4_vllm**, except ovis, which needs transformers 4.x and runs in
# **track4_llm2clip** (PY_OVIS). This is the most GPU-expensive stage of the pipeline, so
# splitting it per member (--slice) or running it in the background is usually better.
#
#   member        script                        model                             note
#   internvl_r32  score_union_hf_4b.py          InternVL3_5-30B-A3B-HF + LoRA     MoE · required
#   llama         score_union_hf_4b.py          Llama-3.2-11B-Vision-Instruct     zero-shot
#   8b            score_union_qwen_4b.py        Qwen3-VL-Reranker-8B              reuses recs
#   qwen3vl_2b    score_union_qwen_4b.py        Qwen3-VL 2B + DoRA                reuses recs
#   pixtral       score_union_pixtral_4b.py     Pixtral-12B-2409                  zero-shot
#   ovis          score_union_ovis.py           Ovis2.5-9B                        zero-shot · tf 4.x
#   jina_m0       score_union_jina.py           jina-reranker-m0 + adapter
#
#   The three fuse_cache dumps (read by IV_CACHE in S3 fuse.py — fixed top-20 column format):
#     internvl_r32 · pixtral · llama32v          dump_fuse_cache.py
#
# Prerequisite: bash ops/04_encode.sh --build   (union_pool.pt has to exist)
#
# Usage:
#   bash ops/05_rerank.sh --list                # members, models, artifact state
#   bash ops/05_rerank.sh --all                 # all seven in sequence (very slow)
#   bash ops/05_rerank.sh internvl_r32 jina_m0  # a selection
#   bash ops/05_rerank.sh 8b --slice 0:500      # query slice (saved as a shard)
#   bash ops/05_rerank.sh --merge 8b            # merge the shards
#   bash ops/05_rerank.sh --smoke jina_m0       # quick check on a few queries
#   bash ops/05_rerank.sh --recs                # assemble recs_*.pt (no GPU)
#   bash ops/05_rerank.sh --fuse                # the three fuse_cache dumps
#   GPU=7 bash ops/05_rerank.sh ovis
#   DRY=1 bash ops/05_rerank.sh --all           # commands only
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
LIST=0; ALL=0; FUSE=0; RECS=0; SMOKE=0; DOMERGE=0; SLICE=""; TARGETS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --list)    LIST=1 ;;
    --all)     ALL=1 ;;
    --fuse)    FUSE=1 ;;
    --recs)    RECS=1 ;;    # assemble recs_*.pt (build_recs.py · no GPU)
    --merge)   DOMERGE=1 ;;
    --slice)   SLICE="$2"; shift ;;
    --smoke)   SMOKE=1 ;;
    --adopted) REP=0 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,36p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) die "unknown argument: $1" ;;
    *)  TARGETS+=("$1") ;;
  esac
  shift
done

RF=(); [ "$REP" = 1 ] && RF=(--rep)
CACHE="assets/cache"; [ "$REP" = 1 ] && CACHE="assets/cache_rep"
RR="$CACHE/s2_rerank"
S2="pipeline/S2_rerank"
VLM="${VLM_MODELS:-assets/model/vlm_models}"
# Adapters come from the deployment (model_rep); with REP=0 they come from the adopted weights
MR="assets/model"; [ "$REP" = 1 ] && MR="assets/model_rep"

MEMBERS=(internvl_r32 llama 8b qwen3vl_2b pixtral ovis jina_m0)


PY_OVIS="${PY_OVIS:-$PY_LLM2CLIP}"

# OUT_DIR=<repo-relative dir> scores the full pool with the adopted weights but writes somewhere
# isolated, leaving both assets/cache and assets/cache_rep untouched. RECS_DIR follows it, so
# 8b/qwen3vl_2b re-score every pair instead of reusing the adopted recs dump.
if [ "$SMOKE" = 1 ] || [ -n "${OUT_DIR:-}" ]; then
  REP=0; RF=()
  CACHE="assets/cache"
  MR="assets/model"
  RR="${OUT_DIR:-ops/smoke/s2_rerank}"
  [ "$SMOKE" = 1 ] || RECS_DIR="${RECS_DIR:-$REPO/$RR}"
  mkdir -p "$REPO/$RR"
  # A copy, not a symlink (1 MB): the adopted cache gets rebuilt by other runs, and a dangling link
  # would kill an isolated job that has no reason to care.
  [ -e "$REPO/$RR/union_pool.pt" ] || cp "$REPO/assets/cache/s1_base/union_pool.pt" "$REPO/$RR/union_pool.pt"
  POOLF="union_pool.pt"
else
  POOLF="../s1_base/union_pool.pt"
fi

# recs live next to the union caches of the mode being run, so this follows $CACHE: a --rep run
# must reuse the recs scored with the model_rep adapters, not the adopted ones. Defined after the
# smoke block so that an isolated run picks up its redirected cache root.
RECS_DIR="${RECS_DIR:-$REPO/$CACHE/s2_rerank}"

# Shared scorer env — output (WORKDIR), candidate pool (POOL_FILE), recs (TRACK4) and the artifact
# suffix. Without them a REP=0 run leaks into $TRACK4/rerank_work (= assets/cache/work/…).
SENV=(env "TRACK4=$RECS_DIR" "WORKDIR=$REPO/$RR" "POOL_FILE=$POOLF" "OUT_SUFFIX=union_cache")
QENV=("${SENV[@]}")

# Slice arguments (0:500 -> --q-start 0 --q-end 500). Supported by the hf/qwen scorers only.
QS=(); if [ -n "$SLICE" ]; then
  QS=(--q-start "${SLICE%%:*}" --q-end "${SLICE##*:}")
fi
# Smoke: pixtral·ovis·jina truncate with --limit, hf·qwen with --q-end
LIM=(); [ "$SMOKE" = 1 ] && LIM=(--limit 8)
QSMOKE=(); [ "$SMOKE" = 1 ] && [ -z "$SLICE" ] && QSMOKE=(--q-end 8)

# ── Per-member re-scoring ──────────────────────────────────────────────────
rr_internvl_r32() {
  run "${SENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY_VLLM" $S2/score_union_hf_4b.py \
    --model "$VLM/InternVL3_5-30B-A3B-HF" --adapter "$MR/reranker/internvl_r32" \
    --name internvl_r32 "${RF[@]+"${RF[@]}"}" "${QS[@]+"${QS[@]}"}" "${QSMOKE[@]+"${QSMOKE[@]}"}"
}
rr_llama() {   # zero-shot — no adapter
  run "${SENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY_VLLM" $S2/score_union_hf_4b.py \
    --model "$VLM/Llama-3.2-11B-Vision-Instruct" \
    --name llama "${RF[@]+"${RF[@]}"}" "${QS[@]+"${QS[@]}"}" "${QSMOKE[@]+"${QSMOKE[@]}"}"
}
# --reuse-recs is an optimisation, not a requirement: it lets the scorer skip (q,c) pairs that are
# already in the recs dump. The dump is produced by `05 --recs` from the union caches, so on a
# from-scratch run it does not exist yet — requiring it would deadlock (recs needs the union cache,
# the union cache would need recs). It is therefore passed only when the file is there.
RU=()                      # filled per member by want_recs
want_recs() {
  RU=()
  if [ -e "$RECS_DIR/$1" ]; then
    RU=(--reuse-recs "$1")
  else
    warn "$RECS_DIR/$1 not found -> scoring the full pool (build it later: bash ops/05_rerank.sh --recs)"
  fi
}
rr_8b() {      # reuses the recs dump when present -> only new (q,c) pairs reach the model
  want_recs recs_8b_p3_k20.pt
  run "${QENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY_VLLM" $S2/score_union_qwen_4b.py \
    --qwen 8b --name 8b "${RU[@]+"${RU[@]}"}" \
    "${RF[@]+"${RF[@]}"}" "${QS[@]+"${QS[@]}"}" "${QSMOKE[@]+"${QSMOKE[@]}"}"
}
# Do not reduce this member's pool to the base top-5 that build_recs consumes: S4a reads the union
# cache directly (tail_refinement.py `cache`/`sig()`) and only falls back to the recs dump when a
# pair is missing, so truncating it costs mAP@10 -0.032 / R@1 -1 / R@10 -1 (measured).
rr_qwen3vl_2b() {
  want_recs recs_2b_dora_k5_p3.pt
  run "${QENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY_VLLM" $S2/score_union_qwen_4b.py \
    --qwen 2b --adapter "$MR/reranker/qwen3vl_2b" --name qwen3vl_2b \
    "${RU[@]+"${RU[@]}"}" \
    "${RF[@]+"${RF[@]}"}" "${QS[@]+"${QS[@]}"}" "${QSMOKE[@]+"${QSMOKE[@]}"}"
}
rr_pixtral() {
  run "${SENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY_VLLM" $S2/score_union_pixtral_4b.py \
    --name pixtral "${RF[@]+"${RF[@]}"}" "${LIM[@]+"${LIM[@]}"}"
}
rr_ovis() {   # track4_llm2clip (transformers 4.x), not track4_vllm — see PY_OVIS above
  need_py "$PY_OVIS" llm2clip
  run "${SENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY_OVIS" $S2/score_union_ovis.py \
    --model "$VLM/Ovis2.5-9B" --name ovis "${RF[@]+"${RF[@]}"}" "${LIM[@]+"${LIM[@]}"}"
}
rr_jina_m0() {
  run "${SENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY_VLLM" $S2/score_union_jina.py \
    --name jina_m0 --ckpt "$MR/reranker/jina_m0" "${RF[@]+"${RF[@]}"}" "${LIM[@]+"${LIM[@]}"}"
}

# ── Listing ────────────────────────────────────────────────────────────────
if [ "$LIST" = 1 ]; then
  say "[05] 7 S2 rerankers · REP=$REP -> $RR   (env=track4_vllm, ovis=track4_llm2clip · GPU=$GPU)"
  printf '  %-14s %-26s %-34s %s\n' "member" "script" "model" "state"
  printf '  %-14s %-26s %-34s %s\n' "-------------" "-------------------------" "---------------------------------" "-----"
  declare -A SC=( [internvl_r32]=score_union_hf_4b [llama]=score_union_hf_4b
                  [8b]=score_union_qwen_4b [qwen3vl_2b]=score_union_qwen_4b
                  [pixtral]=score_union_pixtral_4b [ovis]=score_union_ovis [jina_m0]=score_union_jina )
  declare -A MD=( [internvl_r32]=InternVL3_5-30B-A3B-HF+LoRA [llama]=Llama-3.2-11B-Vision-Instruct
                  [8b]=Qwen3-VL-Reranker-8B [qwen3vl_2b]="Qwen3-VL-2B+DoRA"
                  [pixtral]=Pixtral-12B-2409 [ovis]=Ovis2.5-9B [jina_m0]=jina-reranker-m0 )
  for m in "${MEMBERS[@]}"; do
    f="$RR/${m}_union_cache.pt"
    if [ -e "$f" ]; then s="✓ $(du -h "$f" 2>/dev/null | cut -f1)"; else s="·"; fi
    printf '  %-14s %-26s %-34s %s\n' "$m" "${SC[$m]}" "${MD[$m]}" "$s"
  done
  echo
  say "  fuse_cache (read by IV_CACHE in S3)"
  for d in internvl_r32 pixtral llama32v; do
    if [ -d "$RR/fuse_cache/$d" ]; then ok "$d"; else printf '  \033[33m·\033[0m %s\n' "$d"; fi
  done
  echo
  echo "  prerequisite: $([ -e "$CACHE/s1_base/union_pool.pt" ] && echo '✓ union_pool.pt' || echo '✗ union_pool.pt — bash ops/04_encode.sh --build')"
  echo "  print the commands only: DRY=1 bash ops/05_rerank.sh --all"
  exit 0
fi

# ── Merging shards ─────────────────────────────────────────────────────────
if [ "$DOMERGE" = 1 ]; then
  [ "${#TARGETS[@]}" -gt 0 ] || die "name the member to merge (for example: --merge 8b)"
  say "[05] merging slice shards"
  need_py "$PY_VLLM" vllm
  for m in "${TARGETS[@]}"; do
    run "$PY_VLLM" $S2/merge_union_slices.py --name "$m" "${RF[@]+"${RF[@]}"}"
  done
  exit 0
fi

# ── Assembling recs_*.pt ───────────────────────────────────────────────────
# S4a (tail_refinement) falls back to this dump for any (q,c) missing from the union cache.
# Every input (base score · reranker union scores) is already cached, so **no GPU is used**.
# Consumers hard-code the names: 8b -> recs_8b_p3_k20.pt, qwen3vl_2b -> recs_2b_dora_k5_p3.pt
if [ "$RECS" = 1 ]; then
  say "[05] assembling recs (no GPU)"
  need_path "$CACHE/s1_base/base_score.pt" "bash ops/04_encode.sh --build"
  for n in 8b qwen3vl_2b; do
    need_path "$RR/${n}_union_cache.pt" "bash ops/05_rerank.sh $n"
    run "$PY_ENS" $S2/build_recs.py --name "$n" "${RF[@]+"${RF[@]}"}"
  done
  echo; ok "-> $RR/recs_8b_p3_k20.pt · recs_2b_dora_k5_p3.pt"
  echo "  next: bash ops/05_rerank.sh --fuse"
  exit 0
fi

# ── The three fuse_cache dumps ─────────────────────────────────────────────
if [ "$FUSE" = 1 ]; then
  say "[05] fuse_cache — fixed champion top-20 column format (the S3 fuse.py contract)"
  need_py "$PY_VLLM" vllm
  # dump_fuse_cache derives RECS from its own --rep flag, so it is passed explicitly here to keep
  # the candidate columns on the same base generation as the union caches being used.
  need_path "$RECS_DIR/recs_8b_p3_k20.pt" "assemble the recs dump first: bash ops/05_rerank.sh --recs"
  FENV=(env "RECS=$RECS_DIR/recs_8b_p3_k20.pt")
  MQ=(); [ "$SMOKE" = 1 ] && MQ=(--max-q 8)
  run "${FENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY_VLLM" $S2/dump_fuse_cache.py \
    --model "$VLM/InternVL3_5-30B-A3B-HF" --adapter "$MR/reranker/internvl_r32" \
    --name internvl_r32 "${RF[@]+"${RF[@]}"}" "${MQ[@]+"${MQ[@]}"}"
  run "${FENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY_VLLM" $S2/dump_fuse_cache.py \
    --model "$VLM/Pixtral-12B-2409" --name pixtral --engine vllm "${RF[@]+"${RF[@]}"}" "${MQ[@]+"${MQ[@]}"}"
  run "${FENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" "$PY_VLLM" $S2/dump_fuse_cache.py \
    --model "$VLM/Llama-3.2-11B-Vision-Instruct" --name llama32v "${RF[@]+"${RF[@]}"}" "${MQ[@]+"${MQ[@]}"}"
  echo
  ok "-> $RR/fuse_cache/{internvl_r32,pixtral,llama32v}"
  echo "  next: REP=1 bash run_reproduce.sh best   (S3 fuse+assign -> S4 tail)"
  echo "  the three post-S4b nntail caches need run_reproduce's \$WORK to exist:"
  echo "    for n in internvl_r32 jina_m0 llama; do"
  echo "      \$PY_VLLM pipeline/S4_tail/dump_nntail_cache.py --name \$n --work <WORK> ${RF[*]-}"
  echo "    done"
  exit 0
fi

# ── Run ────────────────────────────────────────────────────────────────────
[ "$ALL" = 1 ] && TARGETS=("${MEMBERS[@]}")
[ "${#TARGETS[@]}" -gt 0 ] || die "name a target. List them with: bash ops/05_rerank.sh --list"
[ "$SMOKE" = 1 ] && warn "smoke mode — a few queries only · output $RR (isolated). Not canonical."
if [ "$SMOKE" = 0 ] && [ "$REP" = 0 ] && [ -z "${OUT_DIR:-}" ]; then
  warn "adopted mode: assets/cache/s2_rerank/*_union_cache.pt is git-tracked content and gets overwritten."
fi
[ -n "$SLICE" ] && [ "${#TARGETS[@]}" -gt 1 ] && die "--slice applies to a single member only"

need_py "$PY_VLLM" vllm
# Check the pool this run actually reads: an OUT_DIR/smoke run owns a copy under $RR, so it must not
# fail because the adopted cache is mid-rebuild by something else.
need_path "$RR/$POOLF" "bash ops/04_encode.sh --build"
if [ "$REP" = 1 ]; then
  need_path "$MR/reranker" "bash ops/03_deploy.sh --adopted"
fi

FAILED=""
for m in "${TARGETS[@]}"; do
  case " ${MEMBERS[*]} " in *" $m "*) ;; *) die "unknown member: $m  (list them with --list)";; esac
  echo; say "[05] $m  (GPU=$GPU${SLICE:+ · slice $SLICE})"
  # One failing member does not stop the rest. Under set -e the whole run dies and the later
  # members are never attempted (a missing scipy for 8b once blocked qwen3vl_2b·pixtral·ovis).
  if ! "rr_$m"; then
    warn "$m failed — continuing"
    FAILED="$FAILED $m"
  fi
done

echo
say "[05] artifacts"
for m in "${TARGETS[@]}"; do
  f="$RR/${m}_union_cache.pt"
  [ -e "$f" ] && ok "$m -> ${m}_union_cache.pt ($(du -h "$f" | cut -f1))" \
              || warn "$m -> missing (if it ran as slices, merge them with --merge $m)"
done
if [ -n "$FAILED" ]; then
  echo; warn "failed members:$FAILED"
fi
echo
# --recs comes first: it is assembled *from* the union caches that just finished, and --fuse reads
# the dump it produces. Skipping it leaves 8b/qwen3vl_2b scoring the full pool on the next run.
echo "  next: bash ops/05_rerank.sh --recs   then   bash ops/05_rerank.sh --fuse"
