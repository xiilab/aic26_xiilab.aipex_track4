#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 02_select.sh — select a model (epoch/step) on the held-out bench
#
# Rule: selection uses **the held-out split only**. Picking an epoch by test mAP would make the
# result test-derived, and open-data epoch selection is known to anti-correlate with Track 4.
#   selection metric = the main bench of heldout_v1/split.json (2,000 queries / 36,773 gallery)
#   the hard bench is diagnostic only and is never used for selection
#
# This step reads the **per-epoch checkpoints** of a heldout run. The local tree holds only the
#    adopted ones, so RUNS_ROOT defaults to the source tree ($SRC_REPO/assets/runs), read-only.
#
# Usage:
#   bash ops/02_select.sh --list                  # print commands and adopted values only
#   bash ops/02_select.sh anchor_tcap             # select one model
#   bash ops/02_select.sh anchor_tcap --epochs 1-12
#   bash ops/02_select.sh --all                   # everything (slow — one eval takes ~16 min)
#   GPU=7 bash ops/02_select.sh metaclip2
#
# The result is written to <run>/heldout_eval.json, which 03_deploy.sh reads.
# The --deploy-rep hook selects and deploys in one go (see the per-command comments below).
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

LIST=0; ALL=0; TARGETS=(); EPOCHS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --list)    LIST=1 ;;
    --all)     ALL=1 ;;
    --epochs)  EPOCHS="$2"; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) die "unknown argument: $1" ;;
    *)  TARGETS+=("$1") ;;
  esac
  shift
done

R="${RUNS_ROOT_SEL:-${RUNS_SRC:-$RUNS_LOCAL}}"
export RUNS_ROOT="$R"
# Heldout run path per model (the first candidate name that exists)
hrun() { resolve_run "$R" "${HELDOUT_CAND[$1]}"; }
HE="train/encoders/eval"

# model -> heldout run · epoch range · selection command
sel_anchor_tcap() {   # SWA family: score per epoch, then average the best range with SWA
  local ep="${EPOCHS:-1-12}"
  run "$PY_TRAIN" $HE/eval_heldout.py --trainer train/encoders/anchor_tcap_heldout/train.py \
      --run "$(hrun anchor_tcap)" --epochs "$ep" --bench main --gpu "$GPU"
  echo "     # optional SWA range diagnosis: $HE/eval_heldout_swa.py --trainer … --run … --epochs $ep"
}
sel_anchor_filip() {
  local ep="${EPOCHS:-1-12}"
  run "$PY_TRAIN" $HE/eval_heldout.py --trainer train/encoders/anchor_filip_heldout/train.py \
      --run "$(hrun anchor_filip)" --epochs "$ep" --bench main --gpu "$GPU"
}
sel_mc2h378_peft() {
  local ep="${EPOCHS:-1-9}"
  run "$PY_TRAIN" $HE/eval_heldout.py --trainer train/encoders/mc2h378_peft_heldout/train.py \
      --run "$(hrun mc2h378_peft)" --epochs "$ep" --bench main --gpu "$GPU"
}
sel_siglip_maxsim() {   # no built-in range: search it instead of scoring epochs (--epochs n/a)
  run "$PY_TRAIN" train/encoders/siglip_maxsim_heldout/search_swa_range.py \
      "$(hrun siglip_maxsim)" --gpu "$GPU"
}
sel_metaclip2() {
  local ep="${EPOCHS:-1-5}"
  # adding --deploy-rep metaclip2 deploys the best epoch straight into model_rep
  run "$PY_TRAIN" $HE/eval_heldout.py --trainer train/encoders/metaclip2/train.py \
      --run "$(hrun metaclip2)" --epochs "$ep" --bench main --gpu "$GPU"
}
sel_beit3_v2() {
  local ep="${EPOCHS:-0-11}"
  run "$PY_BEIT3" train/encoders/beit3/beit3_tool.py eval \
      --run "$(hrun beit3_v2)" --epochs "$ep" --gpu "$GPU"
}
sel_beit3_helip() {
  local ep="${EPOCHS:-0-2}"
  warn "helip uses stage1 as its init; stage1 itself is not a deployment target."
  run "$PY_BEIT3" train/encoders/beit3/beit3_tool.py eval \
      --run "$(hrun beit3_helip)" --epochs "$ep" --gpu "$GPU"
}
sel_metaclip_v1() {
  local ep="${EPOCHS:-1-10}"
  # full FT through open_clip, so it uses its own evaluator. --pretrained is the base weight (.pt).
  run "$PY_BEIT3" $HE/eval_heldout_openclip.py --model ViT-L-14-worldwide-xlmv \
      --pretrained "$REPO/assets/model/vlm_models/MetaCLIP-L14-worldwide/l14_worldwide.pt" \
      --ckpt-dir "$(hrun metaclip_v1)/checkpoints" --epochs "$ep" --gpu "$GPU"
}
sel_llm2clip_anchor5() {
  local rd; rd="$(resolve_run "$R" "${RUN_CAND[llm2clip_anchor5]}" || true)"
  need_path "$rd" "no text-adapter run — train it with train/encoders/llm2clip_anchor5/train_text.py
      (OUT_DIR defaults to assets/runs/llm2clip_text_lora), or point RUNS_ROOT at yours"
  echo "     checkpoints in $(basename "$rd"):"
  # no match returns 1, which under `set -e` would abort the whole run
  ls -1 "$rd" 2>/dev/null | grep -E '^(ep[0-9]+|last)$' | sed 's/^/       /' || true
  echo "     no bench to rank them on — pick one and deploy:"
  echo "       bash ops/03_deploy.sh llm2clip_anchor5 --pick <ep{N}|last>"
}
sel_internvl_r32() {
  run "$PY_VLLM" train/reranker/eval/eval_step.py --member r32 --run "$(hrun internvl_r32)" \
      --steps "${EPOCHS:-step1000,step1500,step2000,step2500,step3000}" --ckpt-subdir "" --gpu "$GPU2"
}
sel_jina_m0() {
  run "$PY_VLLM" train/reranker/eval/eval_step.py --member jina --run "$(hrun jina_m0)" \
      --steps "${EPOCHS:-ex006000,ex007008,ex008000,ex009008,ex010000}" --gpu "$GPU2"
}
sel_qwen3vl_2b() {
  run "$PY_VLLM" train/reranker/eval/eval_step.py --member qw2b --run "$(hrun qwen3vl_2b)" \
      --steps "${EPOCHS:-ex005000,ex006000,ex007000,ex008000,ex009000}" --gpu "$GPU2"
}

MODELS=(anchor_tcap anchor_filip mc2h378_peft siglip_maxsim metaclip2 beit3_v2 beit3_helip
        metaclip_v1 internvl_r32 jina_m0 qwen3vl_2b llm2clip_anchor5)

if [ "$LIST" = 1 ]; then
  say "[02] selection targets · adopted baseline · RUNS_ROOT=$R"
  printf '  %-17s %-14s %-42s %s\n' "model" "adopted" "heldout run (selection input)" "have"
  printf '  %-17s %-14s %-42s %s\n' "-------------" "-------------" "----------------------------------" "----"
  for m in "${MODELS[@]}"; do
    if hp="$(resolve_run "$R" "${HELDOUT_CAND[$m]:-${RUN_CAND[$m]}}")"; then have="✓"; else have="✗"; fi
    ad="${ADOPTED[$m]}"; [ -z "${ad#*:}" ] && ad="$ad<pick>"
    printf '  %-17s %-14s %-42s %s\n' "$m" "$ad" "$(basename "$hp")" "$have"
  done
  echo
  echo "  The three rerankers have no separate heldout run — step checkpoints of the same run"
  echo "  are ranked by pair accuracy. beit3_helip has none either; stage1_heldout is its init."
  echo "  <pick> = no adopted value in the bundle, and only REP=1 needs one (REP=0 reads"
  echo "  assets/model directly): siglip_maxsim takes an SWA range, llm2clip_anchor5 a checkpoint."
  echo
  echo "  print the commands only:  DRY=1 bash ops/02_select.sh <model>"
  echo "  deploy the adopted values without selecting:  bash ops/03_deploy.sh --adopted"
  exit 0
fi

[ "$ALL" = 1 ] && TARGETS=("${MODELS[@]}")
[ "${#TARGETS[@]}" -gt 0 ] || die "name a target. List them with: bash ops/02_select.sh --list"

need_path "$R" "check SRC_REPO, or set RUNS_ROOT"
need_path assets/data/heldout_v1 "bash ops/01_stage_assets.sh --only heldout"

for m in "${TARGETS[@]}"; do
  case " ${MODELS[*]} " in *" $m "*) ;; *) die "unknown model: $m  (list them with --list)";; esac
  echo
  ad="${ADOPTED[$m]}"; [ -z "${ad#*:}" ] && ad="$ad<pick>"
  say "[02] $m  (adopted baseline $ad)"
  "sel_$m"
done
echo
ok "selection done — result JSON: <run>/heldout_eval.json · steps_<member>.json
      (siglip_maxsim writes per-epoch heldout_eval/ep{NN}.json and prints the range instead;
       llm2clip_anchor5 writes none: it has no bench, so 02 only lists its checkpoints)"
echo "  next: bash ops/03_deploy.sh <model> --pick <epoch|step>"
