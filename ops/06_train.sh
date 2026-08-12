#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 06_train.sh — train one encoder or reranker member in the conda env its dependencies pin
#
# Training is optional for reproduction: every adopted adapter already ships under
# assets/model/{encoder,reranker}/, and 04/05 consume those directly. This script exists so a
# member can be **re-trained under the same pinned environment** the adopted weights came from,
# rather than whichever env happens to be active.
#
# The env is not a preference. Picking the wrong one fails in three different ways, and two of
# them are silent — measured on this box (B300, sm_103, driver 580.105.08):
#
#   internvl_r32  torch 2.8 -> `torch._grouped_mm is only supported on ... compute capability = 9.0`
#                 raised inside every MoE forward. train.py's `except: continue` swallows it and the
#                 run ends with 0 optimiser steps and a written-out checkpoint. Needs torch 2.11
#                 (track4_vllm), where the same call returns normally.
#   jina_m0       transformers 5.9 cannot load JinaVLForRanking at all here, and 5.x renamed the
#                 Qwen2-VL keys under the *class* name, which a subclass misses — the tower is then
#                 randomly initialised with no error (this is what score_union_jina.py's KEY_MAP
#                 guard is for). Measured missing_keys: 4.51.3 -> 0, **5.4.0 -> 0**, 5.9.0 -> load
#                 fails. track4_train (5.4.0) is correct and needs no guard.
#   metaclip_v1   needs open_clip 3.3.0, absent from track4_train and track4_vllm -> ImportError.
#   beit3         imports the vendored run_beit3_finetuning, which needs torchscale -> track4_beit3.
#
# CUBLAS_WORKSPACE_CONFIG follows env.sh (unset by default). The adopted weights were produced in a
# container that had it unset; keeping the same policy here as in 04/05 means the GEMM kernels do
# not change between the two. KEEP_CUBLAS_WORKSPACE=1 leaves an inherited value alone.
#
#   bash ops/06_train.sh --list
#   bash ops/06_train.sh anchor_tcap_all
#   bash ops/06_train.sh internvl_r32 -- --lora-r 32 --lr 1e-4 --grad-accum 8
#   GPU=3 bash ops/06_train.sh jina_m0
#   DRY=1 bash ops/06_train.sh --all          # print the plan, no env or data needed
#
# Anything after `--` is appended verbatim to train.py. Runs land in the run directory each
# train.py defaults to (assets/runs/...), never in the source tree.
# ─────────────────────────────────────────────────────────────────────────────
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# ── target -> env ──────────────────────────────────────────────────────────
# Verified by loading the model / running train.py --help in each candidate env, not by reading
# the imports: several scripts import their heavy dependency inside main(), so --help alone passes
# in envs that fail at runtime.
declare -A TRAIN_ENV=(
  [anchor_tcap_all]=TRAIN      [anchor_tcap_heldout]=TRAIN
  [anchor_filip_all]=TRAIN     [anchor_filip_heldout]=TRAIN
  [mc2h378_peft_all]=TRAIN     [mc2h378_peft_heldout]=TRAIN
  [siglip_maxsim_all]=TRAIN    [siglip_maxsim_heldout]=TRAIN
  [metaclip2]=TRAIN
  [metaclip_v1]=BEIT3          # open_clip 3.3.0
  [beit3]=BEIT3                # torchscale + vendored run_beit3_finetuning
  [jina_m0]=TRAIN              # transformers 5.4.0 loads JinaVLForRanking cleanly
  [qwen3vl_2b]=TRAIN
  [internvl_r32]=VLLM          # torch 2.11 — 2.8 has no Blackwell _grouped_mm
)
# Where each target's train.py lives, relative to train/.
declare -A TRAIN_DIR=(
  [anchor_tcap_all]=encoders   [anchor_tcap_heldout]=encoders
  [anchor_filip_all]=encoders  [anchor_filip_heldout]=encoders
  [mc2h378_peft_all]=encoders  [mc2h378_peft_heldout]=encoders
  [siglip_maxsim_all]=encoders [siglip_maxsim_heldout]=encoders
  [metaclip2]=encoders         [metaclip_v1]=encoders          [beit3]=encoders
  [jina_m0]=reranker           [qwen3vl_2b]=reranker           [internvl_r32]=reranker
)
# Listing order — encoders first, then rerankers, heldout next to its all- run.
ORDER=(anchor_tcap_all anchor_tcap_heldout anchor_filip_all anchor_filip_heldout
       mc2h378_peft_all mc2h378_peft_heldout siglip_maxsim_all siglip_maxsim_heldout
       metaclip2 metaclip_v1 beit3 internvl_r32 jina_m0 qwen3vl_2b)

py_for() {   # env key -> interpreter, and the name 00_setup_envs.sh knows it by
  case "$1" in
    TRAIN) printf '%s\t%s\n' "$PY_TRAIN" train ;;
    BEIT3) printf '%s\t%s\n' "$PY_BEIT3" beit3 ;;
    VLLM)  printf '%s\t%s\n' "$PY_VLLM"  vllm  ;;
    *)     die "unknown env key: $1" ;;
  esac
}

usage() {
  say "ops/06_train.sh — train a member in its pinned env"
  printf '  %-24s %-16s %s\n' TARGET ENV SCRIPT
  local t k p n
  for t in "${ORDER[@]}"; do
    k="${TRAIN_ENV[$t]}"; IFS=$'\t' read -r p n <<<"$(py_for "$k")"
    printf '  %-24s %-16s %s\n' "$t" "track4_$n" "train/${TRAIN_DIR[$t]}/$t/train.py"
  done
  printf '\n  bash ops/06_train.sh <target> [-- args passed to train.py]\n'
}

train_one() {
  local t="$1"; shift
  [ -n "${TRAIN_ENV[$t]:-}" ] || die "unknown target: $t
      -> bash ops/06_train.sh --list"
  local script="$REPO/train/${TRAIN_DIR[$t]}/$t/train.py" py name
  IFS=$'\t' read -r py name <<<"$(py_for "${TRAIN_ENV[$t]}")"

  say "[$t] track4_$name"
  need_py "$py" "$name"
  need_path "$script" "the train/ tree is missing — this target has no training script here"

  # GPU selection is per-script and the two ways do not mix:
  #   --gpu    twelve of the fourteen take it, and they index the **physical** device
  #            (`cuda:{physical_gpu}`). Four of those — beit3 · internvl_r32 · jina_m0 ·
  #            qwen3vl_2b — additionally assign os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  #            themselves, before importing torch, so their own default (0, or 6 for beit3) wins
  #            over anything exported here and the job lands on the wrong device without saying so.
  #            Setting CUDA_VISIBLE_DEVICES *and* --gpu is worse still: the process then sees one
  #            device numbered 0 while the script asks for cuda:$GPU, which does not exist.
  #   env      metaclip2 and metaclip_v1 take no --gpu and read CUDA_VISIBLE_DEVICES (metaclip_v1
  #            also honours LOCAL_RANK under torchrun).
  # So: pass --gpu when the script accepts it, otherwise export. Never both.
  if grep -q 'add_argument("--gpu"' "$script"; then
    if printf '%s\n' ${@+"$@"} | grep -qx -- '--gpu'; then
      run "$py" -u "$script" ${@+"$@"}                    # caller chose the device explicitly
    else
      run "$py" -u "$script" --gpu "$GPU" ${@+"$@"}
    fi
  else
    run env CUDA_VISIBLE_DEVICES="$GPU" "$py" -u "$script" ${@+"$@"}
  fi
}

TARGETS=()
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --list|-l) usage; exit 0 ;;
    --all)     TARGETS=("${ORDER[@]}"); shift ;;
    --)        shift; EXTRA=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    -*)        die "unknown option: $1  (train.py options go after --)" ;;
    *)         TARGETS+=("$1"); shift ;;
  esac
done
[ ${#TARGETS[@]} -gt 0 ] || { usage; exit 0; }
# `--` args are meant for a single train.py; forwarding them to every target would be a guess.
[ ${#EXTRA[@]} -eq 0 ] || [ ${#TARGETS[@]} -eq 1 ] || die "-- args apply to one target at a time"

for t in "${TARGETS[@]}"; do train_one "$t" ${EXTRA+"${EXTRA[@]}"}; done
ok "done"
