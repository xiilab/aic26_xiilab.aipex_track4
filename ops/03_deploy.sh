#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 03_deploy.sh — selected checkpoint -> assets/model_rep/{encoder,reranker}/<name>
#
# **model_rep is the only deployment target.** The adopted assets/model/ is never touched: a
# reproduction run overwriting the adopted artifacts would make the md5 comparison meaningless.
#
# Each model has its own deploy.py (tools/promote.py is the generic manual alternative).
#   SWA family (anchor_tcap·anchor_filip·mc2h378_peft): build_swa.py <run> lo hi, then deploy.py <run>
#   metaclip2 : deploy.py <run>                    (checkpoints/swa — the run ships it prebuilt)
#   beit3     : deploy.py <run> --recipe v2|helip --epoch N   (checkpoint-N.pth -> checkpoint-best.pth)
#   metaclip_v1: deploy.py <ckpt_dir> --epoch N    (epoch_N.pt)
#   rerankers : deploy.py <run> --step <t*>
#
# Usage:
#   bash ops/03_deploy.sh --adopted              # deploy all ten adopted values (reproduce the baseline)
#   bash ops/03_deploy.sh metaclip2 --pick 2
#   bash ops/03_deploy.sh anchor_tcap --pick 8-10   # SWA range
#   bash ops/03_deploy.sh jina_m0 --pick ex008000
#   bash ops/03_deploy.sh --verify               # md5 the deployment against the adopted weights
#   DRY=1 bash ops/03_deploy.sh --adopted        # commands only
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

ADOPT=0; VERIFY=0; PICK=""; TARGETS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --adopted) ADOPT=1 ;;
    --verify)  VERIFY=1 ;;
    --pick)    PICK="$2"; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) die "unknown argument: $1" ;;
    *)  TARGETS+=("$1") ;;
  esac
  shift
done

# Deployment reads runs from the **local tree** (build_swa writes into <run>, so it must not be
# the source tree). Point RUNS_ROOT elsewhere to use a different run.
R="${RUNS_ROOT:-$RUNS_LOCAL}"
export RUNS_ROOT="$R"
PY="${PY_DEPLOY:-$PY_ENS}"        # deploy.py only uses shutil, so any python works

# ── Verify mode: md5 the model_rep deployment against the source tree ──────
if [ "$VERIFY" = 1 ]; then
  say "[03] verifying the deployment — assets/model_rep vs $SRC_REPO/assets/model_rep"
  fails=0
  for f in $(cd "$SRC_REPO/assets/model_rep" 2>/dev/null && find . -name 'adapter_model.safetensors' -o -name '*.pth' -o -name '*.pt' | sort); do
    a="$REPO/assets/model_rep/${f#./}"; b="$SRC_REPO/assets/model_rep/${f#./}"
    if [ ! -e "$a" ]; then printf '  \033[33m·\033[0m not deployed %s\n' "${f#./}"; continue; fi
    if [ "$(md5sum "$a" | cut -d' ' -f1)" = "$(md5sum "$b" | cut -d' ' -f1)" ]; then
      ok "${f#./}"
    else
      printf '  \033[31m✗\033[0m %s — md5 mismatch\n' "${f#./}"; fails=$((fails+1))
    fi
  done
  echo
  [ "$fails" -eq 0 ] && ok "no mismatches" || warn "$fails mismatch(es) (expected if a different epoch was chosen)"
  exit 0
fi

# ── Per-model deployment ───────────────────────────────────────────────────
dep_swa() {   # $1=name  $2=range(lo-hi)
  local name="$1" rng="$2" rd lo hi
  rd="$(resolve_run "$R" "${RUN_CAND[$1]}")" || true
  lo="${rng%%-*}"; hi="${rng##*-}"
  need_path "$rd" "check RUNS_ROOT"
  if [ -d "$rd/checkpoints/swa" ] && [ -z "$PICK" ]; then
    # An existing SWA (the adopted one) is not rebuilt. Recomputing needs an explicit --pick range.
    ok "checkpoints/swa present -> skipping build_swa ($(cat "$rd/checkpoints/swa/swa_meta.json" 2>/dev/null | tr -d '\n ' | head -c 60))"
  else
    assert_writable_run "$rd"     # build_swa writes to <run>/checkpoints/swa
    need_path "$rd/checkpoints/ep$(printf '%02d' "$lo")" \
      "ep$(printf '%02d' "$lo")..ep$(printf '%02d' "$hi") are not in this tree.
      Copy those epochs over first to rebuild the SWA (source: $(resolve_run "$RUNS_SRC" "${RUN_CAND[$name]}"))"
    run "$PY" "train/encoders/${name}_all/build_swa.py" "$rd" "$lo" "$hi"
  fi
  run "$PY" "train/encoders/${name}_all/deploy.py" "$rd"
}
# metaclip2 is SWA-deployed like the anchor family, but its scripts live in train/encoders/metaclip2
# (no _all suffix) and the run it ships already carries checkpoints/swa, so nothing is rebuilt here.
# Its deploy.py takes the run directory alone — passing --epoch is what used to break --adopted.
dep_metaclip2()  { run "$PY" train/encoders/metaclip2/deploy.py "${MC2_RUN:-$(resolve_run_with_ckpt "${RUN_CAND[metaclip2]}")}"; }
dep_beit3_v2()   { run "$PY" train/encoders/beit3/deploy.py "$(resolve_run_with_ckpt "${RUN_CAND[beit3_v2]}")"    --recipe v2    --epoch "$1"; }
dep_beit3_helip(){ run "$PY" train/encoders/beit3/deploy.py "$(resolve_run_with_ckpt "${RUN_CAND[beit3_helip]}")" --recipe helip --epoch "$1"; }
dep_metaclip_v1(){ run "$PY" train/encoders/metaclip_v1/deploy.py "$(resolve_run_with_ckpt "${RUN_CAND[metaclip_v1]}")/checkpoints" --epoch "$1"; }
dep_internvl_r32(){ run "$PY" train/reranker/internvl_r32/deploy.py "$(resolve_run_with_ckpt "${RUN_CAND[internvl_r32]}")" --step "$1"; }
dep_jina_m0()    { run "$PY" train/reranker/jina_m0/deploy.py    "$(resolve_run_with_ckpt "${RUN_CAND[jina_m0]}")"    --step "$1"; }
dep_qwen3vl_2b() { run "$PY" train/reranker/qwen3vl_2b/deploy.py "$(resolve_run_with_ckpt "${RUN_CAND[qwen3vl_2b]}")" --step "$1"; }

deploy_one() {
  local m="$1" pick="$2" kind val
  kind="${ADOPTED[$m]%%:*}"; val="${pick:-${ADOPTED[$m]#*:}}"
  say "[03] $m  <- $kind $val"
  case "$m" in
    anchor_tcap|anchor_filip|mc2h378_peft) dep_swa "$m" "$val" ;;
    metaclip2)                             dep_metaclip2 ;;
    beit3_v2|beit3_helip|metaclip_v1)      "dep_$m" "${val#ep}" ;;
    internvl_r32|jina_m0|qwen3vl_2b)       "dep_$m" "$val" ;;
    *) die "unknown model: $m" ;;
  esac
}

MODELS=(anchor_tcap anchor_filip mc2h378_peft metaclip2 beit3_v2 beit3_helip metaclip_v1
        internvl_r32 jina_m0 qwen3vl_2b)

if [ "$ADOPT" = 1 ]; then
  [ "${#TARGETS[@]}" -eq 0 ] || die "do not combine --adopted with an explicit model"
  TARGETS=("${MODELS[@]}"); PICK=""
fi
[ "${#TARGETS[@]}" -gt 0 ] || die "name a target (or use --adopted). List: bash ops/02_select.sh --list"
[ "${#TARGETS[@]}" -gt 1 ] && [ -n "$PICK" ] && die "--pick applies to a single model only"

for m in "${TARGETS[@]}"; do echo; deploy_one "$m" "$PICK"; done

ech
say "[03] deployment result"
find assets/model_rep -mindepth 2 -maxdepth 2 -type d 2>/dev/null \
  | sed "s|assets/model_rep/||" | sort | sed 's/^/  /'
echo
ok "next: bash ops/04_encode.sh --list"