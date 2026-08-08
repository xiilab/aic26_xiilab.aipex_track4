#!/bin/bash
# Dump all 6 encoders for UCA (GPU 7). Sequential, one log file per encoder.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ARTIFACTS:-$REPO/assets/data/benches}"
export CUDA_VISIBLE_DEVICES=7
LLM=${LLM:-$PY_llm2clip}       # peft 0.18 / transformers 4.56
B3=${B3:-$PY_beit3eval}        # provides open_clip
for s in metaclip2 mc2h378_peft anchor_filip anchor_tcap; do
  echo "=== $s $(date +%H:%M:%S)"; $LLM uca_dump_$s.py > logs_uca/$s.log 2>&1 && echo "  OK" || echo "  FAIL (log: logs_uca/$s.log)"
done
echo "=== gme $(date +%H:%M:%S)"; HF_HOME=${HF_CACHE:-$REPO/assets/model/hf_cache} $LLM uca_dump_gme.py > logs_uca/gme.log 2>&1 && echo "  OK" || echo "  FAIL"
echo "=== eva02_pre $(date +%H:%M:%S)"; $B3 uca_dump_eva02_pre.py > logs_uca/eva02_pre.log 2>&1 && echo "  OK" || echo "  FAIL"
echo "=== done $(date +%H:%M:%S)"; ls -la uca_*_feats.pt 2>/dev/null | awk '{printf "  %7.1f MB  %s\n", $5/1048576, $9}'
