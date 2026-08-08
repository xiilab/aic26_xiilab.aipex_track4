#!/bin/bash
# Download the jina-reranker-m0 base model (once, before training).
#
# snapshot_download fetches the repo structure and the small files; large blobs are resumed with
# curl -C -, because HF intermittently stalls at 0 B/s and snapshot_download alone does not finish.
# Fetched: model.safetensors (4.89GB) · tokenizer.json (11MB) · config/preprocessor · custom code (*.py) · *.jinja
#
# HF_CACHE must match the HF_HOME used by the training code (default assets/model/hf_cache).
HF_CACHE=${HF_CACHE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/assets/model/hf_cache}
HUB="$HF_CACHE/hub"
ALLOW="['*.json','*.txt','*.safetensors','*.model','*.py','*.jinja']"

download_model() {
  local MODEL_ID="$1"
  local REPO_DIR="$HUB/models--${MODEL_ID//\//--}"
  echo "================================================================"
  echo "[$(date)] REPO: $MODEL_ID"
  echo "================================================================"

  # ---- (A) snapshot_download: structure + small files + start the large blob (.incomplete); give up after 150s on a stall ----
  echo "[A] snapshot_download (structure/small files) ..."
  HF_HOME="$HF_CACHE" timeout 150 python3 -u -c "
from huggingface_hub import snapshot_download
try:
    p=snapshot_download('$MODEL_ID', cache_dir='$HUB', allow_patterns=$ALLOW)
    print('[A] COMPLETE:', p, flush=True)
except Exception as e:
    print('[A] interrupted (curl will resume):', type(e).__name__, flush=True)
" 2>&1 | grep -vE "%\||it/s|resume_download|warnings.warn"

  # ---- (B) get (file, sha, size) from the API, then resume each LFS file with curl -C - ----
  HF_HOME="$HF_CACHE" python3 -c "
from huggingface_hub import HfApi
info=HfApi().model_info('$MODEL_ID', files_metadata=True)
for s in info.siblings:
    if s.lfs:
        print(s.rfilename, s.lfs.sha256, s.size)
" > /tmp/_dlmap_jina.txt
  while read F SHA EXP; do
    [ -z "$SHA" ] && continue
    local BLOB="$REPO_DIR/blobs/$SHA"
    local URL="https://huggingface.co/$MODEL_ID/resolve/main/$F"
    local SRC="$BLOB.incomplete"; [ -f "$BLOB" ] && SRC="$BLOB"
    echo "[B] $F sha=$SHA expected=$EXP"
    for i in $(seq 1 60); do
      CUR=$(stat -c%s "$SRC" 2>/dev/null || echo 0)
      if [ "$CUR" -ge "$EXP" ]; then echo "[B] $F COMPLETE ($CUR)"; break; fi
      echo "[B] $F try $i: $CUR / $EXP ($(date +%H:%M:%S))"
      curl -sSL -C - -o "$SRC" "$URL" --max-time 1800
      echo "[B] $F curl exit=$? size=$(stat -c%s "$SRC" 2>/dev/null||echo 0)"
      sleep 2
    done
    if [ "$SRC" = "$BLOB.incomplete" ] && [ -f "$SRC" ]; then mv -f "$SRC" "$BLOB"; echo "[B] $F → blob finalized"; fi
  done < /tmp/_dlmap_jina.txt

  # ---- (C) re-run snapshot_download: finalize and verify the completed blobs as symlinks/refs ----
  echo "[C] finalize (snapshot_download → symlink/ref) ..."
  HF_HOME="$HF_CACHE" python3 -u -c "
from huggingface_hub import snapshot_download
print('[C] FINAL:', snapshot_download('$MODEL_ID', cache_dir='$HUB', allow_patterns=$ALLOW), flush=True)
" 2>&1 | grep -vE "%\||it/s|resume_download|warnings.warn"
  echo "[done] $(date) $MODEL_ID  total=$(du -sh "$REPO_DIR" 2>/dev/null | cut -f1)  weights=$(ls $REPO_DIR/snapshots/*/*.safetensors 2>/dev/null | wc -l)"
  echo ""
}

download_model "jinaai/jina-reranker-m0"
echo "########## ALL DONE $(date) ##########"
