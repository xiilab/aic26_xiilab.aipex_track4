"""RSTP gme feats dump — GME-Qwen2-VL-2B-Instruct zero-shot (no adapter).
Uses get_text_embeddings / get_image_embeddings with the image token budget set to 64..256,
with RSTP as the data source.

env: .venv_llm2clip
run: CUDA_VISIBLE_DEVICES=6 HF_HOME=$HF_CACHE $PY_llm2clip rstp_dump_gme.py
out: $ARTIFACTS/rstp_gme_feats.pt  {G, Q, gal_paths, Q_idx}
"""
import os, glob, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")   # overridable from the environment
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root
os.environ.setdefault("HF_HOME", os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import transformers.utils.versions as _v; _v.require_version = lambda *x, **k: None
import transformers; transformers.utils.require_version = lambda *x, **k: None

HERE = os.environ.get("TRACK4", f"{_REPO}/assets/data/benches")
POOL = f"{HERE}/rstp_pool.pt"
T2I = "Find an image that matches the given text."
MIN_TOK, MAX_TOK = 64, 256
BS_IMG, BS_TXT = 32, 64
OUT = f"{HERE}/rstp_gme_feats.pt"

snaps = glob.glob(os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache") + "/hub/models--Alibaba-NLP--gme-Qwen2-VL-2B-Instruct/snapshots/*/")
MODEL = snaps[0] if snaps else "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"

from transformers import AutoModel
gme = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)
try:
    gme.processor.image_processor.min_pixels = MIN_TOK * 28 * 28
    gme.processor.image_processor.max_pixels = MAX_TOK * 28 * 28
    print(f"[pixels] min_tok={MIN_TOK} max_tok={MAX_TOK}", flush=True)
except Exception as e:
    print(f"[pixels] set fail: {e}", flush=True)
print(f"[loaded] {MODEL}", flush=True)

# ---- RSTP data ----
_p = torch.load(POOL, map_location="cpu", weights_only=False)
gal_paths = list(_p["gal_paths"]); caps = list(_p["caps"])
print(f"[rstp-gme] gallery={len(gal_paths)} queries={len(caps)}", flush=True)

t0 = time.time(); Qparts = []
for i in range(0, len(caps), BS_TXT):
    Qparts.append(gme.get_text_embeddings(texts=caps[i:i + BS_TXT], instruction=T2I).float().cpu())
Q = torch.cat(Qparts)
print(f"[Q] {tuple(Q.shape)} {time.time()-t0:.0f}s", flush=True)

t0 = time.time(); Gparts = []
for i in range(0, len(gal_paths), BS_IMG):
    Gparts.append(gme.get_image_embeddings(images=gal_paths[i:i + BS_IMG], is_query=False).float().cpu())
    if (i // BS_IMG) % 20 == 0:
        print(f"  [G] {min(i+BS_IMG, len(gal_paths))}/{len(gal_paths)} ({time.time()-t0:.0f}s)", flush=True)
G = torch.cat(Gparts)[:len(gal_paths)]
print(f"[G] {tuple(G.shape)} {time.time()-t0:.0f}s", flush=True)

torch.save({"G": G, "Q": Q, "gal_paths": gal_paths, "Q_idx": list(range(len(caps)))}, OUT)
print(f"[save] {OUT}  G={tuple(G.shape)} Q={tuple(Q.shape)}", flush=True)
