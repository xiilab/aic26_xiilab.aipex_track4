"""RSTP v28a TTA-views dump (base/hflip/z090) — anchor member.
Feature extraction lives in pipeline/S1_base/encode/encode_anchor_tcap.py; this script only
selects the pool to read and the output path (same views dict format as anchor_tcap_tta_views.pt).

env: .venv_llm2clip
run: CUDA_VISIBLE_DEVICES=6 $PY_llm2clip rstp_dump_anchor_tcap.py
out: $ARTIFACTS/rstp_anchor_tcap_feats.pt  {img:{base,hflip,z090}, txt:{base}, gal_paths, Q_idx}
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")   # overridable from the environment
                                                     # must be set before the torch import below
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root
sys.path.insert(0, f"{_REPO}/pipeline/S1_base/encode")
from encode_anchor_tcap import dump_tta_views

T4 = os.environ.get("TRACK4", f"{_REPO}/assets/data/benches")
POOL = f"{T4}/rstp_pool.pt"         # single source for gal_paths/caps ordering
OUT = f"{T4}/rstp_anchor_tcap_feats.pt"
CKPT = os.environ.get("ENCODER_CKPT", f"{_REPO}/assets/model/encoder/anchor_tcap")   # RUN A ep09
VIEWS = ("base", "hflip", "z090")

dump_tta_views(POOL, OUT, ckpt=CKPT, views=VIEWS, tag="rstp-v28a",
               gallery_batch=64, num_workers=8)
