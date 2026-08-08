"""RSTP metaclip2 feats dump — MetaCLIP2 worldwide-L/14 DoRA ep02.
Mirrors encode_metaclip2.py (metaclip2_infer helpers) with RSTP as the data source.

env: .venv_llm2clip  (peft 0.18 / transformers 4.56, MetaCLIP2 DoRA)
run: CUDA_VISIBLE_DEVICES=6 $PY_llm2clip rstp_dump_metaclip2.py
out: $ARTIFACTS/rstp_metaclip2_feats.pt  {G, Q, gal_paths, Q_idx}
"""
import os, importlib.util
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")   # overridable from the environment
import torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root; all default paths are relative to it

HERE = os.environ.get("TRACK4", f"{_REPO}/assets/data/benches")
POOL = f"{HERE}/rstp_pool.pt"
RUN = f"{HERE}/outputs_baseline/runs/20260609_070733_mc2_l14_distill_ddp_w5"
CKPT = "ep02"
BS = 256
OUT = f"{HERE}/rstp_metaclip2_feats.pt"
dev = "cuda:0"

spec = importlib.util.spec_from_file_location("mc2infer", f"{_REPO}/pipeline/S1_base/encode/metaclip2_infer.py")
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
M = E.import_train_module(E.MODULE_PATH)
ckpt_dir = f"{_REPO}/assets/model/encoder/metaclip2"
print(f"[rstp-metaclip2] MODEL={M.MODEL_NAME} IMG={M.IMAGE_SIZE} ckpt={CKPT}", flush=True)

# ---- RSTP data ----
_p = torch.load(POOL, map_location="cpu", weights_only=False)
gal_paths = list(_p["gal_paths"]); caps = list(_p["caps"])
queries = [{"query_index": str(j), "caption": c, "change": ""} for j, c in enumerate(caps)]
print(f"[rstp-metaclip2] gallery={len(gal_paths)} queries={len(caps)}", flush=True)

tf = E.build_eval_transform(M)
model, tok = E.load_for_inference(M, ckpt_dir, dev)
G, _ = E.encode_gallery(M, model, gal_paths, tf, dev, BS)
print(f"[G] {tuple(G.shape)}", flush=True)
Q, Q_idx, _, _ = E.encode_queries(M, model, queries, tok, dev, BS)
print(f"[Q] {tuple(Q.shape)}", flush=True)

torch.save({"G": G, "Q": Q, "gal_paths": gal_paths, "Q_idx": list(range(len(caps)))}, OUT)
print(f"[save] {OUT}", flush=True)
