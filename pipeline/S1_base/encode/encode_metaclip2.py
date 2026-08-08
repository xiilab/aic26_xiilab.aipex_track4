"""metaclip2 (MetaCLIP2 worldwide-l14, DoRA ep02) -> global feature dump (no labels).

Output: metaclip2_feats.pt {G, Q, G_base, Q_idx}  (member score = norm(Q) @ norm(G).T)
        Default rebuilds assets/cache/s1_base/members/metaclip2_feats.pt; --rep writes to cache_rep.
"""
import argparse, importlib.util, json, os, shutil, time
from pathlib import Path
import torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

MEMBERS= os.environ.get("S1_MEMBERS", f"{_REPO}/assets/cache/s1_base/members")
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
TEST_ROOT= PAB_TEST
ap=argparse.ArgumentParser(); ap.add_argument("--gpu",default="0"); ap.add_argument("--ckpt",default="ep02"); ap.add_argument("--bs",type=int,default=256)
ap.add_argument("--run",default=None,help="run directory (default = the adopted metaclip2 run); set it to score a retrained model")
ap.add_argument("--out",default=None,help="output .pt (default = members/metaclip2_feats.pt)")
ap.add_argument("--limit",type=int,default=None,help="truncate gallery/query for a smoke test")
ap.add_argument("--overwrite",action="store_true",help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep",action="store_true",help="reproduction encoding with model_rep weights -> cache_rep/s1_base/members/metaclip2_feats.pt (+s4_nn)")
a=ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
if a.rep and a.limit:                                         # keep truncated artifacts out of the reproduction cache
    raise SystemExit("[encode_metaclip2] --limit cannot be combined with --rep (write smoke runs elsewhere with --out).")
if a.rep: os.environ.setdefault("ENCODER_CKPT_DIR", f"{_REPO}/assets/model_rep/encoder")   # rep: deployed weight source
CACHE_REP=f"{_REPO}/assets/cache_rep"; REP_MEM=f"{CACHE_REP}/s1_base/members"   # resolve the output path before loading the model
OUT=a.out or (f"{REP_MEM}/metaclip2_feats.pt" if a.rep else f"{MEMBERS}/metaclip2_feats.pt")   # build_base LOADERS['metaclip2']
skip_if_exists(OUT, a.overwrite)
if a.rep: os.makedirs(REP_MEM, exist_ok=True)
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # metaclip2_infer lives in this directory
import metaclip2_infer as E                                       # noqa: E402
dev="cuda:0"
M=E.import_train_module(E.MODULE_PATH)
CKPT_DIR=os.environ.get("ENCODER_CKPT_DIR", f"{_REPO}/assets/model/encoder")
RUN=a.run
ckpt_dir=os.path.join(RUN,"checkpoints",a.ckpt) if RUN else f"{CKPT_DIR}/metaclip2"   # bundled metaclip2 adapter
assert os.path.isdir(ckpt_dir), f"checkpoint not found: {ckpt_dir}"
print(f"[metaclip2] MODEL={M.MODEL_NAME} IMG={M.IMAGE_SIZE} ckpt={a.ckpt}",flush=True)

def load_queries_gallery():
    queries=[]
    with open(f"{TEST_ROOT}/query_text.json","r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            r=json.loads(line)
            queries.append({"query_index":str(r["query_index"]),"caption":r["caption"],"change":r.get("change")})
    gallery_paths=sorted(str(p) for p in Path(f"{TEST_ROOT}/gallery").glob("*.jpg"))
    return queries, gallery_paths

queries,gallery_paths=load_queries_gallery()
if a.limit: queries,gallery_paths=queries[:a.limit],gallery_paths[:a.limit]
tf=E.build_eval_transform(M)
model,tok=E.load_for_inference(M,ckpt_dir,dev)
t0=time.time(); G,G_base=E.encode_gallery(M,model,gallery_paths,tf,dev,a.bs); print(f"gallery {tuple(G.shape)} {time.time()-t0:.0f}s",flush=True)
Q,Q_idx,Q_chg,Q_len=E.encode_queries(M,model,queries,tok,dev,a.bs); print(f"query {tuple(Q.shape)}",flush=True)
torch.save({"G":G,"Q":Q,"G_base":G_base,"Q_idx":Q_idx,"run":RUN,"ckpt":a.ckpt},OUT)
print(f"[save] {OUT}",flush=True)
if a.rep:                                                     # S4b tail-NN uses the same feats
    os.makedirs(f"{CACHE_REP}/s4_nn", exist_ok=True)
    shutil.copy2(OUT, f"{CACHE_REP}/s4_nn/metaclip2_feats.pt")
    print(f"[save] {CACHE_REP}/s4_nn/metaclip2_feats.pt (tail-NN)",flush=True)
