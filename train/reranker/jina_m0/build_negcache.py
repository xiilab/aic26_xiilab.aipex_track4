"""jina_m0/build_negcache — negcache for jina: append action hard negatives to an existing cache.

Adds "same scene and appearance description, different action" negatives (action_negs) so the model
attends to the action. MiniLM caption embeddings supply the top-{SIM_TOPK} neighbours, of which at
most {N_ACTION_NEG} with a different action label are kept. Only train captions and images are used.

Input   the qwen3vl_2b negcache (top-8 variant) and the RECAP CSV
Output  $OUTPUT_DIR/negcache_action_top8a6.pt  → consumed by `train.py`
Run     python build_negcache.py --gpu 6
"""
import argparse,os,csv,glob,time,numpy as np,torch
from transformers import AutoTokenizer,AutoModel
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repository root — all default paths are relative to it
MINING=f"{_REPO}/assets/data/mining"                                                         # input cache — read only
_ap=argparse.ArgumentParser(description="negcache for jina — append action hard negatives to an existing cache")
_ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
_ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", f"{_REPO}/assets/runs/rerank_jina"), help="output directory")
_ap.add_argument("--force", action="store_true", help="rebuild even if the output already exists")
_a=_ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=_a.gpu
OUT_W=_a.out
os.makedirs(OUT_W, exist_ok=True)
if os.path.exists(f"{OUT_W}/negcache_action_top8a6.pt") and not _a.force:
    raise SystemExit(f"already exists: {OUT_W}/negcache_action_top8a6.pt\n  pass --force to rebuild")
SRC_CACHE=f"{MINING}/negcache_hardimg_ep06_poolfull_top8_tau0.85.pt"
RECAP= os.environ.get("RECAP_CSV", f"{_REPO}/assets/data/raw/recaption/train_msr_v2.csv")
WEBP= os.environ.get("PAB_WEBP", f"{_REPO}/assets/data/raw/pab_train/train_webp")
MINI=glob.glob(os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache") + "/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*")[0]
N_ACTION_NEG=6; SIM_TOPK=80; COS_MIN=0.55   # of the top-80 caption neighbours, the 6 best with a different action, cosine floor 0.55
ORIG_STYLE="p00_original"   # caption preset used for the pool
csv.field_size_limit(10**7)

def norm_act(s): return (s or "").strip().lower()
def ann2webp(ann):  # "train/imgs_N/sub/id.jpg" → "Part {N//8+1}/imgs_N/sub/id.webp"
    # Relative to train_webp, matching what qwen3vl_2b/build_negcache.py stores — these strings are
    # both the action_negs written to the cache and the keys matched against `image_path`, so the
    # two must use the same form. No image is opened here (only captions are embedded).
    p=ann.split("/"); n=int(p[1].replace("imgs_",""))
    return f"Part {n//8+1}/{p[1]}/{p[2]}/{os.path.splitext(p[3])[0]}.webp"

# 1) load the pool: caption, action, webp
print("[1] loading the pool ...",flush=True)
caps=[]; acts=[]; webps=[]
with open(RECAP) as f:
    for r in csv.DictReader(f):
        if r["style"] != ORIG_STYLE: continue
        c=(r.get("caption") or "").strip()
        if len(c.split())<4: continue
        caps.append(c); acts.append(norm_act(r.get("normal") or r.get("anomaly")))
        webps.append(ann2webp(r["image_path"]))
print(f"  pool {len(caps):,}",flush=True)
if not caps:
    raise SystemExit(f"✗ 0 {ORIG_STYLE} captions — the style column in {RECAP} must hold preset keys.")
webp2act={w:a for w,a in zip(webps,acts)}   # for exact matching of the ground-truth action

# 2) MiniLM embeddings
tok=AutoTokenizer.from_pretrained(MINI); m=AutoModel.from_pretrained(MINI).cuda().eval()
def embed(texts,bs=1024):
    out=[]
    for i in range(0,len(texts),bs):
        b=tok(texts[i:i+bs],padding=True,truncation=True,max_length=96,return_tensors="pt").to("cuda")
        with torch.no_grad():
            o=m(**b).last_hidden_state; mask=b["attention_mask"].unsqueeze(-1).float()
            e=(o*mask).sum(1)/mask.sum(1)
            out.append(torch.nn.functional.normalize(e,dim=-1).half().cpu())
        if i%51200==0: print(f"  embed {i}/{len(texts)}",flush=True)
    return torch.cat(out)
print("[2] embedding the pool ...",flush=True); t0=time.time()
P=embed(caps).cuda()   # [N,384] half
print(f"  pool embedded in {time.time()-t0:.0f}s",flush=True)

# 3) use the existing cache's positive captions as queries, then keep the neighbours with a different action
print("[3] loading the input cache and embedding the queries ...",flush=True)
cache=torch.load(SRC_CACHE,map_location="cpu"); ex=cache["examples"]
qcaps=[e["pos"] for e in ex]
Q=embed(qcaps).cuda()
print("[4] NN search (action-diff) ...",flush=True); t0=time.time()
added=0; empty=0
for s in range(0,len(ex),2000):
    qe=Q[s:s+2000]
    sims=qe@P.T                                  # [chunk, N]
    topv,topi=sims.topk(SIM_TOPK,dim=1)
    topv=topv.cpu().numpy(); topi=topi.cpu().numpy()
    for r in range(qe.shape[0]):
        e=ex[s+r]
        self_act=webp2act.get(e["image_path"], acts[topi[r,0]])   # exact ground-truth action, falling back to the top-1 estimate
        negs=[]
        for k in range(SIM_TOPK):
            if topv[r,k]<COS_MIN: break
            ai=acts[topi[r,k]]
            if ai and ai!=self_act and ai not in self_act and self_act not in ai:
                w=webps[topi[r,k]]
                if w not in negs: negs.append(w)
            if len(negs)>=N_ACTION_NEG: break
        e["action_negs"]=negs
        if not negs: empty+=1
        added+=len(negs)
    if s%20000==0: print(f"  {s}/{len(ex)} {time.time()-t0:.0f}s",flush=True)
print(f"[5] done: mean action_neg={added/len(ex):.2f}, empty={empty} ({100*empty/len(ex):.1f}%)",flush=True)
cache["meta"]["action_neg"]={"src":"capsim_minilm_v3","n":N_ACTION_NEG,"cos_min":COS_MIN}
OUTP=f"{OUT_W}/negcache_action_top8a6.pt"
torch.save(cache,OUTP)
print(f"[saved] {OUTP}",flush=True)
