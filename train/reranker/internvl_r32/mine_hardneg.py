"""internvl_r32/mine_hardneg — mine r32 hard negatives with a SigLIP2 anchor over a 150k subsample.

Train-set mining: collects the candidates the anchor encoder ranks above the ground
truth, over a 150k subsample of train images and captions. (=infer_v25.encode_test_gallery, infer_v25.encode_test_queries)

In caption-to-image retrieval, the top-ranked non-ground-truth candidates (r2 distractors) become
hard negatives. Candidates with sim >= pos + delta may themselves be correct, so they are treated as
false negatives and dropped.

Output  hardneg_anchor_mined.jsonl {image_id, pos_sim, neg_image_ids, sims}
        hardneg_anchor_mined_emb.pt  embedding cache, reused by build_rescue_pairs and build_antibreak_pairs
Run     python mine_hardneg.py --gpu 6 --bs 48
"""
import argparse, os, sys, time, json, re
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repository root

ANCHOR = os.environ.get("ENCODER_CKPT", f"{_REPO}/assets/model/encoder/siglip_mining")   # bundled anchor adapter
MODULE = f"{ANCHOR}/train_v25_l512.py"                                              # single source for the architecture
JPG = os.environ.get("PAB_JPG", f"{_REPO}/assets/data/raw/pab_train/train_jpg_512")
T4 = os.environ.get("TRACK4", f"{_REPO}/assets/data/mining")   # output root
SUB = f"{T4}/rerank_ft_subsample_150k.jsonl"
MINE = os.environ.get("MINE_DIR", f"{_REPO}/assets/data/mining")   # mining outputs = where the mining data lives
OUT = f"{MINE}/hardneg_anchor_mined.jsonl"
EMB = f"{MINE}/hardneg_anchor_mined_emb.pt"                         # reused by the pair builders

ap = argparse.ArgumentParser(description="mine r32 hard negatives with the SigLIP2 anchor")
ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
ap.add_argument("--out", default=MINE, help="output directory")
ap.add_argument("--force", action="store_true", help="re-mine even if the output already exists")
ap.add_argument("--ckpt", default=ANCHOR, help="anchor adapter directory"); ap.add_argument("--bs", type=int, default=48)
ap.add_argument("--kprime", type=int, default=64); ap.add_argument("--n", type=int, default=8)
ap.add_argument("--delta", type=float, default=0.05)
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
OUT = f"{a.out}/hardneg_anchor_mined.jsonl"
EMB = f"{a.out}/hardneg_anchor_mined_emb.pt"
if os.path.exists(OUT) and not a.force:
    raise SystemExit(f"already exists: {OUT}\n  pass --force to re-mine")
os.makedirs(a.out, exist_ok=True)

import torch                                                         # noqa: E402  (keeps --help light)
import torch.nn.functional as F                                      # noqa: E402
sys.path.insert(0, os.environ.get("TRACK4_CODE", _REPO))
import infer_v25                                                     # noqa: E402
dev = "cuda:0"

def img_path(rel):
    r = rel[len("train/"):] if rel.startswith("train/") else rel
    m = re.search(r"imgs_(\d+)/", r); part = int(m.group(1)) // 8 + 1
    return f"{JPG}/Part {part}/{r}"

rows = [json.loads(l) for l in open(SUB)]
ids = [r["image_id"] for r in rows]; paths = [img_path(r["image_path"]) for r in rows]; caps = [r["caption"] for r in rows]
N = len(rows); print(f"mining pool {N:,}", flush=True)

if os.path.exists(EMB):
    d = torch.load(EMB); I, T = d["I"], d["T"]; print("embedding cache", I.shape, flush=True)
else:
    mod = infer_v25.init(MODULE, EVAL_BATCH_SIZE=a.bs, EVAL_NUM_WORKERS=4)
    print(f"[anchor] MODEL={mod.MODEL_NAME} IMG={mod.IMAGE_SIZE} ckpt={a.ckpt}", flush=True)
    tf = infer_v25.build_eval_transform(mod.IMAGE_SIZE)
    model, tok = infer_v25.load_for_inference(mod.MODEL_NAME, a.ckpt, device=dev)
    t0 = time.time()
    # Train-set mining through the anchor module's encode helpers — the train images / captions
    # are simply passed in the helpers' test_gallery / query-record input layout.
    G, _ = infer_v25.encode_test_gallery(model, paths, tf, device=dev, batch_size=a.bs)
    print(f"[anchor] images {tuple(G.shape)} {time.time()-t0:.0f}s", flush=True)
    queries = [{"caption": c, "query_index": ids[i], "change": ""} for i, c in enumerate(caps)]
    Q, *_ = infer_v25.encode_test_queries(model, queries, tok, device=dev, batch_size=a.bs)
    print(f"[anchor] caps {tuple(Q.shape)} {time.time()-t0:.0f}s", flush=True)
    I = G.float().cpu(); T = Q.float().cpu()            # store raw float32 (avoids half overflow and missed normalisation)
    print(f"  raw norm I~{I[0].norm():.1f} T~{T[0].norm():.1f} (normalised at use)", flush=True)
    torch.save({"I": I, "T": T}, EMB); print("saved embeddings (raw float32)", flush=True)
    I = F.normalize(I, -1); T = F.normalize(T, -1)

Ig = I.float().to(dev); BLK = 1024; fo = open(OUT, "w"); nq = 0; t0 = time.time()
for s in range(0, N, BLK):
    e = min(s + BLK, N)
    sims = (T[s:e].float().to(dev) @ Ig.t())
    possim = sims[torch.arange(e - s), torch.arange(s, e)].clone()
    val, idx = sims.topk(a.kprime, dim=1); val = val.cpu(); idx = idx.cpu(); possim = possim.cpu()
    for bi in range(e - s):
        qi = s + bi; ps = possim[bi].item(); negs = []
        for k in range(a.kprime):
            j = int(idx[bi, k])
            if j == qi: continue
            sv = float(val[bi, k])
            if sv >= ps + a.delta: continue
            negs.append((ids[j], round(sv, 4)))
            if len(negs) >= a.n: break
        if negs:
            fo.write(json.dumps({"image_id": ids[qi], "pos_sim": round(ps, 4),
                                 "neg_image_ids": [x[0] for x in negs], "sims": [x[1] for x in negs]}, ensure_ascii=False) + "\n"); nq += 1
    if (s // BLK) % 20 == 0: print(f"  mine {e:,}/{N:,} | written {nq:,} | {time.time()-t0:.0f}s", flush=True)
fo.close()
print(f"[done] hard negatives for {nq:,} queries → {OUT}", flush=True)
