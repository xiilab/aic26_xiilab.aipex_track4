#!/usr/bin/env python3
"""qwen3vl_2b/build_negcache — hard image-negative cache from the anchor encoder's top-1 failures.

Train-set mining: collects the samples the anchor encoder fails to retrieve
(top-1 ≠ ground truth) over train images and train captions. (=infer.encode_test_gallery, infer.encode_test_queries)

The middle of the three data steps: manifest → **negcache** → training. It is what makes training
see negatives drawn from the same distribution as deployment.

Procedure  encode the working set (train images plus original captions) with the anchor encoder, run
           text-to-image retrieval, and keep **only the samples whose top-1 is not the ground truth**.
           That sample's negatives are the top-{N_IMG_NEG} retrieved results after removing itself and
           any near-duplicate (image-image cos >= NEARDUP_TAU). A "false failure", where every
           distractor is a near-duplicate, is discarded as noise rather than signal.

Two resolutions are involved: encoding reads train_jpg_512, since the anchor takes 512px input and so
needs no downscaling, while the training paths written into the cache point at train_webp, because the
reranker sees 1024px.

Output  NEG_CACHE = {meta, examples:[{image_path, pos, flip_negs, axes, img_negs}]}
        A meta that differs from the current settings triggers an automatic rebuild.
Run     python build_negcache.py --gpu 6      (peft is not used)
"""
import argparse, importlib, json, os, sys, time
import numpy as np
import torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repository root — all default paths are relative to it

# ---- Config — these must match train.py, or the NEG_CACHE path will not line up ----
SEED            = 42
HF_CACHE        = os.environ.get("HF_CACHE", f"{_REPO}/assets/model/hf_cache")
DATA_ROOT       = os.environ.get("PAB_TRAIN", f"{_REPO}/assets/data/raw/pab_train")
IMG_ROOT_512    = f"{DATA_ROOT}/train_jpg_512"     # for anchor encoding (512px input)
IMG_ROOT_WEBP   = f"{DATA_ROOT}/train_webp"        # written into the cache for training (Qwen 1024px)
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", f"{_REPO}/assets/runs/rerank_qw2b")   # the manifest lives here too
RERANK_MANIFEST = f"{OUTPUT_DIR}/pab_manifest_rerank_msrv2_hardneg.jsonl"
N_FLIP_NEG      = 3                                 # cap on Path A flip negatives, taken from the manifest

# anchor encoder — used both to decide failure and to mine negatives
ANCHOR_DIR  = os.environ.get("ENCODER_CKPT", f"{_REPO}/assets/model/encoder/siglip_mining")   # bundled anchor adapter
ANCHOR_CKPT = "ep06"                                                                     # appears in the cache filename and meta
POOL_SIZE   = 0             # size of the working set, which is both the queries and the search pool. 0 = all (~967k; ~2.5 h to encode, ~19k failures)
N_IMG_NEG   = int(os.environ.get("N_IMG_NEG", "5"))   # 5 for qwen3vl_2b, 8 for jina; top{N} is part of the output filename
NEARDUP_TAU = 0.85          # image-image sim >= tau → near-duplicate (possibly the same person) → not used as a negative
TOPK_BUFFER = 10            # headroom for dropping self and near-duplicates: top-(N_IMG_NEG+BUFFER) is fetched
POOL_TAG    = "full" if not POOL_SIZE else str(POOL_SIZE)
NEG_CACHE   = f"{OUTPUT_DIR}/negcache_hardimg_{ANCHOR_CKPT}_pool{POOL_TAG}_top{N_IMG_NEG}_tau{NEARDUP_TAU}.pt"


def _rel(ann, ext):
    """ "train/imgs_N/sub/id.jpg" → "Part {N//8+1}/imgs_N/sub/id{ext}" — the form stored in the
    cache. Keeping it relative to train_webp is what lets the dataset sit anywhere; the trainers
    join it with their own IMG_ROOT_WEBP."""
    p = ann.split("/"); n = int(p[1].replace("imgs_", ""))
    fn = (os.path.splitext(p[3])[0] + ext) if ext else p[3]
    return f"Part {n // 8 + 1}/{p[1]}/{p[2]}/{fn}"


def _map(ann, root, ext):
    """Absolute path under `root` — used to actually read images while mining."""
    return f"{root}/{_rel(ann, ext)}"


def load_working_set():
    """manifest → images that have an orig caption, flips, and a jpg512 file. Each: {ann, pos, flip_negs, axes}."""
    import random
    rng = random.Random(SEED)
    rows = []
    with open(RERANK_MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "_meta" in r or not (r.get("orig_caption") and r.get("flip_captions")):
                continue
            rows.append(r)
    rng.shuffle(rows)
    W = []
    for r in rows:
        if os.path.exists(_map(r["image"], IMG_ROOT_512, "")):       # encodable?
            W.append({"ann": r["image"], "pos": r["orig_caption"],
                      "flip_negs": r["flip_captions"][:N_FLIP_NEG], "axes": (r.get("flip_axes") or [])[:N_FLIP_NEG]})
        if POOL_SIZE and len(W) >= POOL_SIZE:                        # 0 = all
            break
    print(f"[ws] working set M={len(W):,} (this is also the search pool)", flush=True)
    return W


def main():
    global OUTPUT_DIR, RERANK_MANIFEST, NEG_CACHE
    ap = argparse.ArgumentParser(description="build the hard image-negative cache from the anchor's top-1 failures")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--out", default=OUTPUT_DIR, help="output directory")
    ap.add_argument("--force", action="store_true", help="rebuild even when meta matches")
    a = ap.parse_args()
    if a.out != OUTPUT_DIR:
        OUTPUT_DIR = a.out
        RERANK_MANIFEST = f"{OUTPUT_DIR}/pab_manifest_rerank_msrv2_hardneg.jsonl"
        NEG_CACHE = f"{OUTPUT_DIR}/negcache_hardimg_{ANCHOR_CKPT}_pool{POOL_TAG}_top{N_IMG_NEG}_tau{NEARDUP_TAU}.pt"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.environ.setdefault("HF_HOME", HF_CACHE)
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    device = "cuda:0"

    # a cache at NEG_CACHE whose meta differs from this one is rebuilt
    meta = {"method": "top1fail", "anchor_run": os.path.basename(ANCHOR_DIR), "anchor_ckpt": ANCHOR_CKPT,
            "pool": POOL_SIZE, "n_img_neg": N_IMG_NEG, "neardup_tau": NEARDUP_TAU, "seed": SEED}
    if os.path.isfile(NEG_CACHE) and not a.force:
        try:
            if torch.load(NEG_CACHE, map_location="cpu")["meta"] == meta:
                print(f"[negcache] up-to-date → {NEG_CACHE}", flush=True); return
        except Exception:
            pass
        print("[negcache] meta mismatch → rebuilding", flush=True)

    W = load_working_set()
    M = len(W)

    # anchor encoding (train_jpg_512)
    em = importlib.import_module("eval_v25"); infer = em.infer_v25
    run_dir = os.path.abspath(ANCHOR_DIR)
    mod = infer.init(os.path.join(em.HERE, em.MODEL2MODULE[em.read_meta_model_name(run_dir)]),
                     EVAL_BATCH_SIZE=48, EVAL_NUM_WORKERS=4)
    model, tok = infer.load_for_inference(mod.MODEL_NAME, run_dir, device=device)
    eval_tf = infer.build_eval_transform(mod.IMAGE_SIZE)
    print(f"[negcache] IMG_SIZE={mod.IMAGE_SIZE} encoding M={M:,} ...", flush=True)
    # Train-set mining through the anchor module's encode helpers — the train images / captions
    # are simply passed in the helpers' test_gallery / query-record input layout.
    G, _ = infer.encode_test_gallery(model, [_map(w["ann"], IMG_ROOT_512, "") for w in W], eval_tf)   # (M,D) norm
    qrows = [{"query_index": str(i), "caption": w["pos"], "change": ""} for i, w in enumerate(W)]
    Q, Q_idx, _, _ = infer.encode_test_queries(model, qrows, tok)                                     # (M,D)
    del model; torch.cuda.empty_cache()
    order = [int(x) for x in Q_idx]
    if order != list(range(M)):
        Qi = torch.empty_like(Q); Qi[torch.tensor(order)] = Q; Q = Qi

    # decide failure with text→image, then mine negatives (the near-dup filter is image→image)
    Gd = G.to(device); take = N_IMG_NEG + TOPK_BUFFER
    examples = []
    n_fail = n_dropped_neardup = 0
    CH = 1024; t0 = time.time()
    for s in range(0, M, CH):
        sims = Q[s:s+CH].to(device) @ Gd.T                                   # (chunk, M) text→image
        self_ids = torch.arange(s, s + sims.size(0), device=device)
        self_sim = sims[torch.arange(sims.size(0), device=device), self_ids]
        ranks = (sims > self_sim.unsqueeze(1)).sum(dim=1)                     # rank of X (0=top-1)
        topv, topi = sims.topk(min(take, M), dim=1)                          # candidates (chunk, take)
        G_top = Gd[topi]                                                      # (chunk, take, D)
        self_emb = Gd[self_ids]                                              # (chunk, D)
        img_sim = (G_top * self_emb.unsqueeze(1)).sum(-1)                     # (chunk, take) image-image cos
        topi = topi.cpu().numpy(); img_sim = img_sim.cpu().numpy(); ranks = ranks.cpu().numpy()
        for r in range(topi.shape[0]):
            gi = s + r
            if ranks[r] == 0:                                                # top-1 is the GT → not a failure, skip
                continue
            n_fail += 1
            negs = []
            for k in range(topi.shape[1]):
                j = int(topi[r, k])
                if j == gi:
                    continue
                if img_sim[r, k] >= NEARDUP_TAU:                             # near-duplicate → skip
                    continue
                negs.append(j)
                if len(negs) >= N_IMG_NEG:
                    break
            if not negs:                                                     # all near-duplicates = false failure → discard
                n_dropped_neardup += 1; continue
            w = W[gi]
            examples.append({
                "image_path": _rel(w["ann"], ".webp"),        # relative to <PAB_TRAIN>/train_webp
                "pos": w["pos"], "flip_negs": w["flip_negs"], "axes": w["axes"],
                "img_negs": [_rel(W[j]["ann"], ".webp") for j in negs],
                "rank_of_X": int(ranks[r]),
            })
        if (s // CH) % 20 == 0:
            print(f"  [mine] {min(s+CH,M):,}/{M:,} fail={n_fail:,} kept={len(examples):,} ({time.time()-t0:.0f}s)", flush=True)
    del Gd; torch.cuda.empty_cache()

    avg_neg = np.mean([len(e["img_negs"]) for e in examples]) if examples else 0
    print(f"\n[negcache] M={M:,} | failures {n_fail:,} ({100*n_fail/M:.2f}%) | "
          f"near-dup only, discarded {n_dropped_neardup:,} | kept {len(examples):,} | img_neg/example avg={avg_neg:.2f}", flush=True)
    torch.save({"meta": meta, "examples": examples}, NEG_CACHE)
    print(f"[negcache] saved {len(examples):,} examples → {NEG_CACHE}", flush=True)


if __name__ == "__main__":
    main()
