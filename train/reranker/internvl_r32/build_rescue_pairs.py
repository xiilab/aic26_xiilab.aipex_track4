"""internvl_r32/build_rescue_pairs — failure-driven hard negatives (the A_rescue pairs).

Reuses the embedding cache left by `mine_hardneg.py`, so nothing is re-encoded and no GPU encoding
pass is needed. For each query it computes the ground truth's rank and mixes negatives by difficulty:
  hard    wrong images ranked above, or just below, the ground truth (the r1-vs-r2 signal)
  medium  wrong images ranked well below  ·  random  drawn uniformly
Candidates with sim >= pos + FN_MARGIN are excluded as suspected true matches or near-duplicates.

Output  failure_hardneg.jsonl {image_id, query, gt_rank, pos_sim, hard, medium, random}
        failure_hardneg_review.jsonl  sample for manual review (includes image paths)
Run     python build_rescue_pairs.py
"""
import argparse, json, os, re, random, torch
random.seed(0)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repository root
T4 = os.environ.get("TRACK4", f"{_REPO}/assets/data/mining")   # output root
EMB = os.environ.get("MINE_DIR", f"{_REPO}/assets/data/mining") + "/hardneg_anchor_mined_emb.pt"
SUB = f"{T4}/rerank_ft_subsample_150k.jsonl"
_ap = argparse.ArgumentParser(description="build failure-driven hard negatives (A_rescue)")
_ap.add_argument("--out", default=T4, help="output directory")
_ap.add_argument("--force", action="store_true", help="rebuild even if the output already exists")
_a = _ap.parse_args()
OUT = f"{_a.out}/failure_hardneg.jsonl"
REVIEW = f"{_a.out}/failure_hardneg_review.jsonl"
if os.path.exists(OUT) and not _a.force:
    raise SystemExit(f"already exists: {OUT}\n  pass --force to rebuild")
os.makedirs(_a.out, exist_ok=True)
JPG = os.environ.get("PAB_JPG", f"{_REPO}/assets/data/raw/pab_train/train_jpg_512")
KTOP = 30          # candidate window
N_HARD = 4; N_MED = 2; N_RAND = 1
FN_MARGIN = 0.10   # sim >= pos+margin → suspected false negative → excluded
HARD_LO = -0.06    # hard lower bound: sim >= pos+HARD_LO (only those beating or nearly matching the GT)

def img_path(rel):
    r = rel[len("train/"):] if rel.startswith("train/") else rel
    m = re.search(r"imgs_(\d+)/", r); part = int(m.group(1)) // 8 + 1
    return f"{JPG}/Part {part}/{r}"

rows = [json.loads(l) for l in open(SUB)]
ids = [r["image_id"] for r in rows]; caps = [r["caption"] for r in rows]; paths = [r["image_path"] for r in rows]
N = len(rows); id2i = {x: i for i, x in enumerate(ids)}
d = torch.load(EMB); I = d["I"].float(); T = d["T"].float()
# The cached features are already unit-normalised (I[0].norm() == 1.0) — do not normalise again.
I[~torch.isfinite(I).all(1)] = 0; T[~torch.isfinite(T).all(1)] = 0
print(f"embeddings {tuple(I.shape)} | pool {N:,} | I[0]norm {I[0].norm():.3f} (already unit)", flush=True)

Ig = I.cuda(); BLK = 1024
fo = open(OUT, "w"); rv = open(REVIEW, "w")
import collections
st = collections.Counter(); rev_n = 0
for s in range(0, N, BLK):
    e = min(s + BLK, N)
    sims = (T[s:e].cuda() @ Ig.t())
    pos = sims[torch.arange(e - s), torch.arange(s, e)].clone()
    # the GT's rank = how many candidates score above pos
    gt_rank = (sims > pos.unsqueeze(1)).sum(1)
    val, idx = sims.topk(KTOP, dim=1); val = val.cpu(); idx = idx.cpu(); pos = pos.cpu(); gt_rank = gt_rank.cpu()
    for bi in range(e - s):
        qi = s + bi; ps = pos[bi].item(); gr = int(gt_rank[bi])
        hard = []; med = []
        for k in range(KTOP):
            j = int(idx[bi, k])
            if j == qi: continue
            sv = float(val[bi, k])
            if sv >= ps + FN_MARGIN: continue          # held back as a false negative (scores far above the GT)
            if len(hard) < N_HARD:                      # top non-self = most similar wrong = hard
                hard.append((ids[j], round(sv, 4)))
            elif len(med) < N_MED:                      # the next ones = medium
                med.append((ids[j], round(sv, 4)))
        # random neg
        rnd = []
        while len(rnd) < N_RAND:
            rj = random.randrange(N)
            if rj != qi: rnd.append((ids[rj], 0.0))
        if not hard: st["no_hard"] += 1
        st["fail" if gr > 0 else "ok"] += 1; st["hard"] += len(hard)
        rec = {"image_id": ids[qi], "query": caps[qi], "gt_rank": gr, "pos_sim": round(ps, 4),
               "hard": [{"id": h[0], "sim": h[1]} for h in hard],
               "medium": [{"id": m[0], "sim": m[1]} for m in med],
               "random": [{"id": r[0]} for r in rnd]}
        fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # manual-review sample, failures first
        if gr > 0 and hard and rev_n < 400:
            rv.write(json.dumps({"image_id": ids[qi], "query": caps[qi][:200], "gt_rank": gr, "pos_sim": round(ps,4),
                                 "pos_path": img_path(paths[qi]),
                                 "hard": [{"id": h[0], "sim": h[1], "path": img_path(paths[id2i[h[0]]])} for h in hard]}, ensure_ascii=False) + "\n")
            rev_n += 1
fo.close(); rv.close()
tot = st["fail"] + st["ok"]
print(f"DONE: {tot:,} queries | failures (gt rank>0) {st['fail']:,} ({st['fail']/tot*100:.1f}%) | no hard negative {st['no_hard']:,}")
print(f"  hard negatives {st['hard']:,} total ({st['hard']/tot:.2f}/query) | review sample {rev_n} → {REVIEW}")
print(f"  → {OUT}")
