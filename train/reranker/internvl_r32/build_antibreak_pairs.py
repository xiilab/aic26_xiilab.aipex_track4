"""internvl_r32/build_antibreak_pairs — anti-breakage pairs (the B_antibreak pairs).

Collects the cases where the zero-shot reranker **breaks a query the base encoder already got right**,
so training can prevent them. For queries whose base top-1 is the ground truth, the ground truth and
the top-K candidates are scored; whenever a wrong candidate outscores the ground truth, that pair is
recorded. This is the opposite direction to A_rescue (failure recovery), so merging both covers
recovery and regression together.

Output  <--out>/dpo_pairs_antibreak.jsonl {type:"B_antibreak", query, chosen, rejected, s_gt, s_neg}
        merged with A_rescue into `assets/data/mining/dpo_train.jsonl`, the r32 training input
Run     $PY_VLLM build_antibreak_pairs.py --gpu 6 --max-q 6000
"""
import os, sys, json, re, time, argparse, random
ap = argparse.ArgumentParser()
ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES"); ap.add_argument("--max-q", type=int, default=6000)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repository root
ap.add_argument("--topk", type=int, default=3); ap.add_argument("--base", default=os.environ.get("VLM_MODELS", f"{_REPO}/assets/model/vlm_models") + "/Qwen3.6-35B-A3B")
ap.add_argument("--out", default=os.environ.get("TRACK4", f"{_REPO}/assets/data/mining"), help="output directory")
ap.add_argument("--force", action="store_true", help="rebuild even if the output already exists")
a = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
from PIL import Image
random.seed(0)

JPG = os.environ.get("PAB_JPG", f"{_REPO}/assets/data/raw/pab_train/train_jpg_512")
SYS = ('Judge whether the Document (an image) shows exactly the person and scene described by the Query. '
       'Note that the answer can only be "yes" or "no".')
INSTR = "Given a detailed text description of a person, judge whether the candidate image matches it."
def img_path(rel):
    r = rel[len("train/"):] if rel.startswith("train/") else rel
    m = re.search(r"imgs_(\d+)/", r); part = int(m.group(1)) // 8 + 1
    return f"{JPG}/Part {part}/{r}"

# The anchor embedding cache supplies the successful queries and their top-K candidates
T4 = os.environ.get("TRACK4", f"{_REPO}/assets/data/mining")
SUB = f"{T4}/rerank_ft_subsample_150k.jsonl"
d = torch.load(os.environ.get("MINE_DIR", f"{_REPO}/assets/data/mining") + "/hardneg_anchor_mined_emb.pt"); I = d["I"].float(); T = d["T"].float()
I[~torch.isfinite(I).all(1)] = 0; T[~torch.isfinite(T).all(1)] = 0
rows = [json.loads(l) for l in open(SUB)]
ids = [r["image_id"] for r in rows]; caps = {r["image_id"]: r["caption"] for r in rows}
path = {}
POOL = os.environ.get("HARDNEG_POOL", f"{_REPO}/assets/data/mining/hardneg_pool_orig.jsonl")
for l in open(POOL):
    e = json.loads(l); path[e["image_id"]] = e["image_path"]
N = len(ids); Ig = I.cuda()
# successful queries (GT is top-1) plus their top-K candidates, GT excluded
succ = []
BLK = 1024
for s in range(0, N, BLK):
    e = min(s + BLK, N); cs = (T[s:e].cuda() @ Ig.t())
    pos = cs[torch.arange(e - s), torch.arange(s, e)]
    v, idx = cs.topk(a.topk + 1, dim=1); v = v.cpu(); idx = idx.cpu()
    for bi in range(e - s):
        qi = s + bi
        if int(idx[bi, 0]) != qi: continue       # successes only
        cand = [int(idx[bi, k]) for k in range(a.topk + 1) if int(idx[bi, k]) != qi][:a.topk]
        succ.append((qi, cand))
random.shuffle(succ); succ = succ[:a.max_q]
print(f"successful queries {len(succ):,} (scoring top{a.topk} candidates)", flush=True)

proc = AutoProcessor.from_pretrained(a.base, trust_remote_code=True, max_pixels=1280 * 28 * 28)
tok = proc.tokenizer; YES = tok.encode("yes", add_special_tokens=False)[0]; NO = tok.encode("no", add_special_tokens=False)[0]
print("loading the Qwen3.6 zero-shot scorer ...", flush=True)
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(a.base, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True).eval()

@torch.no_grad()
def score(query, stem_path):
    f = img_path(stem_path)
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": [{"type": "text", "text": f"<Instruct>: {INSTR}\n<Query>: {query[:600]}\n<Document>:"},
                                          {"type": "image", "image": f}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    b = proc(text=[text], images=[Image.open(f).convert("RGB")], return_tensors="pt")
    b = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in b.items()}
    last = model(**b).logits[0, -1]
    return torch.sigmoid(last[YES] - last[NO]).item()

OUT = os.path.join(a.out, "dpo_pairs_antibreak.jsonl")
os.makedirs(a.out, exist_ok=True)
if os.path.exists(OUT) and not a.force:
    raise SystemExit(f"already exists: {OUT}\n  pass --force to rebuild")
fo = open(OUT, "w"); nB = 0; t0 = time.time()
for n, (qi, cand) in enumerate(succ):
    iid = ids[qi]
    if iid not in path: continue
    q = caps[iid]; s_gt = score(q, path[iid])
    for cj in cand:
        if ids[cj] not in path: continue
        s_neg = score(q, path[ids[cj]])
        if s_neg > s_gt:        # the VLM ranks a wrong image above the GT = a breakage-inducing pair
            fo.write(json.dumps({"type": "B_antibreak", "query": q, "chosen": path[iid], "rejected": path[ids[cj]],
                                 "s_gt": round(s_gt, 4), "s_neg": round(s_neg, 4)}, ensure_ascii=False) + "\n"); fo.flush(); nB += 1
    if (n + 1) % 100 == 0:
        print(f"  {n+1}/{len(succ)} | B pairs {nB} | {(n+1)/(time.time()-t0):.2f}q/s", flush=True)
fo.close()
print(f"[done] {nB:,} B_antibreak pairs → {OUT}", flush=True)
