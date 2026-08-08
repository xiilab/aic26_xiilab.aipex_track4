#!/usr/bin/env python3
# ============================================================================
# gen_heldout_v1 - builds the held-out bench from PAB train (single file).
#
#   output (assets/data/heldout_v1/ by default):
#     heldout_images.txt   images to exclude from training  <- read by the trainer
#     split.json           exclusion list + main/hard bench + gates  <- read by the evaluator
#   intermediate data (index, embeddings, pairs, components) stays in memory only.
#
#   input (relative to the repository root):
#     ANN_DIR         75 annotation files -> image meta (category/scene/label/action)
#     IMG_ROOT        train_jpg_512      -> DINOv2 embedding source
#     RECAP_CSV_PATH  train_msr_v2.csv   -> query captions (style column = preset key)
#     DINOV2_MODEL    facebook/dinov2-base (assets/model/hf_cache)
#   the challenge test set is never read, by any path.
#
#   pipeline:
#     [1] annotation   -> index
#     [2] DINOv2 224px -> embeddings (N x 768 fp16, kept on GPU)
#     [3] cos >= 0.95  -> every near-duplicate pair (a top-k cut would miss pairs for images with more than k neighbours)
#     [4] union-find   -> connected component = group (the held-out unit)
#     [5] stratified sampling and repair -> query/gallery -> gates -> two files
#   annotation carries no video/clip id, so groups are derived from embeddings. Whole
#   components are removed, so no image of a held-out component remains in train.
#
#   reproducibility:
#     - build_split feeds one random.Random(SEED) to ten shuffles in a fixed order
#       (the consumption points are marked (1)-(10) in the body). Changing the call order,
#       the count, the target length, dict insertion order or in-place use yields a different dataset.
#     - the embeddings are fp16, so a different GPU architecture or EMB_BATCH can flip
#       pair decisions right at the cos 0.95 boundary.
#
#   usage (run from the repository root):
#     python train/gen/gen_heldout_v1.py --gpu 7            # build if missing
#     python train/gen/gen_heldout_v1.py --gpu 7 --force    # rebuild even if present
#   an existing output is skipped. HELDOUT_DIR overrides the output path.
# ============================================================================
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# =========================================================
# input / output
# =========================================================
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _abs(p: str) -> str:
    """Resolve a relative path against the repository root."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


ANN_DIR = os.path.join(_REPO, "assets/data/raw/pab_train/annotation/train")   # imgs_0~74.json (JSONL)
IMG_ROOT = os.path.join(_REPO, "assets/data/raw/pab_train/train_jpg_512")     # Part 1~10/imgs_N/{goal,full,wentwrong}
RECAP_CSV_PATH = _abs(os.environ.get("RECAP_CSV", "assets/data/raw/recaption/train_msr_v2.csv"))
OUT_DIR = _abs(os.environ.get("HELDOUT_DIR", "assets/data/heldout_v1"))

DINOV2_MODEL = "facebook/dinov2-base"
# offline HF cache; an existing HF_HOME is respected.
if "HF_HOME" not in os.environ:
    _c = os.path.join(_REPO, "assets/model/hf_cache")
    if os.path.isdir(f"{_c}/hub/models--{DINOV2_MODEL.replace('/', '--')}"):
        os.environ["HF_HOME"] = _c

# =========================================================
# parameters - changing a value yields a different dataset
# =========================================================
VERSION = "v1"
SEED = 42

EMB_BATCH = 512                     # a different batch can change the fp16 accumulation order
EMB_WORKERS = 16
EMB_SIZE = 224
PAIR_QCHUNK = 8192                  # pair-scan chunk; lower it if GPU memory is tight (the result is unchanged)
EMB_SIM_THR = 0.95                  # group (component) threshold
EMB_GATE_THR = 0.95                 # leak threshold

TARGET_HELDOUT_IMAGES = 45000       # gallery 36,773 + GT siblings + slack (~4.4% of the total)
MAX_COMP_SIZE = 64                  # huge components are excluded from the held-out candidates, avoiding sample bias
N_QUERIES = 2000                    # normal 1000 : anomaly 1000
GALLERY_SIZE = 36773                # GT 2,000 + distractor 34,773
ACTION_BUCKET_MIN = 100             # an action rarer than this goes into the "__tail__" bucket for stratification
REPAIR_MAX_ITERS = 8                # cap on the remove-and-refill repair iterations
HARD_QUERY_N = 2000                 # hard query cap (selected only from components that have siblings)

# query style - preset style keys from the recap CSV, split evenly over 10 (2,000/10=200).
#   p00_original (the source caption) and p11_compound are excluded from queries.
#   the length and order of this tuple feed the shared RNG stream ((8),(9)); changing it changes the bench.
STYLE_SET = ("p01_lexical", "p02_phrasal", "p03_clausal", "p04_diathesis", "p05_involved",
             "p06_informal", "p07_telegraphic", "p08_compact", "p09_narrative", "p10_formal")
ORIGINAL_STYLE = "p00_original"     # the source caption style, which must not leak into the queries (G5)
_ARTIFACTS = ("split.json", "heldout_images.txt")


# =========================================================
# helpers
# =========================================================
def map_local(ann_path: str) -> str:
    """'train/imgs_8/full/10.jpg' -> '{IMG_ROOT}/Part 2/imgs_8/full/10.jpg' (imgs_M -> Part M//8+1)."""
    p = ann_path.split("/")
    assert p[0] == "train" and len(p) == 4, ann_path
    n = int(p[1].replace("imgs_", ""))
    return f"{IMG_ROOT}/Part {n // 8 + 1}/{p[1]}/{p[2]}/{p[3]}"


def md5_of_list(items) -> str:
    """utf-8 + b'\\n' per item, the same rule as heldout_bench.md5_of_list."""
    h = hashlib.md5()
    for x in items:
        h.update(str(x).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def dedup(paths):
    """One image is registered twice in the annotation (train/imgs_0/full/2352.jpg).
    Dedup by path, **preserving order**."""
    seen, out = set(), []
    for x in paths:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================================================
# [1] annotation -> index (in memory)
# =========================================================
def build_index() -> list:
    files = sorted(glob.glob(f"{ANN_DIR}/imgs_*.json"),
                   key=lambda p: int(os.path.basename(p).split(".")[0].replace("imgs_", "")))
    if not files:
        raise SystemExit(f"[error] annotation not found: {ANN_DIR}/imgs_*.json")
    rows, cat, lab = [], Counter(), Counter()
    for f in files:
        for line in open(f, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            img = r["image"]                                  # train/imgs_N/cat/K.jpg
            c = img.split("/")[2]
            nl = (r.get("normal") or "").strip().lower()
            al = (r.get("anomaly") or "").strip().lower()
            lt = "anomaly" if al else ("normal" if nl else "none")
            rows.append({"image": img, "category": c,
                         "scene": (r.get("scene") or "").strip().lower(),
                         "label_type": lt, "action": al if al else nl})
            cat[c] += 1
            lab[lt] += 1
    print(f"  [1] index - {len(files)} files -> {len(rows):,} images  "
          f"cat={dict(cat)}  label={dict(lab)}", flush=True)
    return rows


# =========================================================
# [2] DINOv2 embeddings -> kept on GPU (N x 768 fp16, L2 normalised)
# =========================================================
def embed_images(rows: list):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel

    MEAN = np.array([0.485, 0.456, 0.406], np.float32)     # ImageNet
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    class ImgDS(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            import cv2
            cv2.setNumThreads(1)
            im = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
            if im is None:                                  # a missing image becomes black so the vector is zero
                im = np.zeros((EMB_SIZE, EMB_SIZE, 3), np.uint8)
            im = cv2.resize(im, (EMB_SIZE, EMB_SIZE), interpolation=cv2.INTER_AREA)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            im = (im - MEAN) / STD
            return torch.from_numpy(im.transpose(2, 0, 1))

    paths = [map_local(r["image"]) for r in rows]
    print(f"  [2] embed — {len(paths):,} images · {DINOV2_MODEL} · {EMB_SIZE}px "
          f"· batch {EMB_BATCH}", flush=True)
    model = AutoModel.from_pretrained(DINOV2_MODEL, dtype=torch.float16).to("cuda:0").eval()
    G = torch.zeros((len(paths), model.config.hidden_size), dtype=torch.float16, device="cuda:0")
    dl = DataLoader(ImgDS(paths), batch_size=EMB_BATCH, shuffle=False,
                    num_workers=EMB_WORKERS, pin_memory=True, drop_last=False)
    done, t0 = 0, time.time()
    with torch.inference_mode():
        for x in dl:
            x = x.to("cuda:0", dtype=torch.float16, non_blocking=True)
            f = model(pixel_values=x).last_hidden_state[:, 0]           # CLS
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            G[done:done + f.shape[0]] = f.half()
            done += f.shape[0]
            if done % (EMB_BATCH * 100) == 0:
                el = time.time() - t0
                print(f"      {done:,}/{len(paths):,}  {done / el:.0f} img/s  "
                      f"eta {(len(paths) - done) / max(1e-9, done / el) / 60:.1f}min", flush=True)
    del model
    torch.cuda.empty_cache()
    zero = int((G.abs().sum(1) == 0).sum())
    print(f"      {tuple(G.shape)} (zero-vector {zero}) ({time.time() - t0:.0f}s)", flush=True)
    return G


# =========================================================
# [3] every near-duplicate pair at cos >= EMB_SIM_THR (i<j)
# =========================================================
def find_pairs(G):
    import torch
    N = G.shape[0]
    t0 = time.time()
    print(f"  [3] pairs - N={N:,} cos >= {EMB_SIM_THR} full scan ...", flush=True)
    ii, jj, ss = [], [], []
    for s in range(0, N, PAIR_QCHUNK):
        e = min(s + PAIR_QCHUNK, N)
        sim = G[s:e] @ G.T
        r, c = torch.nonzero(sim >= EMB_SIM_THR, as_tuple=True)
        gr = r + s
        keep = gr < c                                   # upper triangle only
        gr, c = gr[keep], c[keep]
        if gr.numel():
            ii.append(gr.cpu().numpy().astype(np.int32))
            jj.append(c.cpu().numpy().astype(np.int32))
            ss.append(sim[r[keep], c].cpu().numpy().astype(np.float16))
        del sim
    pi = np.concatenate(ii) if ii else np.empty(0, np.int32)
    pj = np.concatenate(jj) if jj else np.empty(0, np.int32)
    ps = (np.concatenate(ss) if ss else np.empty(0, np.float16)).astype(np.float32)
    print(f"      {pi.size:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    return pi, pj, ps


# =========================================================
# [4] union-find -> connected component = group
# =========================================================
class _UF:
    def __init__(self, n):
        self.p = np.arange(n, dtype=np.int64)

    def find(self, x):
        p = self.p
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build_components(pi, pj, ps, n_img: int) -> np.ndarray:
    t0 = time.time()
    m = ps >= EMB_SIM_THR
    a, b = pi[m], pj[m]
    uf = _UF(n_img)
    for x, y in zip(a.tolist(), b.tolist()):
        uf.union(x, y)
    roots = np.array([uf.find(i) for i in range(n_img)], dtype=np.int64)
    _, comp = np.unique(roots, return_inverse=True)
    comp = comp.astype(np.int64)
    sizes = np.bincount(comp)
    multi = sizes[sizes > 1]
    print(f"  [4] components - {sizes.size:,} (multi {multi.size:,}, images {multi.sum():,}"
          f"={100.0 * multi.sum() / comp.size:.2f}%, max {sizes.max():,}) "
          f"({time.time() - t0:.0f}s)", flush=True)
    return comp


def cross_violations(pi, pj, ps, ho_idx, thr: float):
    """A boundary violation is a held-out image linked to a training image at cos >= thr.
    The pair list is exhaustive (never truncated), so this check is exact."""
    m = ps >= thr
    a, b = pi[m], pj[m]
    ho = np.zeros(int(max(a.max(initial=0), b.max(initial=0)) + 1), dtype=bool)
    ho_arr = np.fromiter(ho_idx, dtype=np.int64)
    ho[ho_arr[ho_arr < ho.size]] = True
    ina, inb = ho[a], ho[b]
    cross = ina ^ inb                      # one side held out = a boundary violation
    viol = set(a[cross & ina].tolist()) | set(b[cross & inb].tolist())
    return viol, int(cross.sum()), float(ps[m][cross].max()) if cross.any() else 0.0


# =========================================================
# [5] query style, captions, gates
# =========================================================
def query_style_plan(n: int) -> list:
    """Split n evenly over STYLE_SET (the remainder goes one at a time to the leading styles).
    **Uses no RNG** (deterministic).

    The return length is always n. The caller shuffles it with the shared RNG, so a
    different length would desynchronise the whole RNG stream that follows.
    """
    base, rem = divmod(n, len(STYLE_SET))
    plan = []
    for i, stl in enumerate(STYLE_SET):
        plan += [stl] * (base + (1 if i < rem else 0))
    return plan[:n]


def collect_captions(want: dict) -> dict:
    """Collect (image -> {style: caption}) in a single pass over the recap CSV. want = {image: set(styles)}."""
    if not os.path.exists(RECAP_CSV_PATH):
        raise SystemExit(f"[error] caption CSV not found: {RECAP_CSV_PATH}")
    got = defaultdict(dict)
    t0, n = time.time(), 0
    csv.field_size_limit(min(sys.maxsize, 1 << 31) - 1)
    with open(RECAP_CSV_PATH, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            img = r["image_path"]
            if img in want and r["style"] in want[img]:
                got[img][r["style"]] = r["caption"]
    print(f"      recap CSV {n:,} rows scanned ({time.time() - t0:.0f}s), "
          f"captions found {len(got):,}/{len(want):,} images", flush=True)
    return got


def run_gates(pi, pj, ps, idx, heldout_set, queries, hard_queries,
              gal_easy, gal_hard, ge_names, ghs_names) -> dict:
    """Quality gates. These become `stats.gates` in split.json, and the evaluator
    (heldout_bench.require_gates) enforces `all_pass`. If any gate fails, do not use this split."""
    g = {}
    train_idx = [i for i in range(len(idx)) if i not in heldout_set]
    g["g1_disjoint"] = len(set(train_idx) & heldout_set) == 0

    # G2: boundary near-duplicates - an independent recheck of the repair result.
    viol, cross, bmax = cross_violations(pi, pj, ps, sorted(heldout_set), EMB_GATE_THR)
    g["g2_cross_neardup_pairs"] = int(cross)
    g["g2_boundary_max_cos"] = round(bmax, 4)
    g["g2_sim_mode"] = "embedding(DINOv2)"
    g["g2_gate_thr"] = EMB_GATE_THR
    g["g2_violating_heldout_images"] = int(len(viol))
    g["g2_pass"] = cross == 0

    q_imgs = [q["image"] for q in queries]
    ge_set = {idx[i]["image"] for i in gal_easy}
    gh_set = {idx[i]["image"] for i in gal_hard}
    lt = Counter(q["label_type"] for q in queries)
    g["g3_query_unique"] = len(set(q_imgs)) == len(q_imgs)
    g["g3_query_distinct_components"] = len({q["component"] for q in queries}) == len(queries)
    g["g3_gt_in_gallery_easy"] = all(i in ge_set for i in q_imgs)
    g["g3_gt_in_gallery_hard"] = all(i in gh_set for i in q_imgs)
    g["g4_gallery_size_easy"] = len(gal_easy) == GALLERY_SIZE
    g["g4_gallery_size_hard"] = len(gal_hard) == GALLERY_SIZE
    g["g4_label_balance"] = dict(lt)
    g["g4_label_balanced"] = abs(lt.get("normal", 0) - lt.get("anomaly", 0)) <= 1
    g["g5_query_styles"] = dict(Counter(q["style"] for q in queries))
    g["g5_no_original_style"] = all(q["style"] != ORIGINAL_STYLE for q in queries)

    # G6/G7: style balance and query hygiene (main and hard separately).
    #   this check aborts on violation and is not written into split.json
    #   (the artifact schema is fixed).
    spreads = {}
    for tag, qs, gal in (("main", queries, ge_names), ("hard", hard_queries, ghs_names)):
        c = Counter(q["style"] for q in qs)
        spread = (max(c.values()) - min(c.values())) if c else 0
        spreads[tag] = spread
        if set(c) - set(STYLE_SET):
            raise RuntimeError(f"[G6] {tag}: style outside STYLE_SET {set(c) - set(STYLE_SET)}")
        if spread > 1:
            raise RuntimeError(f"[G6] {tag}: style imbalance (max-min={spread})")
        if len(set(q["qid"] for q in qs)) != len(qs):
            raise RuntimeError(f"[G7] {tag}: duplicate qid")
        if any(not q.get("caption") for q in qs):
            raise RuntimeError(f"[G7] {tag}: empty caption present")
        miss = [q["image"] for q in qs if q["image"] not in set(gal)]
        if miss:
            raise RuntimeError(f"[G7] {tag}: {len(miss)} GT missing from the gallery (e.g. {miss[:2]})")

    fails = [k for k, v in g.items() if k.endswith(("_pass", "_disjoint", "_unique", "_components",
                                                    "_easy", "_hard", "_balanced", "_style"))
             and v is False]
    g["all_pass"] = len(fails) == 0
    bm = (f"boundary cos>={EMB_GATE_THR} pairs 0" if cross == 0 else f"boundary max cos={bmax:.4f}")
    print(f"      gates: disjoint={g['g1_disjoint']}  cross-violation={cross} ({bm})  "
          f"shape={g['g4_gallery_size_easy']}/{g['g4_gallery_size_hard']}  label={dict(lt)}  "
          f"style spread {spreads['main']}/{spreads['hard']}  "
          f"→ all_pass={g['all_pass']}", flush=True)
    if fails:
        raise RuntimeError(f"gate failed: {fails}")
    return g


# =========================================================
# [6] split - component-level held-out sample + query/gallery -> two files
#     (1)-(10) mark where the shared RNG is consumed. Changing the order, count or length yields a different dataset.
# =========================================================
def build_split(idx: list, comp: np.ndarray, pi, pj, ps, out_dir: str) -> dict:
    assert comp.size == len(idx), f"components({comp.size}) != index({len(idx)})"
    print(f"  [5] split - component-level held-out sample (target={TARGET_HELDOUT_IMAGES:,}) ...",
          flush=True)
    rng = random.Random(SEED)                     # one shared RNG; (1)-(10) consume it in order
    total = len(idx)

    # ---- stratification key: (category, action bucket) ----
    act_freq = Counter(r["action"] for r in idx)
    strat_of_img = [(r["category"],
                     r["action"] if act_freq[r["action"]] >= ACTION_BUCKET_MIN else "__tail__")
                    for r in idx]
    train_dist = Counter(strat_of_img)

    # ---- candidate components (size <= MAX_COMP_SIZE) ----
    #   members/cand insertion order follows the first image index of each component, so it is deterministic.
    members = defaultdict(list)
    for i, c in enumerate(comp.tolist()):
        members[c].append(i)
    cand = {c: m for c, m in members.items() if len(m) <= MAX_COMP_SIZE}
    comp_key = {c: Counter(strat_of_img[i] for i in m).most_common(1)[0][0] for c, m in cand.items()}
    by_key = defaultdict(list)
    for c in cand:
        by_key[comp_key[c]].append(c)
    for k in by_key:
        rng.shuffle(by_key[k])                                            # (1)

    # ---- [hard reservation] reserve components that have siblings ----
    #   components with a sibling (same scene, different action) are rare, so random sampling
    #   does not produce a hard bench; reserve all of them.
    hard_comps = [c for c, m in cand.items() if len(m) > 1]
    rng.shuffle(hard_comps)                                               # (2)
    hard_comps = hard_comps[:HARD_QUERY_N]
    hard_comp_set = set(hard_comps)
    print(f"      hard reservation: components with siblings {len(hard_comps):,} / "
          f"images {sum(len(cand[c]) for c in hard_comps):,}", flush=True)
    for k in by_key:
        by_key[k] = [c for c in by_key[k] if c not in hard_comp_set]

    # ---- proportional stratified fill (main) ----
    quota = {k: TARGET_HELDOUT_IMAGES * v / total for k, v in train_dist.items()}
    picked, taken = [], 0
    for k in sorted(by_key, key=lambda k: -quota.get(k, 0)):
        need, got = int(round(quota.get(k, 0))), 0
        while by_key[k] and got < need:
            c = by_key[k].pop()
            picked.append(c)
            got += len(cand[c])
            taken += len(cand[c])
    leftovers = [c for k in by_key for c in by_key[k]]                    # top up the shortfall
    rng.shuffle(leftovers)                                                # (3)
    while taken < TARGET_HELDOUT_IMAGES and leftovers:
        c = leftovers.pop()
        picked.append(c)
        taken += len(cand[c])

    # ---- repair: drop boundary-violating groups whole and refill ----
    #   reverting per image would create new violations with siblings in the same group, so remove per **group**.
    repair_log = []
    for it in range(1, REPAIR_MAX_ITERS + 1):
        allc = picked + hard_comps
        heldout_idx = [i for c in allc for i in cand[c]]
        comp_of_img = {i: c for c in allc for i in cand[c]}
        viol, cross, bmax = cross_violations(pi, pj, ps, heldout_idx, EMB_GATE_THR)
        if not viol:
            repair_log.append({"iter": it, "violating_images": 0, "cross_pairs": 0,
                               "boundary_metric": bmax, "n_heldout": len(heldout_idx)})
            print(f"      repair it{it}: 0 violations (boundary cos>={EMB_GATE_THR} pairs 0) -> held out "
                  f"heldout {len(heldout_idx):,} images", flush=True)
            break
        bad = {comp_of_img[i] for i in viol}
        hard_comps = [c for c in hard_comps if c not in bad]
        picked = [c for c in picked if c not in bad]
        dropped = sum(len(cand[c]) for c in bad)
        taken = sum(len(cand[c]) for c in picked)
        refilled = 0
        while taken < TARGET_HELDOUT_IMAGES and leftovers:
            c = leftovers.pop()
            if c in bad:
                continue
            picked.append(c)
            taken += len(cand[c])
            refilled += len(cand[c])
        repair_log.append({"iter": it, "violating_images": len(viol), "cross_pairs": int(cross),
                           "bad_components": len(bad), "dropped_images": dropped,
                           "refilled_images": refilled})
        print(f"      repair it{it}: violating images {len(viol):,} / pairs {cross:,} -> "
              f"{len(bad):,} groups ({dropped:,} images) removed, {refilled:,} images refilled", flush=True)
    else:
        raise RuntimeError(f"repair did not converge within {REPAIR_MAX_ITERS} iterations.")

    main_idx = [i for c in picked for i in cand[c]]        # fixed before the in-place shuffle at (4)
    hard_idx = [i for c in hard_comps for i in cand[c]]
    heldout_idx = main_idx + hard_idx
    heldout_set = set(heldout_idx)
    print(f"      final: main {len(main_idx):,} images (components {len(picked):,}) + "
          f"hard {len(hard_idx):,} images (components {len(hard_comps):,}) = {len(heldout_idx):,} images "
          f"({100.0 * len(heldout_idx) / total:.2f}% of {total:,})", flush=True)

    # ---- query (GT) selection: at most one per component, normal:anomaly = 1:1 ----
    n_half = N_QUERIES // 2
    pool = {"normal": [], "anomaly": []}                   # the insertion order is the traversal order at (5)
    for c in picked:                     # main components only (the hard reservation is excluded)
        cm = cand[c]                     # not a copy - the shuffle mutates cand[c], and
        rng.shuffle(cm)                  #   that order decides the later sib -> gallery_hard.  (4)
        for i in cm:
            lt = idx[i]["label_type"]
            if lt in pool:
                pool[lt].append((c, i))
                break                       # one candidate per component
    for lt in pool:
        rng.shuffle(pool[lt])                                             # (5)
        if len(pool[lt]) < n_half:
            raise RuntimeError(f"{lt} GT candidates {len(pool[lt])} < {n_half}")
    gt = [x for lt in ("normal", "anomaly") for x in pool[lt][:n_half]]
    gt_comps = {c for c, _ in gt}
    gt_imgs = [i for _, i in gt]
    gt_set = set(gt_imgs)

    # ---- gallery ----
    sib = [i for c in gt_comps for i in cand[c] if i not in gt_set]       # near-duplicate siblings of the GT
    sib_set = set(sib)
    easy_pool = [i for i in main_idx if i not in sib_set and i not in gt_set]
    need_easy = GALLERY_SIZE - len(gt_imgs)
    if len(easy_pool) < need_easy:
        raise RuntimeError(f"not enough easy distractors: {len(easy_pool):,} < {need_easy:,}")
    rng.shuffle(easy_pool)                                                # (6)
    gallery_easy = list(gt_imgs) + easy_pool[:need_easy]
    # hard = GT + every sibling + a random fill for the rest
    hard = list(gt_imgs) + sib[:max(0, GALLERY_SIZE - len(gt_imgs))]
    hard_set = set(hard)
    fill = [i for i in easy_pool if i not in hard_set]
    gallery_hard = hard + fill[:max(0, GALLERY_SIZE - len(hard))]
    n_gt_with_sib = len({c for c, i in gt if len(cand[c]) > 1})

    # ---- [hard set] queries restricted to images that have siblings (diagnostic bench) ----
    hard_gt, hard_sib = [], []
    for c in [c for c in hard_comps if len(cand[c]) > 1][:HARD_QUERY_N]:
        mem = list(cand[c])              # hard_comps is not a target of (4), so shuffle a copy
        rng.shuffle(mem)                                                  # (7)
        hard_gt.append((c, mem[0]))
        hard_sib.extend(mem[1:])
    print(f"      hard set: queries {len(hard_gt):,} + sibling distractors {len(hard_sib):,}", flush=True)

    # ---- query captions (recap CSV) ----
    style_plan = query_style_plan(len(gt_imgs))
    rng.shuffle(style_plan)                                               # (8)
    hard_style_plan = query_style_plan(len(hard_gt)) if hard_gt else []
    rng.shuffle(hard_style_plan)                                          # (9)
    want = defaultdict(set)
    for (c, i), stl in list(zip(gt, style_plan)) + list(zip(hard_gt, hard_style_plan)):
        want[idx[i]["image"]].add(stl)
    caps = collect_captions(want)

    queries, missing = [], 0
    for qn, ((c, i), stl) in enumerate(zip(gt, style_plan), 1):
        cap = caps.get(idx[i]["image"], {}).get(stl)
        if not cap:
            missing += 1
            continue
        queries.append({"qid": f"hq{qn:05d}", "image": idx[i]["image"], "style": stl,
                        "caption": cap, "label_type": idx[i]["label_type"],
                        "category": idx[i]["category"], "scene": idx[i]["scene"],
                        "action": idx[i]["action"], "component": int(c)})
    if missing:
        print(f"      [warn] {missing} queries without a caption excluded -> queries {len(queries):,}", flush=True)
    hard_queries = []
    for qn, ((c, i), stl) in enumerate(zip(hard_gt, hard_style_plan), 1):
        cap = caps.get(idx[i]["image"], {}).get(stl)
        if not cap:
            continue
        hard_queries.append({"qid": f"hh{qn:05d}", "image": idx[i]["image"], "style": stl,
                             "caption": cap, "label_type": idx[i]["label_type"],
                             "category": idx[i]["category"], "scene": idx[i]["scene"],
                             "action": idx[i]["action"], "component": int(c),
                             "n_siblings": len(cand[c]) - 1})

    hard_gt_imgs = [i for _, i in hard_gt]
    _hg, _hs = set(hard_gt_imgs), set(hard_sib)
    hard_pool = [i for i in main_idx if i not in _hg and i not in _hs]
    rng.shuffle(hard_pool)                                                # (10)
    gallery_hardset = (list(hard_gt_imgs) + list(hard_sib)
                       + hard_pool[:max(0, GALLERY_SIZE - len(hard_gt_imgs) - len(hard_sib))])

    # ---- gates ----
    heldout_images = dedup([idx[i]["image"] for i in heldout_idx])
    ge = dedup([idx[i]["image"] for i in gallery_easy])
    gh = dedup([idx[i]["image"] for i in gallery_hard])
    ghs = dedup([idx[i]["image"] for i in gallery_hardset])
    gates = run_gates(pi, pj, ps, idx, heldout_set, queries, hard_queries,
                      gallery_easy, gallery_hard, ge, ghs)

    # ---- distribution report ----
    def top(counter, n, denom):
        return {k: round(100.0 * v / max(1, denom), 2) for k, v in counter.most_common(n)}

    nho = len(heldout_idx)
    dist = {
        "category_train": top(Counter(r["category"] for r in idx), 5, total),
        "category_heldout": top(Counter(idx[i]["category"] for i in heldout_idx), 5, nho),
        "label_train": top(Counter(r["label_type"] for r in idx), 5, total),
        "label_heldout": top(Counter(idx[i]["label_type"] for i in heldout_idx), 5, nho),
        "action_top10_train": top(Counter(r["action"] for r in idx), 10, total),
        "action_top10_heldout": top(Counter(idx[i]["action"] for i in heldout_idx), 10, nho),
    }

    n_cand_imgs = sum(len(m) for m in cand.values())
    stats = {
        "n_heldout": len(heldout_images),
        "n_candidate_images": n_cand_imgs,
        "pct_candidate_of_total": round(100.0 * n_cand_imgs / max(1, total), 2),
        "excluded_from_candidacy": total - n_cand_imgs,   # a huge component (blob) cannot be held out
        "n_components": len(picked),
        "n_queries": len(queries),
        "n_gallery_easy": len(gallery_easy),
        "n_gallery_hard": len(gallery_hard),
        "n_hard_queries": len(hard_queries),
        "n_hard_siblings": len(hard_sib),
        "mean_siblings_per_hard_query": round(
            float(np.mean([q["n_siblings"] for q in hard_queries])) if hard_queries else 0.0, 2),
        "n_gt_with_neardup_sibling": n_gt_with_sib,
        "pct_gt_with_sibling": round(100.0 * n_gt_with_sib / max(1, len(gt_imgs)), 2),
        "n_siblings_total": len(sib),
        "heldout_md5": md5_of_list(sorted(heldout_images)),
        "query_md5": md5_of_list([f"{q['qid']}|{q['image']}|{q['style']}"
                                  for q in queries + hard_queries]),
        "repair_log": repair_log,
        "gates": gates,
        "dist": dist,
    }
    # the key set and its order are the artifact schema. built_at (a timestamp) and config
    #    (absolute paths) vary per run and are left out.
    out = {
        "version": VERSION,
        "stats": stats,
        "queries": queries,
        "queries_hard": hard_queries,
        "gallery_hardset": ghs,
        "heldout_images": heldout_images,
        "gallery_easy": ge,
        "gallery_hard": gh,
    }
    # atomic write: an interrupted run leaves no half-written artifact
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/split.json.partial", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    with open(f"{out_dir}/heldout_images.txt.partial", "w", encoding="utf-8") as f:
        for im in heldout_images:
            f.write(im + "\n")
    for name in _ARTIFACTS:
        os.replace(f"{out_dir}/{name}.partial", f"{out_dir}/{name}")
    print(f"      heldout={len(heldout_images):,} · queries={len(queries):,} · "
          f"hard={len(hard_queries):,} · gallery {len(ge):,}/{len(gh):,}/{len(ghs):,} · "
          f"md5={stats['heldout_md5'][:8]}", flush=True)
    return stats


# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="build the PAB train held-out bench from scratch (single file; the test set is not used)")
    ap.add_argument("--gpu", default="0", help="GPU for the DINOv2 embeddings")
    ap.add_argument("--force", action="store_true", help="rebuild even if it already exists")
    args = ap.parse_args()

    have = [n for n in _ARTIFACTS if os.path.exists(os.path.join(OUT_DIR, n))]
    if len(have) == len(_ARTIFACTS) and not args.force:
        n_ho = sum(1 for _ in open(os.path.join(OUT_DIR, "heldout_images.txt"), "rb"))
        sz = os.path.getsize(os.path.join(OUT_DIR, "split.json")) / 2 ** 20
        print(f"[heldout {VERSION}] already present - skipped")
        print(f"             {os.path.relpath(OUT_DIR, _REPO)}  "
              f"split.json {sz:.1f} MiB · heldout_images.txt {n_ho:,} lines")
        print("             rebuild with --force")
        return

    model_dir = os.path.join(os.environ.get("HF_HOME", ""), "hub",
                             "models--" + DINOV2_MODEL.replace("/", "--"))
    problems = [f"{what} not found: {os.path.relpath(p, _REPO)}"
                for p, what in ((ANN_DIR, "annotation"), (IMG_ROOT, "image root"),
                                (RECAP_CSV_PATH, "caption CSV"), (model_dir, "HF model cache"))
                if not os.path.exists(p)]
    if problems:
        raise SystemExit("[input check failed] fix the following and run again.\n"
                         + "\n".join(f"  - {p}" for p in problems))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    # must be decided before torch is imported

    t0 = time.time()
    print(f"[heldout {VERSION}] out={os.path.relpath(OUT_DIR, _REPO)}  gpu={args.gpu}")
    print(f"             recap={os.path.relpath(RECAP_CSV_PATH, _REPO)}  "
          f"style={len(STYLE_SET)} preset  seed={SEED}")
    rows = build_index()
    G = embed_images(rows)
    pi, pj, ps = find_pairs(G)
    del G
    import torch
    torch.cuda.empty_cache()                          # release the embedding GPU memory
    comp = build_components(pi, pj, ps, len(rows))
    stats = build_split(rows, comp, pi, pj, ps, OUT_DIR)
    print(f"\n[gen_heldout_{VERSION}] done ({time.time() - t0:.0f}s)  "
          f"heldout={stats['n_heldout']:,} · md5={stats['heldout_md5'][:8]}")
    for name in _ARTIFACTS:
        print(f"  → {os.path.relpath(OUT_DIR, _REPO)}/{name}")


if __name__ == "__main__":
    main()
