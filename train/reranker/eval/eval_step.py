#!/usr/bin/env python3
"""eval_step — select a fine-tuned reranker's best **step** by pair accuracy.

Rule  t = argmax  ½·( z(acc_ucc) + z(acc_clean) )      Path B pairs; z is standardised along the step axis
      ucc   = negatives are base top-20 candidates, the distribution deployment actually sees → generalisation
      clean = the untrained tail of the member's own negcache (index >= SKIP)                 → fit to the training objective

## Stages

    1. pair bench    assets/data/benches/rerankstep/rerank_step_pairs.json  (built automatically if absent)
    2. candidate pool <work>/gallery · query_text.json · union_pool.pt   ← the scorer input contract
    3. score per step <work>/<member>_<step>_n<N>_pairs.pt
    4. selection      <work>/steps_<member>.json
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
S2 = os.path.join(PKG, "pipeline", "S2_rerank")
ASSETS = os.path.join(PKG, "assets", "data")
MINING = os.path.join(ASSETS, "mining")
BENCHES = os.path.join(PKG, "assets", "data", "benches")
BENCH_JSON = os.path.join(BENCHES, "rerankstep", "rerank_step_pairs.json")

# member → scorer · ckpt flag · extra args · interpreter · negcache · index where the untrained tail starts
#   SKIP = the number of examples the deployed run consumed (last step in train.log x batch)
MEMBERS = {
    "r32": dict(
        script="score_union_hf_4b.py", ckpt_flag="--adapter",
        extra=["--model", os.environ.get("VLM_MODELS", f"{PKG}/assets/model/vlm_models")
               + "/InternVL3_5-30B-A3B-HF"],
        py=os.environ.get("PY_VLLM", "python"),
        clean=("jsonl", os.environ.get("DPO_POOL", os.path.join(MINING, "dpo_pairs_beit3.jsonl")),
               os.path.join(MINING, "dpo_train.jsonl"), 0),
        ckpt_subdir="", adopted="step2500"),
    "dora": dict(
        script="score_union_qwen_4b.py", ckpt_flag="--adapter", extra=["--qwen", "2b"],
        py=os.environ.get("PY_VLLM", "python"),
        clean=("cache", os.path.join(
            MINING, "negcache_hardimg_ep06_poolfull_top5_tau0.85.pt"), None, 24000),
        ckpt_subdir="checkpoints", adopted="ex007000"),
    "jina": dict(
        script="score_union_jina.py", ckpt_flag="--ckpt", extra=[],
        py=os.environ.get("PY_CONDA", sys.executable),
        clean=("cache", os.path.join(
            MINING, "negcache_action_top8a6.pt"), None, 30000),
        ckpt_subdir="checkpoints", adopted="ex008000"),
}
# ucc-derived pool (base scores, candidates, labels). The source data is assets/data/raw/ucc/; this is the bench derived from it.
UCC_POOL = os.environ.get("UCC_POOL",
                          os.path.join(BENCHES, "ucc", "ucc_champion_pool.pt"))
TRAIN_ROOT = os.environ.get("PAB_TRAIN_JPG",
                            os.path.join(ASSETS, "raw", "pab_train", "train_jpg_512"))
SEED_CACHE = 42        # shuffle seed of the two negcache trainers
SEED_DPO = 0           # sampling seed for the r32 complement set (unrelated to the trainer; local to this script)


# ────────────────────────────── 1. pair bench ──────────────────────────────
def _md5_head(path: str, cap: int = 64 << 20) -> str:
    """md5 over the first `cap` bytes plus the file length, to avoid reading multi-GB mining files whole."""
    h = hashlib.md5()
    h.update(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(cap))
    return h.hexdigest()


def _remap(p: str) -> str:
    """`train/imgs_M/…` or `imgs_M/…` → `<TRAIN_ROOT>/Part N/imgs_M/…`."""
    parts = [x for x in p.split("/") if x]
    if parts and parts[0] == "train":
        parts = parts[1:]
    return os.path.join(TRAIN_ROOT, f"Part {int(parts[0].split('_')[1]) // 8 + 1}", *parts)


def _build_ucc(n: int) -> list[dict]:
    """Take the first positive and the first negative from each query's base top-20 candidates."""
    p = torch.load(UCC_POOL, weights_only=False)
    gal, caps, cand, lab = p["gal_paths"], p["caps"], p["cand"], p["label"]
    out = []
    for i in range(len(caps)):
        pos = [int(c) for c, y in zip(cand[i], lab[i]) if y]
        neg = [int(c) for c, y in zip(cand[i], lab[i]) if not y]
        if pos and neg:
            out.append({"cap": caps[i], "pos": gal[pos[0]], "neg": gal[neg[0]]})
        if len(out) >= n:
            break
    return out


def _build_clean(spec: tuple, n: int) -> list[dict]:
    """Draw pairs from the untrained region: jsonl = full pool minus what was trained; cache = the tail after reproducing the shuffle."""
    kind, src, aux, skip = spec
    if kind == "jsonl":
        trained = set()
        with open(aux, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                trained.add((d["query"], d["chosen"], d["rejected"]))
        comp = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if (d["query"], d["chosen"], d["rejected"]) not in trained:
                    comp.append(d)
        random.Random(SEED_DPO).shuffle(comp)
        print(f"    {len(comp):,} untrained pairs")
        return [{"cap": d["query"], "pos": _remap(d["chosen"]), "neg": _remap(d["rejected"])}
                for d in comp[:n]]
    ex = torch.load(src, map_location="cpu")["examples"]
    random.Random(SEED_CACHE).shuffle(ex)                 # same as the trainer
    held = ex[skip:]
    print(f"    total {len(ex):,} · untrained {len(held):,} (index>={skip:,})")
    return [{"cap": e["pos"], "pos": e["image_path"], "neg": e["img_negs"][0]}
            for e in held if e.get("img_negs")][:n]


def build_bench(n: int, out: str) -> dict:
    """Build ucc plus every clean_<member> split as one reproducible asset."""
    splits, prov = {}, {}
    todo = [("ucc", UCC_POOL, lambda: _build_ucc(n))]
    for m, M in MEMBERS.items():
        todo.append((f"clean_{m}", M["clean"][1], lambda M=M: _build_clean(M["clean"], n)))
    for name, src, fn in todo:
        if not os.path.exists(src):
            print(f"[{name}] ⚠ source missing → skip: {src}")
            continue
        print(f"[{name}] {os.path.basename(src)}")
        splits[name] = fn()
        prov[name] = {"source": src, "md5_head64m": _md5_head(src), "n": len(splits[name])}
        print(f"    → {len(splits[name]):,} pairs")
    doc = {"meta": {"version": "v1",
                    "protocol": "PathB: anchor=caption, pos=GT image, neg=hard distractor",
                    "metric": "pair-acc = P[s(cap,pos) > s(cap,neg)]",
                    "rule": "argmax 1/2*(z(acc_ucc)+z(acc_clean))",
                    "seeds": {"cache_shuffle": SEED_CACHE, "dpo_shuffle": SEED_DPO},
                    "provenance": prov},
           "splits": splits}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)
    print(f"saved → {out}  ({os.path.getsize(out)/1e6:.1f} MB)")
    return doc


# ────────────────────────────── 2. candidate pool ──────────────────────────────
def emit_pool(splits: dict, work: str) -> dict:
    """Pair list → the scorer input contract (gallery symlinks, query_text.json, union_pool.pt).

    Gallery filenames are zero-padded to 8 digits, so `sorted(listdir)` order equals the candidate index.
    """
    imgs, order = {}, []
    for name in sorted(splits):
        for r in splits[name]:
            for p in (r["pos"], r["neg"]):
                if p not in imgs:
                    imgs[p] = len(order)
                    order.append(p)
    gal = os.path.join(work, "gallery")
    os.makedirs(gal, exist_ok=True)
    for p in os.listdir(gal):
        os.unlink(os.path.join(gal, p))
    for p, i in imgs.items():
        os.symlink(p, os.path.join(gal, f"{i:08d}{os.path.splitext(p)[1]}"))

    qorder, union, meta, lines = [], [], {}, []
    for name in sorted(splits):
        for i, r in enumerate(splits[name]):
            q = f"{name}:{i}"
            qorder.append(q)
            union.append([imgs[r["pos"]], imgs[r["neg"]]])
            meta[q] = {"split": name, "pos": imgs[r["pos"]], "neg": imgs[r["neg"]]}
            lines.append(json.dumps({"query_index": q, "caption": r["cap"]}, ensure_ascii=False))
    torch.save({"union": union, "qorder": qorder}, os.path.join(work, "union_pool.pt"))
    with open(os.path.join(work, "query_text.json"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(work, "pairs_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    # report which dataset each gallery image comes from
    src = collections.Counter()
    for p in order:
        t = os.path.realpath(p)
        src["PAB train" if "PAB_Track4/train_" in t else
            "UCF-Crime (ucc)" if "ucc_local" in t else f"other {t.rsplit('/', 3)[0]}"] += 1
    print("[pool] sources " + " · ".join(f"{k} {v:,}" for k, v in src.most_common()))

    listed = sorted(os.listdir(gal))                  # re-check the indices the scorer will see
    assert len(listed) == len(order), f"gallery {len(listed)} ≠ images {len(order)}"
    for i, fn in enumerate(listed):
        assert int(os.path.splitext(fn)[0]) == i, f"index mismatch {fn} ≠ {i}"
    print(f"[pool] {work}\n       gallery {len(order):,} · queries {len(qorder):,} · "
          f"pairs {sum(len(u) for u in union):,}")
    return meta


def subset_pool(work: str, member: str, n: int, meta: dict) -> str:
    """Pool file keeping only the first n queries per split, from `ucc` and `clean_<member>`.

    The gallery is left intact: candidate indices refer to the whole gallery and the scorer opens only
    the images it needs.
    """
    U = torch.load(os.path.join(work, "union_pool.pt"), weights_only=False)
    want = {"ucc", f"clean_{member}"}
    keep_q, keep_u, cnt = [], [], {s: 0 for s in want}
    for q, u in zip(U["qorder"], U["union"]):
        s = meta[q]["split"]
        if s in want and cnt[s] < n:
            cnt[s] += 1
            keep_q.append(q)
            keep_u.append(u)
    name = f"pool_{member}_n{n}.pt"
    torch.save({"union": keep_u, "qorder": keep_q}, os.path.join(work, name))
    print("[pool] " + name + " — " + " · ".join(f"{s} {cnt[s]:,}" for s in sorted(want))
          + f"  ({len(keep_q):,} queries · {2*len(keep_q):,} pairs in total)")
    return name


# ────────────────────────── 3. scoring · 4. selection ──────────────────────────
def score_step(member: str, work: str, pool_file: str, ckpt: str, name: str, gpu: str) -> str:
    """Invoke the pipeline scorer unchanged → <work>/<name>_pairs.pt."""
    M = MEMBERS[member]
    out = os.path.join(work, f"{name}_pairs.pt")
    if os.path.exists(out):
        print(f"    [skip] already present → {os.path.basename(out)}")
        return out
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "WORKDIR": work, "POOL_FILE": pool_file,
                "PAB_TEST": work, "GALLERY": os.path.join(work, "gallery"),
                "QUERY_TEXT": os.path.join(work, "query_text.json"),
                "OUT_SUFFIX": "pairs", "REUSE_EXTRA": ""})
    cmd = [M["py"], os.path.join(S2, M["script"]), "--name", name, M["ckpt_flag"], ckpt] + M["extra"]
    print(f"    $ {M['script']} --name {name} {M['ckpt_flag']} {ckpt} {' '.join(M['extra'])}",
          flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=S2)
    if r.returncode != 0 or not os.path.exists(out):
        raise SystemExit(f"✗ scoring failed (rc={r.returncode}) — {name}")
    print(f"    [ok] {time.time()-t0:.0f}s")
    return out


def measure(cache: str, meta: dict) -> dict:
    """Scorer output {(q,c):score} → pair-acc, margin and 1σ per split."""
    S = torch.load(cache, weights_only=False)["scores"]
    agg = {}
    for q, m in meta.items():
        sp, sn = S.get((q, m["pos"])), S.get((q, m["neg"]))
        if sp is None or sn is None:
            continue
        a = agg.setdefault(m["split"], {"n": 0, "good": 0, "margin": 0.0})
        a["n"] += 1
        a["good"] += int(sp > sn)
        a["margin"] += sp - sn
    return {s: {"n": a["n"], "acc": round(a["good"] / a["n"], 4),
                "margin": round(a["margin"] / a["n"], 4),
                "se": round(math.sqrt((a["good"]/a["n"]) * (1-a["good"]/a["n"]) / a["n"]), 4)}
            for s, a in agg.items()}


def zscore(d: dict) -> dict:
    v = list(d.values())
    if len(v) < 2:
        return {k: 0.0 for k in d}
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) or 1.0
    return {k: (x - m) / s for k, x in d.items()}


def main():
    ap = argparse.ArgumentParser(description="select a reranker's best step by pair accuracy")
    ap.add_argument("--member", choices=list(MEMBERS), help="optional when --build-only is given")
    ap.add_argument("--run", default=None, help="run directory holding the step checkpoints")
    ap.add_argument("--steps", default="", help="comma separated (e.g. ex001000,ex002000)")
    ap.add_argument("--ckpt-subdir", default=None, help="default is per member (empty string for r32)")
    ap.add_argument("--n", type=int, default=400, help="queries per split")
    ap.add_argument("--bench-n", type=int, default=2000, help="pairs per split stored in the bench")
    ap.add_argument("--bench", default=BENCH_JSON)
    ap.add_argument("--work", default=os.environ.get("RUNS_ROOT", f"{PKG}/assets/runs") + "/rerank_step_pairs")
    ap.add_argument("--gpu", default="7")
    ap.add_argument("--adopted", default=None, help="default is the member's deployed step")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the pair bench")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--deploy-rep", default=None, metavar="NAME",
                    help="deploy the selection to assets/model_rep/reranker/<NAME> (tools/promote.py --rep)")
    a = ap.parse_args()

    # 1. bench
    if a.rebuild or not os.path.exists(a.bench):
        doc = build_bench(a.bench_n, a.bench)
    else:
        doc = json.load(open(a.bench, encoding="utf-8"))
        print(f"[bench] {a.bench} — " + " · ".join(
            f"{k} {len(v):,}" for k, v in doc["splits"].items()))
    # 2. candidate pool
    os.makedirs(a.work, exist_ok=True)
    meta_path = os.path.join(a.work, "pairs_meta.json")
    if a.rebuild or not os.path.exists(meta_path):
        meta = emit_pool(doc["splits"], a.work)
    else:
        meta = json.load(open(meta_path, encoding="utf-8"))
    if a.build_only:
        return
    if not a.member:
        raise SystemExit("✗ --member is required unless --build-only is given")

    M = MEMBERS[a.member]
    sub = a.ckpt_subdir if a.ckpt_subdir is not None else M["ckpt_subdir"]
    adopted = a.adopted or M["adopted"]
    clean = f"clean_{a.member}"
    pool_file = subset_pool(a.work, a.member, a.n, meta)
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    if not steps:
        raise SystemExit("✗ --steps is empty")

    # 3. scoring
    rows = []
    for st in steps:
        name = f"{a.member}_{st}_n{a.n}"
        cache = os.path.join(a.work, f"{name}_pairs.pt")
        print(f"\n[{st}]")
        if not a.measure_only:
            ck = os.path.join(a.run, sub, st) if sub else os.path.join(a.run, st)
            if not os.path.exists(ck):
                print(f"    ⚠ ckpt missing → skip: {ck}")
                continue
            score_step(a.member, a.work, pool_file, ck, name, a.gpu)
        if a.score_only or not os.path.exists(cache):
            continue
        m = measure(cache, meta)
        if "ucc" not in m or clean not in m:
            print(f"    ⚠ split missing {list(m)} → skip")
            continue
        rows.append({"step": st, **m})
        print(f"    ucc   acc {m['ucc']['acc']:.4f} ±{m['ucc']['se']:.4f}  margin {m['ucc']['margin']:+.4f}")
        print(f"    clean acc {m[clean]['acc']:.4f} ±{m[clean]['se']:.4f}  margin {m[clean]['margin']:+.4f}")
    if a.score_only or not rows:
        if not rows and not a.score_only:
            print("\nno step was scored.")
        return

    # 4. selection
    zu = zscore({r["step"]: r["ucc"]["acc"] for r in rows})
    zc = zscore({r["step"]: r[clean]["acc"] for r in rows})
    for r in rows:
        r["z_mean"] = round((zu[r["step"]] + zc[r["step"]]) / 2, 4)
    pick = max(rows, key=lambda r: r["z_mean"])["step"]

    print("\n" + "=" * 62)
    print(f"{'step':>12} | {'ucc':>8} | {'clean':>8} | {'z mean':>8}")
    print("-" * 62)
    for r in rows:
        mark = ([" ← pick"] if r["step"] == pick else []) + (["deployed"] if r["step"] == adopted else [])
        print(f"{r['step']:>12} | {r['ucc']['acc']:>8.4f} | {r[clean]['acc']:>8.4f} | "
              f"{r['z_mean']:>8.3f}" + " ".join(mark))
    print("-" * 62)
    print(f"pick = {pick}")
    print(f"  deployed = {adopted} → {'match ✓' if pick == adopted else 'mismatch ✗'}")
    print(f"  ⚠ argmax of clean alone = {max(rows, key=lambda r: r[clean]['acc'])['step']} "
          f"(not to be used on its own — see the docstring)")

    out_path = a.out or os.path.join(a.work, f"steps_{a.member}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"member": a.member, "bench": a.bench, "run": a.run, "n_per_split": a.n,
                   "rule": "argmax 1/2*(z(acc_ucc)+z(acc_clean))",
                   "pick": pick, "adopted": adopted, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"saved → {out_path}")

    if a.deploy_rep:                                 # deploy the selection, only if the inputs are intact
        if not pick:
            raise SystemExit("[deploy-rep] ✗ no step was selected — not deploying")
        src = os.path.join(a.run, sub, pick) if sub else os.path.join(a.run, pick)
        if not a.run or not os.path.exists(src):
            raise SystemExit(f"[deploy-rep] ✗ selected ckpt not found → not deploying: {src}")
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        print(f"[deploy-rep] {pick} → assets/model_rep/reranker/{a.deploy_rep}")
        rc = subprocess.run([sys.executable, os.path.join(_repo, "tools", "promote.py"),
                             "--rep", "--kind", "reranker", "--name", a.deploy_rep, "--src", src]).returncode
        if rc:
            raise SystemExit(f"[deploy-rep] ✗ promote failed (rc={rc})")
        print(f"[deploy-rep] ✓ build the cache_rep cache per model with pipeline/S2_rerank/score_union_*.py --rep --name {a.deploy_rep}")


if __name__ == "__main__":
    main()
