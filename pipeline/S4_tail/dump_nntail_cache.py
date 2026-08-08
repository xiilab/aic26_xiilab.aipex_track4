#!/usr/bin/env python3
"""dump_nntail_cache — score the tail candidates S4b pulls in -> `*_nntail_cache.pt`.

The S2 union re-scoring (`../S2_rerank/score_union_*.py`) only covers (q,c) pairs inside the union
pool, but S4b (NN completion) inserts near-duplicate candidates from outside it at ranks 8-10, so
S4d (cons6 propagation) finds no reranker score for those. This cache fills that gap.

  order:     S3 -> S4a -> S4b -> (this script) -> S4c -> S4d
  rerankers: internvl_r32 · jina_m0 · llama  (the three that reinforce `RK` in S4d)

It collects the (q,c) pairs that are in the answer but not in the union cache, builds a mini pool
from them, runs the matching scorer once over that pool and saves the result as
`{"scores": {(q,c): float}}`.

Output: default assets/cache/s4_tail/ · `--rep` assets/cache_rep/s4_tail/

  # pass the working directory that S4b has been run in
  python dump_nntail_cache.py --name internvl_r32 --work .work --rep
  python dump_nntail_cache.py --name jina_m0      --work .work --rep
  python dump_nntail_cache.py --name llama        --work .work --rep

Environment: PAB_TEST · GALLERY · QUERY_INDEX · VLM_MODELS · PY_VLLM (reranker interpreter)
"""
import argparse
import os
import subprocess
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402
from utils.env import resolve_python                            # noqa: E402

S2 = os.path.join(_REPO, "pipeline", "S2_rerank")
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GALLERY = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
QUERY_INDEX = os.environ.get("QUERY_INDEX", f"{PAB_TEST}/query_index.txt")
VLM = os.environ.get("VLM_MODELS", f"{_REPO}/assets/model/vlm_models")
# The interpreter depends on the model family - the MoE model (internvl_r32) needs torch 2.11+cu130.
# resolve_python() verifies a candidate by actually calling the op before the scorer is started.

# S4b answer (tail_refinement.S4B_NAME); its candidates are what gets scored
S4B_NAME = "answer_tailoverlay_r10ft_m0.5_nn0.8c4_noext.txt"

# reranker -> (script, extra args, adapter flag name; None = zero-shot)
SCORERS = {
    "internvl_r32": ("score_union_hf_4b.py",
                     ["--model", f"{VLM}/InternVL3_5-30B-A3B-HF", "--name", "internvl_r32"], "--adapter"),
    "llama":        ("score_union_hf_4b.py",
                     ["--model", f"{VLM}/Llama-3.2-11B-Vision-Instruct", "--name", "llama"], None),
    "jina_m0":      ("score_union_jina.py", ["--name", "jina_m0"], "--ckpt"),
}


def main():
    ap = argparse.ArgumentParser(description="score the new S4b tail candidates -> nntail cache")
    ap.add_argument("--name", required=True, choices=sorted(SCORERS))
    ap.add_argument("--work", required=True, help="working directory that S4b has been run in (holds the answer and union caches)")
    ap.add_argument("--answer", default=S4B_NAME, help=f"input answer (default {S4B_NAME})")
    ap.add_argument("--out", default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild even if the artifact exists (default: skip)")
    ap.add_argument("--rep", action="store_true", help="reproduction output -> cache_rep/s4_tail (model_rep adapter)")
    a = ap.parse_args()

    cache = f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}"
    out = a.out or f"{cache}/s4_tail/{a.name}_nntail_cache.pt"
    work = os.path.abspath(a.work)
    skip_if_exists(out, a.overwrite)

    gal = sorted(os.listdir(GALLERY))
    gstem = [g[:-4] if g.endswith(".jpg") else g for g in gal]
    s2i = {s: i for i, s in enumerate(gstem)}
    qx = [l.strip() for l in open(QUERY_INDEX) if l.strip()]

    apath = os.path.join(work, a.answer)
    if not os.path.exists(apath):
        raise SystemExit(f"[nntail] answer not found: {apath}\n"
                         f"  Run up to S4b first: python pipeline/S4_tail/tail_refinement.py --stage b")
    rows = [l.split() for l in open(apath) if l.strip()]
    if len(rows) != len(qx):
        raise SystemExit(f"[nntail] answer rows {len(rows)} != queries {len(qx)}")
    ans = [[s2i[t[:-4] if t.endswith('.jpg') else t] for t in r] for r in rows]

    ucache = os.path.join(work, f"{a.name}_union_cache.pt")
    if not os.path.exists(ucache):
        raise SystemExit(f"[nntail] union cache not found: {ucache} - finish the S2 re-scoring first.")
    scored = set(torch.load(ucache, weights_only=False)["scores"])

    # pairs in the answer but not in the union cache = the candidates S4b pulled in
    miss = [[c for c in ans[i] if (qx[i], int(c)) not in scored] for i in range(len(qx))]
    n_miss = sum(len(m) for m in miss)
    print(f"[nntail] {a.name}: unscored {n_miss} pairs / answer {sum(len(r) for r in ans)} pairs", flush=True)
    if n_miss == 0:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        torch.save({"scores": {}}, out)
        print(f"[nntail] nothing to score -> wrote empty cache {out}")
        return

    # mini pool in the scorer's input format ({union, qorder}); kept in a scratch dir so the union
    # cache is left untouched
    scratch = os.path.join(work, f"_nntail_{a.name}")
    os.makedirs(scratch, exist_ok=True)
    torch.save({"union": miss, "qorder": qx}, os.path.join(scratch, "nntail_pool.pt"))

    script, extra, adapter_flag = SCORERS[a.name]
    cmd = [resolve_python(" ".join(extra)), os.path.join(S2, script)] + extra
    if adapter_flag:
        root = "model_rep" if a.rep else "model"
        ck = f"{_REPO}/assets/{root}/reranker/{a.name}"
        if os.path.isdir(ck):
            cmd += [adapter_flag, ck]
    env = dict(os.environ,
               WORKDIR=scratch, POOL_FILE="nntail_pool.pt", OUT_SUFFIX="nntail_raw")
    print(f"[nntail] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, env=env).returncode
    if rc:
        raise SystemExit(f"[nntail] scorer failed (rc={rc})")

    raw = os.path.join(scratch, f"{a.name}_nntail_raw.pt")
    scores = torch.load(raw, weights_only=False)["scores"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"scores": dict(scores)}, out)
    print(f"[nntail] {len(scores)} pairs -> {out}", flush=True)


if __name__ == "__main__":
    main()
