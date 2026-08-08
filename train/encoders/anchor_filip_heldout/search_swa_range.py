#!/usr/bin/env python3
"""Find the SWA epoch range from the held-out metrics.

The range is searched on the **heldout run**, which trained without the held-out images and
therefore has unbiased metrics. It is then applied to the **all run** (the deployed model),
which cannot be scored directly because it has memorised the same bench.

  this script  : <heldout_run> -> range [lo, hi]        (search only)
  build_swa.py : <all_run> lo hi -> checkpoints/swa     (lives in <enc>_all/)

The metrics come from the repository held-out split (assets/data/heldout_v1) scored with
train/encoders/eval/eval_heldout_swa.py; the challenge test set is never read. Per-epoch
results are kept in <run>/heldout_eval/ep{NN}.json, reused when present and computed otherwise.

usage (run from the repository root):
  python train/encoders/<enc>_heldout/search_swa_range.py <heldout_run> [--gpu N]
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
EVALUATOR = os.environ.get(
    "HELDOUT_EVAL", os.path.join(_REPO, "train/encoders/eval/eval_heldout_swa.py"))
TRAINER = os.path.join(HERE, "train.py")
EVAL_SUBDIR = "heldout_eval"      # results stay inside the run directory
HELDOUT_DIR = os.environ.get("HELDOUT_DIR", os.path.join(_REPO, "assets/data/heldout_v1"))

CAP = 10                # highest epoch considered
FLOOR = 2               # ep1 is warmup and is excluded
AFTER = 2               # how many checkpoints past the target to include
BENCH = "hard"          # held-out bench the decision is based on
KEYS = ("R@1", "R@5", "R@10")
NEG = "negmax_cos_mean"


def _abs(p: str) -> str:
    """Resolve a relative path against the repository root."""
    return p if os.path.isabs(p) else os.path.join(_REPO, p)


def load_evals(d: str) -> dict:
    """<run>/heldout_eval/ep*.json -> {epoch: eval dict}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "ep*.json"))):
        try:
            j = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if "ep" in j and BENCH in j:
            out[int(j["ep"])] = j
    return out


def epochs_of(run: str) -> list:
    """The epoch number of a checkpoints/epNN directory."""
    return sorted(int(os.path.basename(x)[2:])
                  for x in glob.glob(os.path.join(run, "checkpoints", "ep[0-9][0-9]")))


def run_eval(run: str, gpu: str, epochs: list) -> None:
    """Pass only the unscored epochs to the evaluator. The evaluator also skips existing results."""
    spec = ",".join(str(e) for e in epochs)
    cmd = [sys.executable, EVALUATOR, "--trainer", TRAINER, "--run", run,
           "--benches", BENCH, "--gpu", gpu, "--epochs", spec]
    print(f"[eval] {len(epochs)} unscored (ep{epochs[0]:02d}-ep{epochs[-1]:02d}) "
          f"-> running the evaluator, 15-20 min per epoch")
    print(f"       {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def se_of(row: dict, key: str) -> float:
    """The 1-sigma standard error of R@k (pt), from the binomial."""
    p = row[key] / 100.0
    return 100 * math.sqrt(max(p * (1 - p), 1e-12) / row["n_query"])


def onset(R: dict, eps: list) -> int:
    """The largest per-key epoch at which 'maximum - 1 sigma' is first reached."""
    out = []
    still_rising = max(FLOOR, eps[-1] - AFTER)
    for k in KEYS:
        c = {e: R[e][BENCH][k] for e in eps}
        m = max(c.values())
        if min(e for e in eps if c[e] == m) == eps[-1]:
            out.append(still_rising)
        else:
            tol = se_of(R[eps[-1]][BENCH], k)
            out.append(max(FLOOR, next(e for e in eps if c[e] >= m - tol)))
    return max(out)


def decide(R: dict) -> tuple:
    """target = max(FLOOR, min(saturation onset, argmin negmax)); range = [target, target+AFTER] intersect [FLOOR, CAP]"""
    eps = [e for e in sorted(R) if e <= CAP]
    if not eps:
        raise SystemExit(f"[error] no scored epoch <= {CAP}")
    a = onset(R, eps)
    n = min(eps, key=lambda e: R[e][BENCH][NEG])
    hi = min(CAP, eps[-1])
    lo = max(FLOOR, min(max(FLOOR, min(a, n)), hi))
    lo = min(lo, max(FLOOR, hi - AFTER))
    return eps, a, n, list(range(lo, min(lo + AFTER, hi) + 1))


def main():
    ap = argparse.ArgumentParser(
        description="search the SWA epoch range from the heldout metrics (the test set is not used)")
    ap.add_argument("run", help="heldout run directory (must hold checkpoints/epNN)")
    ap.add_argument("--gpu", default="0", help="GPU to use if scoring is needed (default 0)")
    args = ap.parse_args()

    run = _abs(args.run)
    problems = []
    if not os.path.isdir(run):
        problems.append(f"path not found: {run}")
    elif not os.path.isdir(os.path.join(run, "checkpoints")):
        problems.append(f"not a run directory (no checkpoints/): {run}")
    elif not glob.glob(os.path.join(run, "checkpoints", "ep[0-9][0-9]")):
        problems.append(f"checkpoints/epNN not found: {run}")
    for path, what in ((TRAINER, "trainer"), (EVALUATOR, "evaluator"),
                       (os.path.join(HELDOUT_DIR, "split.json"), "heldout split")):
        if not os.path.isfile(path):
            problems.append(f"{what} not found: {path}")
    if problems:
        raise SystemExit("[input check failed] fix the following and run again.\n"
                         + "\n".join(f"  - {p}" for p in problems))
    results = os.path.join(run, EVAL_SUBDIR)

    want = [e for e in epochs_of(run) if e <= CAP]      # only needed up to the candidate cap
    R = load_evals(results)
    todo = [e for e in want if e not in R]
    if todo:
        run_eval(run, args.gpu, todo)
        R = load_evals(results)
        still = [e for e in want if e not in R]
        if still:
            raise SystemExit(f"[error] ep{still} '{BENCH}' metrics were not produced: {results}")
        src = f"{len(todo)} newly scored, {len(want) - len(todo)} cached"
    else:
        src = f"all {len(want)} reused from cache"

    eps, ep_onset, ep_neg, W = decide(R)
    print(f"[search] {os.path.relpath(run, _REPO)}")
    print(f"         bench={BENCH}  n_query={R[eps[0]][BENCH]['n_query']:,}  {src}\n")
    print(f"  {'ep':>3}" + "".join(f"{k:>9}" for k in KEYS) + f"{'negmax':>11}")
    for e in eps:
        row = R[e][BENCH]
        print(f"  {e:>3}" + "".join(f"{row[k]:>9.3f}" for k in KEYS)
              + f"{row[NEG]:>11.4f}" + ("  <- min negmax" if e == ep_neg else ""))
    print(f"\n  1σ(R@1) ±{se_of(R[eps[-1]][BENCH], 'R@1'):.3f}"
          f" · onset ep{ep_onset} · min negmax ep{ep_neg} → target ep{W[0]}")
    print(f"\n  SWA range = ep{W[0]:02d}-ep{W[-1]:02d} ({len(W)} ckpt)")
    if len(W) < 2:
        print(f"  a single checkpoint cannot be averaged - use ep{W[0]:02d} as is")
    else:
        enc_all = os.path.basename(HERE).replace("_heldout", "_all")
        print(f"  python train/encoders/{enc_all}/build_swa.py <all_run> {W[0]} {W[-1]}")


if __name__ == "__main__":
    main()
