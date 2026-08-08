"""heldout_bench — shared helpers for the held-out bench.

Used by `eval_heldout.py` (SigLIP2 / MetaCLIP2), `eval_heldout_openclip.py` (eva02 / MetaCLIP v1)
and `beit3/beit3_tool.py eval` (BEiT3). Bench loading, leak re-check, path mapping and scoring all
live here so the evaluators cannot drift apart.

  heldout_images.txt   50,653 images excluded from training ("train/imgs_X/cat/N.jpg")
  split.json           exclusion list + both benches + gates
    main : queries 2,000      / gallery_easy    36,773   representative -> selection metric
    hard : queries_hard 1,762 / gallery_hardset 36,772   siblings added -> diagnostic only

Metrics match the competition scorer (one ground truth per query): mAP@10 = mean(1/rank), R@1/5/10.

Note: `torch` is imported at module load, so callers must set `CUDA_VISIBLE_DEVICES` before
importing this module.
"""
from __future__ import annotations

import hashlib
import json
import os

import torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repository root — all default paths are relative to it

DEFAULT_HELDOUT_DIR = os.environ.get(
    "HELDOUT_DIR",
    os.environ.get("PAB_DATA_INFRA", f"{_REPO}/assets/data") + "/heldout_v1")
PAB_TRAIN = os.environ.get("PAB_TRAIN", f"{_REPO}/assets/data/raw/pab_train")
IMG_ROOT = os.environ.get("PAB_JPG", f"{PAB_TRAIN}/train_jpg_512")


def parse_epochs(spec: str) -> list[int]:
    """Parse "1-12" / "3,5,9" / "1-4,8" into a sorted epoch list."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def map_to_local(ann_path: str, root: str = None) -> str:
    """Annotation path to local path: imgs_M lives under "Part {M // 8 + 1}".

    "train/imgs_8/full/10.jpg" -> "<root>/Part 2/imgs_8/full/10.jpg".
    Same rule as the trainer's `map_annotation_path_to_local()`.
    """
    root = root or IMG_ROOT
    parts = ann_path.split("/")
    assert parts[0] == "train" and len(parts) == 4, ann_path
    n = int(parts[1].replace("imgs_", ""))
    return f"{root}/Part {n // 8 + 1}/{parts[1]}/{parts[2]}/{parts[3]}"


def split_path(heldout_dir: str = None) -> str:
    return os.path.join(heldout_dir or DEFAULT_HELDOUT_DIR, "split.json")


def heldout_list_path(heldout_dir: str = None) -> str:
    return os.path.join(heldout_dir or DEFAULT_HELDOUT_DIR, "heldout_images.txt")


def load_heldout_set(path: str = None) -> set[str]:
    """Read the exclusion list into a set, normalising extensions to .jpg (BEiT3 indexes webp)."""
    path = path or heldout_list_path()
    with open(path, encoding="utf-8") as f:
        return {_as_jpg(ln.strip()) for ln in f if ln.strip()}


def _as_jpg(p: str) -> str:
    stem, ext = os.path.splitext(p)
    return stem + ".jpg" if ext.lower() in (".webp", ".jpeg", ".png") else p


def md5_of_list(items) -> str:
    """md5 over the items, each encoded utf-8 and terminated by a newline byte."""
    h = hashlib.md5()
    for x in items:
        h.update(str(x).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def verify_identity(heldout_dir: str = None, expect: str = None) -> str:
    """Check the exclusion list against `stats.heldout_md5` in `split.json`.

    Makes "same bench" enforced rather than assumed: a changed list stops the caller here.
    Returns the verified md5 so it can be recorded in result JSON and logs.
    """
    sp = split_path(heldout_dir)
    with open(sp, encoding="utf-8") as f:
        stats = json.load(f).get("stats", {})
    want = expect or stats.get("heldout_md5")
    got = md5_of_list(sorted(load_heldout_set(heldout_list_path(heldout_dir))))
    if want and got != want:
        raise SystemExit(
            f"✗ heldout list does not match split.json.\n"
            f"    list {heldout_list_path(heldout_dir)} → {got}\n"
            f"    split.json stats.heldout_md5 → {want}\n"
            f"  Check that both come from the same heldout_v1.")
    print(f"  [heldout] md5 {got} ✓ (matches split.json)")
    return got


def require_gates(heldout_dir: str = None, strict: bool = True) -> dict:
    """Check that every split gate passed.

    With strict=True, training/evaluation does not start unless `all_pass` is True.
    """
    with open(split_path(heldout_dir), encoding="utf-8") as f:
        gates = json.load(f).get("stats", {}).get("gates", {})
    if gates.get("all_pass") is not True:
        msg = f"heldout gates did not pass: {gates}"
        if strict:
            raise SystemExit(f"✗ {msg}")
        print(f"  ⚠ {msg}")
    else:
        print(f"  [heldout] gates all_pass ✓ "
              f"(cross-split near-dup {gates.get('g2_cross_neardup_pairs')} · "
              f"{gates.get('g2_sim_mode')} thr {gates.get('g2_gate_thr')})")
    return gates


def load_bench(bench: str, heldout_dir: str = None, path: str = None):
    """Read split.json into (queries, gallery, gates). bench = main | hard.

    queries[i] = {"qid", "image" (ground truth), "caption", "style", "label_type", "category", ...}
    """
    p = path or split_path(heldout_dir)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    if bench == "main":
        q, g = d["queries"], d["gallery_easy"]
    elif bench == "hard":
        q, g = d["queries_hard"], d["gallery_hardset"]
    else:
        raise ValueError(f"bench must be main | hard (got: {bench})")
    gset = set(g)
    missing = [r["qid"] for r in q if r["image"] not in gset]
    if missing:
        raise RuntimeError(f"[{bench}] {len(missing)} queries whose ground truth is not in the "
                           f"gallery (e.g. {missing[:3]}) — check split.json.")
    return q, g, d.get("stats", {}).get("gates", {})


def report_gates(gates: dict):
    if gates.get("all_pass") is not True:
        print(f"  ⚠ split.json gates all_pass != True → {gates}")


def check_leak(queries, gallery, bench: str, heldout_list: str = None) -> int:
    """Re-check that every bench image is in the exclusion list; return how many are not.

    This inspects the list only, so it does not prove the run honoured it: a run whose training did
    not apply the exclusion passes this check and may still have seen bench images.
    """
    p = heldout_list or heldout_list_path()
    if not os.path.exists(p):
        print(f"  ⚠ exclusion list not found ({p}) → cannot check for leaks")
        return -1
    hold = load_heldout_set(p)
    leak = ({_as_jpg(r["image"]) for r in queries} | {_as_jpg(x) for x in gallery}) - hold
    print(f"  [{bench}] images outside the exclusion list = {len(leak):,} "
          f"{'✓' if not leak else '⚠ leak — these numbers are optimistically biased'}")
    return len(leak)


def score(Q: torch.Tensor, G: torch.Tensor, queries, gallery) -> dict:
    """Score as the competition does: one ground truth per query, mAP@10 = mean(1/rank), R@1/5/10.

    Also reports **margin**, which stays discriminative once the ranking metrics saturate:
    margin = cos(GT) - max cos(hardest negative). R@1 equals the fraction with margin > 0.

    Q and G must be L2-normalised [Q, D] and [N, D].
    """
    gidx = {p: i for i, p in enumerate(gallery)}
    gt = torch.tensor([gidx[r["image"]] for r in queries])
    sims = Q @ G.t()
    top = sims.topk(min(10, G.shape[0]), dim=1).indices
    hit = top == gt[:, None]
    has = hit.any(dim=1)
    rank = hit.float().argmax(dim=1) + 1          # 1 on a miss; masked out by `has`
    n = len(gt)

    rows = torch.arange(n)
    pos = sims[rows, gt]
    masked = sims.clone()
    masked[rows, gt] = float("-inf")
    negmax = masked.max(dim=1).values
    return {
        "mAP@10": 100.0 * torch.where(has, 1.0 / rank, torch.zeros(n)).sum().item() / n,
        "R@1": 100.0 * (has & (rank == 1)).sum().item() / n,
        "R@5": 100.0 * (has & (rank <= 5)).sum().item() / n,
        "R@10": 100.0 * has.sum().item() / n,
        "n_query": n,
        "pos_cos_mean": float(pos.mean()),
        "negmax_cos_mean": float(negmax.mean()),
        "margin_agg": float(pos.mean() - negmax.mean()),
        "margin_mean": float((pos - negmax).mean()),
        "margin_p50": float((pos - negmax).median()),
    }


def print_table(rows, benches, select: str = "main"):
    """Print the per-epoch table and return the best row. Selection uses `main` only."""
    width = 8 + 44 * len(benches)
    print("\n" + "=" * width)
    head = f"{'epoch':>6}"
    for b in benches:
        head += f" | {b + ' mAP@10':>15} {b + ' R@1':>12} {b + ' margin':>13}"
    print(head)
    print("-" * width)
    for r in rows:
        line = f"{r['epoch']:>6}"
        for b in benches:
            line += (f" | {r[b]['mAP@10']:>15.4f} {r[b]['R@1']:>12.4f}"
                     f" {r[b].get('margin_agg', float('nan')):>13.4f}")
        print(line)
    sel = select if select in benches else benches[0]
    best = max(rows, key=lambda r: r[sel]["mAP@10"])
    print("-" * width)
    print(f"best epoch = {best['epoch']}  ({sel} mAP@10 {best[sel]['mAP@10']:.4f})")
    print(f"             {best['ckpt']}")
    if sel != "main":
        print("  ⚠ selected on a bench other than main — hard is diagnostic only.")
    return best, sel

def local_to_ann(path: str) -> str | None:
    """Local image path to annotation path (inverse of `map_to_local`).

    ".../train_webp/Part 2/imgs_8/full/10.webp" -> "train/imgs_8/full/10.jpg"
    The extension is normalised to .jpg so the result can be matched against the exclusion list.
    Returns None when no imgs_* segment is found, leaving the decision to the caller.
    """
    parts = path.replace("\\", "/").split("/")
    for i, seg in enumerate(parts):
        if seg.startswith("imgs_") and i + 2 < len(parts):
            return _as_jpg(f"train/{seg}/{parts[i + 1]}/{parts[i + 2]}")
    return None


def filter_csv_rows(rows, path_key=0, heldout_dir=None, require=True):
    """Drop held-out images from a list of (filepath, caption) rows.

    For the CSV-based trainers (eva02 full-FT, MetaCLIP v1 full-FT). `filepath` is an absolute
    webp/jpg path, so `local_to_ann()` converts it back to annotation form for the comparison.

    With require=True, the list md5 and the gates are checked first and a mismatch aborts.
    """
    if require:
        verify_identity(heldout_dir)
        require_gates(heldout_dir, strict=True)
    hold = load_heldout_set(heldout_list_path(heldout_dir))
    kept, dropped, unmapped = [], 0, 0
    for r in rows:
        ann = local_to_ann(r[path_key])
        if ann is None:
            unmapped += 1
            kept.append(r)
            continue
        if ann in hold:
            dropped += 1
        else:
            kept.append(r)
    print(f"  [heldout] list {len(hold):,} images · rows {len(rows):,} → {len(kept):,} "
          f"(-{dropped:,}, {100.0 * dropped / max(1, len(rows)):.2f}%)")
    if unmapped:
        print(f"  ⚠ {unmapped:,} rows kept as-is: no imgs_* segment in the path")
    if dropped == 0:
        raise RuntimeError(
            "[heldout] 0 rows dropped — the filepath format may not match the exclusion list. "
            f"example={rows[0][path_key]!r} → {local_to_ann(rows[0][path_key])!r}")
    return kept

def still_training(pattern: str = "train_") -> bool:
    """Whether a training process is alive, used to decide when `--watch` may stop.

    Returns True when the lookup fails, so the caller waits: scoring an incomplete checkpoint is
    worse than waiting one extra poll.
    """
    import subprocess
    try:
        return bool(subprocess.run(["pgrep", "-f", pattern],
                                   capture_output=True, text=True).stdout.strip())
    except Exception:
        return True

def dump_bench_inputs(bench: str, queries, gallery, Q, G, out_dir: str,
                      topk: int = 10, heldout_dir: str = None) -> dict:
    """Dump the held-out bench in eval-set shape so the reranker scorers can read it.

    `pipeline/S2_rerank/score_union_*.py` expects three things:
      · a `GALLERY/` directory — candidate index = position in `sorted(os.listdir(GALLERY))`
      · a `QUERY_TEXT` jsonl  — `{"query_index":…, "caption":…}`
      · a `POOL_FILE`         — `{"union": [[cand_idx…]], "qorder": [query_index…]}`

    So the gallery becomes a directory of symlinks, the queries a jsonl, and the candidate pool the
    encoder's top-K. `gt` (qid -> gallery index) and the base scores are saved alongside for scoring.

    Q and G are L2-normalised [Q, D] and [N, D] held-out embeddings from the same encoder.
    """
    img_dir = os.path.join(out_dir, "gallery")
    os.makedirs(img_dir, exist_ok=True)

    # Build unique filenames: the original stems collide because every imgs_X reuses the numbering.
    names = []
    for ann in gallery:
        p = ann.split("/")                     # train/imgs_X/cat/N.jpg
        name = f"{p[1]}_{p[2]}_{p[3]}"         # imgs_X_cat_N.jpg
        names.append(name)
    if len(set(names)) != len(names):
        raise RuntimeError("gallery symlink names collide — check the naming rule.")
    for ann, name in zip(gallery, names):
        dst = os.path.join(img_dir, name)
        if not os.path.lexists(dst):
            os.symlink(map_to_local(ann), dst)

    # The scorer indexes by sorted(listdir), so recompute indices in that order.
    order = sorted(os.listdir(img_dir))
    pos = {n: i for i, n in enumerate(order)}
    perm = torch.tensor([pos[n] for n in names])          # gallery[i] -> scorer index
    Gs = torch.empty_like(G)
    Gs[perm] = G                                          # reorder into scorer order

    qids = [r["qid"] for r in queries]
    with open(os.path.join(out_dir, "query_text.json"), "w", encoding="utf-8") as f:
        for r in queries:
            f.write(json.dumps({"query_index": r["qid"], "caption": r["caption"]},
                               ensure_ascii=False) + "\n")

    sims = Q @ Gs.t()                                     # [Q, N] in scorer order
    top = sims.topk(min(topk, Gs.shape[0]), dim=1).indices
    union = [[int(c) for c in row] for row in top]

    ann2idx = {ann: pos[n] for ann, n in zip(gallery, names)}   # annotation path -> scorer index
    gt = {r["qid"]: ann2idx[r["image"]] for r in queries}

    torch.save({"union": union, "qorder": qids, "tops": {}, "bases": []},
               os.path.join(out_dir, "union_pool.pt"))
    torch.save({"sims": sims, "qorder": qids, "gallery_order": order, "gt": gt},
               os.path.join(out_dir, "base_score.pt"))

    in_pool = sum(1 for i, q in enumerate(qids) if gt[q] in union[i])
    rec = {"bench": bench, "out_dir": out_dir, "topk": topk,
           "n_query": len(qids), "n_gallery": len(order),
           "pairs": sum(len(u) for u in union),
           "gt_in_pool": in_pool, "gt_in_pool_pct": round(100.0 * in_pool / len(qids), 4),
           "heldout_md5": md5_of_list(sorted(load_heldout_set(heldout_list_path(heldout_dir))))}
    with open(os.path.join(out_dir, "bench_inputs.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"  [dump] gallery {len(order):,} symlinks · queries {len(qids):,} · "
          f"candidates {rec['pairs']:,} pairs (top{topk})")
    print(f"  [dump] queries whose GT is in the pool: {in_pool:,}/{len(qids):,} "
          f"({rec['gt_in_pool_pct']:.2f}%) — reranker ceiling")
    print(f"  [dump] → {out_dir}")
    return rec
