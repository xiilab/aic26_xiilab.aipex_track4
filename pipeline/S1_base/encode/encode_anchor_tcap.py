#!/usr/bin/env python3
"""anchor_tcap (SigLIP2-L512 + DoRA) encoder features — the sole producer of this feature family.

Two ways to use it.
  1) As a script: single-view feats dump for arbitrary v28 epochs, in {G, Q, G_base, Q_idx} format.
     Epochs are listed explicitly with --epochs. Test data is loaded once per process and only the
     model is reloaded per epoch (the gallery must still be re-encoded each time).
     Output: v28_{run}_ep{NN}_feats.pt.
     Usage: CUDA_VISIBLE_DEVICES=6 python encode_anchor_tcap.py --run A --epochs 7,8,10,11,12,13
  2) As a library: the entry points imported by the ensemble dumps
     (tools/ensemble/dumps/{uca,rstp}_dump_anchor_tcap.py).
       init_infer / load_encoder      initialize anchor_infer and load a ckpt
       build_tta_transforms           base/hflip/zNNN view transforms
       encode_tta_views               gallery TTA views + query text encoding -> views dict
       dump_tta_views                 pool.pt -> views feats file (one-shot helper for the dumps)
     The views dict format matches anchor_tcap_tta_views.pt:
       {img: {view: G}, txt: {"base": Q}, gal_paths, Q_idx}
"""
import argparse, os, sys, time, gc
import torch
import torchvision.transforms.v2 as Tv2
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

RUNS_ROOT = os.environ.get("RUNS_ROOT", f"{_REPO}/assets/runs")   # run root for epoch sweeps; falls back to the bundled adapter
T4 = os.environ.get("S1_MEMBERS", f"{_REPO}/assets/cache/s1_base/members")
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
TEST_ROOT = PAB_TEST
SIG = f"{_REPO}/train/encoders/anchor_tcap_all"                         # train module: single source for architecture constants
MODULE = f"{SIG}/train.py"
CKPT_ADAPTER = os.environ.get("ENCODER_CKPT", f"{_REPO}/assets/model/encoder/anchor_tcap")   # bundled DoRA adapter
RUNS = {"A": os.path.join(RUNS_ROOT, "v28_l512_A"), "B": os.path.join(RUNS_ROOT, "v28_l512_B")}
DEFAULT_VIEWS = ("base", "hflip", "z090")   # anchor views, same as in anchor_tcap_tta_views.pt
# Architecture flags that determine the inference result. The train module is their single source;
# this file only checks that they exist, failing at load time rather than letting a missing flag
# silently change the dump.
_REQUIRED_CFG = ("POSITION_PI_ENABLED", "NUM_PROBES", "MAX_TEXT_LENGTH",
                 "POSITION_EMBED_NEW_SIZE", "POSITION_PI_TARGET_LEN")


# ── encoder loading ───────────────────────────────────────────────────────
def init_infer(module=MODULE, gallery_batch=64, num_workers=8):
    """Inject the train module into anchor_infer. Returns (anchor_infer, train module)."""
    import anchor_infer
    mod = anchor_infer.init(module, EVAL_BATCH_SIZE=gallery_batch, EVAL_NUM_WORKERS=num_workers)
    missing = [k for k in _REQUIRED_CFG if not hasattr(mod, k)]
    if missing:
        raise AttributeError(f"architecture constants missing from the train module: {missing} ({module})")
    return anchor_infer, mod


def load_encoder(ckpt=CKPT_ADAPTER, module=MODULE, gallery_batch=64,
                 num_workers=8, device="cuda:0"):
    """init_infer plus checkpoint loading. Returns (anchor_infer, mod, model, tokenizer)."""
    infer, mod = init_infer(module, gallery_batch, num_workers)
    model, tok = infer.load_for_inference(mod.MODEL_NAME, ckpt, device=device)
    return infer, mod, model, tok


# ── feature extraction ────────────────────────────────────────────────────
def build_tta_transforms(image_size, views=DEFAULT_VIEWS):
    """TTA view transforms: base = normalize only, hflip, zNNN = center-crop 0.NN then resize."""
    norm = [Tv2.ToDtype(torch.float32, scale=True), Tv2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
    def zoom(r): return [Tv2.CenterCrop(int(round(image_size * r))), Tv2.Resize(image_size, antialias=True)]
    tfs = {}
    for v in views:
        if v == "base":
            tfs[v] = Tv2.Compose(norm)
        elif v == "hflip":
            tfs[v] = Tv2.Compose([Tv2.RandomHorizontalFlip(1.0), *norm])
        elif len(v) == 4 and v[0] == "z" and v[1:].isdigit():
            tfs[v] = Tv2.Compose([*zoom(int(v[1:]) / 100.0), *norm])
        else:
            raise ValueError(f"unknown TTA view: {v!r} (base | hflip | zNNN)")
    return tfs


def encode_tta_views(infer, model, tok, gallery_paths, captions, image_size,
                     views=DEFAULT_VIEWS, tag="anchor_tcap"):
    """Encode the gallery per view and the query captions once. Returns the views dict as stored."""
    tfs = build_tta_transforms(image_size, views)
    img_views = {}
    for name in views:
        I, _ = infer.encode_test_gallery(model, gallery_paths, tfs[name])
        img_views[name] = I.float().cpu()
        print(f"[{tag}] img-view {name} {tuple(img_views[name].shape)}", flush=True)
    queries = [{"caption": c, "query_index": j, "change": ""} for j, c in enumerate(captions)]
    Qf, _, _, _ = infer.encode_test_queries(model, queries, tok)
    return {"img": img_views, "txt": {"base": Qf.float().cpu()},
            "gal_paths": list(gallery_paths), "Q_idx": list(range(len(captions)))}


def dump_tta_views(pool_pt, out_pt, ckpt=CKPT_ADAPTER, views=DEFAULT_VIEWS,
                   tag="anchor_tcap", gallery_batch=64, num_workers=8, device="cuda:0"):
    """pool.pt({gal_paths, caps}) -> TTA views feats file. Shared entry point for the ensemble dumps."""
    pool = torch.load(pool_pt, map_location="cpu", weights_only=False)
    gal_paths, caps = list(pool["gal_paths"]), list(pool["caps"])
    print(f"[{tag}] gallery={len(gal_paths)} queries={len(caps)} ckpt={ckpt}", flush=True)
    infer, mod, model, tok = load_encoder(ckpt=ckpt, gallery_batch=gallery_batch,
                                          num_workers=num_workers, device=device)
    feats = encode_tta_views(infer, model, tok, gal_paths, caps, mod.IMAGE_SIZE, views, tag)
    torch.save(feats, out_pt)
    print(f"[save] {out_pt}  views={list(feats['img'])} txt=base", flush=True)
    return feats


def ckdir(run, ep):
    d = os.path.join(RUNS[run], "checkpoints", f"ep{ep:02d}")
    if os.path.isdir(d): return d
    d2 = os.path.join(RUNS[run], "checkpoints", f"ep{ep}")
    return d2 if os.path.isdir(d2) else CKPT_ADAPTER   # fall back to the bundled adapter when no run dir exists


def dump_test_member(ckpt, out, gallery_batch=64, views=DEFAULT_VIEWS, device="cuda:0"):
    """test-set TTA views -> the build_base member file (anchor_tcap_tta_views.pt).

    `dump_tta_views` targets bench pools and stores `gal_paths`, whereas build_base aligns the
    gallery order via `G_base`, so the test set gets its own path.
    """
    infer, mod, model, tok = load_encoder(ckpt=ckpt, gallery_batch=gallery_batch,
                                          num_workers=4, device=device)
    queries, _qorder, gallery_paths, _ = infer.load_test_data(
        f"{TEST_ROOT}/query_text.json", f"{TEST_ROOT}/query_index.txt",
        f"{TEST_ROOT}/gallery", None)
    print(f"[anchor_tcap] ckpt={ckpt} gallery={len(gallery_paths)} query={len(queries)}", flush=True)
    tfs = build_tta_transforms(mod.IMAGE_SIZE, views)
    img_views, G_base = {}, None
    for name in views:
        t0 = time.time()
        G, G_base = infer.encode_test_gallery(model, gallery_paths, tfs[name])
        img_views[name] = G.float().cpu()
        print(f"[anchor_tcap] img-view {name} {tuple(img_views[name].shape)} ({time.time()-t0:.0f}s)", flush=True)
    Q, Q_idx, _, _ = infer.encode_test_queries(model, queries, tok)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"img": img_views, "txt": {"base": Q.float().cpu()},
                "G_base": G_base, "Q_idx": list(Q_idx),
                "views": list(views), "ckpt": ckpt}, out)
    print(f"[save] {out}  views={list(img_views)} txt=base", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="anchor_tcap encoding — default builds the test-set TTA member file; --run/--epochs runs an epoch sweep")
    ap.add_argument("--run", choices=["A", "B"], help="epoch sweep mode (use with --epochs)")
    ap.add_argument("--epochs", help="epochs to sweep, e.g. 7,8,10,11,12,13")
    ap.add_argument("--gallery-batch", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--rep", action="store_true",
                    help="reproduction encoding with model_rep weights -> cache_rep/s1_base/members")
    ap.add_argument("--out", default=None, help="output for member mode (default = members/anchor_tcap_tta_views.pt)")
    args = ap.parse_args()

    if not (args.run and args.epochs):        # default: the test-set TTA member file (build_base input)
        ckpt = f"{_REPO}/assets/model_rep/encoder/anchor_tcap" if args.rep else CKPT_ADAPTER
        mem = os.environ.get("S1_MEMBERS",
                             f"{_REPO}/assets/{'cache_rep' if args.rep else 'cache'}/s1_base/members")
        _out = args.out or f"{mem}/anchor_tcap_tta_views.pt"
        skip_if_exists(_out, args.overwrite)
        dump_test_member(ckpt, _out,
                         gallery_batch=args.gallery_batch)
        return

    eps = [int(e) for e in args.epochs.split(",")]

    infer, mod = init_infer(gallery_batch=args.gallery_batch, num_workers=4)
    print(f"[v28_{args.run}] MODEL={mod.MODEL_NAME} IMG={mod.IMAGE_SIZE} epochs={eps}", flush=True)

    queries, qorder, gallery_paths, _ = infer.load_test_data(
        f"{TEST_ROOT}/query_text.json", f"{TEST_ROOT}/query_index.txt",
        f"{TEST_ROOT}/gallery", None)
    eval_tf = infer.build_eval_transform(mod.IMAGE_SIZE)
    print(f"[v28_{args.run}] query={len(queries)} gallery={len(gallery_paths)}", flush=True)

    for ep in eps:
        out = os.path.join(T4, f"v28_{args.run}_ep{ep:02d}_feats.pt")
        if os.path.exists(out) and not args.overwrite:
            print(f"[v28_{args.run} ep{ep}] already exists, skipping ({out})", flush=True); continue
        ck = ckdir(args.run, ep)
        assert os.path.isdir(ck), f"checkpoint not found: {ck}"
        t0 = time.time()
        model, tok = infer.load_for_inference(mod.MODEL_NAME, ck, device="cuda:0")
        G, G_base = infer.encode_test_gallery(model, gallery_paths, eval_tf)
        Q, Q_idx, _, _ = infer.encode_test_queries(model, queries, tok)
        torch.save({"G": G, "Q": Q, "G_base": list(G_base), "Q_idx": list(Q_idx),
                    "run": args.run, "epoch": ep, "ckpt": ck}, out)
        del model; gc.collect(); torch.cuda.empty_cache()
        print(f"[v28_{args.run} ep{ep}] G={tuple(G.shape)} Q={tuple(Q.shape)} saved to {out} ({time.time()-t0:.0f}s)", flush=True)

    print(f"[v28_{args.run}] DONE {eps}", flush=True)


if __name__ == "__main__":
    main()
