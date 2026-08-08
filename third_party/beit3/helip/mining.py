"""FastHPM — offline hard-pair mining for BEIT3 retrieval.

Paper §3.2 (Eq. 5): for each anchor pair z_i, pick k hard pairs that maximize
    Σ_{j∈H}  S̃^I(x^I_i, j) · S̃^T(x^T_i, j)
from a uniformly subsampled candidate pool of size C ≪ N. Greedy top-k is
optimal because the objective is a sum of independent per-candidate terms.

Both visual and textual similarities are τ-masked (set to 0 below τ) to avoid
amplifying noise from very-dissimilar candidates (paper: "robust to noise").

Usage ($BEIT3_DATA = the index root created by beit3_tool.py):
    python -m helip.mining \
        --ckpt <stage1 run>/checkpoint-best.pth \
        --data_path $BEIT3_DATA/pab_full_webp \
        --out $BEIT3_DATA/pab_full_webp/helip_hardpairs.npy \
        --k 8 --C 20000 --tau 0.1
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from timm.models import create_model

# Local imports must work when run as `python -m helip.mining` from the
# beit3 directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import modeling_finetune  # noqa: F401 — registers beit3 model
import utils
from datasets import create_dataset_by_split


def _build_model(ckpt_path: str, device: torch.device):
    model = create_model(
        "beit3_large_patch16_384_retrieval",
        pretrained=False,
        drop_path_rate=0.0,
        vocab_size=64010,
        checkpoint_activations=False,
    )
    utils.load_model_and_may_interpolate(ckpt_path, model, "model|module", "")
    model.to(device).eval()
    return model


@torch.no_grad()
def encode_all(model, loader, device):
    img_feats, txt_feats = [], []
    for i, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        tokens = batch["language_tokens"].to(device, non_blocking=True)
        padmask = batch["padding_mask"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            vc, _ = model(image=image, only_infer=True)
            _, lc = model(text_description=tokens, padding_mask=padmask,
                          only_infer=True)
        img_feats.append(vc.float().cpu())
        txt_feats.append(lc.float().cpu())
        if i % 10 == 0:
            print(f"  encoded {(i+1)*image.shape[0]} pairs", flush=True)
    return torch.cat(img_feats, dim=0), torch.cat(txt_feats, dim=0)


def fasthpm(img_feats: torch.Tensor, txt_feats: torch.Tensor,
            k: int, C: int, tau: float, device: torch.device,
            anchor_chunk: int = 256):
    """Compute hard-pair indices and a noise-outlier mask for every anchor.

    Args:
        img_feats, txt_feats: (N, D) L2-normalized features.
        k: number of hard pairs per anchor.
        C: candidate pool size (uniform random subset of [0, N)).
        tau: noise mask threshold; cos similarities below tau are zeroed.
        device: torch device for the matmul.
        anchor_chunk: process this many anchors at a time on GPU.
    Returns:
        hard_pairs: (N, k) int32 — global indices into the dataset.
        outlier_mask: (N,) bool — True iff the anchor is deemed noisy
            (paper §3.2 "Mitigation of Noisy Data": any hard-pair candidate
            has S̃^I·S̃^T == 0 ⇒ a singleton zero-product subset exists ⇒ remove).
    """
    N = img_feats.shape[0]
    assert C <= N, f"C ({C}) cannot exceed N ({N})"
    rng = np.random.default_rng(seed=42)
    sub_idx = rng.choice(N, size=C, replace=False)
    sub_idx_t = torch.from_numpy(sub_idx).long().to(device)

    img_sub = img_feats[sub_idx].to(device)                                # (C, D)
    txt_sub = txt_feats[sub_idx].to(device)                                # (C, D)

    out = np.zeros((N, k), dtype=np.int32)
    outlier = np.zeros((N,), dtype=bool)
    for a0 in range(0, N, anchor_chunk):
        a1 = min(a0 + anchor_chunk, N)
        ia = img_feats[a0:a1].to(device)                                   # (b, D)
        ta = txt_feats[a0:a1].to(device)                                   # (b, D)
        sim_I = ia @ img_sub.t()                                           # (b, C) — cos (feats are normalized)
        sim_T = ta @ txt_sub.t()                                           # (b, C)
        sim_I = torch.where(sim_I >= tau, sim_I, torch.zeros_like(sim_I))
        sim_T = torch.where(sim_T >= tau, sim_T, torch.zeros_like(sim_T))
        score = sim_I * sim_T                                              # (b, C), ≥ 0

        # Exclude self if it lands in sub_idx (cosine = 1).
        self_mask = (sub_idx_t.unsqueeze(0) == torch.arange(a0, a1, device=device).unsqueeze(1))
        score_masked = score.masked_fill(self_mask, float("-inf"))

        topv, topi = score_masked.topk(k, dim=1)                           # (b, k), (b, k)
        out[a0:a1] = sub_idx_t[topi].cpu().numpy().astype(np.int32)

        # Paper noise-cleanup: outlier iff ANY of the k top scores is 0
        # (a singleton {j} with S̃^I·S̃^T = 0 ⇒ unsuitable target pair).
        outlier[a0:a1] = (topv <= 0).any(dim=1).cpu().numpy()

        if (a0 // anchor_chunk) % 20 == 0:
            print(f"  mined {a1}/{N}", flush=True)
    return out, outlier


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="BEIT3 checkpoint (e.g. stage-1 best)")
    p.add_argument("--data_path", required=True)
    p.add_argument("--out", required=True, help="Output .npy path for hard pairs")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--C", type=int, default=20000)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--num_max_bpe_tokens", type=int, default=128)
    p.add_argument("--input_size", type=int, default=384)
    p.add_argument("--sentencepiece_model", required=True)
    p.add_argument("--task", default="356")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save_features", default=None,
                   help="Optional .pt path to also dump (img, txt) features")
    args = p.parse_args()

    # Build a minimal namespace for the dataset/loader helpers.
    class _NS: pass
    ns = _NS()
    for fld in ("data_path", "num_max_bpe_tokens", "input_size", "task",
                "sentencepiece_model", "batch_size", "num_workers"):
        setattr(ns, fld, getattr(args, fld))
    ns.pin_mem = False
    ns.dist_eval = False
    ns.distributed = False
    ns.eval_batch_size = None
    ns.view_aug_mode = "global"           # no random aug — we want deterministic features
    ns.train_interpolation = "bicubic"
    ns.color_jitter = 0.0
    ns.aa = "rand-m9-mstd0.5-inc1"
    ns.reprob = 0.0
    ns.remode = "pixel"
    ns.recount = 1

    device = torch.device(args.device)
    print(f"[HELIP-FastHPM] loading ckpt: {args.ckpt}", flush=True)
    model = _build_model(args.ckpt, device)

    # Reuse the project's dataset factory. Train split (no shuffle) so the
    # i-th batch yields the i-th dataset index in order.
    loader = create_dataset_by_split(ns, split="356", is_train=False)
    # is_train=False uses SequentialSampler — exactly what we want.

    t0 = time.time()
    print(f"[HELIP-FastHPM] encoding {len(loader.dataset)} pairs ...", flush=True)
    img, txt = encode_all(model, loader, device)
    print(f"  encode done in {time.time()-t0:.1f}s. img={tuple(img.shape)} txt={tuple(txt.shape)}",
          flush=True)

    if args.save_features:
        torch.save({"img": img, "txt": txt}, args.save_features)
        print(f"  features → {args.save_features}", flush=True)

    print(f"[HELIP-FastHPM] mining k={args.k}, C={args.C}, tau={args.tau} ...", flush=True)
    t1 = time.time()
    hp, outlier = fasthpm(img, txt, k=args.k, C=args.C, tau=args.tau, device=device)
    n_out = int(outlier.sum())
    print(f"  mining done in {time.time()-t1:.1f}s — outliers={n_out}/{len(outlier)} "
          f"({100.0*n_out/len(outlier):.2f}%)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, hp)
    out_mask_path = os.path.splitext(args.out)[0] + "_outlier.npy"
    np.save(out_mask_path, outlier)
    print(f"[HELIP-FastHPM] hard pairs → {args.out}  shape={hp.shape}", flush=True)
    print(f"[HELIP-FastHPM] outlier mask → {out_mask_path}", flush=True)


if __name__ == "__main__":
    main()
