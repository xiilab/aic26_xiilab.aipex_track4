"""Text Intra-modal Contrastive loss (FG-CLIP 2 §3.2).

For each text anchor T_i in the batch:
  1) Compute cos similarity to every other text in the batch.
  2) Mask out "false negatives": same-image_id captions (if provided) and
     pairs with sim > τ_fn (defaults to 0.95 as in the paper).
  3) Select top-K hardest remaining negatives.
  4) Loss term = log Σ_k exp(S(T_i, T_neg_k))  — a smooth max over the closest
     hard negatives. Minimising this pushes the closest hard negative away
     while remaining differentiable.

Loss is symmetric over rows (no T→I direction here — TIC is purely intra-text).
"""
from typing import Optional
import torch


def tic_loss(text_features: torch.Tensor,
             image_ids: Optional[torch.Tensor] = None,
             top_k: int = 10,
             fn_threshold: float = 0.95,
             logit_scale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute TIC loss for a batch of L2-normalized text features.

    Args:
        text_features: (B, D) — L2-normalized text [CLS] features.
        image_ids: (B,) int tensor or None. If provided, same-id pairs are
            treated as positives and excluded from the negative pool.
        top_k: number of hardest negatives per anchor.
        fn_threshold: cos similarity above this is treated as a positive
            (paraphrase) and excluded.
        logit_scale: optional scalar (e.g. model.logit_scale.exp()) applied
            to similarities before log-sum-exp. None ⇒ no scaling.

    Returns:
        Scalar loss (mean over batch).
    """
    B = text_features.shape[0]
    if B < top_k + 2:
        # Not enough negatives in this batch — return zero.
        return text_features.new_zeros(())

    sim = text_features @ text_features.t()                                # (B, B), in [-1, 1]
    if logit_scale is not None:
        sim = logit_scale.float() * sim

    NEG_INF = torch.finfo(sim.dtype).min

    # Mask self.
    diag = torch.arange(B, device=sim.device)
    sim[diag, diag] = NEG_INF

    # Mask false-negative paraphrases.
    # Use unscaled cosine for thresholding regardless of logit_scale.
    raw_sim = text_features @ text_features.t()
    sim = sim.masked_fill(raw_sim > fn_threshold, NEG_INF)

    # Mask same-image_id (multi-caption-per-image case).
    if image_ids is not None:
        same = image_ids.unsqueeze(0) == image_ids.unsqueeze(1)            # (B, B)
        sim = sim.masked_fill(same, NEG_INF)

    K = min(top_k, B - 1)
    hard_neg_sim = sim.topk(K, dim=1).values                               # (B, K)
    # Drop rows that ended up with no valid negative (all -inf).
    valid = (hard_neg_sim > NEG_INF / 2).any(dim=1)
    if not valid.any():
        return text_features.new_zeros(())

    # log Σ exp(sim) over the K hard negatives, mean over valid anchors.
    return torch.logsumexp(hard_neg_sim[valid], dim=1).mean()
