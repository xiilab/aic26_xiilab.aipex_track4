"""FALCON minibatch construction within a local search space of size M.

Algorithm (paper §3.1, GRIT-VLP extension):
  - Start each mini-batch from a uniformly sampled anchor.
  - At each subsequent step, pick the candidate whose similarity to the most
    recently chosen sample falls at the q-quantile predicted by the scheduler
    for that recent sample's anchor row.
  - Repeat until B samples collected. Remove from candidate pool. Repeat
    until M is exhausted, yielding |M|/B mini-batches.
"""
from typing import List, Tuple
import torch


def build_quantile_features(S: torch.Tensor, m_bins: int = 16) -> torch.Tensor:
    """Reduce |M|×|M| similarity matrix to a sorted softmax-normalized
    |M|×m feature for the scheduler.

    Steps:
      1. Per row, mask self (diagonal).
      2. Take m evenly-spaced quantiles of the masked row.
      3. Sort each row ascending (permutation equivariance).
      4. Row-wise softmax to be scale-robust across training.
    """
    M = S.shape[0]
    # Mask diagonal by setting to -inf for quantile calc
    work = S.clone()
    diag_idx = torch.arange(M, device=S.device)
    work[diag_idx, diag_idx] = float("-inf")

    # quantile points
    qs = torch.linspace(1.0 / (m_bins + 1), m_bins / (m_bins + 1), m_bins,
                        device=S.device)                                  # (m,)

    # finite values per row
    finite_mask = torch.isfinite(work)
    # Replace -inf with a very low value just for sorting-based quantile.
    work_finite = torch.where(finite_mask, work, torch.full_like(work, -1e9))
    sorted_vals, _ = torch.sort(work_finite, dim=1)                       # (M, M)

    # M-1 valid entries per row → indices into sorted positions
    n_valid = M - 1
    idx = (qs * (n_valid - 1)).long().clamp(0, n_valid - 1)               # (m,)
    # off by 1: sorted_vals first column is the -1e9 fillers' position; valid
    # entries occupy positions [1..M-1]. Shift by +1.
    feats = sorted_vals[:, 1 + idx]                                       # (M, m)

    # Sort per row + softmax-normalize
    feats_sorted, _ = torch.sort(feats, dim=1)
    feats_norm = torch.softmax(feats_sorted, dim=1)
    return feats_norm


def build_falcon_minibatches(
    S: torch.Tensor,
    q_per_anchor: torch.Tensor,
    batch_size: int,
) -> List[List[int]]:
    """Construct |M|/B mini-batches from a local pool of M candidates.

    Args:
        S: (M, M) similarity matrix (symmetric, e.g. cos + cos^T).
        q_per_anchor: (M,) per-anchor hardness quantile in (0, 1).
        batch_size: B.
    Returns:
        List of mini-batches; each is a list of local indices in [0, M).
    """
    M = S.shape[0]
    device = S.device
    remaining = torch.ones(M, dtype=torch.bool, device=device)
    batches: List[List[int]] = []
    n_batches = M // batch_size

    for _ in range(n_batches):
        batch: List[int] = []
        # 1) uniform initial anchor from remaining
        remain_idx = remaining.nonzero(as_tuple=False).squeeze(1)
        a0 = remain_idx[torch.randint(0, remain_idx.numel(), (1,), device=device)].item()
        batch.append(a0)
        remaining[a0] = False

        last = a0
        for _ in range(batch_size - 1):
            # similarity from `last` to all remaining
            remain_idx = remaining.nonzero(as_tuple=False).squeeze(1)
            sims_last = S[last, remain_idx]                              # (|R|,)
            sorted_sims, order = torch.sort(sims_last, descending=False)
            n_remain = sorted_sims.numel()
            # q=1 → hardest = most similar = end of ascending sort
            target = int(q_per_anchor[last].item() * (n_remain - 1))
            target = max(0, min(n_remain - 1, target))
            pick_local = order[target].item()
            pick_global = remain_idx[pick_local].item()
            batch.append(pick_global)
            remaining[pick_global] = False
            last = pick_global
        batches.append(batch)
    return batches
