"""Hard Negative Margin Loss (HNML).

Paper Eq. 6:
    ℓ_margin = mean_{j∈B}  max(0,  sim(I_i, T_j) − min_{j'∈H_i^p} sim(I_i, T_{j'}))

Per anchor i (the seeds), we compute the minimum similarity over the i-th
seed's hard pairs (which sit at known positions in the batch), then penalize
any non-hard, non-self batch entry whose similarity to anchor i exceeds it.

The loss reuses the (B, B) logits already computed by ClipLoss; no extra
inner product is needed.
"""
import torch
import torch.nn.functional as F


def hnml_loss(logits_per_image: torch.Tensor,
              n_seeds: int, p: int) -> torch.Tensor:
    """Compute HNML on a HELIP-composite batch.

    Args:
        logits_per_image: (B, B) similarity logits (logit_scale already applied).
        n_seeds: number of seed anchors (B'); first n_seeds positions of the batch.
        p: number of hard pairs per seed; positions [n_seeds, n_seeds + n_seeds*p)
           grouped as `hp[k, j]` at position `n_seeds + k*p + j`.

    Returns:
        Scalar margin loss (image→text direction).
    """
    B = logits_per_image.shape[0]
    assert B == n_seeds * (1 + p), \
        f"batch size mismatch: B={B}, expected n_seeds*(1+p)={n_seeds*(1+p)}"
    device = logits_per_image.device

    # Anchor rows = seeds.
    anchor_sims = logits_per_image[:n_seeds]                              # (n_seeds, B)

    # For each seed k, its hard pair positions in the batch.
    hp_positions = (n_seeds + torch.arange(n_seeds, device=device) * p)
    hp_positions = hp_positions.unsqueeze(1) + torch.arange(p, device=device).unsqueeze(0)
    # (n_seeds, p) — column indices of hard pairs for each seed.

    # m_k = min over j'∈H_k of sim(I_k, T_{j'}).
    hp_sims = torch.gather(anchor_sims, dim=1, index=hp_positions)        # (n_seeds, p)
    m = hp_sims.min(dim=1, keepdim=True).values                           # (n_seeds, 1)

    # Mask: exclude self and own hard pairs from the j∈B sum.
    mask = torch.ones((n_seeds, B), dtype=torch.bool, device=device)
    mask[torch.arange(n_seeds, device=device), torch.arange(n_seeds, device=device)] = False
    mask.scatter_(1, hp_positions, False)

    margin = F.relu(anchor_sims - m)                                       # (n_seeds, B)
    margin = margin * mask.float()
    # Paper Eq. 6 divides by |B|; positions for self and own hard pairs
    # are zero-masked, so they contribute 0 to the sum but still count in |B|.
    return margin.sum(dim=1).mean() / float(B)
