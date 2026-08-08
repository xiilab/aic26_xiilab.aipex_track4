"""FALCON scheduler πϕ: 4-layer residual MLP → Beta(α, β) per anchor."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        h = self.fc1(F.gelu(self.ln1(x)))
        h = self.fc2(F.gelu(self.ln2(h)))
        return x + h


class FalconScheduler(nn.Module):
    """Row-wise MLP that maps a sorted, softmax-normalized quantile feature
    of shape (M, m) to (α, β) parameters of a Beta distribution per anchor.

    See paper §3.2. Sorting rows before MLP enforces permutation equivariance
    over the m quantile bins without heavier set encoders.
    """

    def __init__(self, m_bins: int = 16, hidden: int = 256, depth: int = 4,
                 ab_floor: float = 1e-2):
        super().__init__()
        self.in_proj = nn.Linear(m_bins, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden, 2)
        self.ab_floor = ab_floor

    def forward(self, S_tilde: torch.Tensor):
        """Args:
            S_tilde: (M, m) — sorted softmax-normalized quantile features.
        Returns:
            alpha, beta: (M,) positive tensors.
        """
        h = self.in_proj(S_tilde)
        for blk in self.blocks:
            h = blk(h)
        ab = F.softplus(self.out_proj(h)) + self.ab_floor   # (M, 2)
        return ab[..., 0], ab[..., 1]

    @staticmethod
    def sample_q(alpha: torch.Tensor, beta: torch.Tensor):
        """Sample q ∈ (0,1) per anchor; also return log-prob for REINFORCE."""
        dist = torch.distributions.Beta(alpha, beta)
        q = dist.rsample()                            # differentiable, but we use detach for REINFORCE
        log_p = dist.log_prob(q.clamp(1e-6, 1 - 1e-6))
        return q.detach(), log_p                       # log_p keeps grad wrt (α,β)
