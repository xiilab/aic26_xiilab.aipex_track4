"""gallery_norm — shared pipeline utilities.

  * GALLERY_DIR   gallery path. The submission column order is sorted(os.listdir(GALLERY_DIR)).
                  To move to another test set, set `PAB_TEST` (root) or `TRACK4_GALLERY` /
                  `GALLERY` (gallery directly); every stage picks the change up.
  * normalize()   global min-max normalization of a score matrix to [0,1], inf-safe.
"""
from __future__ import annotations

import os

import torch
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # utils -> pipeline -> repo root

PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GALLERY_DIR = (
    os.environ.get("TRACK4_GALLERY")
    or os.environ.get("GALLERY")
    or f"{PAB_TEST}/gallery"
)


def normalize(s: torch.Tensor) -> torch.Tensor:
    """Global min-max to [0,1]. inf is clipped to the finite minimum; a constant matrix maps to 0."""
    s = s.float()
    if torch.isinf(s).any():
        finite = s[~torch.isinf(s)]
        if finite.numel() == 0:
            return torch.zeros_like(s)
        floor = finite.min().item()
        s = torch.where(torch.isinf(s), torch.full_like(s, floor), s)
    mn, mx = s.min(), s.max()
    denom = mx - mn
    if denom.abs().item() < 1e-12:
        return torch.zeros_like(s)
    return (s - mn) / denom


def get_gallery_files() -> list[str]:
    """Sorted gallery file names, i.e. the column order of the score matrix."""
    return sorted(os.listdir(GALLERY_DIR))
