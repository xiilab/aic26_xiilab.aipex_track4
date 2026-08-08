"""TIC — Text Intra-modal Contrastive loss.

Borrowed from FG-CLIP 2 (arXiv:2510.10921v2, §3.2). Purpose: enforce that the
text encoder maps semantically-similar-but-distinct captions apart, while
preserving the image-text alignment provided by the standard CLIP loss.

L_TIC = mean_i  log Σ_{T_m ∈ HardNeg(T_i)}  exp( S(T_i, T_m) )

  - HardNeg(T_i): top-K most-similar OTHER texts in the batch with
    sim ≤ τ_fn (texts above τ_fn are treated as positives/paraphrases and
    excluded as false negatives).
  - Optionally also exclude same-image_id texts (when the dataloader supplies
    multiple captions per image — pure multistyle case).
"""
from .loss import tic_loss
from .handler import TicRetrievalHandler
