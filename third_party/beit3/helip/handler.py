"""RetrievalHandler variant that adds HNML to the standard ClipLoss.

`ℓ_finetune = ℓ_CLIP + γ · ℓ_margin`  (paper Eq. 7).

Drop-in: same train/eval interface as engine_for_finetuning.RetrievalHandler.
"""
import json

import torch

from engine_for_finetuning import RetrievalHandler
from .loss import hnml_loss


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class HelipRetrievalHandler(RetrievalHandler):
    def __init__(self, n_seeds: int, p: int, gamma: float = 0.3):
        super().__init__()
        self.n_seeds = n_seeds
        self.p = p
        self.gamma = gamma

    def train_batch(self, model, image, language_tokens, padding_mask, image_id, **kwargs):
        loss_clip, vc, lc = model(
            image=image, text_description=language_tokens, padding_mask=padding_mask,
        )
        if torch.isnan(loss_clip).any() or torch.isinf(loss_clip).any():
            print("Loss contains NaN or Inf!")

        # Recompute logits in fp32 for margin stability; same scale as ClipLoss.
        logit_scale = _unwrap(model).logit_scale.exp()
        logits_per_image = (logit_scale * vc.float() @ lc.float().t())
        margin = hnml_loss(logits_per_image, n_seeds=self.n_seeds, p=self.p)

        total = loss_clip + self.gamma * margin
        return {
            "loss": total,
            "loss_clip": loss_clip.detach(),
            "loss_margin": margin.detach(),
        }
