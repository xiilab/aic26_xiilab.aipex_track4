"""RetrievalHandler variant that adds the FG-CLIP 2 TIC loss to ClipLoss.

ℓ_total = ℓ_CLIP + λ · ℓ_TIC          (paper λ ≈ 0.1)

Composable with HELIP (TIC operates on text, HNML on image-text margin), so
this handler can either be used standalone or wrapped further.
"""
import torch

from engine_for_finetuning import RetrievalHandler
from .loss import tic_loss


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class TicRetrievalHandler(RetrievalHandler):
    def __init__(self, lam: float = 0.1, top_k: int = 10,
                 fn_threshold: float = 0.95, scale_logits: bool = False,
                 use_image_id_mask: bool = True):
        super().__init__()
        self.lam = lam
        self.top_k = top_k
        self.fn_threshold = fn_threshold
        self.scale_logits = scale_logits
        self.use_image_id_mask = use_image_id_mask

    def train_batch(self, model, image, language_tokens, padding_mask, image_id, **kwargs):
        loss_clip, vc, lc = model(
            image=image, text_description=language_tokens, padding_mask=padding_mask,
        )
        if torch.isnan(loss_clip).any() or torch.isinf(loss_clip).any():
            print("Loss contains NaN or Inf!")

        ls = _unwrap(model).logit_scale.exp() if self.scale_logits else None
        ids = image_id if self.use_image_id_mask else None
        l_tic = tic_loss(
            lc.float(),
            image_ids=ids,
            top_k=self.top_k,
            fn_threshold=self.fn_threshold,
            logit_scale=ls,
        )

        total = loss_clip + self.lam * l_tic
        return {
            "loss": total,
            "loss_clip": loss_clip.detach(),
            "loss_tic":  l_tic.detach(),
        }
