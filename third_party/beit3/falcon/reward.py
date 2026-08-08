"""ΔITC reward — proxy for cross-modal alignment improvement.

Paper uses ΔMLM. BEIT3 retrieval finetune has no MLM head, so we substitute
the change in contrastive loss measured on a *fixed* held-out subset:

    Δ = L_ITC(model_before; eval_subset) − L_ITC(model_after; eval_subset)

A positive Δ ⇒ the gradient step improved alignment on the held-out set.
"""
from typing import Iterable, List
import torch
import torch.nn.functional as F


@torch.no_grad()
def _itc_loss(image_cls: torch.Tensor, text_cls: torch.Tensor,
              logit_scale: torch.Tensor) -> torch.Tensor:
    logits_per_image = logit_scale * image_cls @ text_cls.t()
    n = image_cls.shape[0]
    target = torch.arange(n, device=image_cls.device)
    return 0.5 * (F.cross_entropy(logits_per_image, target)
                  + F.cross_entropy(logits_per_image.t(), target))


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


class DeltaITCReward:
    """Caches a fixed held-out eval subset and exposes `loss(model)`.

    The eval batch is pulled from `eval_loader` once at construction, kept
    resident on GPU, and re-used every step.
    """

    def __init__(self, model, eval_loader: Iterable, device, n_pairs: int = 64):
        self.device = device
        images: List[torch.Tensor] = []
        tokens: List[torch.Tensor] = []
        padmasks: List[torch.Tensor] = []
        collected = 0
        for batch in eval_loader:
            images.append(batch["image"])
            tokens.append(batch["language_tokens"])
            padmasks.append(batch["padding_mask"])
            collected += batch["image"].shape[0]
            if collected >= n_pairs:
                break
        self.image = torch.cat(images, dim=0)[:n_pairs].to(device, non_blocking=True)
        self.tokens = torch.cat(tokens, dim=0)[:n_pairs].to(device, non_blocking=True)
        self.padmask = torch.cat(padmasks, dim=0)[:n_pairs].to(device, non_blocking=True)

    @torch.no_grad()
    def loss(self, model) -> torch.Tensor:
        """Return L_ITC on the cached eval subset, in eval mode (no dropout).
        Restores train mode on exit.
        """
        was_training = model.training
        model.eval()
        try:
            with torch.amp.autocast("cuda"):
                vc, _ = model(image=self.image, only_infer=True)
                _, lc = model(text_description=self.tokens,
                              padding_mask=self.padmask, only_infer=True)
            logit_scale = _unwrap(model).logit_scale.exp()
            l = _itc_loss(vc.float(), lc.float(), logit_scale.float())
        finally:
            model.train(was_training)
        return l.detach()
