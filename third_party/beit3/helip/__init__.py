"""HELIP: Hard Pair Refinement (Wang et al., NAACL 2025).

Paper: "Getting More Juice Out of Your Data: Hard Pair Refinement Enhances
Visual-Language Models Without Extra Data" (arXiv:2305.05208).

Components:
  - mining.py : FastHPM — offline hard-pair selection. Run once with a
                pretrained BEIT3 ckpt (e.g. stage-1 best).
  - sampler.py: HelipBatchSampler — yields composite batches
                (seeds + p hard pairs per seed).
  - loss.py   : Hard Negative Margin Loss (HNML).
  - handler.py: RetrievalHandler variant that adds HNML to the ClipLoss.
"""
from .sampler import HelipBatchSampler
from .loss import hnml_loss
from .handler import HelipRetrievalHandler
