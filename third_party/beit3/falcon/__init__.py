"""FALCON minibatch construction for BEIT3 retrieval finetune.

Paper: "FALCON: False-Negative Aware Learning of Contrastive Negatives
in Vision-Language Alignment" (arXiv:2505.11192).

Phase A MVP — scheduler+sampler only, reward proxy = ΔITC on fixed eval subset.
"""
from .scheduler import FalconScheduler
from .batch_construction import build_falcon_minibatches, build_quantile_features
from .reward import DeltaITCReward
from .trainer import falcon_train_one_epoch
