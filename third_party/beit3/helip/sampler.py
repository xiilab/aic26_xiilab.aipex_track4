"""HELIP batch sampler — composite batches (seeds + p hard pairs per seed).

For each batch:
  - Pick |B'| = n_seeds unique seed indices.
  - For each seed s_k, randomly choose p hard pairs from H_{s_k}.
  - Final batch = [s_0, s_1, ..., s_{n_seeds-1},
                   hp_{0,0}, hp_{0,1}, ..., hp_{0,p-1},
                   hp_{1,0}, ..., hp_{n_seeds-1, p-1}]
  - Total size: n_seeds * (1 + p). Caller should set batch_size accordingly.

The deterministic position layout lets the loss/handler locate hard-pair
positions without extra metadata.
"""
from typing import Iterator, List
import numpy as np
import torch
from torch.utils.data import Sampler


class HelipBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset_len: int, hard_pairs: np.ndarray,
                 n_seeds: int, p: int, shuffle: bool = True, drop_last: bool = True,
                 seed: int = 0, outlier_mask: np.ndarray = None):
        super().__init__(data_source=None)
        assert hard_pairs.shape[0] == dataset_len, \
            f"hard_pairs.shape[0]={hard_pairs.shape[0]} != dataset_len={dataset_len}"
        assert hard_pairs.shape[1] >= p, \
            f"hard_pairs has only {hard_pairs.shape[1]} candidates per anchor, need ≥ p={p}"
        self.dataset_len = dataset_len
        self.hard_pairs = hard_pairs                                       # (N, k)
        self.n_seeds = n_seeds
        self.p = p
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)
        self._epoch = 0
        # Paper §3.2 noise-cleanup: only non-outlier anchors are eligible as seeds.
        # Outlier anchors are still allowed as hard-pair *targets* (they show up
        # in some other anchor's H_i*), but never as a seed driving the batch.
        if outlier_mask is not None:
            assert outlier_mask.shape == (dataset_len,)
            self.valid_seeds = np.flatnonzero(~outlier_mask.astype(bool))
        else:
            self.valid_seeds = np.arange(dataset_len)

    @property
    def batch_size(self) -> int:
        return self.n_seeds * (1 + self.p)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            rng = np.random.default_rng(self.rng.integers(1 << 30) + self._epoch)
            order = rng.permutation(self.valid_seeds)
        else:
            order = self.valid_seeds.copy()

        k_total = self.hard_pairs.shape[1]
        for start in range(0, len(order), self.n_seeds):
            seeds = order[start:start + self.n_seeds]
            if len(seeds) < self.n_seeds and self.drop_last:
                break
            # Sample p hard pairs per seed.
            picks = self.hard_pairs[seeds]                                 # (n_seeds, k)
            if self.p < k_total:
                col = np.stack([
                    np.random.default_rng().choice(k_total, size=self.p, replace=False)
                    for _ in range(len(seeds))
                ])                                                         # (n_seeds, p)
                hps = np.take_along_axis(picks, col, axis=1)               # (n_seeds, p)
            else:
                hps = picks[:, :self.p]
            batch = list(map(int, seeds.tolist())) + list(map(int, hps.flatten().tolist()))
            yield batch

    def __len__(self) -> int:
        return len(self.valid_seeds) // self.n_seeds
