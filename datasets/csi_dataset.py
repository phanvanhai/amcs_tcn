"""CSI loader for AMCS-TCN (SenseFi-style layout, e.g. UT-HAR).

Expected layout (the .csv files are actually NumPy binaries):
    <data_path>/data/X_{train,val,test}.csv   -> (N, time_steps, antennas * subcarriers)
    <data_path>/label/y_{train,val,test}.csv  -> (N,) integer labels 0..num_classes-1

The last axis must be antenna-major: [ant1 sub1..S | ant2 sub1..S | ...]
(for UT-HAR verified: adjacent-column correlation breaks exactly at 29|30 and 59|60).
Each sample is returned as (antennas, time_steps, subcarriers).

Normalization: min-max scaling with the TRAIN split's min/max applied to every
split (SenseFi normalizes each split with its own statistics, which leaks test info).
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

_cache = {}


def _load_split(data_path, split):
    key = (os.path.abspath(data_path), split)
    if key not in _cache:
        x = np.load(os.path.join(data_path, 'data', f'X_{split}.csv')).astype(np.float32)
        y = np.load(os.path.join(data_path, 'label', f'y_{split}.csv')).astype(np.int64)
        _cache[key] = (x, y)
    return _cache[key]


def _train_stats(data_path):
    key = (os.path.abspath(data_path), 'stats')
    if key not in _cache:
        x, _ = _load_split(data_path, 'train')
        _cache[key] = (float(x.min()), float(x.max()))
    return _cache[key]


class CSIDataset(Dataset):
    def __init__(self, data_path, split, antennas=3, subcarriers=30, time_steps=250):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")

        x, y = _load_split(data_path, split)
        n = len(x)
        expected = (n, time_steps, antennas * subcarriers)
        if x.reshape(n, -1).shape[1] != time_steps * antennas * subcarriers:
            raise ValueError(f"{split}: data shape {x.shape} incompatible with expected {expected} "
                             f"(time_steps={time_steps}, antennas={antennas}, subcarriers={subcarriers})")

        x_min, x_max = _train_stats(data_path)
        x = (x - x_min) / (x_max - x_min)

        # (N, T, A*S) -> (N, T, A, S) -> (N, A, T, S)
        x = x.reshape(n, time_steps, antennas, subcarriers).transpose(0, 2, 1, 3)

        self.x = torch.from_numpy(np.ascontiguousarray(x))
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
