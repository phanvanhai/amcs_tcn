"""CSI loader for AMCS-TCN (SenseFi-style layout, e.g. UT-HAR).

Expected layout (the .csv files are actually NumPy binaries):
    <data_path>/data/X_{train,val,test}.csv   -> (N, time_steps, antennas * subcarriers)
    <data_path>/label/y_{train,val,test}.csv  -> (N,) integer labels 0..num_classes-1

The last axis must be antenna-major: [ant1 sub1..S | ant2 sub1..S | ...]
(for UT-HAR verified: adjacent-column correlation breaks exactly at 29|30 and 59|60).
Each sample is returned as (antennas, time_steps, subcarriers).

Pipeline: optional denoising (datasets/preprocess.py: low-pass -> Hamming -> Kalman, per sample)
then optional scaling (normalize='minmax'): min-max with the TRAIN split's global min/max (computed
after filtering) applied to every split (SenseFi normalizes each split with its own statistics, which
leaks test info). normalize='none' feeds the (filtered) values as-is; the author's repo ships no
loader and the paper names no normalization, so 'none' is the closest to the original code.
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.preprocess import apply_filters

_cache = {}


def _load_raw(data_path, split):
    key = ('raw', os.path.abspath(data_path), split)
    if key not in _cache:
        x = np.load(os.path.join(data_path, 'data', f'X_{split}.csv')).astype(np.float32)
        y = np.load(os.path.join(data_path, 'label', f'y_{split}.csv')).astype(np.int64)
        _cache[key] = (x, y)
    return _cache[key]


def _load_filtered(data_path, split, time_steps, filters, filter_params):
    key = ('filtered', os.path.abspath(data_path), split, tuple(filters),
           json.dumps(filter_params, sort_keys=True))
    if key not in _cache:
        x, y = _load_raw(data_path, split)
        if filters:
            n = len(x)
            x = apply_filters(x.reshape(n, time_steps, -1), filters, filter_params).reshape(x.shape)
        _cache[key] = (x, y)
    return _cache[key]


class CSIDataset(Dataset):
    def __init__(self, data_path, split, antennas=3, subcarriers=30, time_steps=250,
                 filters=(), filter_params=None, normalize='minmax'):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
        if normalize not in ('minmax', 'none'):
            raise ValueError(f"normalize must be 'minmax' or 'none', got {normalize!r}")
        filters, filter_params = list(filters or []), filter_params or {}

        x_raw, _ = _load_raw(data_path, split)
        n = len(x_raw)
        if x_raw.reshape(n, -1).shape[1] != time_steps * antennas * subcarriers:
            raise ValueError(f"{split}: data shape {x_raw.shape} incompatible with expected "
                             f"{(n, time_steps, antennas * subcarriers)} "
                             f"(time_steps={time_steps}, antennas={antennas}, subcarriers={subcarriers})")

        x, y = _load_filtered(data_path, split, time_steps, filters, filter_params)
        if normalize == 'minmax':
            x_train, _ = _load_filtered(data_path, 'train', time_steps, filters, filter_params)
            x_min, x_max = float(x_train.min()), float(x_train.max())
            x = (x - x_min) / (x_max - x_min)

        # (N, T, A*S) -> (N, T, A, S) -> (N, A, T, S)
        x = x.reshape(n, time_steps, antennas, subcarriers).transpose(0, 2, 1, 3)

        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
