"""CSI loaders for AMCS-TCN. Every sample is returned as (antennas, time_steps, subcarriers).

Two on-disk formats (config: DATASETS[...]['format']):

'sensefi' (UT-HAR; the .csv files are actually NumPy binaries):
    <data_path>/data/X_{train,val,test}.csv   -> (N, T, antennas * subcarriers), antenna-major
    <data_path>/label/y_{train,val,test}.csv  -> (N,) integer labels 0..num_classes-1
    (UT-HAR verified antenna-major: adjacent-column correlation breaks exactly at 29|30 and 59|60.)

'sshar' (SSHAR ESP / ASUS, one .npy per receiver per recording):
    <data_path>/room_XX/<device>/rx_YY/subject_ZZ/amp_actAA_posBB_dirCC_repDD.npy -> (ant, sub, T)
    The receivers in `rx_list` are stacked into rx * ant streams (ESP32: 3 rx x 1 ant = 3 streams,
    56 subcarriers). Labels: act in `labels` -> index in that list. Split by repetition, like the
    benchmark (test = dir0 rep > 8, dir!=0 rep > 4) -> 80 / 20. With holdout_val=True one repetition
    per group is moved from train to val (val = dir0 rep 8, dir!=0 rep 4) -> ~70 / 10 / 20; with
    holdout_val=False there is no 'val' split (the caller uses test for validation).

Pipeline (both formats): optional denoising (datasets/preprocess.py, per series, along time, at the
raw rate) -> optional time downsampling (mean over blocks of raw_T // time_steps samples) ->
optional scaling. normalize='minmax' uses the TRAIN split's global min/max (computed after the steps
above) for every split (SenseFi normalizes each split with its own statistics, which leaks test info);
normalize='none' feeds the values as-is.
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.preprocess import apply_filters

_cache = {}

SSHAR_PATTERN = re.compile(r'act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)')


def clear_cache():
    """Free the raw / filtered arrays kept between CSIDataset constructions."""
    _cache.clear()


# ---------------------------------------------------------------- raw loaders -> (N, A, T, S) float32

def _load_sensefi(data_path, split, antennas, subcarriers):
    x = np.load(os.path.join(data_path, 'data', f'X_{split}.csv')).astype(np.float32)
    y = np.load(os.path.join(data_path, 'label', f'y_{split}.csv')).astype(np.int64)
    n = len(x)
    if x.reshape(n, -1).shape[1] % (antennas * subcarriers):
        raise ValueError(f"{split}: data shape {x.shape} incompatible with antennas={antennas}, "
                         f"subcarriers={subcarriers}")
    x = x.reshape(n, -1, antennas, subcarriers).transpose(0, 2, 1, 3)  # (N, T, A, S) -> (N, A, T, S)
    return np.ascontiguousarray(x), y


def sshar_split_of(direction, rep, holdout_val=True):
    """Repetition split: benchmark train/test boundary; optionally last train repetition -> val."""
    last_train = 8 if direction == 0 else 4
    if rep > last_train:
        return 'test'
    return 'val' if holdout_val and rep == last_train else 'train'


def _sshar_index(data_path, device, rx_list, labels, holdout_val=True):
    """[(split, label_idx, [file per rx])] for every recording present in all receivers."""
    root, label_map, index = Path(data_path), {a: i for i, a in enumerate(labels)}, []
    for room in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('room')):
        first = room / device / rx_list[0]
        if not first.is_dir():
            continue
        for subject in sorted(p for p in first.iterdir() if p.is_dir() and p.name.startswith('subject')):
            for f in sorted(subject.glob('*.npy')):
                m = SSHAR_PATTERN.search(f.name)
                if m is None or int(m.group(1)) not in label_map:
                    continue
                act, _, direction, rep = map(int, m.groups())
                files = [room / device / rx / subject.name / f.name for rx in rx_list]
                if not all(p.is_file() for p in files):
                    raise FileNotFoundError(f"{f.name} missing for some receiver in {rx_list} ({subject})")
                index.append((sshar_split_of(direction, rep, holdout_val), label_map[act], files))
    if not index:
        raise FileNotFoundError(f"No SSHAR files for device={device!r}, rx={rx_list}, labels={labels} "
                                f"under {data_path}")
    return index


def _load_sshar(data_path, split, antennas, subcarriers, device, rx_list, labels, holdout_val=True):
    items = [(lab, files) for s, lab, files in _sshar_index(data_path, device, rx_list, labels, holdout_val)
             if s == split]
    if not items:
        raise ValueError(f"SSHAR split {split!r} is empty (holdout_val={holdout_val})")
    first = np.load(items[0][1][0])  # (ant, sub, T)
    ant_per_rx, sub, t_raw = first.shape
    if ant_per_rx * len(rx_list) != antennas or sub != subcarriers:
        raise ValueError(f"SSHAR file shape {first.shape} x {len(rx_list)} rx gives "
                         f"{ant_per_rx * len(rx_list)} streams x {sub} subcarriers, config says "
                         f"antennas={antennas}, subcarriers={subcarriers}")
    x = np.empty((len(items), antennas, t_raw, sub), dtype=np.float32)
    for i, (_, files) in enumerate(items):
        for r, f in enumerate(files):
            a = np.load(f)
            if a.shape != first.shape:
                raise ValueError(f"{f}: shape {a.shape}, expected {first.shape}")
            x[i, r * ant_per_rx:(r + 1) * ant_per_rx] = a.transpose(0, 2, 1)  # (ant, T, sub)
    y = np.array([lab for lab, _ in items], dtype=np.int64)
    return x, y


# ---------------------------------------------------------------- cached pipeline

def _key(*parts):
    return json.dumps(parts, sort_keys=True, default=str)


def _load_raw(data_path, split, fmt, antennas, subcarriers, fmt_kwargs):
    key = _key('raw', os.path.abspath(data_path), split, fmt, antennas, subcarriers, fmt_kwargs)
    if key not in _cache:
        if fmt == 'sensefi':
            _cache[key] = _load_sensefi(data_path, split, antennas, subcarriers)
        elif fmt == 'sshar':
            _cache[key] = _load_sshar(data_path, split, antennas, subcarriers, **fmt_kwargs)
        else:
            raise ValueError(f"Unknown dataset format {fmt!r} (expected 'sensefi' or 'sshar')")
    return _cache[key]


def _load_processed(data_path, split, fmt, antennas, subcarriers, fmt_kwargs, time_steps, filters,
                    filter_params):
    key = _key('proc', os.path.abspath(data_path), split, fmt, antennas, subcarriers, fmt_kwargs,
               time_steps, filters, filter_params)
    if key not in _cache:
        x, y = _load_raw(data_path, split, fmt, antennas, subcarriers, fmt_kwargs)
        n, a, t_raw, s = x.shape
        if filters:  # filter each (sample, stream, subcarrier) series along time, at the raw rate
            x = apply_filters(x.reshape(n * a, t_raw, s), filters, filter_params).reshape(x.shape)
        if t_raw != time_steps:
            if time_steps > t_raw or t_raw % time_steps:
                raise ValueError(f"Raw length T={t_raw} cannot be downsampled to time_steps={time_steps} "
                                 f"(must divide it)")
            f = t_raw // time_steps
            x = x.reshape(n, a, time_steps, f, s).mean(axis=3, dtype=np.float32)
        _cache[key] = (x, y)
    return _cache[key]


class CSIDataset(Dataset):
    def __init__(self, data_path, split, antennas=3, subcarriers=30, time_steps=250,
                 filters=(), filter_params=None, normalize='minmax', fmt='sensefi', fmt_kwargs=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
        if normalize not in ('minmax', 'none'):
            raise ValueError(f"normalize must be 'minmax' or 'none', got {normalize!r}")
        args = (fmt, antennas, subcarriers, fmt_kwargs or {}, time_steps, list(filters or []),
                filter_params or {})

        x, y = _load_processed(data_path, split, *args)
        if normalize == 'minmax':
            x_train, _ = _load_processed(data_path, 'train', *args)
            x_min, x_max = float(x_train.min()), float(x_train.max())
            x = (x - x_min) / (x_max - x_min)

        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
