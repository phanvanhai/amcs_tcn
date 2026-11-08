"""CSI loaders for AMCS-TCN. Every sample is returned as (antennas, time_steps, subcarriers).

Two on-disk formats (config: DATASETS[...]['format']):

'sensefi' (UT-HAR; the .csv files are actually NumPy binaries):
    <data_path>/data/X_{train,val,test}.csv   -> (N, T, antennas * subcarriers), antenna-major
    <data_path>/label/y_{train,val,test}.csv  -> (N,) integer labels 0..num_classes-1
    (UT-HAR verified antenna-major: adjacent-column correlation breaks exactly at 29|30 and 59|60.)

'sshar' (SSHAR ESP / ASUS, one .npy per receiver per recording):
    <data_path>/room_XX/<device>/rx_YY/subject_ZZ/amp_actAA_posBB_dirCC_repDD.npy -> (ant, sub, T)
    The receivers in `rx_list` are stacked into rx * ant streams (ESP32: 3 rx x 1 ant = 3 streams,
    56 subcarriers). Labels: act in `labels` -> index in that list. Two splits (split_by), both 7:1:2:
      'user'   (cross-user, default): user case k of the benchmark (SSHAR_USER_CASES[k] = 8 train
               subjects); the last of those 8 is held out as the val subject, the remaining subjects
               are test. Case 0: train 1,2,3,4,9,10,11 / val 12 / test 13,14 -> 4480 / 640 / 1280.
      'repeat' (same subjects in every set): exactly 7:1:2 per (subject, class): test = dir0 rep 9-10 /
               dir!=0 rep 5 (same test set as the benchmark); val = dir0 rep 8 / dir!=0 rep 4 of one
               direction (dir01 for odd positions, dir02 for even ones); train = the rest.

Pipeline (both formats): optional denoising (datasets/preprocess.py, per series, along time, at the
raw rate) -> optional time downsampling (mean over blocks of raw_T // time_steps samples) ->
optional scaling. normalize='minmax' uses the TRAIN split's global min/max (computed after the steps
above) for every split (SenseFi normalizes each split with its own statistics, which leaks test info);
normalize='none' feeds the values as-is.
"""
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.preprocess import apply_filters

_cache = {}

SSHAR_PATTERN = re.compile(r'act(\d+)_pos(\d+)_dir(\d+)_rep(\d+)')
SSHAR_SPLIT_VERSION = 'v2'  # đổi khi sửa cách chia để cache trên đĩa không bị dùng nhầm

# Các case chia theo người của benchmark (CodeForPaper/benchmark/datasets.py, USER_CASES['sshar']):
# danh sách 8 người dùng để train; người còn lại là test. Ở đây người cuối danh sách được tách làm val.
SSHAR_USER_CASES = [
    [1, 2, 3, 4, 9, 10, 11, 12],
    [1, 3, 4, 9, 11, 12, 13, 14],
    [1, 2, 4, 9, 10, 12, 13, 14],
    [2, 3, 4, 10, 11, 12, 13, 14],
    [1, 2, 4, 9, 10, 11, 13, 14],
]

# Cache mảng thô của từng split ra đĩa (.npy) để lần chạy sau trong cùng phiên nạp nhanh.
# Mặc định bật trên Kaggle (/tmp/csi_cache); đặt CSI_CACHE_DIR="" để tắt, hoặc trỏ thư mục khác.
CACHE_DIR = os.environ.get('CSI_CACHE_DIR', '/tmp/csi_cache' if os.path.isdir('/kaggle/working') else '')
LOAD_WORKERS = int(os.environ.get('CSI_LOAD_WORKERS', '16'))  # số luồng đọc file song song


def _log(msg):
    print(msg, flush=True)


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


def sshar_user_split_of(subject_id, user_case=0):
    """Cross-user 7:1:2 split: 7 train subjects, 1 val subject, the rest test (see module docstring)."""
    train_ids = SSHAR_USER_CASES[user_case]
    if subject_id not in train_ids:
        return 'test'
    return 'val' if subject_id == train_ids[-1] else 'train'


def sshar_split_of(pos, direction, rep):
    """7:1:2 repetition split (see module docstring)."""
    if direction == 0:  # 10 repetitions: 1-7 train, 8 val, 9-10 test
        return 'test' if rep > 8 else 'val' if rep == 8 else 'train'
    # 2 directions x 5 repetitions: rep 5 test; rep 4 of one direction (alternating by pos) val
    if rep == 5:
        return 'test'
    val_dir = 1 if pos % 2 else 2
    return 'val' if rep == 4 and direction == val_dir else 'train'


_index_cache = {}


def _sshar_index(data_path, device, rx_list, labels, split_by='user', user_case=0):
    """[(split, label_idx, [file per rx])] for every recording present in all receivers.

    Built once per argument set; existence is checked with one directory listing per
    (receiver, subject) instead of one stat per file (much faster on network file systems).
    """
    if split_by not in ('user', 'repeat'):
        raise ValueError(f"split_by must be 'user' or 'repeat', got {split_by!r}")
    if split_by == 'user' and not 0 <= user_case < len(SSHAR_USER_CASES):
        raise ValueError(f"user_case must be 0..{len(SSHAR_USER_CASES) - 1}, got {user_case}")
    key = _key(os.path.abspath(data_path), device, rx_list, labels, split_by, user_case)
    if key in _index_cache:
        return _index_cache[key]
    root, label_map, index = Path(data_path), {a: i for i, a in enumerate(labels)}, []
    for room in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('room')):
        first = room / device / rx_list[0]
        if not first.is_dir():
            continue
        for subject in sorted(p.name for p in first.iterdir() if p.is_dir() and p.name.startswith('subject')):
            listing = {rx: set(os.listdir(room / device / rx / subject))
                       if (room / device / rx / subject).is_dir() else set() for rx in rx_list}
            for name in sorted(n for n in listing[rx_list[0]] if n.endswith('.npy')):
                m = SSHAR_PATTERN.search(name)
                if m is None or int(m.group(1)) not in label_map:
                    continue
                act, pos, direction, rep = map(int, m.groups())
                missing = [rx for rx in rx_list if name not in listing[rx]]
                if missing:
                    raise FileNotFoundError(f"{name} missing in {missing} ({room.name}/{subject})")
                files = [room / device / rx / subject / name for rx in rx_list]
                split = (sshar_user_split_of(int(subject.split('_')[1]), user_case) if split_by == 'user'
                         else sshar_split_of(pos, direction, rep))
                index.append((split, label_map[act], files))
    if not index:
        raise FileNotFoundError(f"No SSHAR files for device={device!r}, rx={rx_list}, labels={labels} "
                                f"under {data_path}")
    _index_cache[key] = index
    return index


def sshar_split_subjects(data_path, device, rx_list, labels, split_by='user', user_case=0):
    """{split: sorted subject folder names} for logging / reports."""
    out = {}
    for split, _, files in _sshar_index(data_path, device, rx_list, labels, split_by, user_case):
        out.setdefault(split, set()).add(files[0].parent.name)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _load_sshar(data_path, split, antennas, subcarriers, device, rx_list, labels, split_by='user',
                user_case=0):
    cache = None
    if CACHE_DIR:
        h = hashlib.md5(_key(os.path.abspath(data_path), split, device, rx_list, labels, split_by,
                             user_case, SSHAR_SPLIT_VERSION).encode()).hexdigest()[:12]
        cache = os.path.join(CACHE_DIR, f'sshar_{device}_{split}_{h}')
        if os.path.isfile(cache + '_y.npy'):
            x, y = np.load(cache + '_x.npy'), np.load(cache + '_y.npy')
            _log(f"   [{split}] loaded from cache {cache}_*.npy: {x.shape}")
            return x, y

    t0 = time.time()
    items = [(lab, files) for s, lab, files in _sshar_index(data_path, device, rx_list, labels, split_by, user_case)
             if s == split]
    if not items:
        raise ValueError(f"SSHAR split {split!r} is empty")
    first = np.load(items[0][1][0])  # (ant, sub, T)
    ant_per_rx, sub, t_raw = first.shape
    if ant_per_rx * len(rx_list) != antennas or sub != subcarriers:
        raise ValueError(f"SSHAR file shape {first.shape} x {len(rx_list)} rx gives "
                         f"{ant_per_rx * len(rx_list)} streams x {sub} subcarriers, config says "
                         f"antennas={antennas}, subcarriers={subcarriers}")
    x = np.empty((len(items), antennas, t_raw, sub), dtype=np.float32)

    def read(i):
        for r, f in enumerate(items[i][1]):
            a = np.load(f)
            if a.shape != first.shape:
                raise ValueError(f"{f}: shape {a.shape}, expected {first.shape}")
            x[i, r * ant_per_rx:(r + 1) * ant_per_rx] = a.transpose(0, 2, 1)  # (ant, T, sub)

    n, step = len(items), max(1, len(items) // 10)
    _log(f"   [{split}] reading {n} samples x {len(rx_list)} rx = {n * len(rx_list)} files ({LOAD_WORKERS} threads)...")
    with ThreadPoolExecutor(max_workers=LOAD_WORKERS) as ex:
        for done, _ in enumerate(ex.map(read, range(n)), 1):
            if done % step == 0 or done == n:
                _log(f"   [{split}] {done}/{n} ({time.time() - t0:.0f}s)")
    y = np.array([lab for lab, _ in items], dtype=np.int64)

    if cache:
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.save(cache + '_x.npy', x)
            np.save(cache + '_y.npy', y)  # ghi sau cùng: có _y nghĩa là cache đầy đủ
        except OSError as e:
            _log(f"   [{split}] could not write cache ({e})")
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
