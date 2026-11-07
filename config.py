"""Cấu hình dùng chung cho main.py và run_all.py.

Đổi dataset: thêm một mục vào DATASETS rồi chạy với --dataset <tên>.
"""
from pathlib import Path

_ROOT = Path(__file__).parent

DATASETS = {
    'UT_HAR': {
        'data_path': str((_ROOT / '../../../CodeForPaper/benchmark/data/UT_HAR').resolve()),
        'antennas': 3,
        'subcarriers': 30,
        'time_steps': 250,
        # Tên nhãn theo đúng chỉ số trong y_*.csv (index 0..6).
        # Code gốc dùng thứ tự chữ cái ['bed','fall','pickup','run','sitdown','standup','walk'],
        # nhưng SenseFi ghi thứ tự UT-HAR là lie down, fall, walk, pickup, run, sit down, stand up.
        # Số mẫu (train) mỗi nhãn: [525, 354, 1172, 396, 967, 320, 243] -> nhãn 2 và 4 nhiều nhất,
        # khớp với walk/run (hoạt động liên tục, sinh nhiều cửa sổ) hơn là pickup/sitdown,
        # nên dùng thứ tự của SenseFi.
        'class_names': ['lie_down', 'fall', 'walk', 'pickup', 'run', 'sit_down', 'stand_up'],
        # Trọng số loss theo chỉ số nhãn. Giá trị dưới đây là của tác giả (giảm nhãn 2, 4 nhiều mẫu,
        # tăng nhãn 5, 6 ít mẫu). Có thể đặt 'auto' (nghịch tần suất) hoặc None (không trọng số).
        'class_weights': [1.0, 1.0, 0.8, 1.0, 0.8, 1.5, 2.0],
    },
}

TRAIN = {
    'batch_size': 64,
    'lr': 1e-3,
    'epochs': 100,
    'lr_step': 20,             # StepLR step_size
    'lr_gamma': 0.5,           # StepLR gamma
    'early_stop_patience': 10,
    'noise_std': 0.02,         # độ lớn nhiễu Gaussian (jittering) khi train, 0 để tắt
    'save_dir': 'model',
    'log_dir': 'logs',
}

# Tiền xử lý tín hiệu theo bài báo (mục "CSI processing"): Low-pass -> Hamming -> Kalman, nối tiếp
# theo trục thời gian. Bài báo không cho tham số -> giá trị dưới đây là do mình chọn, có thể chỉnh.
# Đổi khi chạy: --filters lowpass hamming kalman | --filters none (ablation như Fig. 10)
PREPROCESS = {
    'filters': ['lowpass', 'hamming', 'kalman'],
    'filter_params': {
        'lowpass': {'cutoff': 0.2, 'order': 4},   # cutoff: tỉ lệ so với Nyquist (0..1)
        'hamming': {'window': 5},                 # độ dài cửa sổ Hamming (số mẫu)
        'kalman': {'q_ratio': 1.0},               # Q/R; nhỏ hơn -> làm mượt mạnh hơn
    },
}


def find_data_path(search_root='/kaggle/input', max_depth=5):
    """Tìm thư mục chứa data/X_train.csv và label/y_train.csv (dùng trên Kaggle)."""
    import os

    root = os.path.abspath(search_root)
    if not os.path.isdir(root):
        return None
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root, onerror=lambda e: None):
        if (os.path.isfile(os.path.join(dirpath, 'data', 'X_train.csv'))
                and os.path.isfile(os.path.join(dirpath, 'label', 'y_train.csv'))):
            return dirpath
        if dirpath.count(os.sep) - base_depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
    return None


def build_config(dataset='UT_HAR', **overrides):
    """Gộp cấu hình dataset + train, thêm num_classes, áp dụng các giá trị ghi đè khác None."""
    import torch

    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}, available: {list(DATASETS)}")
    cfg = {**TRAIN, **PREPROCESS, **DATASETS[dataset], 'dataset': dataset}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if 'none' in (cfg.get('filters') or []):
        cfg['filters'] = []

    # Đường dẫn mặc định không tồn tại (vd. trên Kaggle) -> tự tìm trong /kaggle/input
    if overrides.get('data_path') is None and not Path(cfg['data_path']).exists():
        found = find_data_path()
        if found is None:
            raise FileNotFoundError(f"Data not found at {cfg['data_path']} nor under /kaggle/input. "
                                    f"Pass --data_path <folder containing data/ and label/>")
        cfg['data_path'] = found
    cfg['num_classes'] = len(cfg['class_names'])
    cfg.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    w = cfg.get('class_weights')
    if isinstance(w, (list, tuple)) and len(w) != cfg['num_classes']:
        raise ValueError(f"class_weights has {len(w)} entries but there are {cfg['num_classes']} classes")
    return cfg


def model_kwargs(cfg):
    """Tham số chung để khởi tạo mọi model (AMCS_TCN, AMCS_TCN_FC, baselines)."""
    return {
        'num_classes': cfg['num_classes'],
        'antennas': cfg['antennas'],
        'subcarriers': cfg['subcarriers'],
        'time_steps': cfg['time_steps'],
    }


def dataset_kwargs(cfg):
    return {
        'antennas': cfg['antennas'],
        'subcarriers': cfg['subcarriers'],
        'time_steps': cfg['time_steps'],
        'filters': cfg.get('filters', []),
        'filter_params': cfg.get('filter_params', {}),
    }


def add_common_args(parser):
    parser.add_argument('--dataset', default='UT_HAR', choices=list(DATASETS))
    parser.add_argument('--data_path', default=None, help='override dataset folder (contains data/ and label/)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--filters', nargs='+', default=None,
                        choices=['lowpass', 'hamming', 'kalman', 'none'],
                        help="bộ lọc tiền xử lý (mặc định theo config.PREPROCESS); 'none' = dữ liệu thô")
    return parser
