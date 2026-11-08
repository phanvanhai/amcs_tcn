"""Cấu hình dùng chung cho main.py và run_all.py.

Đổi dataset: thêm một mục vào DATASETS rồi chạy với --dataset <tên>.
"""
from pathlib import Path

_ROOT = Path(__file__).parent

DATASETS = {
    'UT_HAR': {
        'format': 'sensefi',   # data/X_{split}.csv + label/y_{split}.csv (xem datasets/csi_dataset.py)
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
        'paper_reference': True,   # in số liệu bài báo trong summary.txt (bài báo có chạy UT-HAR)
    },
    # SSHAR, thiết bị ESP32: 3 receiver x 1 anten, 56 subcarrier, 1000 điểm (5 s).
    # room_XX/esp/rx_YY/subject_ZZ/amp_actAA_posBB_dirCC_repDD.npy, mỗi file (1, 56, 1000).
    # 10 người x 8 hành động x 80 mẫu = 6400 mẫu (800/lớp). Chia 7:1:2 -> 4480/640/1280:
    #   split_by='user'   (mặc định, cross-user): case k của benchmark, 7 người train / 1 người val / 2 người test
    #                     case 0: train 1,2,3,4,9,10,11 | val 12 | test 13,14 (đổi bằng --user_case 0..4,
    #                     bảng 5 case: docs/kaggle_run_convention.md mục 4)
    #   split_by='repeat' : chia theo lần lặp, người nào cũng có ở cả 3 tập (xem docs/SSHAR_ESP_protocol.md)
    'SSHAR_ESP': {
        'format': 'sshar',
        'data_path': str((_ROOT / '../../../CodeForPaper/benchmark/data/sshar_amp_interp').resolve()),
        'fmt_kwargs': {'device': 'esp', 'rx_list': ['rx_00', 'rx_01', 'rx_02'],
                       'labels': [1, 2, 3, 4, 5, 6, 7, 8],    # act id trong tên file -> nhãn 0..7
                       'split_by': 'user', 'user_case': 0},
        'antennas': 3,          # = số rx x số anten mỗi rx -> 3 luồng MCS
        'subcarriers': 56,
        'raw_time_steps': 1000,
        # Độ dài đưa vào model; nhỏ hơn 1000 thì lấy trung bình khối (vd. --time_steps 250).
        'time_steps': 1000,     # T=1000, batch 32: ~11 GB activation khi train
        'class_names': ['act01', 'act02', 'act03', 'act04', 'act05', 'act06', 'act07', 'act08'],
        'class_weights': None,  # dữ liệu cân bằng
        'paper_reference': False,
    },
}

# Thiết lập thí nghiệm chung (benchmark): batch 32, 100 epoch, Adam, chọn model theo best val accuracy,
# LR theo bài gốc (AMCS-TCN, Table 1: Adam, lr 0.001; bài dùng batch 64), dừng sớm patience 10 theo
# val acc. StepLR giữ như code gốc của tác giả.
TRAIN = {
    'batch_size': 32,
    'lr': 1e-3,
    'epochs': 100,
    'lr_step': 20,             # StepLR step_size
    'lr_gamma': 0.5,           # StepLR gamma
    'early_stop_patience': 10,   # dừng sớm: 10 epoch liên tiếp val acc không cải thiện (protocol chung); 0 = tắt
    'noise_std': 0.02,         # độ lớn nhiễu Gaussian (jittering) khi train, 0 để tắt
    'save_dir': 'model',
    'log_dir': 'logs',
    # Hiện chỉ chạy 1 seed (42); thêm seed khi cần, vd. [42, 3407, 1024, 2024, 0] (hoặc --seeds ...).
    'seeds': [42],
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
    # Chuẩn hoá sau khi lọc: 'minmax' (min/max toàn cục của tập train, tự thêm) | 'none' (giữ giá trị
    # thô; repo gốc không có data loader, bài báo không nhắc chuẩn hoá). Lưu ý noise_std là độ lớn
    # tuyệt đối: với 'none' (UT-HAR thô std ~5.9) jitter 0.02 yếu hơn nhiều so với khi minmax.
    'normalize': 'minmax',
}


def _is_data_root(path, fmt, fmt_kwargs):
    import os
    if fmt == 'sensefi':
        return (os.path.isfile(os.path.join(path, 'data', 'X_train.csv'))
                and os.path.isfile(os.path.join(path, 'label', 'y_train.csv')))
    if fmt == 'sshar':
        dev, rx = fmt_kwargs['device'], fmt_kwargs['rx_list'][0]
        return any(d.startswith('room') and os.path.isdir(os.path.join(path, d, dev, rx))
                   for d in os.listdir(path))
    raise ValueError(f"Unknown dataset format {fmt!r}")


def find_data_path(fmt='sensefi', fmt_kwargs=None, search_root='/kaggle/input', max_depth=5):
    """Tìm thư mục gốc của dataset theo định dạng (dùng trên Kaggle)."""
    import os

    root = os.path.abspath(search_root)
    if not os.path.isdir(root):
        return None
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, _ in os.walk(root, onerror=lambda e: None):
        if _is_data_root(dirpath, fmt, fmt_kwargs or {}):
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
    split_over = {k: overrides.pop(k) for k in ('split_by', 'user_case') if overrides.get(k) is not None}
    overrides = {k: v for k, v in overrides.items() if k not in ('split_by', 'user_case')}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if split_over:
        if cfg['format'] != 'sshar':
            raise ValueError(f"--split / --user_case only apply to SSHAR datasets (dataset {dataset!r} has a fixed split)")
        cfg['fmt_kwargs'] = {**cfg['fmt_kwargs'], **split_over}
    if 'none' in (cfg.get('filters') or []):
        cfg['filters'] = []

    # Đường dẫn mặc định không tồn tại (vd. trên Kaggle) -> tự tìm trong /kaggle/input
    if overrides.get('data_path') is None and not Path(cfg['data_path']).exists():
        found = find_data_path(cfg['format'], cfg.get('fmt_kwargs'))
        if found is None:
            raise FileNotFoundError(f"Data not found at {cfg['data_path']} nor under /kaggle/input. "
                                    f"Pass --data_path <dataset root folder>")
        cfg['data_path'] = found
    cfg['num_classes'] = len(cfg['class_names'])
    labels = cfg.get('fmt_kwargs', {}).get('labels')
    if labels is not None and len(labels) != cfg['num_classes']:
        raise ValueError(f"fmt_kwargs.labels has {len(labels)} entries but there are {cfg['num_classes']} classes")
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
        'normalize': cfg.get('normalize', 'minmax'),
        'fmt': cfg['format'],
        'fmt_kwargs': cfg.get('fmt_kwargs', {}),
    }


def add_common_args(parser):
    parser.add_argument('--dataset', default='UT_HAR', choices=list(DATASETS))
    parser.add_argument('--data_path', default=None, help='override dataset folder (contains data/ and label/)')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--split', dest='split_by', default=None, choices=['user', 'repeat'],
                        help="SSHAR: 'user' = cross-user (mặc định), 'repeat' = theo lần lặp")
    parser.add_argument('--user_case', type=int, default=None,
                        help='SSHAR cross-user: case 0..4 (mặc định 0: train 1,2,3,4,9,10,11 | val 12 | test 13,14)')
    parser.add_argument('--patience', type=int, default=None,
                        help='dừng sớm sau N epoch liên tiếp val acc không cải thiện (mặc định 10; 0 = tắt)')
    parser.add_argument('--time_steps', type=int, default=None,
                        help='độ dài thời gian đưa vào model (phải chia hết độ dài gốc; giảm bằng trung bình khối)')
    parser.add_argument('--device', default=None)
    parser.add_argument('--filters', nargs='+', default=None,
                        choices=['lowpass', 'hamming', 'kalman', 'none'],
                        help="bộ lọc tiền xử lý (mặc định theo config.PREPROCESS); 'none' = dữ liệu thô")
    parser.add_argument('--normalize', default=None, choices=['minmax', 'none'],
                        help="chuẩn hoá sau lọc (mặc định theo config.PREPROCESS); 'none' = không chuẩn hoá")
    return parser
