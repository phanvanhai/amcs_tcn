"""Chạy mô hình với nhiều seed, lưu chi tiết từng seed và bảng tổng hợp mean ± std.

Hai chế độ (model luôn được chọn theo best val accuracy, có early stop trên val):
    --mode dev    (mặc định) seed config['dev_seeds'] (hiện [42]) -> metric đo trên VAL. Tập test
                  không được nạp, dùng cho mọi điều chỉnh/so sánh.
    --mode final  1 lần, seed config['final_seed'] (42): train, chọn checkpoint theo val rồi đo trên
                  TEST; cùng checkpoint đó cũng được đo trên VAL (file trong seed_<s>/val/).

Cấu trúc thư mục kết quả:
results/<YYYYmmdd-HHMMSS>_<dataset>/
    config.json, environment.json
    <Model>/
        seed_<s>/   best.pth, history.csv, curve.png, metrics.json, test_metrics_and_size.txt,
                    classification_report.txt, per_class_metrics.csv, confusion_matrix.csv,
                    cm.png, cm_normalized.png, predictions.csv
        per_seed.csv          # mỗi seed một dòng (ghi lại sau mỗi seed)
        per_class_mean_std.csv
        summary.json
    summary.csv               # mean ± std mỗi model
    summary.txt               # bảng tổng hợp + so sánh với số liệu bài báo
"""
import csv
import json
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# 把项目根目录加入Python路径，解决模块找不到问题
sys.path.append(str(Path(__file__).parent))

# Windows (cp1252) console/redirect cannot print emoji/Chinese in the logs -> force UTF-8
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, 'reconfigure'):  # không có khi chạy bằng %run trong Jupyter/Kaggle notebook
        _s.reconfigure(encoding='utf-8')

# 导入数据集和训练器
from datasets.csi_dataset import CSIDataset, clear_cache
from trainer import Trainer

# 仅使用 AMCS_TCN_FC。如需对比其他模型，取消下方 import 的注释并在 model_classes 中启用
from models.amcs_tcn_fc import AMCS_TCN_FC
# from models.amcs_tcn import AMCS_TCN
# from models.baselines.a_blstm import A_BLSTM
# from models.baselines.lstm_cnn import LSTM_CNN
# from models.baselines.cnn_gru_cnn import CNN_GRU_CNN
from config import build_config, model_kwargs, dataset_kwargs, add_common_args

# Số liệu bài báo (Zhang et al. 2026) trên public dataset, dùng để tham chiếu.
# Table 3 (AMCS-TCN, 5 seed, mean ± std, %), Table 2 (AMCS+FC, 1 lần), Table 5 (RRMSE, Brier).
# Lưu ý: bài báo tính RRMSE = sqrt(10 * BS) (coi là 10 lớp) -> so Brier thay vì RRMSE.
PAPER_REFERENCE = {
    'AMCS-TCN (GAP), Table 3': {'accuracy': (98.85, 0.12), 'precision': (98.78, 0.15),
                                'recall': (98.66, 0.18), 'f1': (98.68, 0.14)},
    'AMCS+FC, Table 2': {'accuracy': (98.72, None), 'precision': (98.76, None),
                         'recall': (98.69, None), 'f1': (98.76, None)},
    'AMCS-TCN, Table 5': {'rrmse': (0.3950, None), 'brier_score': (0.0156, None)},
}

PCT_METRICS = ['accuracy', 'precision', 'recall', 'f1']
RAW_METRICS = ['brier_score', 'rrmse', 'latency_ms', 'best_epoch', 'epochs_trained', 'train_time_s']


def set_seed(seed=3407):
    """固定所有随机种子，确保每次独立实验的绝对公平与可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def environment_info():
    info = {'python': sys.version.split()[0], 'platform': platform.platform(), 'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available()}
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
        info['cuda'] = torch.version.cuda
    return info


def mean_std(values):
    # np.std 默认 ddof=0 (与原代码一致)
    return float(np.mean(values)), float(np.std(values))


def summarize_model(model_dir, model_name, per_seed, class_names):
    """写 per_seed.csv / per_class_mean_std.csv / summary.json，返回汇总字典。"""
    seeds = [r['seed'] for r in per_seed]
    keys = PCT_METRICS + RAW_METRICS
    write_csv(os.path.join(model_dir, 'per_seed.csv'), ['seed'] + keys + ['params', 'size_mb'],
              [[r['seed']] + [r[k] for k in keys] + [r['params'], r['size_mb']] for r in per_seed])

    summary = {'model': model_name, 'seeds': seeds, 'n_runs': len(per_seed),
               'params': per_seed[-1]['params'], 'size_mb': per_seed[-1]['size_mb']}
    for k in PCT_METRICS:
        summary[k + '_mean'], summary[k + '_std'] = (v * 100 for v in mean_std([r[k] for r in per_seed]))
    for k in RAW_METRICS:
        summary[k + '_mean'], summary[k + '_std'] = mean_std([r[k] for r in per_seed])
    if 'val_accuracy' in per_seed[-1]:  # mode final: cùng checkpoint, đo thêm trên val
        for k in PCT_METRICS:
            summary[f'val_{k}_mean'], summary[f'val_{k}_std'] = (
                v * 100 for v in mean_std([r[f'val_{k}'] for r in per_seed]))

    # 每个类别的 precision / recall / f1 在各 seed 间的 mean ± std
    per_class = {}
    for r in per_seed:
        with open(os.path.join(model_dir, f"seed_{r['seed']}", 'per_class_metrics.csv'), encoding='utf-8') as f:
            for row in csv.DictReader(f):
                d = per_class.setdefault(int(row['class_id']), {'precision': [], 'recall': [], 'f1': []})
                for k in d:
                    d[k].append(float(row[k]))
    rows = []
    for cid in sorted(per_class):
        row = [cid, class_names[cid]]
        for k in ('precision', 'recall', 'f1'):
            m, s = mean_std(per_class[cid][k])
            row += [round(m * 100, 2), round(s * 100, 2)]
        rows.append(row)
    write_csv(os.path.join(model_dir, 'per_class_mean_std.csv'),
              ['class_id', 'class_name', 'precision_mean', 'precision_std', 'recall_mean', 'recall_std',
               'f1_mean', 'f1_std'], rows)

    with open(os.path.join(model_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'per_seed': per_seed}, f, indent=2)
    return summary


def write_run_summary(run_dir, summaries, config):
    write_csv(os.path.join(run_dir, 'summary.csv'),
              ['model', 'n_runs'] + [f'{k}_{s}' for k in PCT_METRICS + RAW_METRICS for s in ('mean', 'std')]
              + ['params', 'size_mb'],
              [[s['model'], s['n_runs']] + [round(s[f'{k}_{t}'], 4) for k in PCT_METRICS + RAW_METRICS
                                            for t in ('mean', 'std')] + [s['params'], round(s['size_mb'], 4)]
               for s in summaries])

    lines = [f"Dataset: {config['dataset']} ({config['num_classes']} classes, {config['antennas']} antennas, "
             f"{config['subcarriers']} subcarriers, T={config['time_steps']}"
             + (f" <- {config['raw_time_steps']} raw" if config.get('raw_time_steps', config['time_steps'])
                != config['time_steps'] else '') + ')',
             f"Data path: {config['data_path']}",
             f"Epochs: {config['epochs']} | batch: {config['batch_size']} | lr: {config['lr']} | "
             f"StepLR({config['lr_step']}, {config['lr_gamma']}) | early stop: {config['early_stop_patience']}",
             f"Filters: {' -> '.join(config.get('filters') or []) or 'none (raw)'} | "
             f"params: {json.dumps({k: v for k, v in config.get('filter_params', {}).items() if k in (config.get('filters') or [])})}",
             f"Normalize: {config.get('normalize', 'minmax')} | noise_std: {config.get('noise_std')}",
             f"Mode: {config['mode']} -> metrics on {config['eval_split'].upper()} "
             f"(model = best val accuracy) | split sizes: {config['split_sizes']}",
             '', f"=== Kết quả trên {config['eval_split'].upper()} (mean ± std qua các seed, std với ddof=0) ==="]
    for s in summaries:
        lines.append(f"\n{s['model']}  (seeds: {s['seeds']})")
        for k in PCT_METRICS:
            lines.append(f"  {k:<10}: {s[k + '_mean']:6.2f} ± {s[k + '_std']:.2f} %")
        if 'val_accuracy_mean' in s:
            lines.append('  VAL (cùng checkpoint): ' + ' | '.join(
                f"{k} {s[f'val_{k}_mean']:.2f} ± {s[f'val_{k}_std']:.2f}" for k in PCT_METRICS))
        lines.append(f"  {'brier':<10}: {s['brier_score_mean']:.4f} ± {s['brier_score_std']:.4f}")
        lines.append(f"  {'rrmse':<10}: {s['rrmse_mean']:.4f} ± {s['rrmse_std']:.4f}")
        lines.append(f"  params    : {s['params']:,} ({s['size_mb']:.3f} MB) | latency: "
                     f"{s['latency_ms_mean']:.2f} ms | best epoch: {s['best_epoch_mean']:.1f} | "
                     f"epochs: {s['epochs_trained_mean']:.1f} | train time: {s['train_time_s_mean']:.0f} s/seed")

    if config.get('paper_reference'):
        lines += ['', '=== Số liệu bài báo (public dataset; split/tiền xử lý có thể khác) ===']
        for name, vals in PAPER_REFERENCE.items():
            parts = [f"{k}={m}" + (f" ± {sd}" if sd is not None else '') for k, (m, sd) in vals.items()]
            lines.append(f"  {name}: " + ', '.join(parts))
    if summaries and config.get('paper_reference'):
        s = summaries[-1]
        lines += ['', f"=== Chênh lệch {s['model']} so với bài báo (điểm %) ==="]
        for ref in ('AMCS+FC, Table 2', 'AMCS-TCN (GAP), Table 3'):
            diffs = [f"{k}: {s[k + '_mean'] - PAPER_REFERENCE[ref][k][0]:+.2f}" for k in PCT_METRICS]
            lines.append(f"  vs {ref}: " + ', '.join(diffs))

    text = '\n'.join(lines) + '\n'
    with open(os.path.join(run_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(text)
    return text


def main():
    import argparse
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument('--mode', default='dev', choices=['dev', 'final'],
                        help="dev: seed config['dev_seeds'], đánh giá trên val | final: seed 42, đánh giá trên test")
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help="ghi đè seed (mặc định: dev -> config['dev_seeds'], final -> config['final_seed'])")
    parser.add_argument('--results_dir', default='results', help='thư mục gốc chứa kết quả')
    parser.add_argument('--run_name', default=None, help='tên thư mục lần chạy (mặc định: thời gian_dataset)')
    args = parser.parse_args()

    # 配置见 config.py (默认 100 轮，严格对齐论文)
    config = build_config(args.dataset, data_path=args.data_path, epochs=args.epochs,
                          batch_size=args.batch_size, lr=args.lr, device=args.device,
                          filters=args.filters, normalize=args.normalize,
                          time_steps=args.time_steps)
    class_names = config['class_names']

    # dev: config['dev_seeds'] -> val | final: seed 42 -> test
    seed_list = args.seeds or (config['dev_seeds'] if args.mode == 'dev' else [config['final_seed']])
    config['mode'] = args.mode
    config['eval_split'] = 'val' if args.mode == 'dev' else 'test'

    run_name = args.run_name or f"{datetime.now():%Y%m%d-%H%M%S}_{config['dataset']}_{args.mode}"
    run_dir = os.path.join(args.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"🖥️ 当前使用的计算设备: {config['device'].upper()}")
    print(f"📁 Results: {os.path.abspath(run_dir)}")

    # 定义要训练的模型类列表 (只存入类名，不提前实例化！)
    model_classes = [
        # ("A_BLSTM", A_BLSTM),
        # ("LSTM_CNN", LSTM_CNN),
        # ("CNN_GRU_CNN", CNN_GRU_CNN),
        # ("AMCS_TCN", AMCS_TCN),
        ("AMCS_TCN_FC", AMCS_TCN_FC)
    ]

    # 数据只加载一次 (数据集本身不依赖随机种子)
    # Model luôn chọn theo val. Tập đánh giá: dev -> val (test không được nạp), final -> test.
    train_set = CSIDataset(config['data_path'], 'train', **dataset_kwargs(config))
    val_set = CSIDataset(config['data_path'], 'val', **dataset_kwargs(config))
    eval_set = val_set if args.mode == 'dev' else CSIDataset(config['data_path'], 'test', **dataset_kwargs(config))
    clear_cache()  # dataset đã giữ bản cần dùng, bỏ bản thô/đã lọc để tiết kiệm RAM
    config['split_sizes'] = {'train': len(train_set), 'val': len(val_set),
                             **({'test': len(eval_set)} if args.mode == 'final' else {})}
    print(f"📦 mode={args.mode} | sizes: {config['split_sizes']} | eval on {config['eval_split'].upper()} | "
          f"seeds: {seed_list} | sample shape: {tuple(train_set.x.shape[1:])} (antennas, T, subcarriers)")
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({**config, 'seeds': seed_list}, f, indent=2)
    with open(os.path.join(run_dir, 'environment.json'), 'w', encoding='utf-8') as f:
        json.dump(environment_info(), f, indent=2)

    summaries = []

    # 自动化串行训练与测试 (外层循环：模型)
    for model_name, ModelClass in model_classes:
        print("=" * 70)
        print(f"🚀 开始评估模型: {model_name} (共需执行 {len(seed_list)} 次独立实验)")
        print("=" * 70)

        model_dir = os.path.join(run_dir, model_name)
        per_seed = []

        # 内层循环：不同的随机种子
        for i, current_seed in enumerate(seed_list):
            print(f"\n---> 正在进行第 {i + 1}/{len(seed_list)} 次实验 (Seed: {current_seed})")

            # 在每一次实验前，严格重置全局随机状态
            set_seed(current_seed)

            # DataLoader 在循环内创建，打乱顺序受当前种子控制
            train_loader = DataLoader(train_set, config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_set, config['batch_size'], shuffle=False)
            eval_loader = DataLoader(eval_set, config['batch_size'], shuffle=False)

            # 动态实例化模型与 Trainer，每个 seed 独立输出目录
            seed_config = {**config, 'output_dir': os.path.join(model_dir, f'seed_{current_seed}')}
            model = ModelClass(**model_kwargs(config))
            trainer = Trainer(model, train_loader, val_loader, eval_loader, seed_config)

            trainer.run()
            metrics_dict = trainer.test(class_names, title=config['eval_split'].upper())
            if args.mode == 'final':  # cùng checkpoint, đo thêm trên val (file trong seed_<s>/val/)
                val_metrics = trainer.test(class_names, loader=val_loader, subdir='val', title='VAL')
                metrics_dict.update({f'val_{k}': val_metrics[k] for k in PCT_METRICS})
            per_seed.append({'seed': current_seed, **metrics_dict})

            # 每个 seed 结束后立即更新汇总 (会话中断也不会丢失已完成的结果)
            summary = summarize_model(model_dir, model_name, per_seed, class_names)
            write_run_summary(run_dir, summaries + [summary], config)

            # 清理显存，防止下一个循环 OOM
            del trainer, model, train_loader, val_loader, eval_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"✨ 第 {i + 1} 次实验执行完毕 (acc={metrics_dict['accuracy']:.4f})，显存已清理。")

        summaries.append(summary)
        print(f"\n🎉 {model_name} {len(seed_list)}次完整评估结束！平均 Accuracy: "
              f"{summary['accuracy_mean']:.2f} ± {summary['accuracy_std']:.2f}%\n")

    text = write_run_summary(run_dir, summaries, config)
    print("\n" + "=" * 75)
    print(text)
    print(f"📄 详细汇总报告已保存至: {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
