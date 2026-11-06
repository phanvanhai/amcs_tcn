import sys
import os
from pathlib import Path
import random
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
from datasets.csi_dataset import CSIDataset
from trainer import Trainer

# 仅使用 AMCS_TCN_FC。如需对比其他模型，取消下方 import 的注释并在 model_classes 中启用
from models.amcs_tcn_fc import AMCS_TCN_FC
# from models.amcs_tcn import AMCS_TCN
# from models.baselines.a_blstm import A_BLSTM
# from models.baselines.lstm_cnn import LSTM_CNN
# from models.baselines.cnn_gru_cnn import CNN_GRU_CNN
from config import build_config, model_kwargs, dataset_kwargs, add_common_args


def set_seed(seed=3407):
    """固定所有随机种子，确保每次独立实验的绝对公平与可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    import argparse
    parser = add_common_args(argparse.ArgumentParser())
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 3407, 1024, 2024, 0])
    args = parser.parse_args()

    # 配置见 config.py (默认 100 轮，严格对齐论文)
    config = build_config(args.dataset, data_path=args.data_path, epochs=args.epochs,
                          batch_size=args.batch_size, lr=args.lr, device=args.device)
    class_names = config['class_names']

    # 定义要求的不同随机种子 (默认 5 个)
    seed_list = args.seeds

    print(f"🖥️ 当前使用的计算设备: {config['device'].upper()}")

    # 定义要训练的模型类列表 (只存入类名，不提前实例化！)
    model_classes = [
        # ("A_BLSTM", A_BLSTM),
        # ("LSTM_CNN", LSTM_CNN),
        # ("CNN_GRU_CNN", CNN_GRU_CNN),
        # ("AMCS_TCN", AMCS_TCN),
        ("AMCS_TCN_FC", AMCS_TCN_FC)
    ]

    # 用于保存最终的测试结果字典 (均值和标准差)
    results_summary = {}

    # 自动化串行训练与测试 (外层循环：模型)
    for model_name, ModelClass in model_classes:
        print("=" * 70)
        print(f"🚀 开始评估模型: {model_name} (共需执行 {len(seed_list)} 次独立实验)")
        print("=" * 70)

        # 用于存储当前模型 5 次实验的各项指标
        metrics_history = {
            'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
            'brier_score': [], 'rrmse': []
        }

        # 内层循环：5 个不同的随机种子
        for i, current_seed in enumerate(seed_list):
            print(f"\n---> 正在进行第 {i + 1}/{len(seed_list)} 次实验 (Seed: {current_seed})")

            # 在每一次实验前，严格重置全局随机状态
            set_seed(current_seed)

            # 将 DataLoader 的实例化移入循环，确保每次实验的数据划分和打乱绝对独立且受当前种子控制
            train_set = CSIDataset(config['data_path'], 'train', **dataset_kwargs(config))
            val_set = CSIDataset(config['data_path'], 'val', **dataset_kwargs(config))
            test_set = CSIDataset(config['data_path'], 'test', **dataset_kwargs(config))

            train_loader = DataLoader(train_set, config['batch_size'], shuffle=True)
            val_loader = DataLoader(val_set, config['batch_size'], shuffle=False)
            test_loader = DataLoader(test_set, config['batch_size'], shuffle=False)

            # 动态实例化模型与 Trainer
            model = ModelClass(**model_kwargs(config))
            trainer = Trainer(model, train_loader, val_loader, test_loader, config)

            # 执行训练
            trainer.run()

            # 执行测试并获取多维指标字典
            metrics_dict = trainer.test(class_names)

            # 记录当前种子的指标
            metrics_history['accuracy'].append(metrics_dict['accuracy'])
            metrics_history['precision'].append(metrics_dict['precision'])
            metrics_history['recall'].append(metrics_dict['recall'])
            metrics_history['f1'].append(metrics_dict['f1'])

            # 如果你的 trainer.py 已经加入了 Brier Score 和 RRMSE，则一并记录
            if 'brier_score' in metrics_dict and 'rrmse' in metrics_dict:
                metrics_history['brier_score'].append(metrics_dict['brier_score'])
                metrics_history['rrmse'].append(metrics_dict['rrmse'])

            # 记录参数量 (参数量是固定的，每次循环覆盖即可)
            params = metrics_dict['params']

            # 清理显存，防止下一个循环 OOM
            del trainer
            del model
            del train_loader
            del val_loader
            del test_loader
            torch.cuda.empty_cache()
            print(f"✨ 第 {i + 1} 次实验执行完毕，显存已清理。")

        # 计算 5 次实验的均值和标准差
        results_summary[model_name] = {
            'acc_mean': np.mean(metrics_history['accuracy']) * 100,
            'acc_std': np.std(metrics_history['accuracy']) * 100,
            'pre_mean': np.mean(metrics_history['precision']) * 100,
            'pre_std': np.std(metrics_history['precision']) * 100,
            'rec_mean': np.mean(metrics_history['recall']) * 100,
            'rec_std': np.std(metrics_history['recall']) * 100,
            'f1_mean': np.mean(metrics_history['f1']) * 100,
            'f1_std': np.std(metrics_history['f1']) * 100,
            'params': params
        }

        if len(metrics_history['brier_score']) > 0:
            results_summary[model_name]['bs_mean'] = np.mean(metrics_history['brier_score'])
            results_summary[model_name]['rrmse_mean'] = np.mean(metrics_history['rrmse'])

        print(
            f"\n🎉 {model_name} {len(seed_list)}次完整评估结束！平均 Accuracy: {results_summary[model_name]['acc_mean']:.2f} ± {results_summary[model_name]['acc_std']:.2f}%\n")

    # 打印最终性能排行榜
    print("\n" + "🏆" * 40)
    print(
        f"                              最终模型性能对比排行榜 ({len(seed_list)}次独立实验均值 ± 标准差)                              ")
    print("🏆" * 40)

    # 按准确率均值从高到低排序
    sorted_results = sorted(results_summary.items(), key=lambda item: item[1]['acc_mean'], reverse=True)

    # 打印表头
    print(f"  {'Rank':<4} | {'Model Name':<15} | {'Accuracy (%)':<15} | {'F1-Score (%)':<15} | {'Parameters':<10}")
    print("-" * 75)
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        acc_str = f"{metrics['acc_mean']:>5.2f} ± {metrics['acc_std']:<4.2f}"
        f1_str = f"{metrics['f1_mean']:>5.2f} ± {metrics['f1_std']:<4.2f}"
        params = metrics['params']
        print(f"  Top {rank}: | {name:<15} | {acc_str:<15} | {f1_str:<15} | {params:>10,}")
    print("=" * 75)

    # 将汇总结果写入总日志文件
    summary_path = os.path.join(config['log_dir'], "experiment_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"模型对比实验最终结果 ({len(seed_list)}次独立实验)\n")
        f.write("=" * 110 + "\n")
        f.write(
            f"{'Rank':<4} | {'Model Name':<15} | {'Accuracy (%)':<15} | {'Precision (%)':<15} | {'Recall (%)':<15} | {'F1-Score (%)':<15} | {'Parameters':<10}\n")
        f.write("-" * 110 + "\n")

        for rank, (name, metrics) in enumerate(sorted_results, 1):
            acc_str = f"{metrics['acc_mean']:>5.2f} ± {metrics['acc_std']:<4.2f}"
            pre_str = f"{metrics['pre_mean']:>5.2f} ± {metrics['pre_std']:<4.2f}"
            rec_str = f"{metrics['rec_mean']:>5.2f} ± {metrics['rec_std']:<4.2f}"
            f1_str = f"{metrics['f1_mean']:>5.2f} ± {metrics['f1_std']:<4.2f}"
            params = metrics['params']

            f.write(
                f"Top {rank} | {name:<15} | {acc_str:<15} | {pre_str:<15} | {rec_str:<15} | {f1_str:<15} | {params:>10,}\n")

        # 如果包含额外指标，则追加打印
        if 'bs_mean' in sorted_results[0][1]:
            f.write("\n" + "=" * 50 + "\n")
            f.write("额外指标分析 (Error Magnitude Metrics)\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Model Name':<15} | {'Brier Score':<15} | {'RRMSE':<15}\n")
            f.write("-" * 50 + "\n")
            for name, metrics in sorted_results:
                f.write(f"{name:<15} | {metrics['bs_mean']:<15.4f} | {metrics['rrmse_mean']:<15.4f}\n")

    print(f"📄 详细汇总报告已保存至: {summary_path}")


if __name__ == "__main__":
    main()