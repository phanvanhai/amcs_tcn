# main.py
import sys
import random
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

from datasets.csi_dataset import CSIDataset
from models.amcs_tcn_fc import AMCS_TCN_FC
# from models.amcs_tcn import AMCS_TCN  # bản GAP, bỏ comment nếu muốn chạy (xem MODELS bên dưới)
from trainer import Trainer
from config import build_config, model_kwargs, dataset_kwargs, add_common_args

'''
#植入随机种子，不同的随机种子可能会有不同的结果，选一个跑分最高的
随机种子有什么用，随机的是什么？
• 权重的“出生点”：神经网络在第一轮训练前的所有权重分布。不同的种子会让模型从群山中不同的位置开始“下山”。
• 数据的“洗牌顺序”：DataLoader 里的 shuffle=True 每一轮怎么打乱数据。
• Dropout 的“屠刀”：每一轮前向传播中，哪些神经元会被随机关闭。
• 数据增强的“随机度”：比如trainer中加的高斯噪声，噪声的强度波动也是由种子生成的。
'''


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 补充这一句，保证底层卷积算法的绝对固定


# Chỉ dùng AMCS_TCN_FC. Muốn chạy bản GAP: bỏ comment import AMCS_TCN, dòng MODELS và --model bên dưới,
# rồi thay `AMCS_TCN_FC(...)` ở phần 模型 bằng `MODELS[args.model](...)`.
# MODELS = {'fc': AMCS_TCN_FC, 'gap': AMCS_TCN}


def main():
    import argparse
    parser = add_common_args(argparse.ArgumentParser())
    # parser.add_argument('--model', default='fc', choices=list(MODELS), help='分类头: fc (默认) 或 gap')
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args()

    # 尝试不同的数字：42, 3407, 114514 等
    set_seed(args.seed)

    # ====================== 配置 (见 config.py) ======================
    # main.py 原本默认 150 轮 (配合 StepLR(20) 平滑衰减)，可用 --epochs 覆盖
    config = build_config(args.dataset, data_path=args.data_path, epochs=args.epochs or 150,
                          batch_size=args.batch_size, lr=args.lr, device=args.device,
                          filters=args.filters, normalize=args.normalize)
    class_names = config['class_names']

    print(f"🖥️ 当前使用的计算设备: {config['device'].upper()}")
    print(f"📐 {config['dataset']}: {config['num_classes']} 类, {config['antennas']} 天线, "
          f"{config['subcarriers']} 子载波, T={config['time_steps']}")

    # ====================== 数据加载 ======================
    print("📦 正在加载数据集...")
    train_set = CSIDataset(config['data_path'], 'train', **dataset_kwargs(config))
    val_set = CSIDataset(config['data_path'], 'val', **dataset_kwargs(config))
    test_set = CSIDataset(config['data_path'], 'test', **dataset_kwargs(config))

    max_label = int(max(s.y.max() for s in (train_set, val_set, test_set)))
    if max_label >= config['num_classes']:
        raise ValueError(f"Label {max_label} found but class_names only has {config['num_classes']} entries")

    train_loader = DataLoader(train_set, config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, config['batch_size'], shuffle=False)
    print("✅ 数据集加载完成！\n")

    # ====================== 模型 ======================
    model = AMCS_TCN_FC(**model_kwargs(config))
    # model = MODELS[args.model](**model_kwargs(config))

    # ====================== 训练 ======================
    # 传入你保留了各类 tricks 的 trainer.py
    trainer = Trainer(model, train_loader, val_loader, test_loader, config)

    # 运行训练循环
    trainer.run()

    # 运行测试，并接收 trainer.py 返回的包含多维指标的字典
    metrics = trainer.test(class_names)

    print("\n✅ 训练与测试全流程执行完毕！")


if __name__ == '__main__':
    main()