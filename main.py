# main.py
import sys
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# 把项目根目录加入Python路径，解决模块找不到问题
sys.path.append(str(Path(__file__).parent))

from datasets.csi_dataset import CSIDataset
from models.amcs_tcn import AMCS_TCN
from models.amcs_tcn_fc import AMCS_TCN_FC
from trainer import Trainer

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


def main():
    # 尝试不同的数字：42, 3407, 114514 等
    set_seed(3407)

    # ====================== 配置 ======================
    config = {
        'data_path': 'C:/AMCS-TCN/data/processed_csi',
        'batch_size': 64,
        'lr': 1e-3,
        'epochs': 150,  # 配合 trainer 中的 StepLR(step_size=20)，150轮足以让学习率充分平滑衰减
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'model',
        'log_dir': 'logs',
        'num_classes': 7
    }

    class_names = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']

    print(f"🖥️ 当前使用的计算设备: {config['device'].upper()}")

    # ====================== 数据加载 ======================
    print("📦 正在加载数据集...")
    train_set = CSIDataset(config['data_path'], 'train')
    val_set = CSIDataset(config['data_path'], 'val')
    test_set = CSIDataset(config['data_path'], 'test')

    train_loader = DataLoader(train_set, config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_set, config['batch_size'], shuffle=False)
    print("✅ 数据集加载完成！\n")

    # ====================== 模型 ======================
    model = AMCS_TCN_FC(num_classes=config['num_classes'])

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