import torch
import torch.nn as nn
import torch.nn.functional as F


class A_BLSTM(nn.Module):
    def __init__(self, num_classes=7, subcarriers=30, antennas=3, time_steps=250, hidden_nodes=200):
        """
        初始化 A-BLSTM (Attention Based Bi-directional LSTM)
        :param num_classes: 动作分类数量
        :param subcarriers: 单根天线的子载波数量 (默认30)
        :param antennas: 天线数量 (默认3)
        :param time_steps: 时间步长 (对齐数据集插值后的 T=250)
        :param hidden_nodes: BLSTM的隐藏层节点数 (论文 4.5 节最优值为 200)
        """
        super().__init__()

        # 1. 原始特征维度: 3根天线 * 30子载波 = 90
        input_dim = antennas * subcarriers

        # 2. 双向 LSTM 网络
        # bidirectional=True 会使输出维度翻倍，即 200 * 2 = 400
        self.blstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_nodes,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 3. 注意力矩阵生成层 (Softmax Regression Layer)
        # 论文要求注意力矩阵的维度与 BLSTM 输出特征维度完全一致
        lstm_out_dim = hidden_nodes * 2
        self.attention_fc = nn.Linear(lstm_out_dim, lstm_out_dim)

        # 4. 展平与分类层
        # 经过 Flatten 后，特征维度为 TimeSteps * 400
        flattened_dim = time_steps * lstm_out_dim
        self.classifier = nn.Linear(flattened_dim, num_classes)

    def forward(self, x):
        """
        :param x: 统一输入张量, 形状须为 (Batch, Antennas=3, TimeSteps=250, Subcarriers=30)
        """
        B, Ant, T, Sub = x.shape

        # 1. 特征融合与维度转换
        # 论文将所有天线和子载波融合为一维向量输入 LSTM: (B, T, 90)
        x = x.permute(0, 2, 1, 3).contiguous()  # 变为 (B, T, Ant, Sub)
        x = x.view(B, T, Ant * Sub)  # 变为 (B, T, 90)

        # 2. 双向 LSTM 特征提取
        # h 的形状: (B, T, 400)
        h, _ = self.blstm(x)

        # 3. 注意力机制 (Attention)
        # 论文强调给每个 feature 和 time step 分配权重 (生成特征级注意力矩阵)
        attention_scores = self.attention_fc(h)  # (B, T, 400)
        attention_weights = F.softmax(attention_scores, dim=1)  # 在时间维度上归一化

        # 4. 特征合并 (Merge)
        # 逐元素相乘 (Element-wise multiplication)
        merged_features = attention_weights * h  # (B, T, 400)

        # 5. 展平 (Flatten Layer)
        flattened = merged_features.view(B, -1)  # (B, T * 400)

        # 6. 分类输出 (Softmax Classification)
        logits = self.classifier(flattened)  # (B, num_classes)
        return logits


# ==========================================
# 本地连通性测试模块
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入参数: Batch=64, 3根天线, 时间步长T=250, 30个子载波
    dummy_input = torch.randn(64, 3, 250, 30).to(device)

    model = A_BLSTM(num_classes=7).to(device)

    # 打印参数量，用于与 AMCS-TCN 进行空间复杂度对比
    total_params = sum(p.numel() for p in model.parameters()) / 1024 / 1024
    print(f"A-BLSTM 模型参数量: {total_params:.2f} MB")

    out = model(dummy_input)
    print(f"输入张量形状: {dummy_input.shape}")
    print(f"输出分类形状: {out.shape} (预期为 [64, 7])")
    print("✅ 网络前向传播测试通过，可直接替换进 main.py！")