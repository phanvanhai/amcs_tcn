import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_GRU_CNN(nn.Module):
    def __init__(self, num_classes=7, subcarriers=30, antennas=3, time_steps=250):
        """
        初始化 Shalaby 等人提出的 CNN-GRU-CNN baseline 模型
        """
        super().__init__()

        # ========================================================
        # 1. 第一阶段：CNN 空间与局部时序特征提取 (2D Convolution)
        # ========================================================
        # 论文规定：128个滤波器，大小为5的卷积核，步长1，以及 batch normalization 和 ReLU
        self.conv1 = nn.Conv2d(in_channels=antennas, out_channels=128,
                               kernel_size=5, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(128)

        # 论文规定：使用大小为2的平均池化 (size 2 pool, size 2 stride) 和 0.6 的 Dropout
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.6)

        # 经过 pool1 后，Time 和 Subcarriers 维度均精确减半
        time_after_pool = time_steps // 2
        sub_after_pool = subcarriers // 2

        # ========================================================
        # 2. 第二阶段：GRU 全局时序特征建模
        # ========================================================
        # 将空间维度展平作为 GRU 的特征输入
        gru_input_dim = 128 * sub_after_pool
        # 论文规定：GRU 层包含 256 个神经元
        self.gru = nn.GRU(input_size=gru_input_dim, hidden_size=256,
                          num_layers=1, batch_first=True)

        # ========================================================
        # 3. 第三阶段：第二层 CNN 抽象特征扩展 (1D Convolution)
        # ========================================================
        # 论文规定：再次使用 128个滤波器，大小为5的核，步长1
        # 注意：此处处理的是 GRU 产生的一维序列特征，因此使用 Conv1d
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128,
                               kernel_size=5, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(128)

        # 论文规定：再次进行大小为2的平均池化
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)

        # ========================================================
        # 4. 第四阶段：全连接分类
        # ========================================================
        time_final = time_after_pool // 2
        flattened_dim = 128 * time_final
        self.classifier = nn.Linear(flattened_dim, num_classes)

    def forward(self, x):
        """
        :param x: 统一输入张量, 形状须为 (Batch, Antennas=3, TimeSteps=250, Subcarriers=30)
        """
        B, Ant, T, Sub = x.shape

        # =======================
        # 1. 第一部分 2D CNN
        # =======================
        # x shape: (B, 3, T, Sub) -> (B, 128, T, Sub)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # shape: (B, 128, T/2, Sub/2)
        x = self.pool1(x)
        x = self.dropout(x)

        # =======================
        # 2. 形状转换与 GRU
        # =======================
        _, C, T_pool, Sub_pool = x.shape
        # 将天线/通道特征与子载波特征融合，以适应 GRU 对时序数据的要求
        x = x.permute(0, 2, 1, 3).contiguous()  # 变为 (B, T_pool, C, Sub_pool)
        x = x.view(B, T_pool, C * Sub_pool)  # 展平特征维度 -> (B, T_pool, 128*15)

        # shape: (B, T_pool, 256)
        x, _ = self.gru(x)

        # =======================
        # 3. 第二部分 1D CNN
        # =======================
        # PyTorch 的 Conv1d 需要输入形状为 (Batch, Channels, Length)
        x = x.permute(0, 2, 1).contiguous()  # -> (B, 256, T_pool)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # shape: (B, 128, T_pool/2)
        x = self.pool2(x)

        # =======================
        # 4. Flatten 与 Classifier
        # =======================
        # 展平送入 Dense 层
        x = x.view(B, -1)  # -> (B, 128 * 62)
        logits = self.classifier(x)

        return logits


# ==========================================
# 本地连通性测试模块
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟在 RTX 4060 等设备上运行时的输入参数: Batch=64, 3根天线, 时间步长T=250, 30个子载波
    dummy_input = torch.randn(64, 3, 250, 30).to(device)

    model = CNN_GRU_CNN(num_classes=7).to(device)

    out = model(dummy_input)
    print(f"输入张量形状: {dummy_input.shape}")
    print(f"输出分类形状: {out.shape} (预期为 [64, 7])")
    print("✅ CNN-GRU-CNN 网络前向传播测试通过，可直接替换进 main.py！")