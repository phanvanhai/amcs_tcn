import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_CNN(nn.Module):
    def __init__(self, num_classes=7, subcarriers=30, antennas=3, time_steps=250):
        """
        初始化 Shang 等人提出的 LSTM-CNN baseline 模型
        """
        super().__init__()
        # 将天线和子载波融合为一维特征向量: 3 * 30 = 90
        input_dim = antennas * subcarriers

        # ========================================================
        # 1. 第一阶段：LSTM 时序特征提取
        # ========================================================
        # 论文 2.2.1 节设定：隐层节点数 r = 28
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=28,
                            num_layers=1, batch_first=True)
        # 论文规定：在每层结尾添加 dropout，丢弃率为 0.2
        self.dropout = nn.Dropout(0.2)

        # ========================================================
        # 2. 第二阶段：2D-CNN 特征提取
        # ========================================================
        # 论文 2.2.2 节设定：5个卷积核，核大小为 30(时间维度) × 28(特征维度)，步长为 1
        # 因为输入特征数恰好是 28，因此在特征维度上是没有滑动的，相当于 1D 卷积的 2D 实现
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=5,
                                kernel_size=(30, 28), stride=1)

        # ========================================================
        # 3. 第三阶段：MLP 分类器
        # ========================================================
        # 经过全局最大池化后，5个卷积核只产生 5 个特征值
        self.classifier = nn.Sequential(
            nn.Linear(5, num_classes)
        )

    def forward(self, x):
        """
        :param x: 统一输入张量, 形状须为 (Batch, Antennas=3, TimeSteps=250, Subcarriers=30)
        """
        B, Ant, T, Sub = x.shape

        # 1. 预处理：融合天线和子载波维度 -> (B, T, 90)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, Ant * Sub)

        # 2. LSTM 层
        # lstm_out shape: (B, T, 28)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        # 3. 维度重构以适配 2D-CNN
        # 论文中要求提升维度至 (b x s x r x 1)，在 PyTorch 中通道数在第2维，即 (B, 1, T, 28)
        cnn_in = lstm_out.unsqueeze(1)

        # 4. 2D 卷积与 ReLU 激活
        # 卷积核高度为30，宽度为28。
        # 卷积后 shape: (B, 5个核, T - 30 + 1, 1) -> 对250步长即为 (B, 5, 221, 1)
        cnn_out = self.conv2d(cnn_in)
        cnn_out = F.relu(cnn_out)

        # 降维，准备进行池化 -> shape: (B, 5, 221)
        cnn_out = cnn_out.squeeze(-1)

        # 5. 全局最大池化 (论文公式 8)
        # 提取时间维度 (dim=2) 上的最大值，得到最具代表性的特征
        # pooled shape: (B, 5)
        pooled, _ = torch.max(cnn_out, dim=2)

        # 6. MLP 分类器输出
        logits = self.classifier(pooled)

        return logits


# ==========================================
# 本地连通性测试模块
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入参数: Batch=64, 3根天线, 时间步长T=250, 30个子载波
    dummy_input = torch.randn(64, 3, 250, 30).to(device)

    model = LSTM_CNN(num_classes=7).to(device)

    # 打印参数量，你会发现它极小
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LSTM-CNN 模型参数量: {total_params} 个")

    out = model(dummy_input)
    print(f"输入张量形状: {dummy_input.shape}")
    print(f"输出分类形状: {out.shape} (预期为 [64, 7])")
    print("✅ LSTM-CNN 网络前向传播测试通过！")