import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 多尺度特征提取与特征图融合注意力 (FMFA)
# ==========================================
class MCS_FMFA_Block(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        # 论文: 1x3, 1x5, 1x7 卷积核，padding='same' 保留时间分辨率
        self.conv_scale1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding='same')
        self.conv_scale2 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding='same')
        self.conv_scale3 = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 1), padding='same')

        # FMFA 注意力权重生成：生成通道数为 1 的门控图
        self.attn_net1 = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.attn_net2 = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.attn_net3 = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # 【特征图归一化层】拯救连乘后信号缩水的关键
        self.fusion_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        m1 = self.conv_scale1(x)
        m2 = self.conv_scale2(x)
        m3 = self.conv_scale3(x)

        n1 = self.attn_net1(m1)
        n2 = self.attn_net2(m2)
        n3 = self.attn_net3(m3)

        # 防止除零错误
        sum_n = n1 + n2 + n3 + 1e-8
        n1_hat, n2_hat, n3_hat = n1 / sum_n, n2 / sum_n, n3 / sum_n

        # 论文公式(1): 逐元素连乘 (Product) 实现尺度特征过滤
        f1i = (n1_hat * m1) * (n2_hat * m2) * (n3_hat * m3)

        # 【拯救梯度】：将缩水的特征重新拉回标准分布
        f1i = self.fusion_bn(f1i)

        return f1i


# ==========================================
# 2. 特征通道注意力 (FCA)
# ==========================================
class FCA_Block(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # 类似 SE-Net，缩减比例 r=16
        reduced_channels = max(1, channels // reduction_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, 1)
        )

        # 【通道特征归一化层】拯救跨天线连乘缩水的关键
        self.fusion_bn = nn.BatchNorm2d(channels)

    def forward(self, f1_list):
        weights = []
        for f1i in f1_list:
            # GAP & GMP: 空间与时间维度的双重池化
            avg_pool = torch.mean(f1i, dim=(2, 3))
            max_pool = torch.amax(f1i, dim=(2, 3))

            avg_out = self.mlp(avg_pool)
            max_out = self.mlp(max_pool)

            # 两个池化分支相加后再经过 ReLU 和 Sigmoid
            w_i = torch.sigmoid(F.relu(avg_out + max_out))
            weights.append(w_i)

        # 跨天线维度归一化
        weights_tensor = torch.cat(weights, dim=1)
        weights_norm = weights_tensor / (torch.sum(weights_tensor, dim=1, keepdim=True) + 1e-8)

        # 论文公式(2): 天线特征逐元素连乘 (Product) 过滤受损物理链路
        f2 = (f1_list[0] * weights_norm[:, 0:1, None, None]) * \
             (f1_list[1] * weights_norm[:, 1:2, None, None]) * \
             (f1_list[2] * weights_norm[:, 2:3, None, None])

        # 【拯救梯度】：再次拉回分布，确保 TCN 能够接收到充沛的信号
        f2 = self.fusion_bn(f2)

        return f2


# ==========================================
# 3. 时间卷积网络底层组件 (TCN)
# ==========================================
class DepthwiseSeparableConv1d(nn.Module):
    """论文图 5 标注的 Separable Convolution 以降低参数量"""

    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class TCN_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        # 图 5 结构: Tanh 激活分支
        self.branch_tanh = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, padding, dilation),
            nn.BatchNorm1d(out_channels),
            nn.Tanh()
        )

        # 图 5 结构: Sigmoid 门控分支
        self.branch_sig = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, padding, dilation),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid()
        )

        # PyTorch 标准的 1D 空间 Dropout (直接丢弃整个特征通道)
        self.spatial_dropout = nn.Dropout1d(0.5)
        self.final_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        # 匹配残差连接的维度
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

        # 图 5 结构: Max Separable Pooling (连接在残差加和之后，执行时序降采样)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        residual = x if self.skip_conv is None else self.skip_conv(x)

        # 门控激活机制 (Gated Activation)
        out_tanh = self.branch_tanh(x)
        out_sig = self.branch_sig(x)
        out = out_tanh * out_sig

        out = self.spatial_dropout(out)
        out = self.final_conv(out)

        out = out + residual
        out = self.max_pool(out)

        return F.relu(out)


class TCN_Backbone(nn.Module):
    def __init__(self, in_channels, tcn_channels=64, dilations=[1, 2, 4, 8]):
        super().__init__()
        layers = []
        for d in dilations:
            # 论文设定：TCN 通道数固定为 64，基础卷积核大小为 3
            layers.append(TCN_ResBlock(in_channels, tcn_channels, dilation=d, kernel_size=3))
            in_channels = tcn_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ==========================================
# 4. AMCS-TCN 主模型入口
# ==========================================
class AMCS_TCN(nn.Module):
    def __init__(self, num_classes=10, cnn_channels=32, subcarriers=30):
        super().__init__()
        # 3 个独立的天线特征流
        self.antenna_streams = nn.ModuleList([MCS_FMFA_Block(1, cnn_channels) for _ in range(3)])

        # 跨天线特征融合
        self.fca_fusion = FCA_Block(cnn_channels, reduction_ratio=16)

        # TCN 时序建模
        tcn_in_channels = cnn_channels * subcarriers
        self.tcn = TCN_Backbone(in_channels=tcn_in_channels, tcn_channels=64, dilations=[1, 2, 4, 8])

        # 全局平均池化 (GAP) 与最终分类器
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        输入要求: (Batch_Size, 3根天线, Time_Steps, 30个子载波)
        """
        # 1. 拆分3根天线并拓展维度适配 2D 卷积: (B, T, Sub) -> (B, 1, T, Sub)
        ant1 = x[:, 0, :, :].unsqueeze(1)
        ant2 = x[:, 1, :, :].unsqueeze(1)
        ant3 = x[:, 2, :, :].unsqueeze(1)

        # 2. 独立多尺度特征提取与 FMFA 连乘融合
        f1_1 = self.antenna_streams[0](ant1)
        f1_2 = self.antenna_streams[1](ant2)
        f1_3 = self.antenna_streams[2](ant3)

        # 3. 跨天线 FCA 评估与连乘过滤 -> 输出: (B, C, T, Sub)
        f2 = self.fca_fusion([f1_1, f1_2, f1_3])

        # 4. 展平空间维度，准备一维 TCN 输入
        B, C, T, Sub = f2.shape
        tcn_in = f2.permute(0, 1, 3, 2).reshape(B, C * Sub, T)

        # 5. TCN 时序特征提取与全局特征压缩
        tcn_out = self.tcn(tcn_in)
        pooled = self.gap(tcn_out).squeeze(-1)

        # 6. 分类输出 (Trainer模块中的 CrossEntropyLoss 已包含 Softmax)
        logits = self.classifier(pooled)
        return logits