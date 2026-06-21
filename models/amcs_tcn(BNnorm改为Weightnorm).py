import torch
import torch.nn as nn
import torch.nn.functional as F
# 新增导入权重归一化工具
from torch.nn.utils import weight_norm

# ==========================================
# 1. 多尺度特征提取与特征图融合注意力 (FMFA) 【完全未改动】
# ==========================================
class MCS_FMFA_Block(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.conv_scale1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding='same')
        self.conv_scale2 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding='same')
        self.conv_scale3 = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 1), padding='same')
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
        self.fusion_bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        m1 = self.conv_scale1(x)
        m2 = self.conv_scale2(x)
        m3 = self.conv_scale3(x)
        n1 = self.attn_net1(m1)
        n2 = self.attn_net2(m2)
        n3 = self.attn_net3(m3)
        sum_n = n1 + n2 + n3 + 1e-8
        n1_hat, n2_hat, n3_hat = n1 / sum_n, n2 / sum_n, n3 / sum_n
        f1i = (n1_hat * m1) * (n2_hat * m2) * (n3_hat * m3)
        f1i = self.fusion_bn(f1i)
        return f1i

# ==========================================
# 2. 特征通道注意力 (FCA) 【完全未改动】
# ==========================================
class FCA_Block(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, 1)
        )
        self.fusion_bn = nn.BatchNorm2d(channels)
    def forward(self, f1_list):
        weights = []
        for f1i in f1_list:
            avg_pool = torch.mean(f1i, dim=(2, 3))
            max_pool = torch.amax(f1i, dim=(2, 3))
            avg_out = self.mlp(avg_pool)
            max_out = self.mlp(max_pool)
            w_i = torch.sigmoid(F.relu(avg_out + max_out))
            weights.append(w_i)
        weights_tensor = torch.cat(weights, dim=1)
        weights_norm = weights_tensor / (torch.sum(weights_tensor, dim=1, keepdim=True) + 1e-8)
        f2 = (f1_list[0] * weights_norm[:, 0:1, None, None]) * \
             (f1_list[1] * weights_norm[:, 1:2, None, None]) * \
             (f1_list[2] * weights_norm[:, 2:3, None, None])
        f2 = self.fusion_bn(f2)
        return f2

# ==========================================
# 3. TCN底层组件【仅移除BN，卷积添加WeightNorm，其余不变】
# ==========================================
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super().__init__()
        # 卷积层包裹weight_norm，移除BN
        self.depthwise = weight_norm(nn.Conv1d(in_channels, in_channels,
                                              kernel_size=kernel_size,
                                              padding=padding, dilation=dilation, groups=in_channels))
        self.pointwise = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1))
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class TCN_ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        # 双分支：删除原BatchNorm1d，卷积已在分离卷积内部做weight_norm
        self.branch_tanh = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, padding, dilation),
            nn.Tanh()
        )
        self.branch_sig = nn.Sequential(
            DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size, padding, dilation),
            nn.Sigmoid()
        )
        self.spatial_dropout = nn.Dropout1d(0.5)
        # 1x1卷积同样添加weight_norm
        self.final_conv = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size=1))
        self.skip_conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size=1)) if in_channels != out_channels else None
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
    def forward(self, x):
        residual = x if self.skip_conv is None else self.skip_conv(x)
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
            layers.append(TCN_ResBlock(in_channels, tcn_channels, dilation=d, kernel_size=3))
            in_channels = tcn_channels
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

# ==========================================
# 4. AMCS-TCN 主模型入口【完全未改动】
# ==========================================
class AMCS_TCN(nn.Module):
    def __init__(self, num_classes=10, cnn_channels=32, subcarriers=30):
        super().__init__()
        self.antenna_streams = nn.ModuleList([MCS_FMFA_Block(1, cnn_channels) for _ in range(3)])
        self.fca_fusion = FCA_Block(cnn_channels, reduction_ratio=16)
        tcn_in_channels = cnn_channels * subcarriers
        self.tcn = TCN_Backbone(in_channels=tcn_in_channels, tcn_channels=64, dilations=[1, 2, 4, 8])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        ant1 = x[:, 0, :, :].unsqueeze(1)
        ant2 = x[:, 1, :, :].unsqueeze(1)
        ant3 = x[:, 2, :, :].unsqueeze(1)
        f1_1 = self.antenna_streams[0](ant1)
        f1_2 = self.antenna_streams[1](ant2)
        f1_3 = self.antenna_streams[2](ant3)
        f2 = self.fca_fusion([f1_1, f1_2, f1_3])
        B, C, T, Sub = f2.shape
        tcn_in = f2.permute(0, 1, 3, 2).reshape(B, C * Sub, T)
        tcn_out = self.tcn(tcn_in)
        pooled = self.gap(tcn_out).squeeze(-1)
        logits = self.classifier(pooled)
        return logits