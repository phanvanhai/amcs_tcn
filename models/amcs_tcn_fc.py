import torch.nn as nn

# MCS_FMFA_Block / FCA_Block / TCN 组件与 GAP 版本完全相同，统一从 amcs_tcn 导入
from models.amcs_tcn import AMCS_TCN_Base


# ==========================================
# AMCS-TCN (FC 分类头版本)
# ==========================================
class AMCS_TCN_FC(AMCS_TCN_Base):
    def __init__(self, num_classes=7, subcarriers=30, antennas=3, time_steps=250, fc_hidden=256, **kwargs):
        super().__init__(num_classes, subcarriers, antennas, time_steps, **kwargs)

        # 【修改点】：移除 GAP，改为全连接层 (FC)
        self.flatten = nn.Flatten()

        # 每个 TCN ResBlock 末尾 MaxPool1d(2, stride=2)，连续向下取整等价于 T // 2^n
        # 例: T=250, 4 层 -> 15，展平维度 64 * 15 = 960
        t_out = time_steps // (2 ** len(self.dilations))
        if t_out < 1:
            raise ValueError(f"time_steps={time_steps} too short for {len(self.dilations)} TCN pooling layers")

        # 添加一个带有 Dropout 的隐藏层以增强拟合能力并防止过拟合
        self.classifier = nn.Sequential(
            nn.Linear(self.tcn_channels * t_out, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden, num_classes)
        )

    def forward(self, x):
        tcn_out = self.extract(x)
        # 6. 【修改点】：展平后直接输入 FC 分类器
        return self.classifier(self.flatten(tcn_out))
