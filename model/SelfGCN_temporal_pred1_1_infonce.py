"""
SelfGCN with Temporal Prediction Module
在骨干网络l10层后提取特征，进行时序切分和预测

`SelfGCN_temporal_pred1_1.py` (work_dir10)
•  有预测模块 TemporalPredictionModule（3层卷积）
•  单向预测：past → future
•  固定gap=3
•  这才是work_dir10实际使用的模型

"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

# 从原始SelfGCN导入所需组件
from model.SelfGCN2 import (
    import_class, conv_init, bn_init, weights_init,
    TemporalConv, MultiScale_TemporalConv, Bi_Inter,
    SelfGCN_Block, unit_tcn, unit_gcn, TCN_GCN_unit
)

class TemporalPredictionModule(nn.Module):
    """
    轻量级时序预测模块
    输入Xpast，预测Xfuture
    """
    def __init__(self, in_channels, hidden_channels=None):
        super(TemporalPredictionModule, self).__init__()
        if hidden_channels is None:
            hidden_channels = in_channels // 2
        
        self.pred_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(in_channels),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
    
    def forward(self, x_past):
        """
        Args:
            x_past: (N*M, C, T/2, V) 前半段特征
        Returns:
            x_pred: (N*M, C, T/2, V) 预测的后半段特征
        """
        return self.pred_net(x_past)


class ModelWithTemporalPrediction(nn.Module):
    """
    带有时序预测的SelfGCN模型
    在l10层后进行时序切分，并预测未来帧特征
    """
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), 
                 in_channels=3, drop_out=0, adaptive=True, pred_hidden_channels=None):
        super(ModelWithTemporalPrediction, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64

        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)

        # 时序预测模块
        self.temporal_pred = TemporalPredictionModule(
            in_channels=base_channel * 4,
            hidden_channels=pred_hidden_channels
        )

        # 新增：用于修复MSE
        # =================【修改 1：在此处添加定义】=================
        self.pred_bn = nn.BatchNorm2d(base_channel * 4)
        bn_init(self.pred_bn, 1)  # 别忘了初始化，否则训练初期不稳定
        # ==========================================================

        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.view(N, T, self.num_point, -1).permute(0, 3, 1, 2).contiguous().unsqueeze(-1)
        N, C, T, V, M = x.size()
        
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # 通过l1-l10层
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # l10层后的特征: (N*M, C, T', V), 其中C=256, T'是下采样后的时间维度
        NM, C_feat, T_feat, V_feat = x.size()
        
        # 分类分支：使用完整特征
        c_new = x.size(1)
        x_cls = x.view(N, M, c_new, -1)
        x_cls = x_cls.mean(3).mean(1)
        x_cls = self.drop_out(x_cls)
        cls_score = self.fc(x_cls)

        # === 修改部分：带 Gap 的时序切分 ===
        x_pred = None
        x_future = None

        # 只在训练时进行时序预测
        if self.training and T_feat >= 6:  # 确保时间够长
            gap = 3  # 设置 gap=3，强迫模型预测更远的未来
            mid = T_feat // 2

            if mid > gap:
                # 切分 Past 和 Future
                x_past_raw = x[:, :, :mid - gap, :]
                x_future_raw = x[:, :, mid + gap:, :]

                # 长度对齐
                min_len = min(x_past_raw.shape[2], x_future_raw.shape[2])
                if min_len > 0:
                    x_past_trim = x_past_raw[:, :, -min_len:, :]
                    x_future_trim = x_future_raw[:, :, :min_len, :]

                    # 预测
                    x_pred = self.temporal_pred(x_past_trim)
                    x_future = x_future_trim

                    # [可选] 对特征进行 L2 归一化，有助于 Cosine Loss 收敛   mse和smooth l1不需要这个
                    x_pred = torch.nn.functional.normalize(x_pred, dim=1)
                    x_future = torch.nn.functional.normalize(x_future, dim=1)
            return cls_score, x_pred, x_future
        else:
            # 测试阶段或时间维度太短，不进行预测
            return cls_score, None, None
