import torch
import torch.nn.functional as F
from torch import nn
import os
import random

class RegionAggregationModule(nn.Module):
    def __init__(self, num_regions=64, dim=512, alpha=100.0, normalize_input=True,
                 continuity_weight=0.1, continuity_kernel=3):  # 移除 vis_dir 参数
        super().__init__()
        self.num_regions = num_regions
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.continuity_weight = continuity_weight  # 连续性约束权重
        self.continuity_kernel = continuity_kernel  # 连续性检测核大小

        # 移除可视化目录相关代码
        # 移除可视化计数器

        # 区域分配卷积层
        self.conv = nn.Conv2d(dim, num_regions, kernel_size=(1, 1), bias=True)

        # 空间连续性检测器
        self.continuity_conv = nn.Conv2d(
            num_regions, num_regions,
            kernel_size=continuity_kernel,
            padding=continuity_kernel // 2,
            groups=num_regions,
            bias=False
        )
        # 固定权重为高斯核，检测空间连续性
        self._init_continuity_kernel()

        # 区域中心参数
        self.centroids = nn.Parameter(torch.rand(num_regions, dim))
        self.init_weights()

    def _init_continuity_kernel(self):
        """初始化空间连续性检测核为高斯分布"""
        k = self.continuity_kernel
        sigma = k * 0.3 + 0.8  # 经验公式计算sigma
        ax = torch.arange(k).float() - k // 2
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx **2 + yy** 2) / (2 * sigma **2))
        kernel = kernel / kernel.sum()  # 归一化

        # 复制到所有通道
        kernel = kernel.view(1, 1, k, k).repeat(
            self.num_regions, 1, 1, 1
        )
        self.continuity_conv.weight.data = kernel
        self.continuity_conv.weight.requires_grad = False  # 固定权重

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, grids):
        N, H, W, C = grids.shape
        grids = grids.permute(0, 3, 1, 2).contiguous()  # [N, C, H, W]

        if self.normalize_input:
            grids = F.normalize(grids, p=2, dim=1)  # 沿通道维度归一化

        # 区域分配概率
        soft_assign = self.conv(grids)  # [N, num_regions, H, W]
        soft_assign = F.softmax(soft_assign, dim=1)  # 区域维度softmax

        # 计算空间连续性损失
        continuity_loss = self.calc_spatial_continuity(soft_assign)

        # 区域特征聚合
        x_flatten = grids.view(N, C, -1)  # [N, C, H*W]
        soft_assign_flat = soft_assign.view(N, self.num_regions, -1)  # [N, num_regions, H*W]

        # 高效残差计算
        centroids_exp = self.centroids.unsqueeze(0).unsqueeze(-1)  # [1, num_regions, C, 1]
        x_exp = x_flatten.unsqueeze(1)  # [N, 1, C, H*W]
        residual = x_exp - centroids_exp  # [N, num_regions, C, H*W]

        # 加权聚合
        residual *= soft_assign_flat.unsqueeze(2)  # [N, num_regions, 1, H*W]
        p = residual.sum(dim=-1)  # [N, num_regions, C] 沿空间位置求和

        # 区域级归一化
        p = F.normalize(p, p=2, dim=2)  # 在特征维度(C)上归一化

        # 移除可视化调用代码

        return p, continuity_loss

    def calc_spatial_continuity(self, assign):
        """
        计算空间连续性损失（基于梯度）
        目标：相邻位置的区域分配应该相似
        """
        # 计算水平和垂直方向的梯度
        grad_x = torch.abs(assign[:, :, :, 1:] - assign[:, :, :, :-1])  # [N, R, H, W-1]
        grad_y = torch.abs(assign[:, :, 1:, :] - assign[:, :, :-1, :])  # [N, R, H-1, W]

        # 确保两个梯度的空间维度匹配
        # 裁剪到最小公共尺寸
        h = min(grad_x.size(2), grad_y.size(2))
        w = min(grad_x.size(3), grad_y.size(3))

        grad_x = grad_x[:, :, :h, :w]
        grad_y = grad_y[:, :, :h, :w]

        # 计算梯度幅度
        grad_magnitude = torch.sqrt(grad_x **2 + grad_y** 2 + 1e-8)

        # 使用对数空间增强小梯度敏感性
        log_grad = torch.log(1 + 100 * grad_magnitude)

        # 计算平均梯度损失
        loss = torch.mean(log_grad)

        return self.continuity_weight * loss

    # 移除 visualize_assignment 方法