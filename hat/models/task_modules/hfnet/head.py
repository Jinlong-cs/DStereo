# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Dict, List

import torch
import horizon_plugin_pytorch as hpp
import torch.nn as nn

from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from torch import Tensor
import torch.nn.functional as F

from hat.models.base_modules.conv_module import ConvModule2d
from hat.models.weight_init import kaiming_init
from hat.registry import OBJECT_REGISTRY

from hat.utils.model_helpers import fx_wrap


__all__ = ["HfVLADLayer"]


class ConvApproxNormalize(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvApproxNormalize, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)  # 增加BatchNorm层

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  # BatchNorm层可以在一定程度上起到归一化的作用
        return x


@OBJECT_REGISTRY.register
class HfVLADLayer(nn.Module):
    def __init__(self, feature_dim, n_clusters, dimensionality_reduction):
        """
        Args:
            n_clusters : int
                The number of clusters
            feature_dim : int
                Dimension of descriptors
        """
        super(HfVLADLayer, self).__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim

        self.conv0 = nn.Conv2d(feature_dim, 128, kernel_size=1)
        self.conv1 = ConvModule2d(128, n_clusters, kernel_size=1)
        self.dim_reduc = nn.Conv2d(n_clusters * 128, 
                                                dimensionality_reduction, kernel_size=1)
        
        self.approx_norm1 = ConvApproxNormalize(n_clusters * 128, n_clusters * 128)
        self.approx_norm2 = ConvApproxNormalize(dimensionality_reduction, dimensionality_reduction)
        
        self.clusters = nn.Parameter(torch.rand(n_clusters, 128, 1, 1))
        self.clusters.data = nn.init.xavier_uniform_(self.clusters.data)

        self.softmax = nn.Softmax(dim=1)

    @fx_wrap()
    def l2_normalize(self, x, dim=-1, eps=1e-12):
        """使用torch.sqrt手动实现L2归一化"""
        square_sum = torch.sum(torch.pow(x,exponent=2), dim=dim, keepdim=True)
        norm = torch.sqrt(square_sum + eps)
        normalized_x = torch.div(x, norm)
        return normalized_x

    def forward(self, features_inputs: List[Tensor]) -> Tensor:
        assert len(features_inputs)==1
        feature_map = features_inputs[0]
        # feature_map形状: [batch_size, feature_dim, H, W]
        feature_map = self.conv0(feature_map) # 降维
    
        # 计算每个像素点属于各个聚类的概率
        memberships = self.conv1(feature_map)  # 形状: [batch_size, n_clusters, H, W]
        memberships = self.softmax(memberships)  # 应用Softmax进行归一化
        
        # 计算残差: 聚类中心减去特征图，然后乘以成员关系概率
        feature_map_expanded = feature_map.unsqueeze(1)  # 插入新维度以进行广播，形状: [batch_size, feature_dim, 1, H, W]
        
        residuals = torch.sub(self.clusters, feature_map_expanded) # 形状: [batch_size, n_clusters, feature_dim, H, W]
        residuals = torch.mul(residuals,memberships.unsqueeze(2))  # 将成员关系概率应用于残差，形状: [batch_size, n_clusters, feature_dim, H, W]
        
        # 对残差在空间维度(H, W)上求和，得到VLAD描述符
        descriptor = torch.sum(residuals, dim=3, keepdim=False) 
        descriptor = torch.sum(descriptor, dim=3, keepdim=False)   # 形状: [batch_size, n_clusters, feature_dim]
        
        # 手动进行L2归一化，保证描述符在每个维度的长度为1
        descriptor = F.normalize(descriptor, dim=2)  # 形状: [batch_size, n_clusters, feature_dim]
        descriptor = descriptor.view(descriptor.size(0), -1)  # 展平，形状: [batch_size, n_clusters * feature_dim]
        descriptor = F.normalize(descriptor, dim=1)  # 再次归一化
        
        # # 使用1x1卷积进行降维
        # descriptor = descriptor.unsqueeze(-1).unsqueeze(-1)  # 调整形状以匹配1x1卷积的输入要求，形状: [batch_size, n_clusters * feature_dim, 1, 1]
        # descriptor = self.dim_reduc(descriptor)  # 1x1卷积降维，形状: [batch_size, dimensionality_reduction, 1, 1]
        # descriptor = descriptor.squeeze(-1).squeeze(-1)  # 移除最后两个维度，形状: [batch_size, dimensionality_reduction]
        
        # # 最终描述符的L2归一化
        # descriptor = F.normalize(descriptor, dim=1)  # 形状: [batch_size, dimensionality_reduction]

        return descriptor