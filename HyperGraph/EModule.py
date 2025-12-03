import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import math
from HyperGraph.hypergraph import HyperGCN
from HyperGraph.mixste import LearnableGraphConv, SGA
import torch.nn.functional as F


class Enhance_Module(nn.Module):
    def __init__(self, num_joints=17, in_channels=2, feature_dim=64, dataset='h36m'):
        super().__init__()
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.dataset = dataset
        
        # 输入投影层，将2D坐标提升到更高维度
        self.input_proj = nn.Linear(in_channels, feature_dim)
        
        # 超图卷积层集成了关节、部位和身体三个层次的特征增强
        self.hypergcn = HyperGCN(
            dim_in=feature_dim, 
            dim_out=feature_dim, 
            num_nodes=num_joints,
            neighbour_num=4,
            use_partscale=True,
            use_bodyscale=True,
            dataset=dataset
        )
        
        # 批归一化层 - 注意：应该对特征维度(feature_dim)进行归一化，而不是关节数
        self.bn = nn.BatchNorm1d(feature_dim)
        self.relu = nn.ReLU()
        
        # 用于z_enhanced_1的线性变换层
        self.residual_linear = nn.Linear(feature_dim, feature_dim)
        
        # 输出投影层，将增强的特征映射回原始维度
        self.output_proj = nn.Linear(feature_dim, in_channels)
    
        # 初始化权重
        self._init_weights()
        
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1.414)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=1.414)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, x):
        # x: [BatchSize, Number_of_Views, Number_of_Frames, Number_of_Joints, 2]
        batch_size, num_views, num_frames, num_joints, in_channels = x.shape
        
        # 重塑输入以处理每个视角的每一帧
        # [BatchSize * Number_of_Views * Number_of_Frames, Number_of_Joints, 2]
        x_reshaped = rearrange(x, 'b v f j c -> (b v f) j c')
        
        # 提升特征维度 - 这是Z_{l-1}
        x_proj = self.input_proj(x_reshaped)
        
        # 这里我们将帧数设置为1，因为我们一次处理一帧
        x_hyper = x_proj.unsqueeze(1)  # [B*V*F, 1, J, C]
        
        # 使用HyperGCN进行多粒度特征增强
        # HyperGCN内部已经处理了关节、部位和身体三个层次的特征提取和融合
        z_enhanced = self.hypergcn(x_hyper)
        
        # 移除临时维度
        z_enhanced = z_enhanced.squeeze(1)
        
        # 应用Linear模块处理Z_{l-1}
        z_linear = self.residual_linear(x_proj)
        
        # 第一个残差连接：Z_joint + Z_part + Z_body + Linear(Z_{l-1})
        z_enhanced = z_enhanced + z_linear
        
        # 重塑输入以应用BatchNorm1d
        z_enhanced_reshaped = rearrange(z_enhanced, '(b v f) j c -> (b v f j) c', 
                                      b=batch_size, v=num_views, f=num_frames)
        
        # 应用批归一化
        z_enhanced_reshaped = self.bn(z_enhanced_reshaped)
        
        # 第二个残差连接：在ReLU前连接Z_{l-1}
        x_proj_reshaped = rearrange(x_proj, '(b v f) j c -> (b v f j) c', 
                                  b=batch_size, v=num_views, f=num_frames)
        z_enhanced_with_residual = z_enhanced_reshaped + x_proj_reshaped
        
        # 激活函数
        z_enhanced_with_residual = self.relu(z_enhanced_with_residual)
        
        # 恢复关节维度，为了输出投影做准备
        # [B*V*F, J, C]
        z_enhanced_3d = rearrange(z_enhanced_with_residual, '(b v f j) c -> (b v f) j c', 
                                 b=batch_size, v=num_views, f=num_frames)
        
        # 投影回原始维度
        z_enhanced_proj = self.output_proj(z_enhanced_3d)
        
        # 恢复原始形状
        z_enhanced = rearrange(z_enhanced_proj, '(b v f) j c -> b v f j c', 
                              b=batch_size, v=num_views, f=num_frames)
        
        return z_enhanced


class GraphConvWrapper(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, adj):
        super().__init__()
        self.gconv = LearnableGraphConv(in_features, out_features, adj)
    
    def forward(self, x):
        return self.gconv(x)