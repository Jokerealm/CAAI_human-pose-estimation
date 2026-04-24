## SGraFormer with Dynamic Hypergraph Neural Network Integration
## Based on SGraFormer_raw.py and ST-DHGNN concepts from Proposal.md

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.layers import DropPath

from common.opt import opts
from model.Temporal_encoder import Temporal__features
from functools import partial

opt = opts().parse()
device = torch.device("cuda")


class DynamicHypergraphBlock(nn.Module):
    """
    动态超图模块 - 实现重规划和超边消除机制
    论文建议：k=3, prune_ratio=0.0, 骨架约束
    """
    def __init__(self, in_channels, out_channels, num_joints=17, k_neighbors=3, prune_ratio=0.0):
        """
        参数:
            in_channels: 输入特征维度 C
            out_channels: 输出特征维度
            num_joints: 关节数量 N (SGraFormer中为17)
            k_neighbors: 动态图构建时的K近邻数量
            prune_ratio: 超边消除比例 (例如0.2表示消除方差最大的20%超边)
        """
        super(DynamicHypergraphBlock, self).__init__()
        self.k = k_neighbors
        self.prune_ratio = prune_ratio
        self.num_joints = num_joints
        
        # 线性变换层 (对应公式中的 Theta) - 简化为单层以提升稳定性
        self.theta = nn.Linear(in_channels, out_channels)
        
        # 激活函数
        self.act = nn.GELU()
        
        # 归一化
        self.norm = nn.LayerNorm(out_channels)
        
        # 可学习的超边权重
        self.edge_weight = nn.Parameter(torch.ones(1))
        
        # Dropout for regularization - 降低到0.02以增强学习能力
        self.dropout = nn.Dropout(0.02)

    def construct_dynamic_hypergraph(self, x):
        """
        重规划模块 (Re-planning Module) with Skeletal Bias
        基于论文建议：添加骨架约束，优先连接物理相邻的关节
        输入 x: (B, N, C)
        输出 H: (B, N, N) 关联矩阵
        """
        B, N, C = x.shape
        
        # 1. 计算成对欧氏距离矩阵
        x_norm = (x**2).sum(-1).view(B, N, 1)
        y_norm = x_norm.view(B, 1, N)
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, x.transpose(1, 2))
        dist = dist.clamp(min=1e-12)
        
        # 2. 应用骨架约束（Skeletal Bias）
        # 定义Human3.6M的17关节骨架连接（静态邻接矩阵）
        # 0:Hip, 1:RHip, 2:RKnee, 3:RAnkle, 4:LHip, 5:LKnee, 6:LAnkle,
        # 7:Spine, 8:Thorax, 9:Neck, 10:Head, 11:LShoulder, 12:LElbow, 13:LWrist,
        # 14:RShoulder, 15:RElbow, 16:RWrist
        skeleton_edges = [
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
            (8, 11), (11, 12), (12, 13),  # Left arm
            (8, 14), (14, 15), (15, 16),  # Right arm
        ]
        
        # 创建骨架邻接矩阵
        skeleton_adj = torch.zeros(N, N, device=x.device)
        for i, j in skeleton_edges:
            skeleton_adj[i, j] = 1.0
            skeleton_adj[j, i] = 1.0
        
        # 应用骨架偏置：降低物理相邻关节的距离
        # dist = dist * (1 - 0.5 * skeleton_adj)
        # 扩展到batch维度
        skeleton_bias = (1.0 - 0.5 * skeleton_adj).unsqueeze(0).expand(B, -1, -1)
        dist = dist * skeleton_bias
        
        # 3. K-NN 选择（现在会优先选择骨架相邻的关节）
        _, knn_idx = torch.topk(dist, k=self.k, dim=-1, largest=False)
        
        # 3. 构建关联矩阵 H
        H = torch.zeros(B, N, N, device=x.device)
        
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, self.k)
        hyperedge_idx = torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, self.k)
        
        H[batch_idx, knn_idx, hyperedge_idx] = 1.0
        
        return H

    def hyperedge_elimination(self, x, H):
        """
        超边消除模块 (Hyperedge Elimination Module)
        基于方差剔除不稳定的超边
        """
        B, N, M = H.shape
        
        # 计算超边度数 D_e (B, M)
        D_e = H.sum(dim=1).clamp(min=1.0)
        
        # 1. 聚合超边特征 (计算均值)
        sum_features = torch.bmm(H.transpose(1, 2), x)
        mean_features = sum_features / D_e.unsqueeze(-1)
        
        # 2. 计算方差
        sum_sq_features = torch.bmm(H.transpose(1, 2), x**2)
        mean_sq_features = sum_sq_features / D_e.unsqueeze(-1)
        
        variance = mean_sq_features - mean_features**2
        total_variance = variance.mean(dim=-1)
        
        # 3. 阈值剪枝
        num_keep = int(M * (1 - self.prune_ratio))
        
        _, keep_indices = torch.topk(total_variance, k=num_keep, dim=1, largest=False)
        
        mask = torch.zeros_like(total_variance)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, num_keep)
        mask[batch_idx, keep_indices] = 1.0
        
        H_pruned = H * mask.unsqueeze(1)
        
        return H_pruned

    def forward(self, x):
        """
        前向传播
        x: (B_total, N, C)
        """
        # 1. 重规划: 动态构建图
        H = self.construct_dynamic_hypergraph(x)
        
        # 2. 消除: 剪枝
        if self.training or self.prune_ratio > 0:
            H = self.hyperedge_elimination(x, H)
        
        # 3. 超图卷积
        x_theta = self.theta(x)
        
        # 构建度矩阵
        D_v = H.sum(dim=2).clamp(min=1.0)
        D_e = H.sum(dim=1).clamp(min=1.0)
        
        # 归一化因子
        D_v_inv = D_v.pow(-0.5).unsqueeze(2)
        D_e_inv = D_e.pow(-1.0).unsqueeze(1)
        
        # 消息传递: 节点 -> 超边
        x_norm = x_theta * D_v_inv
        Y = torch.bmm(H.transpose(1, 2), x_norm)
        Y = Y * D_e_inv.transpose(1, 2)
        
        # 消息传递: 超边 -> 节点
        Z = torch.bmm(H, Y)
        Z = Z * D_v_inv
        
        # 归一化和激活
        Z = self.norm(Z)
        Z = self.act(Z)
        Z = self.dropout(Z)
        
        # 残差连接
        if x.shape[-1] == Z.shape[-1]:
            return x + Z
        else:
            return Z


class MultiLevelHypergraph(nn.Module):
    """
    多层级超图：Joint-Level + Part-Level + Global-Level
    参考：
    1. HyperDiff: joint, part, global三种层级
    2. ST-DHGNN: head-edge, leg-edge, trunk-edge等
    """
    def __init__(self, embed_dim, num_joints=17, k=3):
        super().__init__()
        self.num_joints = num_joints
        self.embed_dim = embed_dim
        self.per_joint_dim = embed_dim // num_joints
        
        # 定义身体部位分组（Part-Level）
        self.part_groups = {
            'head': [9, 10],  # 头部
            'trunk': [0, 7, 8],  # 躯干
            'left_arm': [8, 14, 15, 16],  # 左臂
            'right_arm': [8, 11, 12, 13],  # 右臂
            'left_leg': [0, 4, 5, 6],  # 左腿
            'right_leg': [0, 1, 2, 3],  # 右腿
        }
        
        # Joint-Level HyperGraph（细粒度）
        self.joint_transform = nn.Linear(self.per_joint_dim, self.per_joint_dim)
        self.joint_norm = nn.LayerNorm(self.per_joint_dim)
        self.k = k
        
        # Part-Level HyperGraph（中粒度）
        self.part_transforms = nn.ModuleDict({
            name: nn.Linear(self.per_joint_dim, self.per_joint_dim)
            for name in self.part_groups.keys()
        })
        self.part_norm = nn.LayerNorm(self.per_joint_dim)
        
        # Global-Level HyperGraph（粗粒度）
        self.global_transform = nn.Linear(self.per_joint_dim, self.per_joint_dim)
        self.global_norm = nn.LayerNorm(self.per_joint_dim)
        
        # 融合三个层级
        self.fusion = nn.Linear(self.per_joint_dim * 3, self.per_joint_dim)
        
        # 骨架邻接矩阵（用于Joint-Level）
        self.register_buffer('skeleton_adj', self._build_skeleton())
    
    def _build_skeleton(self):
        adj = torch.zeros(17, 17)
        connections = [
            (0, 1), (1, 2), (2, 3),  # Right leg
            (0, 4), (4, 5), (5, 6),  # Left leg
            (0, 7), (7, 8), (8, 9), (9, 10),  # Spine
            (8, 11), (11, 12), (12, 13),  # Right arm
            (8, 14), (14, 15), (15, 16),  # Left arm
        ]
        for i, j in connections:
            adj[i, j] = adj[j, i] = 1.0
        adj.fill_diagonal_(1.0)
        return adj
    
    def joint_level_hypergraph(self, x):
        """
        Joint-Level: 每个关节连接其k近邻（骨架约束）
        x: [B*F, J, C]
        """
        # 计算距离并应用骨架偏置
        dist = torch.cdist(x, x)
        skeleton_bias = self.skeleton_adj.unsqueeze(0)
        dist = dist * (1.0 - 0.5 * skeleton_bias)
        
        # k-NN
        _, knn_idx = torch.topk(dist, k=self.k, dim=-1, largest=False)
        
        # 邻居聚合
        x_neighbors = torch.gather(
            x.unsqueeze(2).expand(-1, -1, self.k, -1),
            1,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
        )
        x_agg = x_neighbors.mean(dim=2)
        
        # 变换
        x_out = self.joint_transform(x_agg)
        x_out = self.joint_norm(x_out)
        
        return x_out
    
    def part_level_hypergraph(self, x):
        """
        Part-Level: 身体部位级别的超边
        x: [B*F, J, C]
        """
        BF, J, C = x.shape
        x_part = torch.zeros_like(x)
        
        for part_name, indices in self.part_groups.items():
            # 提取该部位的关节
            part_x = x[:, indices, :]  # [B*F, len(indices), C]
            
            # 部位内聚合
            part_agg = part_x.mean(dim=1, keepdim=True)  # [B*F, 1, C]
            
            # 变换
            part_feat = self.part_transforms[part_name](part_agg)  # [B*F, 1, C]
            
            # 广播回该部位的所有关节
            for idx in indices:
                x_part[:, idx, :] = part_feat.squeeze(1)
        
        x_part = self.part_norm(x_part)
        return x_part
    
    def global_level_hypergraph(self, x):
        """
        Global-Level: 所有关节在一个超边中
        x: [B*F, J, C]
        """
        # 全局聚合
        x_global = x.mean(dim=1, keepdim=True)  # [B*F, 1, C]
        
        # 变换
        x_global = self.global_transform(x_global)  # [B*F, 1, C]
        
        # 广播到所有关节
        x_global = x_global.expand(-1, self.num_joints, -1)  # [B*F, J, C]
        
        x_global = self.global_norm(x_global)
        return x_global
    
    def forward(self, x):
        """
        x: [B, F, J*C]
        返回: [B, F, J*C]
        """
        B, F, JC = x.shape
        C = self.per_joint_dim
        
        # Reshape to [B*F, J, C]
        x_reshaped = x.view(B, F, self.num_joints, C).view(B*F, self.num_joints, C)
        
        # 三个层级的超图
        x_joint = self.joint_level_hypergraph(x_reshaped)  # 细粒度
        x_part = self.part_level_hypergraph(x_reshaped)    # 中粒度
        x_global = self.global_level_hypergraph(x_reshaped)  # 粗粒度
        
        # 融合三个层级
        x_multi = torch.cat([x_joint, x_part, x_global], dim=-1)  # [B*F, J, 3*C]
        x_fused = self.fusion(x_multi)  # [B*F, J, C]
        
        # Reshape back
        x_out = x_fused.view(B, F, JC)
        
        return x_out




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HybridSpatialEncoder(nn.Module):
    """
    混合空间编码器：Semantic Attention + 多层级HyperGraph（并行）
    """
    def __init__(self, base_encoder, embed_dim, num_joints=17, k=3):
        super().__init__()
        self.base_encoder = base_encoder  # 原始的Semantic Attention
        self.multi_level_hg = MultiLevelHypergraph(embed_dim, num_joints, k)
        self.alpha_hg = nn.Parameter(torch.tensor(0.1))  # 可学习融合权重
        
    def forward(self, x, hop, *args):
        """
        x: [B, C, F, J]
        hop: hop特征
        *args: MSA等
        """
        # Semantic Attention（主分支）
        x_sa, hop_out, *other_outputs = self.base_encoder(x, hop, *args)
        
        # 多层级HyperGraph（辅助分支）
        # Joint-Level + Part-Level + Global-Level
        x_hg = self.multi_level_hg(x_sa)
        
        # 并行残差融合
        x_fused = x_sa + self.alpha_hg * x_hg
        
        return x_fused, hop_out, *other_outputs


class SGraFormer_HyperGraph(nn.Module):
    """
    SGraFormer + 多层级HyperGraph
    在每个Spatial encoder后添加HyperGraph辅助（多层级）
    """
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 k_neighbors=3, prune_ratio=0.0, num_heads=8, mlp_ratio=2., 
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0.2, norm_layer=None):
        """
        多层级HyperGraph：在每个视点的Spatial encoder后添加辅助
        """
        super().__init__()
        
        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3
        
        # 导入原始SGraFormer的Spatial encoders
        from model.Spatial_encoder import First_view_Spatial_features, Spatial_features
        
        # 创建基础的Semantic Attention编码器
        base_SF1 = First_view_Spatial_features(
            num_frame, num_joints, in_chans, embed_dim_ratio, depth,
            num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate, norm_layer
        )
        base_SF2 = Spatial_features(
            num_frame, num_joints, in_chans, embed_dim_ratio, depth,
            num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate, norm_layer
        )
        base_SF3 = Spatial_features(
            num_frame, num_joints, in_chans, embed_dim_ratio, depth,
            num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate, norm_layer
        )
        base_SF4 = Spatial_features(
            num_frame, num_joints, in_chans, embed_dim_ratio, depth,
            num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate, norm_layer
        )
        
        # 包装为混合编码器（多层级HyperGraph）
        self.SF1 = HybridSpatialEncoder(base_SF1, embed_dim, num_joints, k=k_neighbors)
        self.SF2 = HybridSpatialEncoder(base_SF2, embed_dim, num_joints, k=k_neighbors)
        self.SF3 = HybridSpatialEncoder(base_SF3, embed_dim, num_joints, k=k_neighbors)
        self.SF4 = HybridSpatialEncoder(base_SF4, embed_dim, num_joints, k=k_neighbors)
        
        # 多视点融合（保留）
        self.view_pos_embed = nn.Parameter(torch.zeros(1, 4, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)
        
        self.conv = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.conv_hop = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.conv_norm = nn.LayerNorm(embed_dim)
        self.conv_hop_norm = nn.LayerNorm(embed_dim)
        
        # 时序编码器（保留）
        self.TF = Temporal__features(
            num_frame, num_joints, in_chans, embed_dim_ratio, depth,
            num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate, norm_layer
        )
        
        # 输出头（保留）
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        
        # Hop参数（保留）
        self.hop_w0 = nn.Parameter(torch.ones(17, 17))
        self.hop_w1 = nn.Parameter(torch.ones(17, 17))
        self.hop_w2 = nn.Parameter(torch.ones(17, 17))
        self.hop_w3 = nn.Parameter(torch.ones(17, 17))
        self.hop_w4 = nn.Parameter(torch.ones(17, 17))
        self.hop_global = nn.Parameter(torch.ones(17, 17))
        
        self.linear_hop = nn.Linear(8, 2)
        self.edge_embedding = nn.Linear(17*17*4, 17*17)
        
    def forward(self, x, hops):
        """
        x: [B, F, V, J, 2]
        hops: hop矩阵
        """
        b, f, v, j, c = x.shape
        
        # Edge embedding
        edge_embedding = self.edge_embedding(hops[0].reshape(1, -1))
        
        # Global hop features
        x_hop_global = x.unsqueeze(3).repeat(1, 1, 1, 17, 1, 1)
        x_hop_global = x_hop_global - x_hop_global.permute(0, 1, 2, 4, 3, 5)
        x_hop_global = torch.sum(x_hop_global ** 2, dim=-1)
        hop_global = x_hop_global / torch.sum(x_hop_global, dim=-1).unsqueeze(-1)
        
        hops = hops.unsqueeze(1).unsqueeze(2).repeat(1, f, v, 1, 1, 1)
        hops1 = hop_global * hops[:, :, :, 0]
        hops2 = hop_global * hops[:, :, :, 1]
        hops3 = hop_global * hops[:, :, :, 2]
        hops4 = hop_global * hops[:, :, :, 3]
        hops = torch.cat((hops1, hops2, hops3, hops4), dim=-1)
        
        # 分离视点
        x1 = x[:, :, 0].permute(0, 3, 1, 2)
        x2 = x[:, :, 1].permute(0, 3, 1, 2)
        x3 = x[:, :, 2].permute(0, 3, 1, 2)
        x4 = x[:, :, 3].permute(0, 3, 1, 2)
        
        hop1 = hops[:, :, 0].permute(0, 3, 1, 2)
        hop2 = hops[:, :, 1].permute(0, 3, 1, 2)
        hop3 = hops[:, :, 2].permute(0, 3, 1, 2)
        hop4 = hops[:, :, 3].permute(0, 3, 1, 2)
        
        # 混合编码器（Semantic Attention + HyperGraph，多层级）
        x1, hop1, MSA1, MSA2, MSA3, MSA4 = self.SF1(x1, hop1, edge_embedding)
        x2, hop2, MSA1, MSA2, MSA3, MSA4 = self.SF2(x2, hop2, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        x3, hop3, MSA1, MSA2, MSA3, MSA4 = self.SF3(x3, hop3, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        x4, hop4, MSA1, MSA2, MSA3, MSA4 = self.SF4(x4, hop4, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        
        # 多视点融合
        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)), dim=1) + self.view_pos_embed
        x = self.pos_drop(x)
        x = self.conv(x).squeeze(1) + x1 + x2 + x3 + x4
        x = self.conv_norm(x)
        
        hop = torch.cat((hop1.unsqueeze(1), hop2.unsqueeze(1), hop3.unsqueeze(1), hop4.unsqueeze(1)), dim=1) + self.view_pos_embed
        hop = self.pos_drop(hop)
        hop = self.conv(hop).squeeze(1) + hop1 + hop2 + hop3 + hop4
        hop = self.conv_norm(hop)
        
        x = x * hop
        
        # 时序编码
        x = self.TF(x)
        
        # 输出
        x = self.head(x)
        x = x.view(b, opt.frames, j, -1)
        
        return x



