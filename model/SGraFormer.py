"""
Hybrid SGraFormer with Part-Level HyperGraph
结合SGraFormer的Global Semantic Attention和HyperGraph的Part-Level建模
"""

import torch
import torch.nn as nn
from einops import rearrange

from common.opt import opts
from model.Spatial_encoder import First_view_Spatial_features, Spatial_features
from model.Temporal_encoder import Temporal__features

opt = opts().parse()
device = torch.device("cuda")


class PartLevelHyperGraphModule(nn.Module):
    """
    Part-Level HyperGraph模块
    基于人体部位（躯干、左臂、右臂、左腿、右腿）构建超图
    """
    def __init__(self, embed_dim, num_joints=17):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        
        # 定义人体部位（Human3.6M 17关节）
        # 0:Hip, 1:RHip, 2:RKnee, 3:RAnkle, 4:LHip, 5:LKnee, 6:LAnkle,
        # 7:Spine, 8:Thorax, 9:Neck, 10:Head, 11:LShoulder, 12:LElbow, 13:LWrist,
        # 14:RShoulder, 15:RElbow, 16:RWrist
        self.body_parts = {
            'torso': [0, 7, 8, 9, 10],  # Hip, Spine, Thorax, Neck, Head
            'left_arm': [8, 11, 12, 13],  # Thorax, LShoulder, LElbow, LWrist
            'right_arm': [8, 14, 15, 16],  # Thorax, RShoulder, RElbow, RWrist
            'left_leg': [0, 4, 5, 6],  # Hip, LHip, LKnee, LAnkle
            'right_leg': [0, 1, 2, 3],  # Hip, RHip, RKnee, RAnkle
        }
        
        # 为每个部位创建超图卷积
        self.part_convs = nn.ModuleDict({
            part_name: nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(0.02),
                nn.Linear(embed_dim, embed_dim)
            ) for part_name in self.body_parts.keys()
        })
        
        # 部位间交互（跨部位超边）
        self.inter_part_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.02),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 归一化
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        x: (B, N, C) - Batch, Joints, Channels
        """
        B, N, C = x.shape
        
        # 初始化输出
        x_out = torch.zeros_like(x)
        
        # 1. Part-level aggregation（部位内聚合）
        for part_name, joint_indices in self.body_parts.items():
            # 提取该部位的关节特征
            part_features = x[:, joint_indices, :]  # (B, num_joints_in_part, C)
            
            # 部位内聚合：计算部位中心特征
            part_center = part_features.mean(dim=1, keepdim=True)  # (B, 1, C)
            
            # 通过部位特定的卷积
            part_refined = self.part_convs[part_name](part_center)  # (B, 1, C)
            
            # 广播回该部位的所有关节
            part_refined = part_refined.expand(-1, len(joint_indices), -1)
            
            # 累加到输出
            x_out[:, joint_indices, :] += part_refined
        
        # 2. Inter-part interaction（部位间交互）
        # 计算全局特征
        global_feature = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        global_refined = self.inter_part_conv(global_feature)  # (B, 1, C)
        global_refined = global_refined.expand(-1, N, -1)  # (B, N, C)
        
        # 3. 融合
        x_out = x_out + global_refined
        x_out = self.norm(x_out)
        
        return x_out


class HybridSpatialBlock(nn.Module):
    """
    混合空间块：Semantic Attention (主) + Part-Level HyperGraph (辅)
    """
    def __init__(self, spatial_encoder, embed_dim, num_joints=17, alpha=0.1):
        super().__init__()
        
        # 主路径：原始的Semantic Attention
        self.semantic_attention = spatial_encoder
        
        # 辅助路径：Part-Level HyperGraph
        self.hypergraph = PartLevelHyperGraphModule(embed_dim, num_joints)
        
        # 可学习的融合权重
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, x, hops, *args):
        """
        x: (b, c, f, j) - 原始SGraFormer格式
        """
        b, c, f, j = x.shape
        
        # 主路径：Semantic Attention
        x_sa, hops_out, *msa_outputs = self.semantic_attention(x, hops, *args)
        # x_sa: (b, f, j*embed_dim_ratio)
        
        # 准备HyperGraph输入
        # 需要reshape为 (b*f, j, embed_dim_ratio)
        embed_dim_ratio = x_sa.shape[-1] // j
        x_hg_input = x_sa.view(b, f, j, embed_dim_ratio)
        x_hg_input = x_hg_input.view(b * f, j, embed_dim_ratio)
        
        # 辅助路径：Part-Level HyperGraph
        x_hg = self.hypergraph(x_hg_input)  # (b*f, j, embed_dim_ratio)
        
        # Reshape回原始格式
        x_hg = x_hg.view(b, f, j * embed_dim_ratio)
        
        # 融合：主路径 + α * 辅助路径
        x_out = x_sa + torch.sigmoid(self.alpha) * x_hg
        
        return x_out, hops_out, *msa_outputs


class HybridSGraFormer(nn.Module):
    """
    混合SGraFormer：结合Global Semantic Attention和Part-Level HyperGraph
    """
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=None,
                 use_hypergraph=True, hypergraph_alpha=0.1):
        super().__init__()
        
        self.use_hypergraph = use_hypergraph
        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3
        
        # 创建原始的Spatial encoders
        sf1_base = First_view_Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                                num_heads, mlp_ratio, qkv_bias, qk_scale,
                                                drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        sf2_base = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                     num_heads, mlp_ratio, qkv_bias, qk_scale,
                                     drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        sf3_base = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                     num_heads, mlp_ratio, qkv_bias, qk_scale,
                                     drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        sf4_base = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                     num_heads, mlp_ratio, qkv_bias, qk_scale,
                                     drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        
        if use_hypergraph:
            # 使用混合块（Semantic Attention + HyperGraph）
            self.SF1 = HybridSpatialBlock(sf1_base, embed_dim_ratio, num_joints, hypergraph_alpha)
            self.SF2 = HybridSpatialBlock(sf2_base, embed_dim_ratio, num_joints, hypergraph_alpha)
            self.SF3 = HybridSpatialBlock(sf3_base, embed_dim_ratio, num_joints, hypergraph_alpha)
            self.SF4 = HybridSpatialBlock(sf4_base, embed_dim_ratio, num_joints, hypergraph_alpha)
        else:
            # 纯Semantic Attention（baseline）
            self.SF1 = sf1_base
            self.SF2 = sf2_base
            self.SF3 = sf3_base
            self.SF4 = sf4_base
        
        # Multi-view fusion (保持不变)
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
        
        # Temporal encoder (保持不变)
        self.TF = Temporal__features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                     num_heads, mlp_ratio, qkv_bias, qk_scale,
                                     drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        
        # Output head (保持不变)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )
        
        # Hop parameters (保持不变)
        self.hop_w0 = nn.Parameter(torch.ones(17, 17))
        self.hop_w1 = nn.Parameter(torch.ones(17, 17))
        self.hop_w2 = nn.Parameter(torch.ones(17, 17))
        self.hop_w3 = nn.Parameter(torch.ones(17, 17))
        self.hop_w4 = nn.Parameter(torch.ones(17, 17))
        self.hop_global = nn.Parameter(torch.ones(17, 17))
        
        self.linear_hop = nn.Linear(8, 2)
        self.edge_embedding = nn.Linear(17*17*4, 17*17)
        
    def forward(self, x, hops):
        b, f, v, j, c = x.shape
        
        edge_embedding = self.edge_embedding(hops[0].reshape(1, -1))
        
        # Global feature
        x_hop_global = x.unsqueeze(3).repeat(1, 1, 1, 17, 1, 1)
        x_hop_global = x_hop_global - x_hop_global.permute(0, 1, 2, 4, 3, 5)
        x_hop_global = torch.sum(x_hop_global ** 2, dim=-1)
        hop_global = x_hop_global / torch.sum(x_hop_global, dim=-1).unsqueeze(-1)
        hops = hops.unsqueeze(1).unsqueeze(2).repeat(1, f, v, 1, 1, 1)
        hops1 = hop_global * hops[:, :, :, 0]
        hops2 = hop_global * hops[:, :, :, 1]
        hops3 = hop_global * hops[:, :, :, 2]
        hops4 = hop_global * hops[:, :, :, 3]
        hops_cat = torch.cat((hops1, hops2, hops3, hops4), dim=-1)
        
        x1 = x[:, :, 0].permute(0, 3, 1, 2)
        x2 = x[:, :, 1].permute(0, 3, 1, 2)
        x3 = x[:, :, 2].permute(0, 3, 1, 2)
        x4 = x[:, :, 3].permute(0, 3, 1, 2)
        
        hop1 = hops_cat[:, :, 0].permute(0, 3, 1, 2)
        hop2 = hops_cat[:, :, 1].permute(0, 3, 1, 2)
        hop3 = hops_cat[:, :, 2].permute(0, 3, 1, 2)
        hop4 = hops_cat[:, :, 3].permute(0, 3, 1, 2)
        
        # Semantic graph transformer encoder (with optional HyperGraph)
        x1, hop1, MSA1, MSA2, MSA3, MSA4 = self.SF1(x1, hop1, edge_embedding)
        x2, hop2, MSA1, MSA2, MSA3, MSA4 = self.SF2(x2, hop2, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        x3, hop3, MSA1, MSA2, MSA3, MSA4 = self.SF3(x3, hop3, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        x4, hop4, MSA1, MSA2, MSA3, MSA4 = self.SF4(x4, hop4, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        
        # Multi-view cross-channel fusion
        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)), dim=1) + self.view_pos_embed
        x = self.pos_drop(x)
        x = self.conv(x).squeeze(1) + x1 + x2 + x3 + x4
        x = self.conv_norm(x)
        
        hop = torch.cat((hop1.unsqueeze(1), hop2.unsqueeze(1), hop3.unsqueeze(1), hop4.unsqueeze(1)), dim=1) + self.view_pos_embed
        hop = self.pos_drop(hop)
        hop = self.conv(hop).squeeze(1) + hop1 + hop2 + hop3 + hop4
        hop = self.conv_norm(hop)
        
        x = x * hop
        
        # Temporal transformer encoder
        x = self.TF(x)
        
        # Output head
        x = self.head(x)
        x = x.view(b, opt.frames, j, -1)
        
        return x
