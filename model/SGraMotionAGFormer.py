## SGraMotionAGFormer: 结合SGraFormer和MotionAGFormer的3D人体姿态估计模型

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.layers import DropPath
import torch.nn.functional as F
from collections import OrderedDict

from common.opt import opts
from model.modules.AGF import AGF_Attention, MultiScaleTCN
from model.modules.graph import GCN
from model.Spatial_encoder import First_view_Spatial_features, Spatial_features
from model.Temporal_encoder import Temporal__features

opt = opts().parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCNFormerBranch(nn.Module):
    """
    GCNFormer分支，用于每个视角的空间特征提取
    使用GCN进行空间特征提取
    """
    def __init__(self, dim, num_joints=17, num_heads=8, mlp_ratio=4., act_layer=nn.GELU, 
                 attn_drop=0., drop=0., drop_path=0., use_layer_scale=True, 
                 layer_scale_init_value=1e-5, n_frames=27):
        super().__init__()
        
        # 使用GCN进行空间特征提取
        # 确保mode='spatial'，这样GCN会在初始化时创建adj属性
        self.spatial_gcn = GCN(dim_in=dim, dim_out=dim, num_nodes=num_joints, mode='spatial')
        
        # 时间特征提取
        self.temporal_tcn = MultiScaleTCN(in_channels=dim, out_channels=dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # Layer norm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # Layer scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
    
    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        residual = x
        x = self.norm1(x)
        # 获取输入张量的维度
        b, t, j, c = x.shape
        # 直接应用GCN进行空间特征提取
        # GCN期望输入格式为[B, T, J, C]
        x = self.spatial_gcn(x)  # [B, T, J, C]
        
        # 应用时间TCN处理空间特征
        temporal_outputs = []
        for i in range(j):
            joint_x = x[:, :, i, :]  # [B, T, C]
            # 为joint_x添加一个维度以匹配MultiScaleTCN的输入要求 [B, T, J, C]
            joint_x_expanded = joint_x.unsqueeze(2)  # [B, T, 1, C]
            temporal_out = self.temporal_tcn(joint_x_expanded)
            # 移除添加的维度
            temporal_out = temporal_out.squeeze(2)
            temporal_outputs.append(temporal_out.unsqueeze(2))
        x = torch.cat(temporal_outputs, dim=2)  # [B, T, J, C]
        
        # 应用层缩放和残差连接
        if self.use_layer_scale:
            x = residual + self.drop_path(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x)
        else:
            x = residual + self.drop_path(x)
        
        # MLP部分
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        if self.use_layer_scale:
            x = residual + self.drop_path(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * x)
        else:
            x = residual + self.drop_path(x)
        
        return x


class SGraMotionAGFormer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3  #### output dimension is num_joints * 3
        
        ## Spatial_features (Transformer分支)
        self.SF1 = First_view_Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                               num_heads, mlp_ratio, qkv_bias, qk_scale,
                                               drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF2 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF3 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF4 = Spatial_features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                    num_heads, mlp_ratio, qkv_bias, qk_scale,
                                    drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        
        ## GCNFormer分支 (每个视角对应一个)
        self.gcn_branch1 = GCNFormerBranch(embed_dim_ratio, num_joints=num_joints, num_heads=num_heads, mlp_ratio=mlp_ratio, act_layer=nn.GELU,
                                         attn_drop=attn_drop_rate, drop=drop_rate, drop_path=drop_path_rate, n_frames=num_frame)
        self.gcn_branch2 = GCNFormerBranch(embed_dim_ratio, num_joints=num_joints, num_heads=num_heads, mlp_ratio=mlp_ratio, act_layer=nn.GELU,
                                         attn_drop=attn_drop_rate, drop=drop_rate, drop_path=drop_path_rate, n_frames=num_frame)
        self.gcn_branch3 = GCNFormerBranch(embed_dim_ratio, num_joints=num_joints, num_heads=num_heads, mlp_ratio=mlp_ratio, act_layer=nn.GELU,
                                         attn_drop=attn_drop_rate, drop=drop_rate, drop_path=drop_path_rate, n_frames=num_frame)
        self.gcn_branch4 = GCNFormerBranch(embed_dim_ratio, num_joints=num_joints, num_heads=num_heads, mlp_ratio=mlp_ratio, act_layer=nn.GELU,
                                         attn_drop=attn_drop_rate, drop=drop_rate, drop_path=drop_path_rate, n_frames=num_frame)
        
        # 添加GCN输出融合层
        self.gcn_fusion_layer = nn.Linear(embed_dim_ratio, embed_dim_ratio)
        
        # 嵌入层，用于GCN分支的输入
        self.joints_embed_gcn = nn.Linear(in_chans, embed_dim_ratio)
        
        ## MVF (多视角融合)
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

        # Time Serial
        self.TF = Temporal__features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                      num_heads, mlp_ratio, qkv_bias, qk_scale,
                                      drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        self.hop_w0 = nn.Parameter(torch.ones(17, 17))
        self.hop_w1 = nn.Parameter(torch.ones(17, 17))
        self.hop_w2 = nn.Parameter(torch.ones(17, 17))
        self.hop_w3 = nn.Parameter(torch.ones(17, 17))
        self.hop_w4 = nn.Parameter(torch.ones(17, 17))

        self.hop_global = nn.Parameter(torch.ones(17, 17))

        self.linear_hop = nn.Linear(8, 2)
        
        self.edge_embedding = nn.Linear(17*17*4, 17*17)
        
        self.linear_att = nn.Linear(2, 64)
        nn.init.kaiming_uniform_(self.linear_att.weight, nonlinearity='relu')
        self.agf_att = AGF_Attention(64)
        
        self.linear_contrastive = nn.Linear(17*32, 17*32)
        nn.init.kaiming_uniform_(self.linear_contrastive.weight, nonlinearity='relu')
        self.temperature = 0.5
        
        self.linear_att2 = nn.Linear(2, 64)
        nn.init.kaiming_uniform_(self.linear_att2.weight, nonlinearity='relu')
        
        self.temporal_att = AGF_Attention(32, mode="temporal", mixer_type="attention")  # Transformer
        self.temporal_tcn = AGF_Attention(32, mode="temporal", mixer_type="ms-tcn")  # TCN
        self.temporal_embed = nn.Parameter(torch.randn(1, opt.frames, 1, 32))

    def forward(self, x, hops): # hop:[b, 4, 17, 17]
        b, f, v, j, c = x.shape # x:[b, 27, 4, 17, 2]

        edge_embedding = self.edge_embedding(hops[0].reshape(1, -1)) # [1, 289]

        ###############golbal feature#################
        x_hop_global = x.unsqueeze(3).repeat(1, 1, 1, 17, 1, 1) # [b, 27, 4, 17, 17, 2]
        x_hop_global = x_hop_global - x_hop_global.permute(0, 1, 2, 4, 3, 5) # [b, 27, 4, 17, 17, 2]
        x_hop_global = torch.sum(x_hop_global ** 2, dim=-1) # [b, 27, 4, 17, 17]
        hop_global = x_hop_global / torch.sum(x_hop_global, dim=-1).unsqueeze(-1) # [b, 27, 4, 17, 17]
        hops = hops.unsqueeze(1).unsqueeze(2).repeat(1, f, v, 1, 1, 1) # [b, 27, 4, 4, 17, 17]
        hops1 = hop_global * hops[:, :, :, 0] # [b, 27, 4, 17, 17]
        hops2 = hop_global * hops[:, :, :, 1]
        hops3 = hop_global * hops[:, :, :, 2]
        hops4 = hop_global * hops[:, :, :, 3]
        hops = torch.cat((hops1,hops2,hops3,hops4), dim=-1) # [b, 27, 4, 17, 17*4]

        x1 = x[:, :, 0] # [b, 27, 17, 2]
        x2 = x[:, :, 1]
        x3 = x[:, :, 2]
        x4 = x[:, :, 3]
        
        x1_t = x1.permute(0, 3, 1, 2) # [b, f, 17, 2] -> [b, 2, f, 17]
        x2_t = x2.permute(0, 3, 1, 2)
        x3_t = x3.permute(0, 3, 1, 2)
        x4_t = x4.permute(0, 3, 1, 2)

        # 准备GCN分支的输入
        x1_gcn = self.joints_embed_gcn(x1)  # [b, 27, 17, 32]
        x2_gcn = self.joints_embed_gcn(x2)  # [b, 27, 17, 32]
        x3_gcn = self.joints_embed_gcn(x3)  # [b, 27, 17, 32]
        x4_gcn = self.joints_embed_gcn(x4)  # [b, 27, 17, 32]

        # GCN分支处理 - 确保不再传递graph参数
        x1_gcn = self.gcn_branch1(x1_gcn)  # [b, 27, 17, 32]
        x2_gcn = self.gcn_branch2(x2_gcn)  # [b, 27, 17, 32]
        x3_gcn = self.gcn_branch3(x3_gcn)  # [b, 27, 17, 32]
        x4_gcn = self.gcn_branch4(x4_gcn)  # [b, 27, 17, 32]
        
        hop1 = hops[:, :, 0] # [b, 27, 17, 17*4]
        hop2 = hops[:, :, 1]
        hop3 = hops[:, :, 2]
        hop4 = hops[:, :, 3]

        hop1 = hop1.permute(0, 3, 1, 2) # [b, 68, 27, 17]
        hop2 = hop2.permute(0, 3, 1, 2)
        hop3 = hop3.permute(0, 3, 1, 2)
        hop4 = hop4.permute(0, 3, 1, 2)

        ### Semantic graph transformer encoder (Transformer分支)
        x1_t, hop1, MSA1, MSA2, MSA3, MSA4 = self.SF1(x1_t, hop1, edge_embedding)
        x2_t, hop2, MSA1, MSA2, MSA3, MSA4 = self.SF2(x2_t, hop2, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        x3_t, hop3, MSA1, MSA2, MSA3, MSA4 = self.SF3(x3_t, hop3, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        x4_t, hop4, MSA1, MSA2, MSA3, MSA4 = self.SF4(x4_t, hop4, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        
        # 融合Transformer和GCN分支的输出
        # 需要将Transformer分支的输出重塑为 [b, 27, 17, 32] 以便进行特征融合
        x1_t_reshaped = x1_t.reshape(b, f, j, -1)  # [b, 27, 17, 32]
        x2_t_reshaped = x2_t.reshape(b, f, j, -1)  # [b, 27, 17, 32]
        x3_t_reshaped = x3_t.reshape(b, f, j, -1)  # [b, 27, 17, 32]
        x4_t_reshaped = x4_t.reshape(b, f, j, -1)  # [b, 27, 17, 32]
        
        # 特征融合：使用残差连接代替简单拼接
        # 先应用GCN特征的融合层
        b, f, j, c = x1_gcn.shape
        x1_gcn = x1_gcn.permute(0, 3, 1, 2).contiguous().view(b, c, f, j)
        x1_gcn = nn.Conv2d(c, c, kernel_size=1).to(x1_gcn.device)(x1_gcn)
        x1_gcn = x1_gcn.view(b, -1, f, j).permute(0, 2, 3, 1).contiguous()
        
        x2_gcn = x2_gcn.permute(0, 3, 1, 2).contiguous().view(b, c, f, j)
        x2_gcn = nn.Conv2d(c, c, kernel_size=1).to(x2_gcn.device)(x2_gcn)
        x2_gcn = x2_gcn.view(b, -1, f, j).permute(0, 2, 3, 1).contiguous()
        
        x3_gcn = x3_gcn.permute(0, 3, 1, 2).contiguous().view(b, c, f, j)
        x3_gcn = nn.Conv2d(c, c, kernel_size=1).to(x3_gcn.device)(x3_gcn)
        x3_gcn = x3_gcn.view(b, -1, f, j).permute(0, 2, 3, 1).contiguous()
        
        x4_gcn = x4_gcn.permute(0, 3, 1, 2).contiguous().view(b, c, f, j)
        x4_gcn = nn.Conv2d(c, c, kernel_size=1).to(x4_gcn.device)(x4_gcn)
        x4_gcn = x4_gcn.view(b, -1, f, j).permute(0, 2, 3, 1).contiguous()
        
        # 融合双流特征
        x1_fused = x1_t_reshaped + x1_gcn  # 使用残差连接
        x2_fused = x2_t_reshaped + x2_gcn
        x3_fused = x3_t_reshaped + x3_gcn
        x4_fused = x4_t_reshaped + x4_gcn
        
        # 重塑回 [b, 27, 544] 以便后续处理
        x1 = x1_fused.reshape(b, f, -1)  # [b, 27, 544]
        x2 = x2_fused.reshape(b, f, -1)  # [b, 27, 544]
        x3 = x3_fused.reshape(b, f, -1)  # [b, 27, 544]
        x4 = x4_fused.reshape(b, f, -1)  # [b, 27, 544]
        
        x1_c = self.linear_contrastive(x1) # 
        x2_c = self.linear_contrastive(x2)
        x3_c = self.linear_contrastive(x3)
        x4_c = self.linear_contrastive(x4)
        
        x1_c = F.normalize(x1_c, p=2, dim=-1)
        x2_c = F.normalize(x2_c, p=2, dim=-1)
        x3_c = F.normalize(x3_c, p=2, dim=-1)
        x4_c = F.normalize(x4_c, p=2, dim=-1)
        # Positive pairs (same frame, different views)
        positive_pairs_1 = torch.sum(x1_c * x2_c, dim=-1)  # x1 with x2  [b, 27]
        positive_pairs_2 = torch.sum(x1_c * x3_c, dim=-1)  # x1 with x3
        positive_pairs_3 = torch.sum(x1_c * x4_c, dim=-1)  # x1 with x4
        
        positive_pairs_1 = torch.sigmoid(positive_pairs_1)  # [b, 27]
        positive_pairs_2 = torch.sigmoid(positive_pairs_2)  # [b, 27]
        positive_pairs_3 = torch.sigmoid(positive_pairs_3)
        
        # contrastive loss (InfoNCE)
        positive_pairs = torch.cat((positive_pairs_1, positive_pairs_2, positive_pairs_3), dim=-1) # [b, 81]
        positive_pairs = torch.exp(positive_pairs / self.temperature)
        positive_pairs = positive_pairs / positive_pairs.sum(dim=-1, keepdim=True)
        loss_contrastive = -torch.log(positive_pairs)

        ### Multi-view cross-channel fusion
        x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)), dim=1) + self.view_pos_embed # [b, 4, 27, 544]
        x = self.pos_drop(x) # [b, 4, 27, 544]
        x = self.conv(x).squeeze(1) + x1 + x2 + x3 + x4 # [b, 27, 544]
        x = self.conv_norm(x) # [b, 27, 544]

        hop = torch.cat((hop1.unsqueeze(1), hop2.unsqueeze(1), hop3.unsqueeze(1), hop4.unsqueeze(1)), dim=1) + self.view_pos_embed # [b, 4, 27, 544]
        hop = self.pos_drop(hop) # [b, 4, 27, 544]
        hop = self.conv(hop).squeeze(1) + hop1 + hop2 + hop3 + hop4 # [b, 27, 544]
        hop = self.conv_norm(hop) # [b, 27, 544]

        x = x * hop # [b, 27, 544]

        ### Temporal transformer encoder
        x = x.view(b, f, j, -1)  # [b, 27, 17, 32]
        x_att = self.temporal_att(x)
        x_tcn = self.temporal_tcn(x)
        x = x + x_att + x_tcn + self.temporal_embed 
        x = x.reshape(b, f, -1)
        
        x = self.TF(x) # [b, 27, 544]
        
        x = self.head(x) # [b, 27, 51]
        x = x.view(b, opt.frames, j, -1) # [b, 27, 17, 3]
        return x, loss_contrastive

