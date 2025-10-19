## SGraAGFormer: 融合SGraFormer的多视角处理和MotionAGFormer的并行双流结构

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.layers import DropPath
import torch.nn.functional as F
from common.opt import opts

from model.modules.AGF import AGF_Attention, GCN
from model.Spatial_encoder import First_view_Spatial_features, Spatial_features
from model.Temporal_encoder import Temporal__features

opt = opts().parse()
device = torch.device("cuda")


class SGraAGFormer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0.2, attn_drop_rate=0.1, drop_path_rate=0.2, norm_layer=None):
        """
        Args:
            num_frame (int): 输入帧数量
            num_joints (int): 关节点数量
            in_chans (int): 输入通道数，2D关节点有2个通道：(x,y)
            embed_dim_ratio (int): 嵌入维度比例
            depth (int): Transformer深度
            num_heads (int): 注意力头数量
            mlp_ratio (int): MLP隐藏层维度与嵌入维度的比例
            qkv_bias (bool): 是否启用qkv的偏置
            qk_scale (float): qk缩放因子
            drop_rate (float): dropout率
            attn_drop_rate (float): 注意力dropout率
            drop_path_rate (float): 随机深度率
            norm_layer: 归一化层
        """
        super().__init__()

        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3  #### 输出维度是num_joints * 3
        
        ## Transformer流 - 使用SGraFormer的空间编码器
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

        ## GCN流 - 使用AGFormer的GCN结构
        self.gcn_embed = nn.Linear(in_chans, embed_dim_ratio)
        self.gcn_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.gcn_norm = nn.LayerNorm(embed_dim_ratio)
        
        # 创建4个视角的GCN流处理模块
        self.gcn_layers = nn.ModuleList([
            nn.Sequential(
                GCN(embed_dim_ratio, embed_dim_ratio, num_nodes=num_joints, mode='spatial'),
                GCN(embed_dim_ratio, embed_dim_ratio, num_nodes=num_frame, mode='temporal', 
                    use_temporal_similarity=False, temporal_connection_len=3)
            ) for _ in range(4)
        ])

        ## MVF - 多视角融合模块
        self.view_pos_embed = nn.Parameter(torch.zeros(1, 4, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        self.conv = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )

        # 双流融合权重
        self.fusion_weight = nn.Parameter(torch.ones(2), requires_grad=True)
        
        # Time Serial - 时间编码器
        self.TF = Temporal__features(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                     num_heads, mlp_ratio, qkv_bias, qk_scale,
                                     drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        # SGraFormer中的hop相关参数
        self.hop_w0 = nn.Parameter(torch.ones(17, 17))
        self.hop_w1 = nn.Parameter(torch.ones(17, 17))
        self.hop_w2 = nn.Parameter(torch.ones(17, 17))
        self.hop_w3 = nn.Parameter(torch.ones(17, 17))
        self.hop_w4 = nn.Parameter(torch.ones(17, 17))

        self.hop_global = nn.Parameter(torch.ones(17, 17))

        self.linear_hop = nn.Linear(8, 2)
        self.edge_embedding = nn.Linear(17*17*4, 17*17)
        
        # 对比学习相关
        self.linear_contrastive = nn.Linear(17*32, 17*32)
        nn.init.kaiming_uniform_(self.linear_contrastive.weight, nonlinearity='relu')
        self.temperature = 0.5
        
        # 时间特征提取
        self.temporal_att = AGF_Attention(32, mode="temporal", mixer_type="attention")  # Transformer
        self.temporal_tcn = AGF_Attention(32, mode="temporal", mixer_type="ms-tcn")  # TCN
        self.temporal_embed = nn.Parameter(torch.randn(1, opt.frames, 1, 32))

    def forward(self, x, hops): # hop:[b, 4, 17, 17]
        b, f, v, j, c = x.shape # x:[b, 27, 4, 17, 2]

        edge_embedding = self.edge_embedding(hops[0].reshape(1, -1)) # [1, 289]

        ###############global feature#################
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

        # 提取每个视角的数据
        x_views = [x[:, :, i] for i in range(4)]  # [b, 27, 17, 2] for each view
        hop_views = [hops[:, :, i] for i in range(4)]  # [b, 27, 17, 17*4] for each view

        # Transformer流处理
        x_tf = []
        hop_tf = []
        
        # 第一个视角的处理
        x1_permuted = x_views[0].permute(0, 3, 1, 2)  # [b, 2, 27, 17]
        hop1_permuted = hop_views[0].permute(0, 3, 1, 2)  # [b, 68, 27, 17]
        x1_tf, hop1_tf, MSA1, MSA2, MSA3, MSA4 = self.SF1(x1_permuted, hop1_permuted, edge_embedding)
        x_tf.append(x1_tf)
        hop_tf.append(hop1_tf)
        
        # 其余视角的处理
        for i in range(1, 4):
            x_permuted = x_views[i].permute(0, 3, 1, 2)  # [b, 2, 27, 17]
            hop_permuted = hop_views[i].permute(0, 3, 1, 2)  # [b, 68, 27, 17]
            if i == 1:
                x_i_tf, hop_i_tf, MSA1, MSA2, MSA3, MSA4 = self.SF2(x_permuted, hop_permuted, MSA1, MSA2, MSA3, MSA4, edge_embedding)
            elif i == 2:
                x_i_tf, hop_i_tf, MSA1, MSA2, MSA3, MSA4 = self.SF3(x_permuted, hop_permuted, MSA1, MSA2, MSA3, MSA4, edge_embedding)
            else:
                x_i_tf, hop_i_tf, MSA1, MSA2, MSA3, MSA4 = self.SF4(x_permuted, hop_permuted, MSA1, MSA2, MSA3, MSA4, edge_embedding)
            x_tf.append(x_i_tf)
            hop_tf.append(hop_i_tf)

        # GCN流处理
        x_gcn = []
        for i in range(4):
            xv = x_views[i]  # [b, 27, 17, 2]
            # 线性投影
            xv_embedded = self.gcn_embed(xv)  # [b, 27, 17, 32]
            xv_embedded = xv_embedded + self.gcn_pos_embed.unsqueeze(1)  # 添加位置编码
            xv_embedded = self.gcn_norm(xv_embedded)
            
            # 通过GCN层处理
            for gcn_layer in self.gcn_layers[i]:
                if hasattr(gcn_layer, 'mode') and gcn_layer.mode == 'temporal':
                    # 确保neighbour_num不会超过实际帧数
                    current_frames = xv_embedded.shape[1]
                    if hasattr(gcn_layer, 'neighbour_num'):
                        # 临时保存原始neighbour_num
                        original_neighbour_num = gcn_layer.neighbour_num
                        # 设置一个安全的neighbour_num值
                        gcn_layer.neighbour_num = min(original_neighbour_num, current_frames - 1)
                        xv_embedded = gcn_layer(xv_embedded)
                        # 恢复原始值
                        gcn_layer.neighbour_num = original_neighbour_num
                    else:
                        xv_embedded = gcn_layer(xv_embedded)
                else:
                    xv_embedded = gcn_layer(xv_embedded)
            
            # 重塑为与Transformer流相同的形状 [b, 27, 544]
            xv_reshaped = xv_embedded.reshape(b, f, -1)
            x_gcn.append(xv_reshaped)

        # 对比学习损失计算（使用Transformer流的输出）
        x1_c = self.linear_contrastive(x_tf[0])
        x2_c = self.linear_contrastive(x_tf[1])
        x3_c = self.linear_contrastive(x_tf[2])
        x4_c = self.linear_contrastive(x_tf[3])
        
        x1_c = F.normalize(x1_c, p=2, dim=-1)
        x2_c = F.normalize(x2_c, p=2, dim=-1)
        x3_c = F.normalize(x3_c, p=2, dim=-1)
        x4_c = F.normalize(x4_c, p=2, dim=-1)
        
        # 正样本对（相同帧，不同视角）
        positive_pairs_1 = torch.sum(x1_c * x2_c, dim=-1)  # x1 with x2  [b, 27]
        positive_pairs_2 = torch.sum(x1_c * x3_c, dim=-1)  # x1 with x3
        positive_pairs_3 = torch.sum(x1_c * x4_c, dim=-1)  # x1 with x4
        
        positive_pairs_1 = torch.sigmoid(positive_pairs_1)  # [b, 27]
        positive_pairs_2 = torch.sigmoid(positive_pairs_2)  # [b, 27]
        positive_pairs_3 = torch.sigmoid(positive_pairs_3)
        
        # 对比损失 (InfoNCE)
        positive_pairs = torch.cat((positive_pairs_1, positive_pairs_2, positive_pairs_3), dim=-1) # [b, 81]
        positive_pairs = torch.exp(positive_pairs / self.temperature)
        positive_pairs = positive_pairs / positive_pairs.sum(dim=-1, keepdim=True)
        loss_contrastive = -torch.log(positive_pairs)

        # 双流融合：将Transformer流和GCN流的特征进行融合
        x_fused_views = []
        for i in range(4):
            # 加权融合
            weights = F.softmax(self.fusion_weight, dim=0)
            x_fused = weights[0] * x_tf[i] + weights[1] * x_gcn[i]
            x_fused_views.append(x_fused)

        ### 多视角跨通道融合 (使用SGraFormer的多视角融合模块)
        x = torch.cat([xf.unsqueeze(1) for xf in x_fused_views], dim=1) + self.view_pos_embed # [b, 4, 27, 544]
        x = self.pos_drop(x) # [b, 4, 27, 544]
        
        # 融合所有视角
        x = self.conv(x).squeeze(1)
        for i in range(4):
            x = x + x_fused_views[i]
        x = x / (4 + 1)  # 平均
        
        # Temporal transformer encoder
        x = x.view(b, f, j, -1)  # [b, 27, 17, 32]
        x_att = self.temporal_att(x)
        x_tcn = self.temporal_tcn(x)
        x = x + x_att + x_tcn + self.temporal_embed 
        x = x.reshape(b, f, -1)
        
        x = self.TF(x) # [b, 27, 544]
        
        x = self.head(x) # [b, 27, 51]
        x = x.view(b, opt.frames, j, -1) # [b, 27, 17, 3]
        return x, loss_contrastive


if __name__ == '__main__':
    # 简单的测试代码
    b, f, v, j, c = 2, 27, 4, 17, 2
    x = torch.randn(b, f, v, j, c).to(device)
    hops = torch.randn(b, 4, j, j).to(device)
    
    model = SGraAGFormer(num_frame=f, num_joints=j, in_chans=c).to(device)
    out, loss = model(x, hops)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Contrastive loss shape: {loss.shape}")