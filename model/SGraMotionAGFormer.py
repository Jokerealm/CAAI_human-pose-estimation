from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.layers import DropPath
import torch.nn.functional as F

from common.opt import opts
from model.AGF import AGF_Attention, MultiScaleTCN
from model.modules.attention import Attention
from model.modules.graph import GCN
from model.modules.mlp import MLP

opt = opts().parse()


class AGFormerBlock(nn.Module):
    """
    Implementation of AGFormer block.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, qkv_bias=False, qk_scale=None, use_layer_scale=True, layer_scale_init_value=1e-5,
                 mode='spatial', mixer_type="attention", use_temporal_similarity=True,
                 temporal_connection_len=1, neighbour_num=4, n_frames=243):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        if mixer_type == 'attention':
            self.mixer = Attention(dim, dim, num_heads, qkv_bias, qk_scale, attn_drop,
                                   proj_drop=drop, mode=mode)
        elif mixer_type == 'graph':
            self.mixer = GCN(dim, dim,
                             num_nodes=17 if mode == 'spatial' else n_frames,
                             neighbour_num=neighbour_num,
                             mode=mode,
                             use_temporal_similarity=use_temporal_similarity,
                             temporal_connection_len=temporal_connection_len)
        elif mixer_type == "ms-tcn":
            self.mixer = MultiScaleTCN(in_channels=dim, out_channels=dim)
        else:
            raise NotImplementedError("AGFormer mixer_type is either attention or graph")
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep GraphFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MotionAGFormerBlock(nn.Module):
    """
    Implementation of MotionAGFormer block. It has two ST and TS branches followed by adaptive fusion.
    """

    def __init__(self, dim, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop=0., drop_path=0.,
                 num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                 use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                 temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243):
        super().__init__()
        self.hierarchical = hierarchical
        dim = dim // 2 if hierarchical else dim

        # ST Attention branch
        self.att_spatial = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                         qk_scale, use_layer_scale, layer_scale_init_value,
                                         mode='spatial', mixer_type="attention",
                                         use_temporal_similarity=use_temporal_similarity,
                                         neighbour_num=neighbour_num,
                                         n_frames=n_frames)
        self.att_temporal = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads, qkv_bias,
                                          qk_scale, use_layer_scale, layer_scale_init_value,
                                          mode='temporal', mixer_type="attention",
                                          use_temporal_similarity=use_temporal_similarity,
                                          neighbour_num=neighbour_num,
                                          n_frames=n_frames)

        # ST Graph branch
        if graph_only:
            self.graph_spatial = GCN(dim, dim,
                                     num_nodes=17,
                                     mode='spatial')
            if use_tcn:
                self.graph_temporal = MultiScaleTCN(in_channels=dim, out_channels=dim)
            else:
                self.graph_temporal = GCN(dim, dim,
                                          num_nodes=n_frames,
                                          neighbour_num=neighbour_num,
                                          mode='temporal',
                                          use_temporal_similarity=use_temporal_similarity,
                                          temporal_connection_len=temporal_connection_len)
        else:
            self.graph_spatial = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                               qkv_bias,
                                               qk_scale, use_layer_scale, layer_scale_init_value,
                                               mode='spatial', mixer_type="graph",
                                               use_temporal_similarity=use_temporal_similarity,
                                               temporal_connection_len=temporal_connection_len,
                                               neighbour_num=neighbour_num,
                                               n_frames=n_frames)
            self.graph_temporal = AGFormerBlock(dim, mlp_ratio, act_layer, attn_drop, drop, drop_path, num_heads,
                                                qkv_bias,
                                                qk_scale, use_layer_scale, layer_scale_init_value,
                                                mode='temporal', mixer_type="ms-tcn" if use_tcn else 'graph',
                                                use_temporal_similarity=use_temporal_similarity,
                                                temporal_connection_len=temporal_connection_len,
                                                neighbour_num=neighbour_num,
                                                n_frames=n_frames)

        self.use_adaptive_fusion = use_adaptive_fusion
        if self.use_adaptive_fusion:
            self.fusion = nn.Linear(dim * 2, 2)
            self._init_fusion()

    def _init_fusion(self):
        self.fusion.weight.data.fill_(0)
        self.fusion.bias.data.fill_(0.5)

    def forward(self, x):
        """
        x: tensor with shape [B, T, J, C]
        """
        if self.hierarchical:
            B, T, J, C = x.shape
            x_attn, x_graph = x[..., :C // 2], x[..., C // 2:]

            x_attn = self.att_temporal(self.att_spatial(x_attn))
            x_graph = self.graph_temporal(self.graph_spatial(x_graph + x_attn))
        else:
            x_attn = self.att_temporal(self.att_spatial(x))
            x_graph = self.graph_temporal(self.graph_spatial(x))

        if self.hierarchical:
            x = torch.cat((x_attn, x_graph), dim=-1)
        elif self.use_adaptive_fusion:
            alpha = torch.cat((x_attn, x_graph), dim=-1)
            alpha = self.fusion(alpha)
            alpha = alpha.softmax(dim=-1)
            x = x_attn * alpha[..., 0:1] + x_graph * alpha[..., 1:2]
        else:
            x = (x_attn + x_graph) * 0.5

        return x


class create_layers(nn.Module):
    """
    generates MotionAGFormer layers
    """
    def __init__(self, dim, n_layers, mlp_ratio=4., act_layer=nn.GELU, attn_drop=0., drop_rate=0., drop_path_rate=0.,
                  num_heads=8, use_layer_scale=True, qkv_bias=False, qk_scale=None, layer_scale_init_value=1e-5,
                  use_adaptive_fusion=True, hierarchical=False, use_temporal_similarity=True,
                  temporal_connection_len=1, use_tcn=False, graph_only=False, neighbour_num=4, n_frames=243):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(MotionAGFormerBlock(dim=dim,
                                              mlp_ratio=mlp_ratio,
                                              act_layer=act_layer,
                                              attn_drop=attn_drop,
                                              drop=drop_rate,
                                              drop_path=drop_path_rate,
                                              num_heads=num_heads,
                                              use_layer_scale=use_layer_scale,
                                              layer_scale_init_value=layer_scale_init_value,
                                              qkv_bias=qkv_bias,
                                              qk_scale=qk_scale,
                                              use_adaptive_fusion=use_adaptive_fusion,
                                              hierarchical=hierarchical,
                                              use_temporal_similarity=use_temporal_similarity,
                                              temporal_connection_len=temporal_connection_len,
                                              use_tcn=use_tcn,
                                              graph_only=graph_only,
                                              neighbour_num=neighbour_num,
                                              n_frames=n_frames))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class MotionSpatialEncoder(nn.Module):
    """
    基于MotionAGFormerBlock的空间编码器，替代原SGraFormer中的Spatial_features
    """
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio

        # 空间嵌入
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))

        # Hop图嵌入
        self.hop_to_embedding = nn.Linear(68, embed_dim)
        self.hop_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 使用MotionAGFormerBlock替代原有的Block
        self.layers = create_layers(
            dim=embed_dim,
            n_layers=depth,
            mlp_ratio=mlp_ratio,
            act_layer=nn.GELU,
            attn_drop=attn_drop_rate,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_heads=num_heads,
            use_layer_scale=True,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            use_adaptive_fusion=True,
            hierarchical=False,
            use_temporal_similarity=True,
            neighbour_num=4,
            n_frames=num_frame
        )

        self.Spatial_norm = norm_layer(embed_dim)
        self.hop_norm = norm_layer(embed_dim)

    def forward(self, x, hops):
        b, _, f, p = x.shape  # b是批次大小，f是帧数，p是关节数
        x_input = x
        x = rearrange(x, 'b c f p  -> (b f) p  c')

        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)

        # 为了适应MotionAGFormer的输入格式[B, T, J, C]，我们需要调整形状
        x = x.view(b, f, p, -1)

        # 应用MotionAGFormer的层
        x = self.layers(x)

        # 恢复原始形状并应用规范化
        x = rearrange(x, 'b f p c -> (b f) p c')
        x = self.Spatial_norm(x)
        x = rearrange(x, '(b f) w c -> b f (w c)', f=f)

        # 处理hop图
        hops = rearrange(hops, 'b c f p  -> (b f) p  c')
        hops = self.hop_to_embedding(hops)
        hops += self.hop_pos_embed
        hops = self.pos_drop(hops)
        hops = self.hop_norm(hops)
        hops = rearrange(hops, '(b f) w c -> b f (w c)', f=f)

        if x.size() == x_input.size():
            x = x + x_input
        return x, hops


class MotionTemporalEncoder(nn.Module):
    """
    基于MotionAGFormerBlock的时间编码器，替代原SGraFormer中的Temporal__features
    """
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints  # 时间嵌入维度是关节数*空间嵌入维度比

        # 时间位置编码
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # 使用MotionAGFormerBlock替代原有的Block
        self.layers = create_layers(
            dim=embed_dim_ratio,  # 修改为使用embed_dim_ratio而非embed_dim
            n_layers=depth,
            mlp_ratio=mlp_ratio,
            act_layer=nn.GELU,
            attn_drop=attn_drop_rate,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_heads=num_heads,
            use_layer_scale=True,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            use_adaptive_fusion=True,
            hierarchical=False,
            use_temporal_similarity=True,
            neighbour_num=4,
            n_frames=num_frame
        )

        self.Temporal_norm = norm_layer(embed_dim)

    def forward(self, x):
        b = x.shape[0]
        f = x.shape[1]
        j = 17  # 标准人体关节数
        c = x.shape[2] // j  # 计算每个关节的通道数
        x_input = x
        # 调整形状以适应MotionAGFormer的输入格式[B, T, J, C]
        x = x.view(b, f, j, c)
        x += self.Temporal_pos_embed.view(1, f, j, c)
        x = self.pos_drop(x)

        # 应用MotionAGFormer的层
        x = self.layers(x)

        # 恢复原始形状并应用规范化
        x = x.view(b, f, -1)
        x = self.Temporal_norm(x)
        if x.size() == x_input.size():
            x = x + x_input
        return x


class SGraMotionAGFormer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """
        SGraMotionAGFormer模型：结合SGraFormer的多视图框架和MotionAGFormer的先进特性
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3  # 输出维度是关节数*3

        # 使用基于MotionAGFormer的空间编码器替代原有的SF1-SF4
        self.SF1 = MotionSpatialEncoder(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                       num_heads, mlp_ratio, qkv_bias, qk_scale,
                                       drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF2 = MotionSpatialEncoder(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                       num_heads, mlp_ratio, qkv_bias, qk_scale,
                                       drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF3 = MotionSpatialEncoder(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                       num_heads, mlp_ratio, qkv_bias, qk_scale,
                                       drop_rate, attn_drop_rate, drop_path_rate, norm_layer)
        self.SF4 = MotionSpatialEncoder(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                       num_heads, mlp_ratio, qkv_bias, qk_scale,
                                       drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        # MVF部分：多视图融合，增强跨视图信息交互
        self.view_pos_embed = nn.Parameter(torch.zeros(1, 4, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        # 引入图注意力增强的多视图融合
        self.conv = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )

        self.conv_norm = nn.LayerNorm(embed_dim)

        # 时间序列处理：使用基于MotionAGFormer的时间编码器
        self.TF = MotionTemporalEncoder(num_frame, num_joints, in_chans, embed_dim_ratio, depth,
                                       num_heads, mlp_ratio, qkv_bias, qk_scale,
                                       drop_rate, attn_drop_rate, drop_path_rate, norm_layer)

        # 输出头
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

        # 额外的模块用于增强多视图融合
        self.linear_hop = nn.Linear(8, 2)
        self.edge_embedding = nn.Linear(17*17*4, 17*17)
        self.linear_att = nn.Linear(2, 64)
        nn.init.kaiming_uniform_(self.linear_att.weight, nonlinearity='relu')
        self.agf_att = AGF_Attention(64)
        
        # 对比学习组件
        self.linear_contrastive = nn.Linear(17*32, 17*32)
        nn.init.kaiming_uniform_(self.linear_contrastive.weight, nonlinearity='relu')
        self.temperature = 0.5

        # 时间特征融合模块
        self.temporal_att = AGF_Attention(32, mode="temporal", mixer_type="attention"
                                          use_temporal_similarity=True,
                                          temporal_connection_len=3,
                                          neighbour_num=6)  # Transformer
        self.temporal_tcn = AGF_Attention(32, mode="temporal", mixer_type="ms-tcn")  # TCN
        self.temporal_embed = nn.Parameter(torch.randn(1, opt.frames, 1, 32))

    def forward(self, x, hops):  # hop:[b, 4, 17, 17]
        b, f, v, j, c = x.shape  # x:[b, 27, 4, 17, 2]

        # 提取边缘特征
        edge_embedding = self.edge_embedding(hops[0].reshape(1, -1))  # [1, 289]

        ###############全局特征#################
        x_hop_global = x.unsqueeze(3).repeat(1, 1, 1, 17, 1, 1)  # [b, 27, 4, 17, 17, 2]
        x_hop_global = x_hop_global - x_hop_global.permute(0, 1, 2, 4, 3, 5)  # [b, 27, 4, 17, 17, 2]
        x_hop_global = torch.sum(x_hop_global ** 2, dim=-1)  # [b, 27, 4, 17, 17]
        hop_global = x_hop_global / torch.sum(x_hop_global, dim=-1).unsqueeze(-1)  # [b, 27, 4, 17, 17]
        hops = hops.unsqueeze(1).unsqueeze(2).repeat(1, f, v, 1, 1, 1)  # [b, 27, 4, 4, 17, 17]
        hops1 = hop_global * hops[:, :, :, 0]  # [b, 27, 4, 17, 17]
        hops2 = hop_global * hops[:, :, :, 1]
        hops3 = hop_global * hops[:, :, :, 2]
        hops4 = hop_global * hops[:, :, :, 3]
        hops = torch.cat((hops1,hops2,hops3,hops4), dim=-1)  # [b, 27, 4, 17, 17*4]

        # 处理每个视图的数据
        x1 = x[:, :, 0]  # [b, 27, 17, 2]
        x2 = x[:, :, 1]
        x3 = x[:, :, 2]
        x4 = x[:, :, 3]
        
        x1 = x1.permute(0, 3, 1, 2)  # [b, f, 17, 2] -> [b, 2, f, 17]
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x4 = x4.permute(0, 3, 1, 2)

        # 提取每个视图的hop特征
        hop1 = hops[:, :, 0]  # [b, 27, 17, 17*4]
        hop2 = hops[:, :, 1]
        hop3 = hops[:, :, 2]
        hop4 = hops[:, :, 3]

        hop1 = hop1.permute(0, 3, 1, 2)  # [b, 68, 27, 17]
        hop2 = hop2.permute(0, 3, 1, 2)
        hop3 = hop3.permute(0, 3, 1, 2)
        hop4 = hop4.permute(0, 3, 1, 2)

        # 使用基于MotionAGFormer的空间编码器处理每个视图
        x1, hop1 = self.SF1(x1, hop1)
        x2, hop2 = self.SF2(x2, hop2)
        x3, hop3 = self.SF3(x3, hop3)
        x4, hop4 = self.SF4(x4, hop4)
        
        # 对比学习
        x1_c = self.linear_contrastive(x1)
        x2_c = self.linear_contrastive(x2)
        x3_c = self.linear_contrastive(x3)
        x4_c = self.linear_contrastive(x4)
        
        x1_c = F.normalize(x1_c, p=2, dim=-1)
        x2_c = F.normalize(x2_c, p=2, dim=-1)
        x3_c = F.normalize(x3_c, p=2, dim=-1)
        x4_c = F.normalize(x4_c, p=2, dim=-1)
        
        # 正样本对（相同帧，不同视图）
        positive_pairs_1 = torch.sum(x1_c * x2_c, dim=-1)  # x1 with x2  [b, 27]
        positive_pairs_2 = torch.sum(x1_c * x3_c, dim=-1)  # x1 with x3
        positive_pairs_3 = torch.sum(x1_c * x4_c, dim=-1)  # x1 with x4
        
        positive_pairs_1 = torch.sigmoid(positive_pairs_1)  # [b, 27]
        positive_pairs_2 = torch.sigmoid(positive_pairs_2)  # [b, 27]
        positive_pairs_3 = torch.sigmoid(positive_pairs_3)
        
        # 对比损失 (InfoNCE)
        positive_pairs = torch.cat((positive_pairs_1, positive_pairs_2, positive_pairs_3), dim=-1)  # [b, 81]
        positive_pairs = torch.exp(positive_pairs / self.temperature)
        positive_pairs = positive_pairs / positive_pairs.sum(dim=-1, keepdim=True)
        loss_contrastive = -torch.log(positive_pairs + 1e-10).mean()

        # 多视图跨通道融合
        x_views = torch.stack([x1, x2, x3, x4], dim=1)  # [b, 4, f, c]
        # 计算注意力权重
        view_weights = F.softmax(self.view_weights, dim=0).view(1, 4, 1, 1)  # 视图权重
        x = x_views * view_weights  # 加权
        x = x.sum(dim=1)  # [b, f, c]

        # 保留卷积融合但调整方式
        conv_out = self.conv(x_views + self.view_pos_embed).squeeze(1)
        x = x + conv_out
        # x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)), dim=1) + self.view_pos_embed  # [b, 4, 27, 544]
        # x = self.pos_drop(x)  # [b, 4, 27, 544]
        # x = self.conv(x).squeeze(1) + x1 + x2 + x3 + x4  # [b, 27, 544]
        # x = self.conv_norm(x)  # [b, 27, 544]

        # Hop图特征融合
        hop = torch.cat((hop1.unsqueeze(1), hop2.unsqueeze(1), hop3.unsqueeze(1), hop4.unsqueeze(1)), dim=1) + self.view_pos_embed  # [b, 4, 27, 544]
        hop = self.pos_drop(hop)  # [b, 4, 27, 544]
        hop = self.conv(hop).squeeze(1) + hop1 + hop2 + hop3 + hop4  # [b, 27, 544]
        hop = self.conv_norm(hop)  # [b, 27, 544]

        # 融合空间特征和Hop图特征
        x = x * hop  # [b, 27, 544]

        # 时间特征处理
        # 在forward方法中修改这部分代码
        x = x.view(b, f, j, -1)  # [b, 27, 17, 32]
        # 确保数据在多GPU环境下的连续性
        if x.is_cuda and torch.cuda.device_count() > 1:
            x = x.contiguous()
        x_att = self.temporal_att(x)
        x = x + x_att + self.temporal_embed 
        x = x.reshape(b, f, -1)
        
        # 使用基于MotionAGFormer的时间编码器进一步处理
        x = self.TF(x)  # [b, 27, 544]
        
        # 输出3D姿态
        x = self.head(x)  # [b, 27, 51]
        x = x.view(b, opt.frames, j, -1)  # [b, 27, 17, 3]
        return x, loss_contrastive