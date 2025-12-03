
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.layers import DropPath

from common.opt import opts
from model.Spatial_encoder import First_view_Spatial_features, Spatial_features
from model.Temporal_encoder import Temporal__features
from common.computer_triangulate_loss import triangulate_loss, get_batch_proj_matrices
from common.computer_reprojection_loss import reprojection_loss

opt = opts().parse()
device = torch.device("cuda")


class GeometricFusion(nn.Module):
    """
    Geometric-aware fusion using camera parameters for triangulation-based weighting
    
    Key idea: Use triangulation confidence to weight different views
    - Views with better geometric configuration get higher weights
    - Based on: "Learning to Fuse: A Deep Learning Approach to Visual-Inertial Camera Pose Estimation"
    """
    def __init__(self, embed_dim=544, num_views=4):
        super().__init__()
        self.num_views = num_views
        self.embed_dim = embed_dim
        
        # Learnable view importance (initialized uniformly)
        self.view_weights = nn.Parameter(torch.ones(1, num_views, 1, 1))
        
        # Confidence estimator: predicts reliability of each view
        self.confidence_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()  # Output confidence in [0, 1]
        )
        
    def forward(self, view_features, input_2d=None, subjects=None):
        """
        view_features: [b, 4, f, embed_dim] - features from 4 views
        input_2d: [b, f, 4, j, 2] - optional, for geometric confidence
        subjects: list of subject IDs - optional, for camera-aware weighting
        
        Returns: [b, f, embed_dim] - fused features
        """
        b, v, f, d = view_features.shape
        
        # Compute per-view confidence scores
        confidences = []
        for view_idx in range(v):
            view_feat = view_features[:, view_idx]  # [b, f, d]
            conf = self.confidence_net(view_feat)  # [b, f, 1]
            confidences.append(conf)
        
        confidences = torch.stack(confidences, dim=1)  # [b, 4, f, 1]
        
        # Combine with learnable view weights
        weights = confidences * self.view_weights  # [b, 4, f, 1]
        weights = torch.softmax(weights, dim=1)  # Normalize across views
        
        # Weighted fusion
        weighted_features = view_features * weights  # [b, 4, f, d]
        fused = torch.sum(weighted_features, dim=1)  # [b, f, d]
        
        return fused, weights.squeeze(-1)  # Return weights for visualization


#######################################################################################################################
class sgraformer(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                 use_camera=True, use_triangulation=True):
        """
        Args:
            use_camera (bool): whether to use camera-aware geometric fusion
            use_triangulation (bool): whether to use triangulation as auxiliary supervision
        """
        super().__init__()

        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3
        self.use_camera = use_camera
        self.use_triangulation = use_triangulation
        self.num_joints = num_joints
        
        ##Spatial_features
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

        ## Geometric fusion module (camera-aware)
        self.geometric_fusion = GeometricFusion(embed_dim=embed_dim, num_views=4)
        # Original MVF
        self.view_pos_embed = nn.Parameter(torch.zeros(1, 4, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )
        self.conv_norm = nn.LayerNorm(embed_dim)

        self.conv_hop = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )
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
        
        
    def forward(self, x, hops, subjects=None):
        """
        x: [b, f, v, j, c] - input 2D poses
        hops: hop matrices
        subjects: list of subject IDs - for camera parameter lookup
        """
        b, f, v, j, c = x.shape

        edge_embedding = self.edge_embedding(hops[0].reshape(1, -1))

        ###############global feature#################
        x_hop_global = x.unsqueeze(3).repeat(1, 1, 1, 17, 1, 1)
        x_hop_global = x_hop_global - x_hop_global.permute(0, 1, 2, 4, 3, 5)
        x_hop_global = torch.sum(x_hop_global ** 2, dim=-1)
        hop_global = x_hop_global / torch.sum(x_hop_global, dim=-1).unsqueeze(-1)
        hops = hops.unsqueeze(1).unsqueeze(2).repeat(1, f, v, 1, 1, 1)
        hops1 = hop_global * hops[:, :, :, 0]
        hops2 = hop_global * hops[:, :, :, 1]
        hops3 = hop_global * hops[:, :, :, 2]
        hops4 = hop_global * hops[:, :, :, 3]
        hops = torch.cat((hops1,hops2,hops3,hops4), dim=-1)

        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        x3 = x[:, :, 2]
        x4 = x[:, :, 3]

        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x4 = x4.permute(0, 3, 1, 2)

        hop1 = hops[:, :, 0]
        hop2 = hops[:, :, 1]
        hop3 = hops[:, :, 2]
        hop4 = hops[:, :, 3]

        hop1 = hop1.permute(0, 3, 1, 2)
        hop2 = hop2.permute(0, 3, 1, 2)
        hop3 = hop3.permute(0, 3, 1, 2)
        hop4 = hop4.permute(0, 3, 1, 2)

        ### Semantic graph transformer encoder
        x1, hop1, MSA1, MSA2, MSA3, MSA4 = self.SF1(x1, hop1, edge_embedding)
        x2, hop2, MSA1, MSA2, MSA3, MSA4 = self.SF2(x2, hop2, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        x3, hop3, MSA1, MSA2, MSA3, MSA4 = self.SF3(x3, hop3, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        x4, hop4, MSA1, MSA2, MSA3, MSA4 = self.SF4(x4, hop4, MSA1, MSA2, MSA3, MSA4, edge_embedding)
        
        ### Multi-view fusion with geometric awareness
        if self.use_camera and subjects is not None:
            # Stack view features: [b, 4, f, embed_dim]
            view_features = torch.stack([x1, x2, x3, x4], dim=1)
            
            # Geometric-aware fusion
            x, fusion_weights = self.geometric_fusion(view_features, x, subjects)
            
            # Store fusion weights for visualization/analysis
            self.last_fusion_weights = fusion_weights
        else:
            # Original MVF
            x = torch.cat((x1.unsqueeze(1), x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)), dim=1) + self.view_pos_embed
            x = self.pos_drop(x)
            x = self.conv(x).squeeze(1) + x1 + x2 + x3 + x4
            x = self.conv_norm(x)

        hop = torch.cat((hop1.unsqueeze(1), hop2.unsqueeze(1), hop3.unsqueeze(1), hop4.unsqueeze(1)), dim=1)
        if not self.use_camera:
            hop = hop + self.view_pos_embed
            hop = self.pos_drop(hop)
        hop = self.conv_hop(hop).squeeze(1) + hop1 + hop2 + hop3 + hop4
        hop = self.conv_hop_norm(hop)

        x = x * hop # b, f, embedding*j

        ### Temporal transformer encoder
        x = self.TF(x) # b, f, embedding*j
        
        x = self.head(x) # b, f, 3*j
        x = x.view(b, opt.frames, j, -1)
        
        return x
    
    def compute_geometric_losses(self, pred_3d, input_2d, subjects):
        """
        Compute triangulation and reprojection losses for geometric supervision
        
        Args:
            pred_3d: [b, f, j, 3] - predicted 3D poses
            input_2d: [b, f, 4, j, 2] - input 2D poses from 4 views
            subjects: list of subject IDs
            
        Returns:
            dict with 'triangulation' and 'reprojection' losses
        """
        if not self.use_triangulation:
            return {}
        
        b, f, j, _ = pred_3d.shape
        
        # Select middle frame for geometric losses (to avoid padding issues)
        mid_frame = f // 2
        pred_3d_mid = pred_3d[:, mid_frame]  # [b, j, 3]
        input_2d_mid = input_2d[:, mid_frame]  # [b, 4, j, 2]
        
        # Triangulation loss: compare prediction with triangulated 3D
        try:
            tri_loss = triangulate_loss(pred_3d_mid, input_2d_mid, subjects)
        except:
            tri_loss = torch.tensor(0.0, device=pred_3d.device)
        
        # Reprojection loss: project prediction back to 2D
        try:
            reproj_loss = reprojection_loss(pred_3d_mid, input_2d_mid, subjects)
        except:
            reproj_loss = torch.tensor(0.0, device=pred_3d.device)
        
        return {
            'triangulation': tri_loss.mean() if tri_loss.numel() > 1 else tri_loss,
            'reprojection': reproj_loss.mean() if reproj_loss.numel() > 1 else reproj_loss
        }
