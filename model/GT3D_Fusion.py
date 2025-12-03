"""
GT 3D Fusion Module for Ablation Study

This module provides different fusion strategies to inject GT 3D pose
into the model at various positions.

Fusion Positions:
1. after_spatial: After spatial encoders, before MVF
2. after_mvf: After MVF, before temporal (RECOMMENDED)
3. after_temporal: After temporal, before head
4. in_temporal: Inside temporal blocks

Fusion Methods:
- residual: Simple weighted addition
- gating: Learnable gating mechanism
- cross_attention: Cross-attention fusion
"""

import torch
import torch.nn as nn
from einops import rearrange


class GT3DFusionModule(nn.Module):
    """Flexible GT 3D fusion module"""
    
    def __init__(self, embed_dim, num_joints=17, fusion_method='gating'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.fusion_method = fusion_method
        
        # Project GT 3D (J*3) to embed_dim
        # Use 2-layer MLP with LayerNorm to ensure proper scale
        self.gt_proj = nn.Sequential(
            nn.Linear(num_joints * 3, embed_dim),
            nn.LayerNorm(embed_dim),  # This ensures output has std~1
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)  # Final LayerNorm for consistent scale
        )
        
        # Use default Xavier initialization (not too small)
        for m in self.gt_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        if fusion_method == 'residual':
            # Start with small but reasonable weight
            self.alpha = nn.Parameter(torch.ones(1) * 0.05)
            
        elif fusion_method == 'gating':
            # Gating mechanism with conservative initialization
            self.gate_proj = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim)
            )
            # Initialize gate to output moderate values
            nn.init.xavier_uniform_(self.gate_proj[-1].weight, gain=0.1)
            nn.init.constant_(self.gate_proj[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12
            
            # Scale factor - start small but not too small
            self.fusion_scale = nn.Parameter(torch.ones(1) * 0.1)
            
        elif fusion_method == 'cross_attention':
            # Cross-attention
            self.num_heads = 8
            self.cross_attn = nn.MultiheadAttention(
                embed_dim, 
                num_heads=self.num_heads,
                batch_first=True,
                dropout=0.1
            )
            self.norm = nn.LayerNorm(embed_dim)
            self.fusion_scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x, gt_3d):
        """
        Args:
            x: Feature tensor (B, F, embed_dim) or (B, F, J, embed_dim_ratio)
            gt_3d: GT 3D pose (B, F, J, 3)
        Returns:
            Fused feature with same shape as x
        """
        B, F = gt_3d.shape[:2]
        
        # Normalize GT 3D to reasonable scale
        # 1. Center around root joint
        gt_3d_centered = gt_3d - gt_3d[:, :, :1, :]
        
        # 2. Normalize by fixed scale (Human36M typical range ~1000mm)
        gt_3d_normalized = gt_3d_centered / 1000.0  # Now in range ~[-1, 1]
        
        # Flatten and project
        gt_3d_flat = gt_3d_normalized.reshape(B, F, -1)
        gt_3d_embed = self.gt_proj(gt_3d_flat)
        # After LayerNorm in gt_proj, gt_3d_embed should have std~1
        
        # Handle different input shapes
        original_shape = x.shape
        if len(x.shape) == 4:  # (B, F, J, embed_dim_ratio)
            x = x.reshape(B, F, -1)  # (B, F, J*embed_dim_ratio)
        
        # Apply fusion method
        if self.fusion_method == 'residual':
            x_fused = x + self.alpha * gt_3d_embed
            
        elif self.fusion_method == 'gating':
            gate = torch.sigmoid(self.gate_proj(torch.cat([x, gt_3d_embed], dim=-1)))
            # Use fusion_scale to control the strength
            x_fused = x + self.fusion_scale * gate * gt_3d_embed
            
        elif self.fusion_method == 'cross_attention':
            # x as query, gt_3d as key/value
            attn_out, _ = self.cross_attn(x, gt_3d_embed, gt_3d_embed)
            x_fused = x + self.fusion_scale * self.norm(attn_out)
        
        # Restore original shape
        if len(original_shape) == 4:
            x_fused = x_fused.reshape(original_shape)
        
        return x_fused


class TemporalGT3DFusion(nn.Module):
    """Fusion inside temporal blocks"""
    
    def __init__(self, embed_dim, num_joints=17):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Project GT 3D
        self.gt_proj = nn.Linear(num_joints * 3, embed_dim)
        
        # Cross-attention for temporal fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads=8,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x, gt_3d):
        """
        Args:
            x: (B, F, embed_dim)
            gt_3d: (B, F, J, 3)
        """
        B, F = gt_3d.shape[:2]
        gt_3d_flat = gt_3d.reshape(B, F, -1)
        gt_3d_embed = self.gt_proj(gt_3d_flat)
        
        # Cross-attention
        attn_out, _ = self.cross_attn(x, gt_3d_embed, gt_3d_embed)
        x = x + self.norm(attn_out)
        
        return x
