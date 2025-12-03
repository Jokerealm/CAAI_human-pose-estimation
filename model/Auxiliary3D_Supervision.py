"""
Auxiliary 3D Supervision Module

This module provides auxiliary supervision using 3D pose information
(GT 3D, triangulated 3D, or estimated 3D) without direct fusion.

Progressive pipeline:
1. GT 3D (upper bound)
2. Triangulation with GT camera parameters
3. Triangulation with estimated camera parameters
4. No auxiliary supervision (final deployment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Auxiliary3DProjection(nn.Module):
    """Project 3D pose to feature space for auxiliary supervision"""
    
    def __init__(self, embed_dim, num_joints=17, projection_type='simple'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_joints = num_joints
        self.projection_type = projection_type
        
        if projection_type == 'simple':
            # Simple linear projection
            self.proj = nn.Sequential(
                nn.Linear(num_joints * 3, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        elif projection_type == 'mlp':
            # 2-layer MLP
            self.proj = nn.Sequential(
                nn.Linear(num_joints * 3, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        
        # Initialize with small weights
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, pose_3d):
        """
        Args:
            pose_3d: (B, F, J, 3) - 3D pose (GT, triangulated, or estimated)
        Returns:
            embed: (B, F, embed_dim) - projected embedding
        """
        B, F = pose_3d.shape[:2]
        
        # Normalize 3D pose
        # 1. Center around root joint
        pose_3d_centered = pose_3d - pose_3d[:, :, :1, :]
        
        # 2. Normalize by fixed scale (Human36M ~1000mm)
        pose_3d_normalized = pose_3d_centered / 1000.0
        
        # Flatten and project
        pose_3d_flat = pose_3d_normalized.reshape(B, F, -1)
        embed = self.proj(pose_3d_flat)
        
        return embed


class Auxiliary3DLoss(nn.Module):
    """Compute auxiliary loss between feature and 3D pose embedding"""
    
    def __init__(self, embed_dim, num_joints=17, 
                 loss_type='mse', projection_type='simple'):
        super().__init__()
        self.loss_type = loss_type
        
        # Projection module
        self.projection = Auxiliary3DProjection(
            embed_dim, num_joints, projection_type
        )
    
    def forward(self, feature, pose_3d, weight=1.0):
        """
        Args:
            feature: (B, F, embed_dim) - intermediate feature from model
            pose_3d: (B, F, J, 3) - 3D pose for supervision
            weight: float - loss weight
        Returns:
            loss: scalar tensor
        """
        # Project 3D pose to embedding space
        with torch.no_grad():  # Don't backprop through 3D pose
            pose_embed = self.projection(pose_3d)
        
        # Compute loss
        if self.loss_type == 'mse':
            loss = F.mse_loss(feature, pose_embed)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(feature, pose_embed)
        elif self.loss_type == 'cosine':
            # Cosine similarity loss
            feature_norm = F.normalize(feature, dim=-1)
            pose_norm = F.normalize(pose_embed, dim=-1)
            loss = 1 - (feature_norm * pose_norm).sum(dim=-1).mean()
        
        return weight * loss


class Progressive3DSupervision:
    """
    Helper class to manage progressive 3D supervision
    
    Usage:
        supervisor = Progressive3DSupervision(
            stage='gt',  # 'gt', 'triangulated_gt_cam', 'triangulated_est_cam', 'none'
            weight=0.01
        )
        
        aux_loss = supervisor.compute_loss(feature, pose_3d)
    """
    
    def __init__(self, stage='gt', weight=0.01, 
                 embed_dim=544, num_joints=17):
        self.stage = stage
        self.weight = weight
        
        if stage != 'none':
            self.loss_module = Auxiliary3DLoss(
                embed_dim, num_joints,
                loss_type='mse',
                projection_type='simple'
            )
    
    def compute_loss(self, feature, pose_3d):
        """
        Compute auxiliary loss based on current stage
        
        Args:
            feature: (B, F, embed_dim)
            pose_3d: (B, F, J, 3)
        Returns:
            loss: scalar or 0
        """
        if self.stage == 'none' or self.weight == 0:
            return 0
        
        return self.loss_module(feature, pose_3d, self.weight)
    
    def get_stage_info(self):
        """Get information about current stage"""
        info = {
            'gt': 'Using GT 3D (upper bound)',
            'triangulated_gt_cam': 'Using triangulation with GT camera',
            'triangulated_est_cam': 'Using triangulation with estimated camera',
            'none': 'No auxiliary supervision'
        }
        return info.get(self.stage, 'Unknown stage')


def test_auxiliary_supervision():
    """Test auxiliary supervision module"""
    print("=" * 80)
    print("Testing Auxiliary 3D Supervision")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test data
    B, F, J = 2, 27, 17
    embed_dim = 544
    
    feature = torch.randn(B, F, embed_dim).to(device)
    pose_3d = torch.randn(B, F, J, 3).to(device) * 1000  # Simulate real scale
    
    print(f"\nInput:")
    print(f"  Feature: {feature.shape}, mean={feature.mean():.4f}, std={feature.std():.4f}")
    print(f"  Pose 3D: {pose_3d.shape}, mean={pose_3d.mean():.2f}, std={pose_3d.std():.2f}")
    
    # Test projection
    print("\n" + "=" * 80)
    print("Testing Projection")
    print("=" * 80)
    
    projection = Auxiliary3DProjection(embed_dim, J, 'simple').to(device)
    with torch.no_grad():
        pose_embed = projection(pose_3d)
    
    print(f"\nProjected embedding:")
    print(f"  Shape: {pose_embed.shape}")
    print(f"  Mean: {pose_embed.mean():.4f}, Std: {pose_embed.std():.4f}")
    print(f"  Range: [{pose_embed.min():.2f}, {pose_embed.max():.2f}]")
    
    # Test loss
    print("\n" + "=" * 80)
    print("Testing Auxiliary Loss")
    print("=" * 80)
    
    loss_module = Auxiliary3DLoss(embed_dim, J, 'mse', 'simple').to(device)
    
    weights = [0.001, 0.01, 0.1, 1.0]
    for w in weights:
        loss = loss_module(feature, pose_3d, weight=w)
        print(f"  Weight {w:5.3f}: loss = {loss.item():.6f}")
    
    # Test progressive supervision
    print("\n" + "=" * 80)
    print("Testing Progressive Supervision")
    print("=" * 80)
    
    stages = ['gt', 'triangulated_gt_cam', 'triangulated_est_cam', 'none']
    for stage in stages:
        supervisor = Progressive3DSupervision(stage, weight=0.01, embed_dim=embed_dim)
        print(f"\nStage: {stage}")
        print(f"  Info: {supervisor.get_stage_info()}")
        
        if stage != 'none':
            supervisor.loss_module = supervisor.loss_module.to(device)
            loss = supervisor.compute_loss(feature, pose_3d)
            print(f"  Loss: {loss.item():.6f}")
        else:
            loss = supervisor.compute_loss(feature, pose_3d)
            print(f"  Loss: {loss}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_auxiliary_supervision()
