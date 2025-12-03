"""
Bone Length Consistency Loss for 3D Human Pose Estimation

Based on recent CVPR/ICCV papers:
- "Exploiting Temporal Contexts with Strided Transformer for 3D Human Pose Estimation" (TMM 2023)
- "MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation" (CVPR 2022)

Key insight: Human bone lengths should remain constant across frames
"""
import torch
import torch.nn as nn


# Human3.6M skeleton structure (17 joints)
H36M_SKELETON = [
    [0, 1],   # Hip -> RHip
    [0, 4],   # Hip -> LHip
    [1, 2],   # RHip -> RKnee
    [2, 3],   # RKnee -> RAnkle
    [4, 5],   # LHip -> LKnee
    [5, 6],   # LKnee -> LAnkle
    [0, 7],   # Hip -> Spine
    [7, 8],   # Spine -> Thorax
    [8, 9],   # Thorax -> Neck/Nose
    [9, 10],  # Neck -> Head
    [8, 11],  # Thorax -> LShoulder
    [11, 12], # LShoulder -> LElbow
    [12, 13], # LElbow -> LWrist
    [8, 14],  # Thorax -> RShoulder
    [14, 15], # RShoulder -> RElbow
    [15, 16], # RElbow -> RWrist
]


class BoneLengthLoss(nn.Module):
    """
    Bone length consistency loss
    Enforces that bone lengths remain constant across frames
    """
    def __init__(self, skeleton=None):
        super(BoneLengthLoss, self).__init__()
        self.skeleton = skeleton if skeleton is not None else H36M_SKELETON
        
        # Pre-compute bone pairs as tensors for efficient GPU computation
        self.register_buffer('bone_pairs', torch.tensor(self.skeleton, dtype=torch.long))
    
    def forward(self, pred_3d):
        """
        Args:
            pred_3d: [B, F, J, 3] predicted 3D poses
        
        Returns:
            bone length consistency loss
        """
        B, F, J, _ = pred_3d.shape
        
        # Extract bone start and end points: [B, F, num_bones, 3]
        bone_starts = pred_3d[:, :, self.bone_pairs[:, 0], :]
        bone_ends = pred_3d[:, :, self.bone_pairs[:, 1], :]
        
        # Compute bone vectors: [B, F, num_bones, 3]
        bone_vectors = bone_ends - bone_starts
        
        # Compute bone lengths: [B, F, num_bones]
        bone_lengths = torch.sqrt(torch.sum(bone_vectors ** 2, dim=-1) + 1e-8)
        
        # Compute mean bone length across frames: [B, 1, num_bones]
        mean_bone_lengths = torch.mean(bone_lengths, dim=1, keepdim=True)
        
        # Consistency loss: variance of bone lengths across frames
        bone_length_variance = torch.mean((bone_lengths - mean_bone_lengths) ** 2)
        
        return bone_length_variance


class BoneLengthSymmetryLoss(nn.Module):
    """
    Bone length symmetry loss
    Enforces that left and right limbs have similar bone lengths
    """
    def __init__(self):
        super(BoneLengthSymmetryLoss, self).__init__()
        
        # Define symmetric bone pairs (left-right)
        # Format: [left_bone, right_bone] where each bone is [start_joint, end_joint]
        self.symmetric_pairs = [
            [[0, 1], [0, 4]],     # Hip to RHip vs LHip
            [[1, 2], [4, 5]],     # RHip-RKnee vs LHip-LKnee
            [[2, 3], [5, 6]],     # RKnee-RAnkle vs LKnee-LAnkle
            [[8, 11], [8, 14]],   # Thorax-LShoulder vs RShoulder
            [[11, 12], [14, 15]], # LShoulder-LElbow vs RShoulder-RElbow
            [[12, 13], [15, 16]], # LElbow-LWrist vs RElbow-RWrist
        ]
    
    def forward(self, pred_3d):
        """
        Args:
            pred_3d: [B, F, J, 3] predicted 3D poses
        
        Returns:
            bone length symmetry loss
        """
        total_loss = 0.0
        
        for left_bone, right_bone in self.symmetric_pairs:
            # Compute left bone length
            left_vec = pred_3d[:, :, left_bone[1], :] - pred_3d[:, :, left_bone[0], :]
            left_length = torch.sqrt(torch.sum(left_vec ** 2, dim=-1) + 1e-8)
            
            # Compute right bone length
            right_vec = pred_3d[:, :, right_bone[1], :] - pred_3d[:, :, right_bone[0], :]
            right_length = torch.sqrt(torch.sum(right_vec ** 2, dim=-1) + 1e-8)
            
            # Symmetry loss: difference between left and right
            total_loss += torch.mean((left_length - right_length) ** 2)
        
        return total_loss / len(self.symmetric_pairs)


class BoneDirectionLoss(nn.Module):
    """
    Bone direction consistency loss
    Enforces smooth bone direction changes across frames
    """
    def __init__(self, skeleton=None):
        super(BoneDirectionLoss, self).__init__()
        self.skeleton = skeleton if skeleton is not None else H36M_SKELETON
        self.register_buffer('bone_pairs', torch.tensor(self.skeleton, dtype=torch.long))
    
    def forward(self, pred_3d):
        """
        Args:
            pred_3d: [B, F, J, 3] predicted 3D poses
        
        Returns:
            bone direction smoothness loss
        """
        B, F, J, _ = pred_3d.shape
        
        if F < 2:
            return torch.tensor(0.0, device=pred_3d.device)
        
        # Extract bone vectors: [B, F, num_bones, 3]
        bone_starts = pred_3d[:, :, self.bone_pairs[:, 0], :]
        bone_ends = pred_3d[:, :, self.bone_pairs[:, 1], :]
        bone_vectors = bone_ends - bone_starts
        
        # Normalize bone vectors: [B, F, num_bones, 3]
        bone_lengths = torch.sqrt(torch.sum(bone_vectors ** 2, dim=-1, keepdim=True) + 1e-8)
        bone_directions = bone_vectors / bone_lengths
        
        # Compute direction change between consecutive frames: [B, F-1, num_bones, 3]
        direction_diff = bone_directions[:, 1:, :, :] - bone_directions[:, :-1, :, :]
        
        # Smoothness loss: penalize large direction changes
        smoothness_loss = torch.mean(direction_diff ** 2)
        
        return smoothness_loss


@torch.jit.script
def compute_bone_length_loss_fast(pred_3d, bone_pairs):
    """
    JIT-compiled fast bone length consistency loss
    
    Args:
        pred_3d: [B, F, J, 3]
        bone_pairs: [num_bones, 2]
    
    Returns:
        scalar loss
    """
    # Extract bone endpoints
    bone_starts = pred_3d[:, :, bone_pairs[:, 0], :]
    bone_ends = pred_3d[:, :, bone_pairs[:, 1], :]
    
    # Compute bone lengths
    bone_vectors = bone_ends - bone_starts
    bone_lengths = torch.sqrt(torch.sum(bone_vectors * bone_vectors, dim=-1) + 1e-8)
    
    # Consistency: variance across frames
    mean_lengths = torch.mean(bone_lengths, dim=1, keepdim=True)
    variance = torch.mean((bone_lengths - mean_lengths) ** 2)
    
    return variance
