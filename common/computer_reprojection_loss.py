import json
import torch
import numpy as np

with open(r'./common/camera.json') as f:
    camera_parameter = json.load(f)


def project_point_radial_batch(P, R, T, f, c, k, p):
    """
    批量投影3D点到2D，包含径向和切向畸变
    P: (B, J, 3) 世界坐标系中的3D点
    返回: (B, J, 2) 像素空间中的2D点
    """
    # P: (B, J, 3), 转换为 (3, B*J)
    B, J = P.shape[0], P.shape[1]
    P_flat = P.reshape(-1, 3).T  # (3, B*J)
    
    # 旋转和平移
    X = torch.mm(R, P_flat - T)  # (3, B*J)
    XX = X[:2, :] / (X[2, :] + 1e-9)  # (2, B*J)
    
    # 径向畸变
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2  # (B*J,)
    N = B * J
    radial = 1 + torch.einsum('ij,ij->j', k.repeat((1, N)), torch.stack([r2, r2 ** 2, r2 ** 3]))
    
    # 切向畸变
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]
    
    # 应用畸变
    XXX = XX * (radial + tan).repeat([2, 1]) + torch.outer(torch.tensor([p[1], p[0]], device=P.device), r2)
    
    # 投影到像素坐标
    Proj = (f * XXX) + c  # (2, B*J)
    Proj = Proj.T.reshape(B, J, 2)  # (B, J, 2)
    
    return Proj


def zero_the_root(pose, root_idx):
    """根节点中心化"""
    if isinstance(pose, np.ndarray):
        root_pose = pose[:, root_idx:root_idx+1, :]
        pose = pose - root_pose
        pose = np.delete(pose, root_idx, 1)
        return pose, root_pose.squeeze(1)

    elif torch.is_tensor(pose):
        pose1 = pose.clone()
        # 处理根节点为零的情况
        for i in range(pose.shape[0]):
            if torch.sum(torch.abs(pose1[i, root_idx, :])) == 0:
                pose1[i, root_idx, :] = (pose1[i, 1, :] + pose1[i, 4, :]) / 2
        root_pose = pose1[:, root_idx:root_idx+1, :]
        pose1 = pose1 - root_pose
        return pose1
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def reprojection_loss(gt_3D, target_2D, subject):
    """
    GPU优化的重投影损失：将预测的3D姿态投影回2D（包含畸变），与输入2D姿态比较
    gt_3D: (B, J, 3) 预测的3D姿态 - GPU tensor
    target_2D: (B, 4, J, 2) 输入的2D姿态（4个视角） - GPU tensor
    """
    device = gt_3D.device
    batch_size = gt_3D.shape[0]
    J = gt_3D.shape[1]
    
    # 预分配结果张量 - 在GPU上
    project_points_batch = torch.zeros(batch_size, 4, J, 2, device=device, dtype=gt_3D.dtype)
    input_2D = target_2D.clone()
    
    # 对每个相机视角
    for cam_idx in range(4):
        # 对batch中的每个样本（保持循环以处理不同subject的相机参数）
        for batch_idx in range(batch_size):
            subject_id = int(subject[batch_idx][1:])
            
            # 获取相机参数 - 直接在GPU上创建
            R, T, f, c, k, p, name = camera_parameter[str((subject_id, cam_idx + 1))]
            R = torch.tensor(R, dtype=torch.float32, device=device)
            T = torch.tensor(T, dtype=torch.float32, device=device)
            f = torch.tensor(f, dtype=torch.float32, device=device)
            c = torch.tensor(c, dtype=torch.float32, device=device)
            k = torch.tensor(k, dtype=torch.float32, device=device)
            p = torch.tensor(p, dtype=torch.float32, device=device)
            
            # 投影单个样本的所有关节点
            gt_3D_sample = gt_3D[batch_idx:batch_idx+1] * 1000  # (1, J, 3)
            proj_point = project_point_radial_batch(gt_3D_sample, R, T, f, c, k, p)  # (1, J, 2)
            
            # 归一化到 [-1, 1] 范围
            w = 1000
            h = 1002 if (cam_idx == 0 or cam_idx == 3) else 1000
            proj_point = proj_point / w * 2 - torch.tensor([1, h / w], device=device)
            
            project_points_batch[batch_idx, cam_idx] = proj_point.squeeze(0)
        
        # 根节点中心化（向量化处理整个batch）
        project_points_batch[:, cam_idx, :, :] = zero_the_root(project_points_batch[:, cam_idx, :, :], 0)
        input_2D[:, cam_idx, :, :] = zero_the_root(input_2D[:, cam_idx, :, :], 0)
    
    # 计算L1损失（向量化）
    loss_batch = torch.mean(
        torch.norm(project_points_batch - input_2D, p=1, dim=-1),  # (B, 4, J)
        dim=-1  # (B, 4)
    )
    
    # 对视角维度求平均
    return torch.mean(loss_batch, dim=1)  # (B,)