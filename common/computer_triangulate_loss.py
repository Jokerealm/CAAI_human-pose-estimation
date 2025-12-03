import json
import torch
import numpy as np

with open(r'./common/camera.json') as f:
    camera_parameter = json.load(f)

# 预计算投影矩阵缓存 - 使用double精度以匹配原始实现
_proj_matrix_cache = {}

def get_batch_proj_matrices(subjects, device):
    """
    Generate Projection Matrices P = K[R|T] for the whole batch with caching
    使用double精度以完全匹配原始实现
    Returns: (B, 4, 3, 4)
    """
    batch_P = []
    for subj in subjects:
        subj_id = int(subj[1:])
        cache_key = (subj_id, str(device))
        
        if cache_key not in _proj_matrix_cache:
            view_P = []
            for cam_idx in range(1, 5):
                params = camera_parameter[str((subj_id, cam_idx))]
                # 使用double精度匹配原始实现
                R = np.array(params[0], dtype=np.double)
                T = np.array(params[1], dtype=np.double)
                f = params[2]
                c = params[3]
                
                # K matrix
                K = np.array([[f[0][0], 0., c[0][0]], 
                             [0., f[1][0], c[1][0]], 
                             [0., 0., 1.0]], dtype=np.double)
                
                # Extrinsics [R | -RT]
                T = np.dot(R, np.negative(T))
                if len(T.shape) < 2:
                    T = np.expand_dims(T, axis=-1)
                
                # P = K @ [R | T]
                P = np.dot(K, np.concatenate((R, T), axis=1))
                view_P.append(P)
            
            # 转换为torch张量并缓存
            _proj_matrix_cache[cache_key] = torch.from_numpy(np.stack(view_P)).to(device)
        
        batch_P.append(_proj_matrix_cache[cache_key])
    
    return torch.stack(batch_P)  # (B, 4, 3, 4)

def zero_the_root_vectorized(pose, root_idx=0):
    # pose: (B, J, 3)
    is_zero = torch.sum(torch.abs(pose[:, root_idx, :]), dim=-1) == 0
    
    if torch.any(is_zero):
        estimated = (pose[:, 1, :] + pose[:, 4, :]) * 0.5
        pose[is_zero, root_idx, :] = estimated[is_zero]
    
    return pose - pose[:, root_idx:root_idx+1, :]

def triangulate_loss(output_3D, input_2D_points, subject):
    """
    GPU优化的三角化损失 - 完全在GPU上运行，支持梯度传播
    output_3D: (B, J, 3) - GPU tensor
    input_2D_points: (B, 4, J, 2) Normalized [-1, 1] - GPU tensor
    """
    device = output_3D.device
    B, J, _ = output_3D.shape
    
    # 1. Denormalization - 在GPU上处理
    bias = torch.tensor([[[1, 1.002]], [[1, 1]], [[1, 1]], [[1, 1.002]]], 
                        device=device, dtype=input_2D_points.dtype)  # (4, 1, 2)
    input_2D = (input_2D_points + bias) / 2 * 1000
    
    # 2. Get Projection Matrices: (B, 4, 3, 4) - 在GPU上
    proj_mats = get_batch_proj_matrices(subject, device)
    
    # 3. 批量构建DLT系统并求解
    triangulate_3d_points_batch = []
    
    for batch_idx in range(B):
        proj_matrix = proj_mats[batch_idx]  # (4, 3, 4)
        input_2D_sample = input_2D[batch_idx]  # (4, J, 2)
        
        # 预扩展投影矩阵以减少重复计算
        proj_row2_exp = proj_matrix[:, 2:3].expand(4, 2, 4)  # (4, 2, 4)
        proj_row01 = proj_matrix[:, :2]  # (4, 2, 4)
        
        # 对每个关节点进行三角化
        triangulate_3d_points = []
        for joint_idx in range(J):
            # 构建DLT方程 A*X=0
            uv = input_2D_sample[:, joint_idx, :].view(4, 2, 1)  # (4, 2, 1)
            A = proj_row2_exp * uv - proj_row01  # (4, 2, 4)
            
            # SVD求解 - 在GPU上
            A_flat = A.view(-1, 4)  # (8, 4)
            
            # 使用 torch.linalg.svd (支持GPU和自动微分)
            _, _, Vh = torch.linalg.svd(A_flat, full_matrices=True)
            
            # 齐次坐标解（最小奇异值对应的右奇异向量）
            # Vh 的形状是 (4, 4)，最后一行对应最小奇异值
            point_3d_homo = -Vh[-1, :]  # 添加负号以匹配原始实现
            
            # 转换为欧几里得坐标 - 使用符号判断确保一致性
            w = point_3d_homo[3]
            if torch.abs(w) < 1e-10:
                # 如果w接近0，使用默认值
                triangulate_3d = point_3d_homo[:3] / 1000
            else:
                triangulate_3d = point_3d_homo[:3] / w / 1000
            
            triangulate_3d_points.append(triangulate_3d)
        
        triangulate_3d_points = torch.stack(triangulate_3d_points, dim=0)  # (J, 3)
        
        # 处理根节点为零的情况
        if torch.sum(torch.abs(triangulate_3d_points[0, :])) < 1e-9:
            triangulate_3d_points[0, :] = (triangulate_3d_points[1, :] + triangulate_3d_points[4, :]) * 0.5
        
        triangulate_3d_points_batch.append(triangulate_3d_points)
    
    triangulate_3d_points_batch = torch.stack(triangulate_3d_points_batch, dim=0)  # (B, J, 3)
    
    # 4. 根节点中心化 - 向量化
    root = triangulate_3d_points_batch[:, 0:1, :]
    triangulate_3d_points_batch = triangulate_3d_points_batch - root
    
    # 5. 计算L1损失 - 向量化
    loss_batch = torch.mean(
        torch.norm(triangulate_3d_points_batch - output_3D, p=1, dim=-1),
        dim=-1
    )
    
    return loss_batch