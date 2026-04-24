# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from tqdm import tqdm


from common.opt import opts
from common.h36m_dataset import Human36mDataset
from common.Mydataset import Fusion
from common.utils import *
from model.SGraFormer_HyperGraph import SGraFormer_HyperGraph
from model.SGraFormer import sgraformer


H36M_JOINTS = [
    'Hip',           # 0
    'RHip',          # 1  
    'RKnee',         # 2
    'RFoot',         # 3
    'LHip',          # 4
    'LKnee',         # 5
    'LFoot',         # 6
    'Spine',         # 7
    'Thorax',        # 8
    'Neck/Nose',     # 9
    'Head',          # 10
    'LShoulder',     # 11
    'LElbow',        # 12
    'LWrist',        # 13
    'RShoulder',     # 14
    'RElbow',        # 15
    'RWrist'         # 16
]


H36M_SKELETON = [
    [0, 1],   # Hip -> RHip
    [1, 2],   # RHip -> RKnee  
    [2, 3],   # RKnee -> RFoot
    [0, 4],   # Hip -> LHip
    [4, 5],   # LHip -> LKnee
    [5, 6],   # LKnee -> LFoot
    [0, 7],   # Hip -> Spine
    [7, 8],   # Spine -> Thorax
    [8, 9],   # Thorax -> Neck/Nose
    [9, 10],  # Neck/Nose -> Head
    [8, 11],  # Thorax -> LShoulder
    [11, 12], # LShoulder -> LElbow
    [12, 13], # LElbow -> LWrist
    [8, 14],  # Thorax -> RShoulder
    [14, 15], # RShoulder -> RElbow
    [15, 16]  # RElbow -> RWrist
]


COLORS = {
    'left': '#FF4444',      # 红色 - 左侧
    'right': '#4444FF',     # 蓝色 - 右侧  
    'center': '#666666'     # 灰色 - 中心部位
}


SKELETON_COLORS = [
    COLORS['right'],    # [0, 1] Hip -> RHip
    COLORS['right'],    # [1, 2] RHip -> RKnee
    COLORS['right'],    # [2, 3] RKnee -> RFoot
    COLORS['left'],     # [0, 4] Hip -> LHip
    COLORS['left'],     # [4, 5] LHip -> LKnee
    COLORS['left'],     # [5, 6] LKnee -> LFoot
    COLORS['center'],   # [0, 7] Hip -> Spine
    COLORS['center'],   # [7, 8] Spine -> Thorax
    COLORS['center'],   # [8, 9] Thorax -> Neck/Nose
    COLORS['center'],   # [9, 10] Neck/Nose -> Head
    COLORS['left'],     # [8, 11] Thorax -> LShoulder
    COLORS['left'],     # [11, 12] LShoulder -> LElbow
    COLORS['left'],     # [12, 13] LElbow -> LWrist
    COLORS['right'],    # [8, 14] Thorax -> RShoulder
    COLORS['right'],    # [14, 15] RShoulder -> RElbow
    COLORS['right']     # [15, 16] RElbow -> RWrist
]


def load_sgraformer_raw(model_path, device):
    """加载SGraFormer_raw模型 (baseline)"""
    print(f"Loading SGraFormer_raw from: {model_path}")
    
    # 创建模型 - 使用与main.py相同的参数
    model = sgraformer(num_frame=27, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                      num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理DataParallel保存的模型
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除module.前缀（如果存在）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # 尝试加载兼容的参数
    model_dict = model.state_dict()
    compatible_dict = {}
    incompatible_keys = []
    
    for k, v in new_state_dict.items():
        if k in model_dict.keys():
            if v.shape == model_dict[k].shape:
                compatible_dict[k] = v
            else:
                incompatible_keys.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
        else:
            incompatible_keys.append(f"{k}: not in model")
    
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print(f'✅ SGraFormer_raw loaded {len(compatible_dict)}/{len(model_dict)} compatible parameters')
    if incompatible_keys and len(incompatible_keys) <= 5:
        for key in incompatible_keys:
            print(f'     - {key}')
    
    model.to(device)
    model.eval()
    return model


def load_sgraformer_hypergraph(model_path, device):
    """加载SGraFormer_HyperGraph模型"""
    print(f"Loading SGraFormer_HyperGraph from: {model_path}")
    
    # 创建模型 - 使用与main_hnn.py相同的参数
    # model = SGraFormer_HyperGraph(
    #     num_frame=27,
    #     num_joints=17,
    #     in_chans=2,
    #     embed_dim_ratio=32,
    #     depth=6,
    #     k_neighbors=3,
    #     prune_ratio=0.0,
    #     num_heads=8,
    #     mlp_ratio=2.,
    #     drop_rate=0.,
    #     attn_drop_rate=0.,
    #     drop_path_rate=0.1,
    # )
    model = sgraformer(num_frame=27, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                      num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理DataParallel保存的模型
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # 移除module.前缀（如果存在）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # 尝试加载兼容的参数
    model_dict = model.state_dict()
    compatible_dict = {}
    incompatible_keys = []
    
    for k, v in new_state_dict.items():
        if k in model_dict.keys():
            if v.shape == model_dict[k].shape:
                compatible_dict[k] = v
            else:
                incompatible_keys.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
        else:
            incompatible_keys.append(f"{k}: not in model")
    
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print(f'✅ SGraFormer_HyperGraph loaded {len(compatible_dict)}/{len(model_dict)} compatible parameters')
    if incompatible_keys and len(incompatible_keys) <= 5:
        for key in incompatible_keys:
            print(f'     - {key}')
    
    model.to(device)
    model.eval()
    return model


def draw_skeleton_3d_clean(ax, pose_3d, colors=SKELETON_COLORS):
    """在3D坐标系中绘制骨架 - 带网格和透明背景"""
    ax.clear()
    
    # 设置透明背景
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # 设置网格线颜色和透明度
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # 绘制骨骼连接
    for i, (start_idx, end_idx) in enumerate(H36M_SKELETON):
        start_point = pose_3d[start_idx]
        end_point = pose_3d[end_idx]
        ax.plot3D([start_point[0], end_point[0]], 
                  [start_point[1], end_point[1]], 
                  [start_point[2], end_point[2]], 
                  color=colors[i], linewidth=3)
    
    # 绘制关节点
    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], 
               c='black', s=50, alpha=0.8)
    
    # 设置相同的坐标轴范围以便比较
    max_range = np.array([pose_3d[:, 0].max() - pose_3d[:, 0].min(),
                          pose_3d[:, 1].max() - pose_3d[:, 1].min(),
                          pose_3d[:, 2].max() - pose_3d[:, 2].min()]).max() / 2.0
    
    mid_x = (pose_3d[:, 0].max() + pose_3d[:, 0].min()) * 0.5
    mid_y = (pose_3d[:, 1].max() + pose_3d[:, 1].min()) * 0.5
    mid_z = (pose_3d[:, 2].max() + pose_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置视角
    ax.view_init(elev=15, azim=45)
    
    # 显示网格
    ax.grid(True, alpha=0.3)
    
    # 移除坐标轴标签和刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')


def create_comparison_visualization(raw_pose, hypergraph_pose, gt_pose, subject, action, frame_idx, output_dir):
    """创建三模型对比可视化图像 - 1行3列布局，透明背景"""
    
    # 创建主体文件夹
    subject_dir = os.path.join(output_dir, subject)
    os.makedirs(subject_dir, exist_ok=True)
    
    # 创建1行3列的子图，设置透明背景
    fig = plt.figure(figsize=(15, 5), facecolor='none')
    fig.patch.set_alpha(0.0)
    
    # SGraFormer_raw结果
    ax1 = fig.add_subplot(131, projection='3d')
    draw_skeleton_3d_clean(ax1, raw_pose)
    
    # SGraFormer_HyperGraph结果  
    ax2 = fig.add_subplot(132, projection='3d')
    draw_skeleton_3d_clean(ax2, hypergraph_pose)
    
    # Ground Truth
    ax3 = fig.add_subplot(133, projection='3d')
    draw_skeleton_3d_clean(ax3, gt_pose)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存图像，透明背景
    comparison_path = os.path.join(subject_dir, f'{action}_frame_{frame_idx:04d}_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight', 
                facecolor='none', transparent=True)
    plt.close(fig)
    
    return comparison_path


def calculate_mpjpe(pred_pose, gt_pose):
    """计算MPJPE误差 (Mean Per Joint Position Error)"""
    return np.sqrt(np.sum((pred_pose - gt_pose) ** 2, axis=1)).mean()


def calculate_pck(pred_pose, gt_pose, threshold=150.0):
    """计算PCK (Percentage of Correct Keypoints)"""
    distances = np.sqrt(np.sum((pred_pose - gt_pose) ** 2, axis=1))
    return (distances < threshold).mean()


def input_augmentation_raw(input_2D, hops, model):
    """SGraFormer_raw的测试时数据增强"""
    # SGraFormer_raw不使用hops参数
    if input_2D.dim() == 6:
        input_2D_non_flip = input_2D[:, 0]
    elif input_2D.dim() == 5:
        input_2D_non_flip = input_2D
    else:
        input_2D_non_flip = input_2D.unsqueeze(1)
    
    output_3D_non_flip = model(input_2D_non_flip, hops)  # 保持接口一致
    return input_2D_non_flip, output_3D_non_flip


def input_augmentation_hypergraph(input_2D, hops, model):
    """SGraFormer_HyperGraph的测试时数据增强"""
    if input_2D.dim() == 6:
        input_2D_non_flip = input_2D[:, 0]
    elif input_2D.dim() == 5:
        input_2D_non_flip = input_2D
    else:
        input_2D_non_flip = input_2D.unsqueeze(1)
    
    output_3D_non_flip = model(input_2D_non_flip, hops)
    return input_2D_non_flip, output_3D_non_flip


def compare_three_models(raw_model, hypergraph_model, output_dir, max_samples=None):
    """对比三个模型的结果：SGraFormer_raw, SGraFormer_HyperGraph, GT"""
    
    print("=" * 80)
    print("🎨 H36M Three-Model Comparison Visualization")
    print("=" * 80)
    print(f"Output: {output_dir}")
    if max_samples:
        print(f"Max samples: {max_samples}")
    else:
        print("Processing all samples")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置选项
    opt = opts().parse()
    opt.dataset = 'h36m'
    opt.keypoints = 'cpn_ft_h36m_dbb'
    opt.actions = '*'
    opt.stride = 1
    opt.crop_uv = 0
    opt.test_augmentation = True
    opt.pad = 13  # 中间帧
    
    # 加载数据集
    print("\n📂 Loading H36M dataset...")
    root_path = opt.root_path
    dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'
    dataset = Human36mDataset(dataset_path, opt)
    
    # 创建测试数据加载器
    test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                  shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Dataset loaded: {len(test_data)} samples")
    
    # 统计计数器
    subject_counts = {}
    action_counts = {}
    total_processed = 0
    
    print("\n🔄 Processing samples...")
    
    device = next(raw_model.parameters()).device
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader, desc="Creating comparisons")):
            if max_samples and total_processed >= max_samples:
                break
                
            # 解包数据
            batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, start, end, hops = data
            
            # 转换为模型设备
            input_2D = input_2D.to(device)
            gt_3D = gt_3D.to(device)
            hops = hops.to(device)
            
            # 获取主体名称和动作名称
            subject_name = subject[0]
            action_name = action[0].replace(' ', '_')
            
            # 统计计数
            if subject_name not in subject_counts:
                subject_counts[subject_name] = 0
                action_counts[subject_name] = {}
            if action_name not in action_counts[subject_name]:
                action_counts[subject_name][action_name] = 0
            
            # 如果设置了max_samples，则不限制每个动作的样本数
            if max_samples is None and action_counts[subject_name][action_name] >= 3:
                continue
            
            # SGraFormer_raw推理
            if opt.test_augmentation:
                input_2D_raw, output_3D_raw = input_augmentation_raw(input_2D, hops, raw_model)
            else:
                output_3D_raw = raw_model(input_2D, hops)
            
            # SGraFormer_HyperGraph推理
            if opt.test_augmentation:
                input_2D_hg, output_3D_hg = input_augmentation_hypergraph(input_2D, hops, hypergraph_model)
            else:
                output_3D_hg = hypergraph_model(input_2D, hops)
            
            # 处理输出 - SGraFormer_raw
            if output_3D_raw.shape[1] != 1:
                output_3D_raw = output_3D_raw[:, opt.pad].unsqueeze(1)
            output_3D_raw[:, :, 1:, :] -= output_3D_raw[:, :, :1, :]
            output_3D_raw[:, :, 0, :] = 0
            
            # 处理输出 - SGraFormer_HyperGraph
            if output_3D_hg.shape[1] != 1:
                output_3D_hg = output_3D_hg[:, opt.pad].unsqueeze(1)
            output_3D_hg[:, :, 1:, :] -= output_3D_hg[:, :, :1, :]
            output_3D_hg[:, :, 0, :] = 0
            
            # 处理GT
            out_target = gt_3D.clone()
            out_target[:, :, 0] = 0
            
            # 转换为numpy数组
            raw_pose = output_3D_raw[0, 0].cpu().numpy()  # [17, 3]
            hypergraph_pose = output_3D_hg[0, 0].cpu().numpy()  # [17, 3]
            gt_pose = out_target[0, 0].cpu().numpy()  # [17, 3]
            
            # 创建对比可视化
            try:
                comparison_path = create_comparison_visualization(
                    raw_pose, hypergraph_pose, gt_pose, 
                    subject_name, action_name, 
                    action_counts[subject_name][action_name], output_dir
                )
                
                subject_counts[subject_name] += 1
                action_counts[subject_name][action_name] += 1
                total_processed += 1
                
                if total_processed % 100 == 0:
                    print(f"  Processed {total_processed} samples...")
                
            except Exception as e:
                print(f"  ❌ Error creating comparison for {subject_name} {action_name}: {e}")
                continue
    
    print(f"\n✅ Comparison visualization completed!")
    print(f"📊 Summary:")
    print(f"  Total processed: {total_processed} samples")
    for subject, count in subject_counts.items():
        print(f"  {subject}: {count} samples")
        if subject in action_counts:
            for action, action_count in action_counts[subject].items():
                print(f"    - {action}: {action_count} samples")
    print(f"📁 Results saved in: {output_dir}")
    
    # 保存处理统计
    save_comparison_summary(subject_counts, action_counts, total_processed, output_dir)


def save_comparison_summary(subject_counts, action_counts, total_processed, output_dir):
    """保存对比处理统计信息"""
    summary_path = os.path.join(output_dir, 'comparison_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("H36M Three-Model Comparison Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Models compared:\n")
        f.write("1. baseline\n")
        f.write("2. ours\n")
        f.write("3. Ground Truth\n\n")
        
        f.write(f"Total processed samples: {total_processed}\n\n")
        
        # 按主体统计
        f.write("Samples by Subject:\n")
        f.write("-" * 30 + "\n")
        for subject, count in sorted(subject_counts.items()):
            f.write(f"{subject}: {count} samples\n")
        
        f.write("\nDetailed breakdown by Action:\n")
        f.write("-" * 50 + "\n")
        
        # 按主体和动作详细统计
        for subject in sorted(subject_counts.keys()):
            f.write(f"\n{subject}:\n")
            if subject in action_counts:
                for action, count in sorted(action_counts[subject].items()):
                    f.write(f"  {action}: {count} samples\n")
    
    print(f"📋 Comparison summary saved: {summary_path}")


def main():
    """
    主函数 - 对比SGraFormer_raw、SGraFormer_HyperGraph和GT的结果
    """
    
    # ==================== 模型配置 ====================
    # 模型文件路径
    raw_model_path = 'checkpoint/0111_1657_48_27/model_19_2908.pth'  # SGraFormer_raw
    hypergraph_model_path = 'checkpoint/gt/model_53_1137.pth'  # SGraFormer_HyperGraph
    
    # ==================== 可视化配置 ====================
    output_dir = 'dataset/h36m_comparison_final'
    max_samples = 500  # 处理所有样本，设置为None；如果要限制数量，设置为具体数字如1000
    
    # ==================== 设备配置 ====================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== 加载模型 ====================
    print("=" * 80)
    print("🚀 H36M Three-Model Comparison Visualization (Final Version)")
    print("=" * 80)
    print(f"SGraFormer_raw model: {raw_model_path}")
    print(f"SGraFormer_HyperGraph model: {hypergraph_model_path}")
    print(f"Output directory: {output_dir}")
    print("Features: Grid lines + Transparent background + All samples")
    print("=" * 80)
    
    # 检查模型文件
    if not os.path.exists(raw_model_path):
        print(f"❌ SGraFormer_raw model file not found: {raw_model_path}")
        return
    
    if not os.path.exists(hypergraph_model_path):
        print(f"❌ SGraFormer_HyperGraph model file not found: {hypergraph_model_path}")
        return
    
    # 加载SGraFormer_raw模型
    try:
        print("\n📥 Loading SGraFormer_raw model...")
        raw_model = load_sgraformer_raw(raw_model_path, device)
    except Exception as e:
        print(f"❌ Failed to load SGraFormer_raw model: {e}")
        return
    
    # 加载SGraFormer_HyperGraph模型
    try:
        print("\n📥 Loading SGraFormer_HyperGraph model...")
        hypergraph_model = load_sgraformer_hypergraph(hypergraph_model_path, device)
    except Exception as e:
        print(f"❌ Failed to load SGraFormer_HyperGraph model: {e}")
        return
    
    # ==================== 开始对比可视化 ====================
    print("\n🎨 Starting three-model comparison...")
    compare_three_models(raw_model, hypergraph_model, output_dir, max_samples)


if __name__ == '__main__':
    main()