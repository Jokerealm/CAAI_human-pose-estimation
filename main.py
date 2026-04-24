import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
CUDA_ID = [0]

import torch
import logging
import random
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter

from common.utils import *
from common.utils_3dhp import define_actions_3dhp, define_error_list_3dhp
from common.opt import opts
from common.h36m_dataset import Human36mDataset
from common.Mydataset import Fusion
from common.MPI_dataset_hops import Fusion_3dhp
from common.SKI_dataset import Fusion_ski

# from model.SGraFormer import sgraformer
from model.Auxiliary3D_Supervision import Auxiliary3DLoss

from common.computer_triangulate_loss import triangulate_loss
from common.computer_reprojection_loss import reprojection_loss
import torch.nn.functional as F
from model.SGraFormer import SGraFormer_HyperGraph

device = torch.device("cuda")


# ============================================================================
# 辅助监督相关函数 - 用于多卡DataParallel兼容
# ============================================================================

# 全局变量存储中间特征（用于hook）
_intermediate_features = {}


def register_feature_hook(model, position='mvf'):
    """
    注册forward hook来捕获中间特征
    这个方法与DataParallel完全兼容
    
    Args:
        model: 模型（可能被DataParallel包装）
        position: 'mvf' 或 'temporal'
    
    Returns:
        hook_handle: hook句柄，用于后续移除
    """
    global _intermediate_features
    _intermediate_features.clear()
    
    def hook_fn(module, input, output):
        """Hook函数，捕获模块的输出"""
        # 存储输出（detach避免梯度问题）
        _intermediate_features[position] = output.detach()
    
    # 获取实际模型（处理DataParallel包装）
    actual_model = model.module if hasattr(model, 'module') else model
    
    # 根据位置注册hook到对应的模块
    try:
        if position == 'mvf':
            # Hook到MVF之后的LayerNorm
            if hasattr(actual_model, 'conv_norm'):
                handle = actual_model.conv_norm.register_forward_hook(hook_fn)
            else:
                print(f"Warning: Model doesn't have 'conv_norm' module")
                return None
        elif position == 'temporal':
            # Hook到Temporal Encoder
            if hasattr(actual_model, 'TF'):
                handle = actual_model.TF.register_forward_hook(hook_fn)
            else:
                print(f"Warning: Model doesn't have 'TF' module")
                return None
        else:
            print(f"Warning: Unknown position '{position}'")
            return None
        
        return handle
    except Exception as e:
        print(f"Error registering hook: {e}")
        return None


def get_intermediate_feature(position='mvf'):
    """
    获取hook捕获的中间特征
    
    Args:
        position: 'mvf' 或 'temporal'
    
    Returns:
        feature: 捕获的特征张量，如果没有则返回None
    """
    global _intermediate_features
    return _intermediate_features.get(position, None)


def get_model_attr(model, attr_name):
    """
    获取模型属性，处理DataParallel包装
    
    Args:
        model: 模型
        attr_name: 属性名
    
    Returns:
        属性值
    """
    if hasattr(model, 'module'):
        return getattr(model.module, attr_name)
    else:
        return getattr(model, attr_name)


def set_model_attr(model, attr_name, value):
    """
    设置模型属性，处理DataParallel包装
    
    Args:
        model: 模型
        attr_name: 属性名
        value: 属性值
    """
    if hasattr(model, 'module'):
        setattr(model.module, attr_name, value)
    else:
        setattr(model, attr_name, value)


# ============================================================================

# 设置随机种子以确保结果可复现
def set_random_seed(seed):
    """
    设置所有相关的随机种子以确保实验可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡训练时使用
    # 确保CUDA卷积操作的确定性
    cudnn.deterministic = True
    cudnn.benchmark = False  # 关闭自动寻找最优卷积算法，确保确定性



def train(opt, actions, train_loader, model, optimizer, epoch, writer, adaptive_weight=None, auxiliary_3d_loss=None):
    """训练一个epoch"""
    return step('train', opt, actions, train_loader, model, optimizer, epoch, writer, adaptive_weight, auxiliary_3d_loss)


def val(opt, actions, val_loader, model):
    """验证一个epoch"""
    with torch.no_grad():
        # 对于测试，需要使用测试集的actions
        if opt.dataset == 'SKI':
            from common.utils_ski import define_actions_SKI
            test_actions = define_actions_SKI(opt.actions, train=False)
        elif opt.dataset.startswith('3dhp'):
            from common.utils_3dhp import define_actions_3dhp
            test_actions = define_actions_3dhp(opt.actions, train=False)
        else:
            test_actions = actions
        
        return step('test', opt, test_actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None, writer=None, adaptive_weight=None, auxiliary_3d_loss=None):
    
    loss_all = {'loss': AccumLoss()}
    action_error_sum = define_error_list(actions)

    if split == 'train':
        model.train()
    else:
        model.eval()

    TQDM = tqdm(enumerate(dataLoader), total=len(dataLoader), ncols=100)
    for i, data in TQDM:
        # 根据数据集类型解包数据
        if opt.dataset.startswith('SKI'):
            # SKI数据集返回6个值
            gt_3D, input_2D, action, start, end, hops = data
            batch_cam = None
            subject = None
            scale = None
            bb_box = None
            [input_2D, gt_3D, hops] = get_varialbe(split, [input_2D, gt_3D, hops])
        else:
            # H36M和3DHP数据集返回10个值
            batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, start, end, hops = data
            [input_2D, gt_3D, batch_cam, scale, bb_box, hops] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box, hops])

        if split == 'train':
            # ============ 训练模式 ============
            
            # 如果使用辅助监督，注册hook来捕获中间特征
            hook_handle = None
            if hasattr(opt, 'use_auxiliary_3d') and opt.use_auxiliary_3d and auxiliary_3d_loss is not None:
                position = getattr(opt, 'aux_3d_position', 'mvf')
                hook_handle = register_feature_hook(model, position=position)
            
            # Forward pass
            output_3D = model(input_2D, hops)
            
            # 移除hook
            if hook_handle is not None:
                hook_handle.remove()
            
            # Prepare target
            out_target = gt_3D.clone()
            out_target[:, :, 0] = 0

            # 计算主loss
            main_loss = mpjpe_cal(output_3D, out_target)
            
            # 计算辅助loss（如果启用）
            if hasattr(opt, 'use_auxiliary_3d') and opt.use_auxiliary_3d and auxiliary_3d_loss is not None:
                aux_weight = getattr(opt, 'aux_3d_weight', 0.01)
                if aux_weight > 0:
                    # 获取hook捕获的中间特征
                    position = getattr(opt, 'aux_3d_position', 'mvf')
                    intermediate_feature = get_intermediate_feature(position=position)
                    
                    if intermediate_feature is not None:
                        # 计算辅助loss
                        try:
                            aux_loss = auxiliary_3d_loss(intermediate_feature, gt_3D, weight=aux_weight)
                            loss = main_loss + aux_loss
                            
                            # 显示辅助loss（每10个iteration）
                            if i % 10 == 0:
                                TQDM.set_description(f'Epoch [{epoch}/{opt.nepoch}]')
                                TQDM.set_postfix({
                                    "main": f"{main_loss.item():.4f}",
                                    "aux": f"{aux_loss.item():.4f}",
                                    "total": f"{loss.item():.4f}"
                                })
                        except Exception as e:
                            print(f"\nWarning: Error computing auxiliary loss: {e}")
                            print(f"  intermediate_feature shape: {intermediate_feature.shape if intermediate_feature is not None else None}")
                            print(f"  gt_3D shape: {gt_3D.shape}")
                            print(f"  Skipping auxiliary loss for this batch")
                            loss = main_loss
                    else:
                        # 中间特征为None，只使用主loss
                        if i == 0:  # 只在第一个iteration警告一次
                            print(f"\nWarning: intermediate_feature is None (position={position})")
                            print(f"  This may indicate that the hook is not working correctly")
                        loss = main_loss
                else:
                    loss = main_loss
            else:
                loss = main_loss
            
            # 更新进度条
            if i % 10 == 0:
                TQDM.set_description(f'Epoch [{epoch}/{opt.nepoch}]')
                TQDM.set_postfix({"l": loss.item()})

            # 记录loss
            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            # ============ 测试模式 ============
            if opt.dataset == 'h36m':
                # H36M使用test-time augmentation
                input_2D, output_3D = input_augmentation(input_2D, hops, model)
            else:
                # 3DHP和SKI直接推理
                output_3D = model(input_2D, hops)
            
            # Prepare output
            if output_3D.shape[1] != 1:
                output_3D = output_3D[:, opt.pad].unsqueeze(1)
            output_3D[:, :, 1:, :] -= output_3D[:, :, :1, :]
            output_3D[:, :, 0, :] = 0
            
            # Prepare target
            out_target = gt_3D.clone()
            if opt.dataset.startswith('SKI'):
                # SKI数据集的target处理
                if out_target.shape[1] != 1:
                    out_target = out_target[:, opt.pad].unsqueeze(1)
            out_target[:, :, 0] = 0
            
            # 确保内存连续
            output_3D = output_3D.contiguous()
            out_target = out_target.contiguous()
            
            # 计算误差
            action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, opt.dataset, subject)

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        p1, p2, pck, auc = print_error(opt.dataset, action_error_sum, opt.train)
        return p1, p2, pck, auc


def input_augmentation(input_2D, hops, model, subject=None):
    # 检查input_2D的维度
    if input_2D.dim() == 6:
        # 6D: (b, aug, f, v, j, c) - H36M with augmentation
        # 取第一个augmentation
        input_2D_non_flip = input_2D[:, 0]  # -> (b, f, v, j, c)
    elif input_2D.dim() == 5:
        # 5D: (b, f, v, j, c) - H36M without augmentation or already processed
        input_2D_non_flip = input_2D
    else:
        # 4D: (b, v, j, c) - SKI/3DHP without temporal dimension
        # 需要添加temporal维度
        input_2D_non_flip = input_2D.unsqueeze(1)  # -> (b, 1, v, j, c)
    
    output_3D_non_flip = model(input_2D_non_flip, hops)

    return input_2D_non_flip, output_3D_non_flip

# 定义一个函数来处理模型状态字典，确保单卡和多卡训练的一致性
def process_state_dict(state_dict, is_multi_gpu_model):
    """
    处理模型状态字典，确保在单卡和多卡环境之间的兼容性
    Args:
        state_dict: 模型的状态字典
        is_multi_gpu_model: 当前加载的模型是否是多GPU训练的
            (即是否有'module.'前缀)
    Returns:
        处理后的状态字典
    """
    new_state_dict = {}
    
    # 检测输入的state_dict是否包含'module.'前缀
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    for k, v in state_dict.items():
        if is_multi_gpu_model:
            # 如果当前是多GPU环境，但权重没有'module.'前缀，则添加前缀
            if not k.startswith('module.'):
                new_state_dict[f'module.{k}'] = v
            else:
                new_state_dict[k] = v
        else:
            # 如果当前是单GPU环境，但权重有'module.'前缀，则移除前缀
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
    
    return new_state_dict

if __name__ == '__main__':
    opt = opts().parse()
    root_path = opt.root_path
    opt.manualSeed = 1
    set_random_seed(opt.manualSeed)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    # 根据数据集类型加载不同的数据和动作定义
    if opt.dataset == 'h36m':
        # H36M数据集
        dataset_path = root_path + 'data_3d_' + opt.dataset + '.npz'
        dataset = Human36mDataset(dataset_path, opt)
        actions = define_actions(opt.actions)
        
        if opt.train:
            train_data = Fusion(opt=opt, train=True, dataset=dataset, root_path=root_path)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                           shuffle=True, num_workers=int(opt.workers), pin_memory=True)
        
        test_data = Fusion(opt=opt, train=False, dataset=dataset, root_path=root_path)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    
    elif opt.dataset.startswith('3dhp'):
        # 3DHP数据集
        actions = define_actions_3dhp(opt.actions, 1) if opt.train == 1 else define_actions_3dhp(opt.actions, 0)
        
        if opt.train:
            train_data = Fusion_3dhp(opt=opt, train=1)
            print('train_data:', len(train_data))
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                           shuffle=True, num_workers=int(opt.workers), pin_memory=True)
        
        test_data = Fusion_3dhp(opt=opt, train=0)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    
    elif opt.dataset.startswith('SKI'):
        # SKI数据集
        actions = define_actions_SKI(opt.actions)
        
        if opt.train:
            train_data = Fusion_ski(opt=opt, train=1)
            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size,
                                                           shuffle=True, num_workers=int(opt.workers), pin_memory=True)
        
        test_data = Fusion_ski(opt=opt, train=0)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size,
                                                      shuffle=False, num_workers=int(opt.workers), pin_memory=True)
    
    else:
        raise ValueError(f"Unknown dataset: {opt.dataset}. Supported datasets: h36m, 3dhp, SKI")

    # model = sgraformer(num_frame=opt.frames, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
    #                   num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, 
    #                   drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1)
    print("\n📦 Creating SGraFormer_HyperGraph model...")
    
    # 论文建议参数：k=3, depth=3, prune_ratio=0.0
    k_neighbors = 3
    prune_ratio = 0.0
    depth = 4
    
    print(f"  🔧 HyperGraph Parameters:")
    print(f"     - K-neighbors: {k_neighbors}")
    print(f"     - Prune ratio: {prune_ratio}")
    print(f"     - Depth: {depth}")
    
    model = SGraFormer_HyperGraph(
        num_frame=opt.frames,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=32,
        depth=depth,
        k_neighbors=k_neighbors,
        prune_ratio=prune_ratio,
        num_heads=8,
        mlp_ratio=2.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1
    )


    # 检查是否使用多GPU
    is_multi_gpu = torch.cuda.device_count() > 1
    
    # 先将模型移至GPU，再应用DataParallel
    model = model.to(device)
    
    if is_multi_gpu:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=CUDA_ID)

    # 计算参数量
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"\n📊 模型参数:")
    # print(f"  总参数: {total_params / 1e6:.2f}M")
    # print(f"  可训练参数: {trainable_params / 1e6:.2f}M")

    
    model_dict = model.state_dict()
    if opt.previous_dir != '':
        print('pretrained model path:', opt.previous_dir)
        model_path = opt.previous_dir
        
        # 加载预训练模型，不指定map_location，让PyTorch自动处理
        pre_dict = torch.load(model_path)
        
        # 处理状态字典，确保与当前环境（单卡/多卡）兼容
        state_dict = process_state_dict(pre_dict, is_multi_gpu)
        
        # 过滤掉不匹配的键
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
        
        # 更新模型状态字典
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        lr = opt.lr
        print(f"Successfully loaded pretrained model. Loaded {len(state_dict)}/{len(model_dict)} parameters.")

    # Check if we're only training the refinement module
    if hasattr(opt, 'train_refinement_only') and opt.train_refinement_only:
        # Get the actual model (handle DataParallel wrapping)
        actual_model = model.module if hasattr(model, 'module') else model
        
        # Step 1: Freeze all parameters
        for param in actual_model.parameters():
            param.requires_grad = False
        
        # Step 2: Unfreeze the refinement module parameters
        if hasattr(actual_model, 'temporal_refine'):
            for param in actual_model.temporal_refine.parameters():
                param.requires_grad = True

            # Step 3: Create optimizer with only refinement module parameters
            refinement_param = list(actual_model.temporal_refine.parameters())
            # Use a reduced learning rate for refinement-only training
            refinement_lr = opt.lr * getattr(opt, 'refinement_lr_ratio', 0.1)
            optimizer = optim.AdamW(refinement_param, lr=refinement_lr, weight_decay=0.01)
            print(f"  Using learning rate: {refinement_lr} (base lr: {opt.lr})")
        else:
            print("Error: Model doesn't have 'temporal_refine' attribute!")
            exit(1)
    else:
        # Original behavior: train all parameters
        all_param = []
        all_param += list(model.parameters())

        optimizer = optim.AdamW(all_param, lr=opt.lr, weight_decay=0.01)

    ## tensorboard
    # writer = SummaryWriter("runs/nin")
    writer = None
    flag = 0
    adaptive_weight = define_adaptive_weight()
    
    # ============================================================================
    # 初始化辅助3D监督模块（如果启用）
    # ============================================================================
    auxiliary_3d_loss = None
    if hasattr(opt, 'use_auxiliary_3d') and opt.use_auxiliary_3d:
        print("\n" + "=" * 80)
        print("Initializing Auxiliary 3D Supervision")
        print("=" * 80)
        print(f"  Stage: {getattr(opt, 'aux_3d_stage', 'gt')}")
        print(f"  Weight: {getattr(opt, 'aux_3d_weight', 0.01)}")
        print(f"  Position: {getattr(opt, 'aux_3d_position', 'mvf')}")
        
        embed_dim = 32 * 17  # embed_dim_ratio * num_joints
        auxiliary_3d_loss = Auxiliary3DLoss(
            embed_dim=embed_dim,
            num_joints=17,
            loss_type='mse',
            projection_type='simple'
        ).to(device)
        
        print("  ✅ Auxiliary 3D supervision initialized")
        print("=" * 80 + "\n")
    
    # ============================================================================
    lr = opt.lr
    best_epoch = 0
    for epoch in range(1, opt.nepoch + 1):
        if opt.train:
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch, writer, adaptive_weight, auxiliary_3d_loss)

        p1, p2, pck, auc = val(opt, actions, test_dataloader, model)

        if opt.train:
            save_model_epoch(opt.checkpoint, epoch, model)

            if p1 < opt.previous_best_threshold:
                best_epoch = epoch
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model)
                opt.previous_best_threshold = p1

        if opt.train == 0:
            if opt.dataset.startswith('3dhp') or opt.dataset.startswith('SKI'):
                print('pck: %.2f, auc: %.2f, p1: %.2f, p2: %.2f' % (pck, auc, p1, p2))
            else:
                print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:
            if opt.dataset.startswith('3dhp') or opt.dataset.startswith('SKI'):
                logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, pck: %.2f, auc: %.2f, best_epoch: %d, best_p1: %.2f' % 
                           (epoch, lr, loss, p1, p2, pck, auc, best_epoch, opt.previous_best_threshold))
                print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, pck: %.2f, auc: %.2f, best: %d/%.2f' % 
                     (epoch, lr, loss, p1, p2, pck, auc, best_epoch, opt.previous_best_threshold))
            else:
                logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, best_epoch: %d, best_p1: %.2f' % 
                           (epoch, lr, loss, p1, p2, best_epoch, opt.previous_best_threshold))
                print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, best: %d/%.2f' % 
                     (epoch, lr, loss, p1, p2, best_epoch, opt.previous_best_threshold))

        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay

    print(opt.checkpoint)