"""
main_hnn.py - 使用SGraFormer_HyperGraph模型的训练脚本
基于main.py，仅替换模型为动态超图版本
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
CUDA_ID = [0,1]

import torch
import logging
import random
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch.backends.cudnn as cudnn

from common.utils import *
from common.utils_3dhp import define_actions_3dhp, define_error_list_3dhp
from common.opt import opts
from common.h36m_dataset import Human36mDataset
from common.Mydataset import Fusion
from common.MPI_dataset_hops import Fusion_3dhp
from common.SKI_dataset import Fusion_ski

# ============================================================================
# 使用HyperGraph模型
# ============================================================================
from model.SGraFormer_HyperGraph import SGraFormer_HyperGraph
# ============================================================================

from common.loss import loss_limb_var

device = torch.device("cuda")


def set_random_seed(seed):
    """设置所有相关的随机种子以确保实验可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def train(opt, actions, train_loader, model, optimizer, epoch):
    """训练一个epoch"""
    return step('train', opt, actions, train_loader, model, optimizer, epoch)


def val(opt, actions, val_loader, model):
    """验证一个epoch"""
    with torch.no_grad():
        if opt.dataset == 'SKI':
            from common.utils_ski import define_actions_SKI
            test_actions = define_actions_SKI(opt.actions, train=False)
        elif opt.dataset.startswith('3dhp'):
            from common.utils_3dhp import define_actions_3dhp
            test_actions = define_actions_3dhp(opt.actions, train=False)
        else:
            test_actions = actions
        
        return step('test', opt, test_actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None):
    
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
            gt_3D, input_2D, action, start, end, hops = data
            batch_cam = None
            subject = None
            scale = None
            bb_box = None
            [input_2D, gt_3D, hops] = get_varialbe(split, [input_2D, gt_3D, hops])
        else:
            batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, start, end, hops = data
            [input_2D, gt_3D, batch_cam, scale, bb_box, hops] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box, hops])

        if split == 'train':
            # ============ 训练模式 ============
            
            # Prepare target
            out_target = gt_3D.clone()
            out_target[:, :, 0] = 0
            
            # Forward pass (hops参数保留兼容性，但HyperGraph模型不使用)
            output_3D = model(input_2D, hops)
            
            # 计算loss
            loss_mpjpe = mpjpe_cal(output_3D, out_target)
            # loss_bone_var = loss_limb_var(output_3D)
            # main_loss = 0.01*loss_mpjpe + 0.99 * loss_bone_var
            loss = loss_mpjpe
            
            # 更新进度条
            TQDM.set_description(f'Epoch {epoch}/{opt.nepoch}')
            TQDM.set_postfix({"loss": f"{loss.item():.4f}"})

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
        input_2D_non_flip = input_2D[:, 0]
    elif input_2D.dim() == 5:
        input_2D_non_flip = input_2D
    else:
        input_2D_non_flip = input_2D.unsqueeze(1)
    
    output_3D_non_flip = model(input_2D_non_flip, hops)

    return input_2D_non_flip, output_3D_non_flip


def process_state_dict(state_dict, is_multi_gpu_model):
    """处理模型状态字典，确保在单卡和多卡环境之间的兼容性"""
    new_state_dict = {}
    
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    for k, v in state_dict.items():
        if is_multi_gpu_model:
            if not k.startswith('module.'):
                new_state_dict[f'module.{k}'] = v
            else:
                new_state_dict[k] = v
        else:
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
                            filename=os.path.join(opt.checkpoint, 'train_hnn.log'), level=logging.INFO)

    print("\n" + "=" * 80)
    print("🚀 SGraFormer_HyperGraph Training")
    print("=" * 80)

    # 根据数据集类型加载不同的数据和动作定义
    if opt.dataset == 'h36m':
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

    # ============================================================================
    # 创建HyperGraph模型
    # ============================================================================
    print("\n📦 Creating SGraFormer_HyperGraph model...")
    
    # 论文建议参数：k=3, depth=3, prune_ratio=0.0
    k_neighbors = 3
    prune_ratio = 0.0
    depth = 6
    
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
    # ============================================================================

    # 检查是否使用多GPU
    is_multi_gpu = torch.cuda.device_count() > 1
    
    # 先将模型移至GPU，再应用DataParallel
    model = model.to(device)
    
    if is_multi_gpu:
        print(f"  🔧 Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model, device_ids=CUDA_ID)
    else:
        print(f"  🔧 Using single GPU")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    
    # Load pretrained weights
    if opt.previous_dir != '':
        print(f'\n📥 Loading pretrained model: {opt.previous_dir}')
        try:
            pre_dict = torch.load(opt.previous_dir)
            state_dict = process_state_dict(pre_dict, is_multi_gpu)
            model_dict = model.state_dict()
            
            # 尝试加载兼容的参数
            compatible_dict = {}
            incompatible_keys = []
            
            for k, v in state_dict.items():
                if k in model_dict.keys():
                    if v.shape == model_dict[k].shape:
                        compatible_dict[k] = v
                    else:
                        incompatible_keys.append(f"{k}: {v.shape} vs {model_dict[k].shape}")
                else:
                    incompatible_keys.append(f"{k}: not in model")
            
            model_dict.update(compatible_dict)
            model.load_state_dict(model_dict, strict=False)
            
            print(f'  ✅ Loaded {len(compatible_dict)}/{len(model_dict)} compatible parameters')
            if incompatible_keys:
                print(f'  ⚠️  Skipped {len(incompatible_keys)} incompatible parameters')
                if len(incompatible_keys) <= 10:
                    for key in incompatible_keys:
                        print(f'     - {key}')
        except Exception as e:
            print(f'  ❌ Error loading pretrained model: {e}')
            print(f'  ℹ️  Starting from scratch')

    # Setup optimizer
    lr = opt.lr
    all_param = list(model.parameters())
    optimizer = optim.AdamW(all_param, lr=lr, weight_decay=0.1)

    print(f"\n⚙️  Optimizer: AdamW")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: 0.1")
    
    print("\n" + "=" * 80)
    print("🎯 Starting Training...")
    print("=" * 80 + "\n")

    best_epoch = 0
    for epoch in range(1, opt.nepoch + 1):
        if opt.train:
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch)

        p1, p2, pck, auc = val(opt, actions, test_dataloader, model)

        if opt.train:
            save_model_epoch(opt.checkpoint, epoch, model)

            if p1 < opt.previous_best_threshold:
                best_epoch = epoch
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model)
                opt.previous_best_threshold = p1
                print(f"  💾 Best model saved (epoch {epoch})")

        if opt.train == 0:
            if opt.dataset.startswith('3dhp') or opt.dataset.startswith('SKI'):
                print('pck: %.2f, auc: %.2f, p1: %.2f, p2: %.2f' % (pck, auc, p1, p2))
            else:
                print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:
            if opt.dataset.startswith('3dhp') or opt.dataset.startswith('SKI'):
                log_msg = 'epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, pck: %.2f, auc: %.2f, best_epoch: %d, best_p1: %.2f' % \
                          (epoch, lr, loss, p1, p2, pck, auc, best_epoch, opt.previous_best_threshold)
                print_msg = 'e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, pck: %.2f, auc: %.2f, best: %d/%.2f' % \
                           (epoch, lr, loss, p1, p2, pck, auc, best_epoch, opt.previous_best_threshold)
            else:
                log_msg = 'epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, best_epoch: %d, best_p1: %.2f' % \
                          (epoch, lr, loss, p1, p2, best_epoch, opt.previous_best_threshold)
                print_msg = 'e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f, best: %d/%.2f' % \
                           (epoch, lr, loss, p1, p2, best_epoch, opt.previous_best_threshold)
            
            logging.info(log_msg)
            print(print_msg)

        # 手动更新学习率（与main.py一致）
        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
            lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
            lr *= opt.lr_decay

    print("\n" + "=" * 80)
    print("✅ Training Completed!")
    print(f"📁 Checkpoint: {opt.checkpoint}")
    print(f"🏆 Best Epoch: {best_epoch}")
    print(f"📊 Best P1: {opt.previous_best_threshold:.2f}")
    print("=" * 80)
