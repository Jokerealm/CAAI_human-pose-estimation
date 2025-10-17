import os
import torch
import logging
import random
import torch.optim as optim
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from common.loss import combined_depth_consistency_loss

from common.utils import *
from common.opt import opts
from common.h36m_dataset import Human36mDataset
from common.Mydataset import Fusion

from model.SGraFormer_loss import sgraformer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
CUDA_ID = [0]
device = torch.device("cuda")


def train(opt, actions, train_loader, model, optimizer, epoch, writer, adaptive_weight=None):
    return step('train', opt, actions, train_loader, model, optimizer, epoch, writer, adaptive_weight)


def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)


def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None, writer=None, adaptive_weight=None):
    loss_all = {'loss': AccumLoss(), 'mpjpe_loss': AccumLoss(),
                'spatial_rank_loss': AccumLoss(), 'temporal_rank_loss': AccumLoss()}
    action_error_sum = define_error_list(actions)

    if split == 'train':
        model.train()
    else:
        model.eval()

    TQDM = tqdm(enumerate(dataLoader), total=len(dataLoader), ncols=100)
    for i, data in TQDM:
        batch_cam, gt_3D, input_2D, action, subject, scale, bb_box, start, end, hops = data

        [input_2D, gt_3D, batch_cam, scale, bb_box, hops] = get_varialbe(split, [input_2D, gt_3D, batch_cam, scale, bb_box, hops])

        if split == 'train':
            output_3D = model(input_2D, hops)
        elif split == 'test':
            input_2D, output_3D = input_augmentation(input_2D, hops, model)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        if split == 'train':
            # 计算MPJPE损失
            mpjpe_loss = mpjpe_cal(output_3D, out_target)

            # 计算空间和时间关系深度一致性损失
            lambda_spatial = 20  # 空间损失权重
            lambda_temporal = 20  # 时间损失权重
            
            # 计算组合深度一致性损失
            depth_consistency_loss, spatial_rank_loss_val, temporal_rank_loss_val = combined_depth_consistency_loss(
                output_3D, out_target, lambda_spatial=lambda_spatial, lambda_temporal=lambda_temporal
            )
            
            # 总损失 = MPJPE损失 + 深度一致性损失
            loss = mpjpe_loss + depth_consistency_loss
            
        
            TQDM.set_description(f'Epoch [{epoch}/{opt.nepoch}]')
            loss_scalar = loss.mean().item()
            mpjpe_scalar = mpjpe_loss.mean().item()
            spatial_rank_scalar = spatial_rank_loss_val.mean().item()
            temporal_rank_scalar = temporal_rank_loss_val.mean().item()
            
            TQDM.set_postfix({"l": loss_scalar, "mpjpe": mpjpe_scalar,
                              "sr": spatial_rank_scalar, "tr": temporal_rank_scalar}
                             )

            N = input_2D.size(0)
            # 更新损失累加器时也要确保使用标量
            loss_all['spatial_rank_loss'].update(spatial_rank_loss_val.mean().item() * N, N)
            loss_all['temporal_rank_loss'].update(temporal_rank_loss_val.mean().item() * N, N)
            loss_all['loss'].update(loss_scalar * N, N)
            loss_all['mpjpe_loss'].update(mpjpe_scalar * N, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        elif split == 'test':
            if output_3D.shape[1] != 1:
                output_3D = output_3D[:, opt.pad].unsqueeze(1)
            output_3D[:, :, 1:, :] -= output_3D[:, :, :1, :]
            output_3D[:, :, 0, :] = 0
            action_error_sum = test_calculation(output_3D, out_target, action, action_error_sum, opt.dataset, subject)

    if split == 'train':
        return loss_all['loss'].avg
    elif split == 'test':
        p1, p2 = print_error(opt.dataset, action_error_sum, opt.train)
        return p1, p2


def input_augmentation(input_2D, hops, model):
    input_2D_non_flip = input_2D[:, 0]
    output_3D_non_flip = model(input_2D_non_flip, hops)  # 忽略验证时的对比损失

    return input_2D_non_flip, output_3D_non_flip


if __name__ == '__main__':
    opt = opts().parse()
    root_path = opt.root_path
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    root_path = opt.root_path
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

    model = sgraformer(num_frame=opt.frames, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                      num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=CUDA_ID).to(device)
    model = model.to(device)

    model_dict = model.state_dict()
    if opt.previous_dir != '':
        print('pretrained model path:', opt.previous_dir)
        model_path = opt.previous_dir

        pre_dict = torch.load(model_path)

        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    all_param = []
    lr = opt.lr
    all_param += list(model.parameters())

    optimizer = optim.AdamW(all_param, lr=lr, weight_decay=0.1)

    ## tensorboard
    # writer = SummaryWriter("runs/nin")
    writer = None
    flag = 0


    for epoch in range(1, opt.nepoch + 1):
        if opt.train:
            loss = train(opt, actions, train_dataloader, model, optimizer, epoch, writer)

        p1, p2 = val(opt, actions, test_dataloader, model)

        if opt.train:
            save_model_epoch(opt.checkpoint, epoch, model)

            if p1 < opt.previous_best_threshold:
                opt.previous_name = save_model(opt.previous_name, opt.checkpoint, epoch, p1, model)
                opt.previous_best_threshold = p1

        if opt.train == 0:
            print('p1: %.2f, p2: %.2f' % (p1, p2))
            break
        else:
            logging.info('epoch: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))
            print('e: %d, lr: %.7f, loss: %.4f, p1: %.2f, p2: %.2f' % (epoch, lr, loss, p1, p2))

        if epoch % opt.large_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay_large
                lr *= opt.lr_decay_large
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opt.lr_decay
                lr *= opt.lr_decay

    print(opt.checkpoint)