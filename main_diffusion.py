import os
import torch
import logging
import random
import torch.optim as optim
from tqdm import tqdm

from common.utils import *
from common.opt import opts
from common.h36m_dataset import Human36mDataset
from common.Mydataset import Fusion

from model.SGraDiFormer import sgraformer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
CUDA_ID = [0, 1, 2]
device = torch.device("cuda")

# 预训练模型路径
PRETRAINED_PATH = '/data16t/guanz/coding/SGraFormer/checkpoint/baseline/model_34_2781.pth'

def train(opt, actions, train_loader, model, optimizer, epoch, writer):
    return step('train', opt, actions, train_loader, model, optimizer, epoch, writer)

def val(opt, actions, val_loader, model):
    with torch.no_grad():
        return step('test', opt, actions, val_loader, model)

def step(split, opt, actions, dataLoader, model, optimizer=None, epoch=None, writer=None):
    loss_all = {'loss': AccumLoss()}
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
            # 使用diffusion精炼模型进行训练
            output_3D = model(input_2D, hops, use_diffusion=True, diffusion_steps=250)
        elif split == 'test':
            input_2D, output_3D = input_augmentation(input_2D, hops, model)

        out_target = gt_3D.clone()
        out_target[:, :, 0] = 0

        if split == 'train':
            loss = mpjpe_cal(output_3D, out_target)

            TQDM.set_description(f'Epoch [{epoch}/{opt.nepoch}]')
            TQDM.set_postfix({"l": loss.item()})

            N = input_2D.size(0)
            loss_all['loss'].update(loss.detach().cpu().numpy() * N, N)

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

# 添加input_augmentation函数以支持验证
def input_augmentation(input_2D, hops, model):
    input_2D_non_flip = input_2D[:, 0]
    output_3D_non_flip = model(input_2D_non_flip, hops, use_diffusion=True)
    
    return input_2D_non_flip, output_3D_non_flip

def main():
    # Parse arguments
    opt = opts().parse()
    
    # 设置随机种子
    opt.manualSeed = 1
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    # 设置日志
    if opt.train:
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S',
                            filename=os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
    
    # 准备数据集
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
    
    # 创建模型
    model = sgraformer(num_frame=opt.frames, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                      num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)
    
    # 多GPU设置
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=CUDA_ID).to(device)
    model = model.to(device)
    
    # 加载预训练模型
    model_dict = model.state_dict()
    print('pretrained model path:', PRETRAINED_PATH)
    
    pre_dict = torch.load(PRETRAINED_PATH)
    
    # 过滤参数并加载
    model_dict = model.state_dict()
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    print(f"Successfully loaded {len(state_dict)} parameters from checkpoint")
    
    # 冻结原始模型参数，只训练diffusion部分
    print("Freezing original SGraFormer parameters...")
    
    # 手动初始化diffusion_refiner，确保它在收集参数前被创建
    print("Initializing diffusion refiner...")
    dummy_input = torch.zeros(1, opt.frames, 17, 2).to(device)  # 假的2D输入
    dummy_x_2d = torch.zeros(1, 1, opt.frames, 17, 2).to(device)  # 假的多视图2D输入
    with torch.no_grad():
        # 调用模型的forward方法来触发diffusion_refiner的初始化
        # 这里假设model.forward方法接受x_2d和use_diffusion=True参数
        if hasattr(model, 'module'):  # 处理DataParallel包装的情况
            model.module.forward(dummy_input, dummy_x_2d, use_diffusion=True)
        else:
            model.forward(dummy_input, dummy_x_2d, use_diffusion=True)
    
    # 现在收集diffusion相关参数
    diffusion_params = []
    has_diffusion_params = False
    
    for name, param in model.named_parameters():
        # 只让diffusion相关参数可训练
        if 'diffusion_refiner' in name:
            param.requires_grad = True
            diffusion_params.append(param)
            print(f"Trainable parameter: {name}")
            has_diffusion_params = True
        else:
            param.requires_grad = False
    
    # 如果仍然没有找到diffusion参数，发出警告
    if not has_diffusion_params:
        print("Warning: No diffusion_refiner parameters found! Check if the attribute name is correct.")
    
    # 创建优化器，只优化diffusion参数
    optimizer = optim.AdamW(diffusion_params, lr=opt.lr, weight_decay=0.1)
    
    # tensorboard
    writer = None
    flag = 0
    opt.previous_best_threshold = float('inf')
    opt.previous_name = ''
    
    # 训练循环
    lr = opt.lr
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

if __name__ == '__main__':
    main()