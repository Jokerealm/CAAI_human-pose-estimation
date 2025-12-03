import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import os

def pck(predicted, target, debug=False): 
    assert predicted.shape == target.shape 
    threshold = 150.0 / 1000 

    batch_size = predicted.shape[0] * 1.0
    frame_num = predicted.shape[1] * 1.0 
    joints_num = predicted.shape[-2] * 1.0 
    
    dis = torch.norm(predicted - target, dim=len(target.shape)-1) 

    if debug:
        print(f"\n[PCK Debug]")
        print(f"  Shape: {predicted.shape}")
        print(f"  Threshold: {threshold * 1000:.2f}mm")
        print(f"  Distance stats: min={dis.min().item()*1000:.2f}mm, max={dis.max().item()*1000:.2f}mm, mean={dis.mean().item()*1000:.2f}mm")
        print(f"  Joints within threshold: {(dis < threshold).sum().item()} / {dis.numel()}")
    
    t = torch.Tensor([threshold]).cuda()
    out = (dis < t).float() * 1 
    pck = out.sum() / batch_size / joints_num / frame_num 

    return pck 


def auc(predicted, target): 
    assert predicted.shape == target.shape 
    dis = torch.norm(predicted - target, dim=len(target.shape)-1) 
    outall = 0 
    threshold = 150 

    batch_size = predicted.shape[0] * 1.0
    frame_num = predicted.shape[1] * 1.0 
    joints_num = predicted.shape[-2] * 1.0 


    for i in range(threshold): 
        t = torch.Tensor([float(i)/1000])
        t = t.cuda()
        out = (dis < t).float() * 1 
        outall += out.sum() / batch_size / joints_num / frame_num 

    outall = outall / threshold 
    
    return outall


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):

    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def test_calculation(predicted, target, action, error_sum, data_type, subject):
    # 确保error_sum中包含pck和auc字段
    if isinstance(error_sum, dict):
        for action_name, metrics in error_sum.items():
            if isinstance(metrics, dict):
                if 'pck' not in metrics:
                    metrics['pck'] = AccumLoss()
                if 'auc' not in metrics:
                    metrics['auc'] = AccumLoss()
    
    # 处理Tensor类型的action (SKI数据集)
    if torch.is_tensor(action):
        action_name = str(action[0].item())
        # 计算p1和p2误差
        error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)
        
        # 使用提供的pck和auc函数计算指标
        try:
            # 保留数据在当前设备上，pck和auc函数内部会处理设备问题
            # 计算pck和auc
            pck_value = pck(predicted, target)
            auc_value = auc(predicted, target)
            
            # 更新统计信息
            error_sum[action_name]['pck'].update(pck_value.item(), 1)
            error_sum[action_name]['auc'].update(auc_value.item(), 1)
        except Exception as e:
            print(f"计算PCK/AUC时出错: {e}")
    else:
        # 原始字符串列表处理逻辑
        # 计算p1和p2误差
        error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)
        
        # 对整个批次计算pck和auc
        try:
            # 保留数据在当前设备上，pck和auc函数内部会处理设备问题
            
            # 计算pck和auc
            pck_value = pck(predicted, target)
            auc_value = auc(predicted, target)
            
            # 对每个动作更新pck和auc
            if len(set(list(action))) == 1:
                end_index = action[0].find(' ')
                if end_index != -1:
                    action_name = action[0][:end_index]
                else:
                    action_name = action[0]
                error_sum[action_name]['pck'].update(pck_value.item(), 1)
                error_sum[action_name]['auc'].update(auc_value.item(), 1)
            else:
                # 对于多个不同的动作，只更新一次（避免重复累加）
                # 收集所有唯一的action名称
                unique_actions = set()
                for i in range(len(action)):
                    end_index = action[i].find(' ')
                    if end_index != -1:
                        action_name = action[i][:end_index]
                    else:
                        action_name = action[i]
                    unique_actions.add(action_name)
                
                # 对每个唯一的action只更新一次
                for action_name in unique_actions:
                    if action_name in error_sum and isinstance(error_sum[action_name], dict):
                        if 'pck' in error_sum[action_name] and 'auc' in error_sum[action_name]:
                            error_sum[action_name]['pck'].update(pck_value.item(), 1)
                            error_sum[action_name]['auc'].update(auc_value.item(), 1)
        except Exception as e:
            print(f"计算PCK/AUC时出错: {e}")

    return error_sum

# 原有的calculate_pck_auc函数已被test_calculation中的新逻辑替代


def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    # 处理 Tensor 类型的 action
    if torch.is_tensor(action):
        # 对于 SKI 数据集，假设 action 是一个包含键名的张量
        # 获取第一个元素作为 action_name
        action_name = str(action[0].item())
        action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * num, num)
    else:
        # 原始字符串列表处理逻辑
        if len(set(list(action))) == 1:
            end_index = action[0].find(' ')
            if end_index != -1:
                action_name = action[0][:end_index]
            else:
                action_name = action[0]

            action_error_sum[action_name]['p1'].update(torch.mean(dist).item() * num, num)
        else:
            for i in range(num):
                end_index = action[i].find(' ')
                if end_index != -1:
                    action_name = action[i][:end_index]
                else:
                    action_name = action[i]

                action_error_sum[action_name]['p1'].update(dist[i].item(), 1)

    return action_error_sum


def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)

    # 处理 Tensor 类型的 action
    if torch.is_tensor(action):
        # 对于 SKI 数据集，假设 action 是一个包含键名的张量
        # 获取第一个元素作为 action_name
        action_name = str(action[0].item())
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        # 原始字符串列表处理逻辑
        if len(set(list(action))) == 1:
            end_index = action[0].find(' ')
            if end_index != -1:
                action_name = action[0][:end_index]
            else:
                action_name = action[0]
            action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
        else:
            for i in range(num):
                end_index = action[i].find(' ')
                if end_index != -1:
                    action_name = action[i][:end_index]
                else:
                    action_name = action[i]
                action_error_sum[action_name]['p2'].update(np.mean(dist), 1)

    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def define_actions(action):
    actions = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Photo", "Posing", "Purchases",
               "Sitting", "SittingDown", "Smoking", "Waiting",
               "WalkDog", "Walking", "WalkTogether"]

    if action == "All" or action == "all" or action == '*':
        return actions

    if not action in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]


def define_actions_SKI(action):
    """
    定义 SKI 数据集的 actions
    基于数据集文件中的实际动作编号和结构
    
    数据集特点：
    - 每个动作包含6个相机视角
    - 每个动作的帧数各不相同（如动作103:145帧、124:143帧、202:133帧）
    - 每个相机视角数据包含'2D'、'3D'和'frame'三个键
    - 2D和3D数据均为列表类型，每个元素为numpy.ndarray格式
    - 每个动作有17个关节点
    """
    # SKI 数据集的训练动作列表 - 基于实际数据统计
    train_actions = ["103", "124", "202", "221", "302", "110", "115", "207", "214", "309"]
    # SKI 数据集的测试动作列表
    test_actions = ["412", "405"]
    # 所有动作的完整列表
    all_actions = train_actions + test_actions
    # 返回所有动作
    if action == "All" or action == "all" or action == '*':
        return all_actions
    # 返回仅训练动作
    elif action == "train" or action == "Train":
        return train_actions
    # 返回仅测试动作
    elif action == "test" or action == "Test":
        return test_actions
    # 检查单个动作是否有效
    if str(action) not in all_actions:
        raise ValueError(f"Unrecognized SKI action: {action}. Available actions: {', '.join(all_actions)}")
    return [str(action)]
def get_ski_action_info(action_id):
    """
    获取SKI数据集特定动作的基本信息
    
    参数:
    - action_id: 动作编号字符串
    
    返回:
    - 字典，包含动作的相机数量、大致帧数等信息
    """
    # 基于数据分析得到的帧数统计
    action_frame_stats = {
        "103": 145,
        "124": 143,
        "202": 133,
    }
    
    return {
        "num_cameras": 6,  # 所有动作都有6个相机视角
        "approx_frames": action_frame_stats.get(action_id, "未知"),
        "num_joints": 17,  # 每个动作有17个关节点
    }


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]:
                          {'p1': AccumLoss(), 'p2': AccumLoss(), 'pck': AccumLoss(), 'auc': AccumLoss()}
                      for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def get_varialbe(split, target):
    num = len(target)
    var = []
    # 检查是否可以使用CUDA
    use_cuda = torch.cuda.is_available()
    
    for i in range(num):
        if split == 'train':
            temp = Variable(target[i], requires_grad=False).contiguous()
        else:
            temp = Variable(target[i]).contiguous()
        
        # 根据设备类型选择合适的tensor类型
        if use_cuda:
            temp = temp.cuda().type(torch.cuda.FloatTensor)
        else:
            temp = temp.type(torch.FloatTensor)
            
        var.append(temp)

    return var


def print_error(data_type, action_error_sum, is_train):
    # 调用更新后的print_error_action函数，获取所有四个指标的值
    mean_error_p1, mean_error_p2, pck, auc = 0, 0, 0, 0
    mean_error_p1, mean_error_p2, pck, auc = print_error_action(action_error_sum, is_train)
    return mean_error_p1, mean_error_p2, pck, auc


def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0, 'pck': 0.0, 'auc': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss(), 'pck': AccumLoss(), 'auc': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^10} {3:=^10} {4:=^10}".format("Action", "p#1 mm", "p#2 mm", "PCK", "AUC"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        # 获取并更新p1和p2值
        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)
        
        mean_error_each['pck'] = action_error_sum[action]['pck'].avg * 100.0
        mean_error_all['pck'].update(mean_error_each['pck'], 1)

        mean_error_each['auc'] = action_error_sum[action]['auc'].avg * 100.0
        mean_error_all['auc'].update(mean_error_each['auc'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f} {2:>10.2f} {3:>10.2f}".format(
                    mean_error_each['p1'], mean_error_each['p2'], 
                    mean_error_each['pck'], mean_error_each['auc']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}".format(
            "Average", mean_error_all['p1'].avg, mean_error_all['p2'].avg,
            mean_error_all['pck'].avg, mean_error_all['auc'].avg))

    # 返回所有四个指标的值
    return mean_error_all['p1'].avg, mean_error_all['p2'].avg, mean_error_all['pck'].avg, mean_error_all['auc'].avg


def save_model(previous_name, save_dir, epoch, data_threshold, model):
    """
    保存模型的最佳检查点，确保保存时移除'module.'前缀，
    使单卡和多卡训练的模型可以互相加载
    """
    if os.path.exists(previous_name):
        os.remove(previous_name)
    
    # 获取模型状态字典
    state_dict = model.state_dict()
    
    # 保存前移除'module.'前缀，确保检查点在单卡环境中也能正确加载
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除'module.'前缀
        else:
            new_state_dict[k] = v
    
    # 保存处理后的状态字典
    model_path = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    torch.save(new_state_dict, model_path)
    
    return model_path


def save_model_epoch(save_dir, epoch, model):
    """
    保存每个epoch的模型检查点，确保保存时移除'module.'前缀，
    使单卡和多卡训练的模型可以互相加载
    """
    # 获取模型状态字典
    state_dict = model.state_dict()
    
    # 保存前移除'module.'前缀，确保检查点在单卡环境中也能正确加载
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 移除'module.'前缀
        else:
            new_state_dict[k] = v
    
    # 保存处理后的状态字典
    torch.save(new_state_dict, '%s/epoch_%d.pth' % (save_dir, epoch))

def define_adaptive_weight():
    train_list = ['S1', 'S5', 'S6', 'S7', 'S8']
    actions = ['Directions 1', 'Directions', 'Discussion 1', 'Discussion', 'Eating 2', 'Eating', 'Greeting 1',
               'Greeting', 'Phoning 1', 'Phoning', 'Posing 1', 'Posing', 'Purchases 1', 'Purchases', 'Sitting 1',
               'Sitting 2', 'SittingDown 2', 'SittingDown', 'Smoking 1', 'Smoking', 'Photo 1', 'Photo', 'Waiting 1',
               'Waiting', 'Walking 1', 'Walking', 'WalkDog 1', 'WalkDog', 'WalkTogether 1', 'WalkTogether',
               'Directions 2', 'Discussion 2', 'Discussion 3', 'Eating 1', 'Greeting 2', 'Photo 2', 'Sitting',
               'SittingDown 1', 'Waiting 2', 'Posing 2', 'Waiting 3', 'Phoning 2', 'Walking 2', 'WalkTogether 2']
    act_dict = {}
    for act in actions:
        act_dict[act] = np.ones(6400)

    adaptive_weight = {}
    for sbj in train_list:
        adaptive_weight[sbj] = act_dict

    return adaptive_weight


def get_adaptive_weight(adaptive_weight, subject, action, start, end):
    N = len(subject)
    weights = torch.zeros((1, N))
    for idx in range(N):
        weights[0, idx] = sum(adaptive_weight[subject[idx]][action[idx]][start[idx]:end[idx]]) / (end[idx] - start[idx])
    return weights

def fil_ex(se, min=0.1, max=0.9):
    se_num = se.shape[0]
    se_sort = np.sort(se)
    se_sort_file = se_sort[int(se_num * min):int(se_num * max)]
    mean, var = np.mean(se_sort_file), np.sqrt(np.var(se_sort_file))

    if var < 2:  ##threshold
        return mean, var
    else:
        return mean, torch.tensor(1e-10)
    
def update_adaptive_weight(adaptive_weight, subject, action, start, end, loss_batch):
    N = len(subject)
    loss_batch_for_mean_var = loss_batch.detach().cpu().numpy()
    mean, var = fil_ex(loss_batch_for_mean_var, min=0.05,
                       max=0.95)  ## Calculate the mean and variance of the loss between 0.05-0.95 after sorting
    for idx in range(N):
        temp_weight = torch.exp(-(loss_batch[idx] - mean) * var).detach().cpu().numpy()
        adaptive_weight[subject[idx]][action[idx]][start[idx]:end[idx]] *= temp_weight

    return adaptive_weight, mean, var

def get_adaptive_weight(adaptive_weight, subject, action, start, end):
    N = len(subject)
    weights = torch.zeros((1, N))
    for idx in range(N):
        weights[0, idx] = sum(adaptive_weight[subject[idx]][action[idx]][start[idx]:end[idx]]) / (end[idx] - start[idx])
    return weights


