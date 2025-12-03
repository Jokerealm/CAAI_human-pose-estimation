import torch
from torch.autograd import Variable
import numpy as np
import os
import hashlib

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
    has_cuda = torch.cuda.is_available()
    
    if split == 'train':
        for i in range(num):
            if has_cuda:
                temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            else:
                temp = Variable(target[i], requires_grad=False).contiguous().type(torch.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            if has_cuda:
                temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            else:
                temp = Variable(target[i]).contiguous().type(torch.FloatTensor)
            var.append(temp)
    return var


def print_error(data_type, action_error_sum, is_train, opt=None):
    """
    打印错误信息，返回p1, p2, pck, auc四个指标
    """
    mean_error_p1, mean_error_p2, pck, auc = print_error_action(action_error_sum, is_train, data_type)
    return mean_error_p1, mean_error_p2, pck, auc


def print_error_action(action_error_sum, is_train, data_type):
    """
    计算并打印每个动作的错误
    注意：pck和auc函数返回的是0-1之间的比例值，需要乘以100转换为百分比
    """
    mean_error_each = {'p1': 0.0, 'p2': 0.0, 'pck': 0.0, 'auc': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss(), 'pck': AccumLoss(), 'auc': AccumLoss()}

    if not is_train:
        print("{0:=^12} {1:=^10} {2:=^8} {3:=^8} {4:=^8}".format("Action", "p#1 mm", "p#2 mm", "PCK%", "AUC%"))

    for action, value in action_error_sum.items():
        if not is_train:
            print("{0:<12} ".format(action), end="")
            
        # p1和p2已经是mm单位
        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        # pck和auc是0-1之间的比例，需要乘以100转换为百分比
        try:
            mean_error_each['pck'] = action_error_sum[action]['pck'].avg * 100.0
            mean_error_all['pck'].update(mean_error_each['pck'], 1)
        except (KeyError, AttributeError):
            mean_error_each['pck'] = 0.0
            mean_error_all['pck'].update(mean_error_each['pck'], 1)

        try:
            mean_error_each['auc'] = action_error_sum[action]['auc'].avg * 100.0
            mean_error_all['auc'].update(mean_error_each['auc'], 1)
        except (KeyError, AttributeError):
            mean_error_each['auc'] = 0.0
            mean_error_all['auc'].update(mean_error_each['auc'], 1)

        if not is_train:
            print("{0:>6.2f} {1:>10.2f} {2:>10.2f} {3:>10.2f}".format(
                mean_error_each['p1'], mean_error_each['p2'], 
                mean_error_each['pck'], mean_error_each['auc']))

    if not is_train:
        print("{0:<12} {1:>6.2f} {2:>10.2f} {3:>10.2f} {4:>10.2f}".format("Average", 
            mean_error_all['p1'].avg, mean_error_all['p2'].avg,
            mean_error_all['pck'].avg, mean_error_all['auc'].avg))

    return mean_error_all['p1'].avg, mean_error_all['p2'].avg, \
           mean_error_all['pck'].avg, mean_error_all['auc'].avg


def define_error_list(actions):
    """
    为每个动作定义错误累加器
    """
    error_sum = {}
    error_sum.update({actions[i]: 
        {'p1': AccumLoss(), 'p2': AccumLoss(), 'pck': AccumLoss(), 'auc': AccumLoss()} 
        for i in range(len(actions))})
    return error_sum


def define_actions_SKI(action, train):
    """
    定义SKI数据集的动作列表
    """
    if train:
        actions = ["SKI"]  # 训练集动作
    else:
        actions = ["SKI"]  # 测试集动作
    return actions


def save_model(previous_name, save_dir, epoch, data_threshold, model):
    """
    保存模型
    """
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(), '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))
    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)

    return previous_name


def deterministic_random(min_value, max_value, data):
    """
    确定性随机数生成
    """
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


# 从utils.py导入需要的函数
from common.utils import (
    pck, auc, mpjpe_cal, test_calculation,
    mpjpe_by_action_p1, mpjpe_by_action_p2
)
