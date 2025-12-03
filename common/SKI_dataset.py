import torch
import h5py
import numpy as np
import torch.utils.data as data

from common.opt import opts


class ChunkedGenerator:

    def __init__(self, batch_size, poses_3d, poses_2d,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug=False,
                 out_all=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))

        pairs = []
        self.saved_index = {}
        start_index = 0

        for key in poses_2d.keys():
            assert poses_3d is None or poses_2d[key].shape[0] == poses_3d[key].shape[0]
            n_chunks = (poses_2d[key].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[key].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            reverse_augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            keys = np.tile(np.array(key).reshape([1, 1]), (len(bounds - 1), 1))
            pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, reverse_augment_vector))
            if reverse_aug:
                pairs += list(zip(keys, bounds[:-1], bounds[1:], augment_vector, ~reverse_augment_vector))
            if augment:
                if reverse_aug:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, ~reverse_augment_vector))
                else:
                    pairs += list(zip(keys, bounds[:-1], bounds[1:], ~augment_vector, reverse_augment_vector))
            end_index = start_index + poses_3d[key].shape[0]
            self.saved_index[key] = [start_index, end_index]
            start_index = start_index + poses_3d[key].shape[0]

        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[key].shape[-2], poses_3d[key].shape[-1]))
        self.batch_2d = np.empty(
            (batch_size, chunk_length + 2 * pad, poses_2d[key].shape[-3], poses_2d[key].shape[-2],
             poses_2d[key].shape[-1]))  # 创建空的矩阵，等会装数据

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.state = None

        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        """
        :return:当前状态，seq_name, start_3d, end_3d, flip, reverse
        """
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    # seq_name, start_3d, end_3d, flip, reverse
    def get_batch(self, seq_i, start_3d, end_3d):
        seq = seq_i
        seq_name = seq[0]
        start_2d = start_3d - self.pad - self.causal_shift  # 开始位置
        end_2d = end_3d + self.pad - self.causal_shift

        seq_2d = self.poses_2d[seq_name].copy()
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d

        if pad_left_2d != 0 or pad_right_2d != 0:
            self.batch_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0), (0, 0)),
                                   'edge')
        else:
            self.batch_2d = seq_2d[low_2d:high_2d]

        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_name].copy()
            if self.out_all:
                low_3d = low_2d
                high_3d = high_2d
                pad_left_3d = pad_left_2d
                pad_right_3d = pad_right_2d
            else:
                low_3d = max(start_3d, 0)
                high_3d = min(end_3d, seq_3d.shape[0])
                pad_left_3d = low_3d - start_3d
                pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                self.batch_3d = np.pad(seq_3d[low_3d:high_3d],
                                       ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                self.batch_3d = seq_3d[low_3d:high_3d]
        # self.batch_3d = self.batch_3d[13]
        # print(self.batch_2d.shape)

        return self.batch_3d.copy(), self.batch_2d.copy(), seq, low_2d, high_2d


class Fusion_ski(data.Dataset):
    def __init__(self, opt, train=0):

        self.hop1 = torch.tensor([[0, 1,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0]])

        self.hop2 = torch.tensor([[0,	0,	1,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	1,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	1,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	1,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	0,	0,	1],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0]])

        self.hop3 = torch.tensor([[0,	0,	0,	1,	0,	0,	1,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	1,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	1,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0]])

        self.hop4 = torch.tensor([[0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	1,	0,	0],
                            [0,	0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1],
                            [0,	0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	1,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	0,	1,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0,	1,	0,	0],
                            [0,	1,	0,	0,	1,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0],
                            [1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	0,	0,	0],
                            [0,	0,	0,	0,	0,	0,	0,	1,	0,	1,	0,	1,	0,	0,	0,	0,	0]])
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad

        if self.train:
            self.dataset = np.load('dataset/SKI-pose/SKI_train.npz', allow_pickle=True)['metadata'][()]
        else:
            self.dataset = np.load('dataset/SKI-pose/SKI_test.npz', allow_pickle=True)['metadata'][()]

        if self.train:
            self.prepare_data()
            self.poses_train, self.poses_train_2d = self.fetch()
            self.generator = ChunkedGenerator(opt.batch_size // opt.stride, self.poses_train,
                                              self.poses_train_2d, self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              out_all=opt.out_all)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.prepare_data()
            self.poses_train, self.poses_train_2d = self.fetch()
            self.generator = ChunkedGenerator(opt.batch_size // opt.stride, self.poses_train,
                                              self.poses_train_2d, self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              out_all=opt.out_all)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))

    def prepare_data(self):
        cam_idx = [0, 1, 2, 3]
        for seq in self.dataset.keys():
            positions_2d_pairs = []
            positions_3d_pairs = []
            min_length = min([len(self.dataset[seq][cam_idx[0]]['2D']), len(self.dataset[seq][cam_idx[1]]['2D']),
                             len(self.dataset[seq][cam_idx[2]]['2D']), len(self.dataset[seq][cam_idx[3]]['2D'])])
            positions_2d_pairs.append([self.dataset[seq][cam_idx[0]]['2D'][:min_length],
                                       self.dataset[seq][cam_idx[1]]['2D'][:min_length],
                                       self.dataset[seq][cam_idx[2]]['2D'][:min_length],
                                       self.dataset[seq][cam_idx[3]]['2D'][:min_length]])
            positions_3d_pairs.append(self.dataset[seq][cam_idx[0]]['3D'][:min_length])
            # print(len(positions_3d_pairs))
            self.dataset[seq]['Pose_2D'] = np.array(positions_2d_pairs).squeeze(0).transpose((1, 0, 2, 3))
            self.dataset[seq]['Pose_3D'] = np.array(positions_3d_pairs).squeeze(0)
            self.dataset[seq]['frame'] = np.array(self.dataset[seq][cam_idx[0]]['frame'])

    def hop_normalize(self, x1, x2, x3, x4):
        x1 = x1 / torch.sum(x1, dim=1)
        x2 = x2 / torch.sum(x1, dim=1)
        x3 = x3 / torch.sum(x1, dim=1)
        x4 = x4 / torch.sum(x1, dim=1)
        return torch.cat((x1.unsqueeze(0), x2.unsqueeze(0), x3.unsqueeze(0), x4.unsqueeze(0)), dim=0)


    def fetch(self):

        out_poses_3d = {}
        out_poses_2d = {}
        for s in self.dataset.keys():
            poses_2d = self.dataset[s]['Pose_2D']
            out_poses_2d[s] = poses_2d

            poses_3d = self.dataset[s]['Pose_3D']
            out_poses_3d[s] = poses_3d

        return out_poses_3d, out_poses_2d

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]
        gt_3D, input_2D, seq, low_2d, high_2d = self.generator.get_batch(seq_name, start_3d, end_3d)
        hops = self.hop_normalize(self.hop1, self.hop2, self.hop3, self.hop4)

        return gt_3D, input_2D, seq, low_2d, high_2d, hops

