import numpy as np
import copy

from common.h36m_dataset import Skeleton, MocapDataset


mpi_inf_3dhp_skeleton = Skeleton(parents=[],
                         joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                         joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

class Mpi_inf_3dhp_Dataset(MocapDataset):
    def __init__(self, path, opt):
        super().__init__(fps=50, skeleton=mpi_inf_3dhp_skeleton)
        self.train_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
        self.test_list = ['S7', 'S8']

        # Load serialized dataset
        data = np.load(path, allow_pickle=True)['positions_3d'].item()

        self._data = {}
        for subject, actions in data.items():
            self._data[subject] = {}
            for action_name, positions in actions.items():
                self._data[subject][action_name] = {
                    'positions': positions,
                }