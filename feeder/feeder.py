# np
import numpy as np

# torch
import torch

# operation
from .tools import *

class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        mode: must be train or test
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        temporal_downsample_step: Step for down sampling the output sequence
        mean_subtraction: The value of bias should be subtracted from output data
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 mode,
                 random_choose=False,
                 random_shift=False,
                 window_size=-1,
                 temporal_downsample_step=1,
                 mean_subtraction=0,
                 normalization=False,
                 debug=False):
        self.debug = debug
        self.mode = mode
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.window_size = window_size
        self.mean_subtraction = mean_subtraction
        self.temporal_downsample_step = temporal_downsample_step
        self.normalization = normalization

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        # load label
        self.label = np.load(self.label_path)

        # load data
        self.data = np.load(self.data_path)
        self.data = np.expand_dims(self.data,axis=4)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(
            axis=2, keepdims=True).mean(
                axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # get data
        data_numpy = self.data[index]
        label = self.label[index]

        # normalization
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map

        # processing
        if self.temporal_downsample_step != 1:
            if self.mode == 'train':
                data_numpy = downsample(data_numpy,self.temporal_downsample_step)
            else:
                data_numpy = temporal_slice(data_numpy,self.temporal_downsample_step)
        if self.mode == 'train':
            if self.random_shift:
                data_numpy = random_shift(data_numpy)
            if self.random_choose:
                data_numpy = random_choose(data_numpy, self.window_size)

        # mean subtraction
        if self.mean_subtraction != 0:
            data_numpy = mean_subtractor(data_numpy, self.mean_subtraction)

        return data_numpy, label
