# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools
import json


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
        data_ratio: Float between 0 and 1. If set, use only a fraction of the data (default: 1.0)
        obs_ratio: Observation ratio cho Early Action Prediction (student chỉ thấy
                   obs_ratio của số frame HỢP LỆ, phần còn lại zero-pad). Mặc định 1.0.
    """

    def __init__(self,
                 data_path,
                 label_path,
                 text_path,
                 random_rot=False,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 p_interval=1,
                 debug=False,
                 data_ratio=1.0,
                 mmap=True,
                 bone=False,
                 obs_ratio=1.0):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_rot = random_rot
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.p_interval = p_interval
        self.data_ratio = data_ratio
        self.bone = bone
        self.obs_ratio = obs_ratio
        self.text_list = self.load_text(text_path)
        self.load_data(mmap)

    def load_text(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        text_list = [data[str(i)][0] for i in range(len(data))]
        return text_list

    def label_to_text(self, label):
        return self.text_list[label]  # string

    def load_data(self, mmap):
        # data: N C V T M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # apply data ratio
        if 0 < self.data_ratio < 1.0:
            num_samples = int(len(self.label) * self.data_ratio)
            indices = np.random.choice(len(self.label), num_samples, replace=False)
            self.label = [self.label[i] for i in indices]
            self.data = self.data[indices]
            self.sample_name = [self.sample_name[i] for i in indices]

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        text = self.label_to_text(label)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)

        # processing
        if self.window_size > 0:
            data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        if self.bone:
            from .bone_pairs import cobot_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in cobot_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy

        data_teacher = data_numpy  # (C, T, V, M), teacher luôn thấy chuỗi đầy đủ

        # Early Action Prediction: student chỉ thấy obs_ratio của số frame HỢP LỆ
        if self.obs_ratio < 1.0:
            C_dim, T_dim, V_dim, M_dim = data_numpy.shape
            # Đếm frame hợp lệ SAU crop/resize (frame toàn 0 là padding ở cuối chuỗi)
            valid_t = int((np.abs(data_numpy).sum(axis=(0, 2, 3)) != 0).sum())
            if valid_t == 0:
                valid_t = T_dim
            t_obs = max(1, int(valid_t * self.obs_ratio))
            # Partial Sequence: chi giu cac frame da quan sat (cat ngan, khong pad ve T).
            # Do dai co dinh t_fix = int(T * obs_ratio) de ghep batch; neu chuoi day du
            # khong co padding thi t_obs == t_fix, tuc la cat ngan chinh xac.
            t_fix = max(1, int(T_dim * self.obs_ratio))
            t_obs = min(t_obs, t_fix)
            data_student_slice = np.zeros((C_dim, t_fix, V_dim, M_dim), dtype=data_numpy.dtype)
            data_student_slice[:, :t_obs, :, :] = data_numpy[:, :t_obs, :, :]
        else:
            data_student_slice = data_numpy

        # (C, T, V, M) -> (T, V, C)
        if data_student_slice.shape[-1] == 1:
            data_student = data_student_slice.squeeze(-1).transpose(1, 2, 0)  # (T, V, C)
        else:
            data_student = data_student_slice.mean(-1).transpose(1, 2, 0)

        return data_teacher, data_student, label, text