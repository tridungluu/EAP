import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class FeederEAP(Dataset):
    def __init__(self, data_path, label_path, ratio=1.0, target_time=96):
        self.data_path = data_path
        self.label_path = label_path
        self.ratio = float(ratio)
        self.target_time = int(target_time)
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.data_path}")
        if not os.path.exists(self.label_path):
            raise FileNotFoundError(f"Không tìm thấy file nhãn: {self.label_path}")

        # Nạp dữ liệu vị trí khung xương: (N, C, T, V) hoặc (N, C, T, V, M)
        data = np.load(self.data_path)
        if data.ndim == 5:
            # Lấy chủ thể người đầu tiên
            data = data[..., 0]

        if data.ndim != 4:
            raise ValueError(f"Dữ liệu cần có dạng (N, C, T, V), nhưng nhận được shape: {data.shape}")

        self.data = data

        # Nạp nhãn cử chỉ
        with open(self.label_path, 'rb') as f:
            labels = pickle.load(f)
        if isinstance(labels, tuple):
            labels = labels
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Dữ liệu gốc: (C=3, T=96, V=48)
        full_sample = np.array(self.data[index], dtype=np.float32)
        C, T, V = full_sample.shape

        # Đảm bảo khớp chuẩn target_time (96 frames)
        if T != self.target_time:
            indices = np.linspace(0, T - 1, self.target_time).astype(int)
            full_sample = full_sample[:, indices, :]

        # Cắt theo tỷ lệ quan sát (ratio)
        cutoff_T = max(1, int(self.target_time * self.ratio))

        # Tạo chuỗi dở dang cho Student (che các frame tương lai)
        partial_sample = np.copy(full_sample)
        partial_sample[:, cutoff_T:, :] = 0.0

        label = self.labels[index]

        return (
            torch.tensor(partial_sample, dtype=torch.float32),
            torch.tensor(full_sample, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )