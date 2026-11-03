# feeder/feeder_eap.py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

class FeederEAP(Dataset):
    """
    Feeder cho bài toán Early Action Prediction (EAP) tương thích với datacobot_v2.
    Đọc dữ liệu từ cặp file: position.npy và label.pkl.
    """
    def __init__(self, data_path, label_path, ratio=1.0, target_time=64):
        """
        data_path: Đường dẫn tới file position .npy (vd: 'datacobot_v2/xsub/train_position.npy')
        label_path: Đường dẫn tới file label .pkl (vd: 'datacobot_v2/xsub/train_label.pkl')
        ratio: Tỷ lệ quan sát video (1.0 cho Teacher; 0.2, 0.4, 0.6, 0.8 cho Student)
        target_time: Số khung hình chuẩn hóa (mặc định T = 64)
        """
        super(FeederEAP, self).__init__()
        self.data_path = data_path
        self.label_path = label_path
        self.ratio = ratio
        self.target_time = target_time
        
        self.load_data()

    def load_data(self):
        # 1. Nạp tọa độ khung xương từ file .npy
        self.data = np.load(self.data_path, mmap_mode='r')
        
        # 2. Nạp nhãn từ file .pkl
        with open(self.label_path, 'rb') as f:
            raw_labels = pickle.load(f)
            
        if isinstance(raw_labels, tuple):
            if len(raw_labels) < 2:
                raise ValueError("File nhãn dạng tuple phải chứa (sample_names, labels).")
            self.labels = raw_labels[1]
        elif isinstance(raw_labels, dict) and 'labels' in raw_labels:
            self.labels = raw_labels['labels']
        else:
            self.labels = raw_labels

        self.labels = np.array(self.labels, dtype=np.int64)

        # 3. Chuẩn hóa shape về (N, C, T, V)
        # datacobot_v2 có shape (N, C, T, V, M) với M=1
        if len(self.data.shape) == 5:
            self.num_samples, self.C, self.T, self.V, self.M = self.data.shape
        elif len(self.data.shape) == 4:
            self.num_samples, self.C, self.T, self.V = self.data.shape
            self.M = 1
        else:
            raise ValueError(f"Shape dữ liệu không hợp lệ: {self.data.shape}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # Lấy mẫu thứ index, chuyển về dạng (C, T, V)
        if len(self.data.shape) == 5:
            skeleton = np.array(self.data[index, :, :, :, 0], dtype=np.float32)
        else:
            skeleton = np.array(self.data[index], dtype=np.float32)

        label = self.labels[index]
        C, T, V = skeleton.shape

        # Cắt chuỗi theo tỷ lệ quan sát dở dang (Observation Ratio)
        observed_frames = max(1, int(T * self.ratio))
        partial_skeleton = skeleton[:, :observed_frames, :]  # (C, T_obs, V)

        # Đồng nhất chiều thời gian về target_time (T = 64)
        if observed_frames < self.target_time:
            # Lặp lại frame cuối để đệm (padding) cho đủ target_time
            pad_len = self.target_time - observed_frames
            last_frame = partial_skeleton[:, -1:, :]
            padding = np.repeat(last_frame, pad_len, axis=1)
            padded_skeleton = np.concatenate([partial_skeleton, padding], axis=1)
        else:
            # Lấy mẫu đều các frame
            idx = np.linspace(0, observed_frames - 1, self.target_time, dtype=int)
            padded_skeleton = partial_skeleton[:, idx, :]

        # Chuyển đổi sang PyTorch Tensor
        skeleton_tensor = torch.from_numpy(padded_skeleton).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return skeleton_tensor, label_tensor, observed_frames