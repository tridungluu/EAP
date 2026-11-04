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

        # 1. Nạp dữ liệu vị trí khung xương: (N, C, T, V) hoặc (N, C, T, V, M)
        data = np.load(self.data_path, mmap_mode=None)
        if data.ndim == 5:
            # Lấy chủ thể người/bàn tay đầu tiên: (N, C, T, V)
            data = data[..., 0]

        if data.ndim != 4:
            raise ValueError(f"Dữ liệu cần có dạng (N, C, T, V), nhưng nhận được shape: {data.shape}")

        # Ép kiểu float32 sẵn từ đầu để không tốn chi phí ép kiểu lặp lại trong __getitem__
        self.data = np.ascontiguousarray(data, dtype=np.float32)

        # 2. Nạp nhãn cử chỉ từ file pkl
        with open(self.label_path, 'rb') as f:
            label_obj = pickle.load(f)

        # Trích xuất nhãn số nguyên:
        # File pkl của skeleton thường lưu dạng tuple (sample_names, label_list)
        if isinstance(label_obj, (tuple, list)):
            if len(label_obj) == 2 and isinstance(label_obj[0], (list, np.ndarray)) and isinstance(label_obj[1], (list, np.ndarray)):
                # Kiểm tra phần tử nào chứa số nguyên để lấy đúng nhãn
                first_elem_0 = label_obj[0][0] if len(label_obj[0]) > 0 else None
                first_elem_1 = label_obj[1][0] if len(label_obj[1]) > 0 else None

                if isinstance(first_elem_1, (int, np.integer)):
                    raw_labels = label_obj[1]
                elif isinstance(first_elem_0, (int, np.integer)):
                    raw_labels = label_obj[0]
                else:
                    # Mặc định lấy phần tử thứ 2 nếu phần tử thứ 1 là chuỗi tên file
                    raw_labels = label_obj[1]
            else:
                raw_labels = label_obj
        elif isinstance(label_obj, dict):
            raw_labels = label_obj.get("labels", label_obj.get("label", []))
        else:
            raw_labels = label_obj

        self.labels = np.asarray(raw_labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Dữ liệu gốc: (C=3, T, V=48)
        full_sample = self.data[index]
        C, T, V = full_sample.shape

        # Đảm bảo khớp chuẩn target_time (96 frames)
        if T != self.target_time:
            indices = np.linspace(0, T - 1, self.target_time).astype(int)
            full_sample = full_sample[:, indices, :]
        else:
            full_sample = np.copy(full_sample)

        # Cắt theo tỷ lệ quan sát (ratio) cho bài toán Early Action Prediction
        cutoff_T = max(1, int(self.target_time * self.ratio))

        # Tạo chuỗi dở dang cho Student (che các frame tương lai bằng 0)
        partial_sample = np.copy(full_sample)
        partial_sample[:, cutoff_T:, :] = 0.0

        label = self.labels[index]

        # Trả về bộ ba (partial, full, label)
        return (
            torch.from_numpy(partial_sample),
            torch.from_numpy(full_sample),
            torch.tensor(label, dtype=torch.long)
        )