import numpy as np

# Nạp file dữ liệu val hoặc train của bạn
data = np.load("datacobot_v2/xsub/val_position.npy") # (N, C, T, V)
# Tính độ dịch chuyển trung bình (variance) của từng khớp qua thời gian
motion_var = np.var(data, axis=(0, 1, 2)) # Shape: (48,)

print("Top 5 khớp ít di chuyển nhất (ứng viên sáng giá cho root/gốc):", np.argsort(motion_var)[:5])