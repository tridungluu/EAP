import torch
import torch.nn as nn

class SkeletonDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3, num_joints=48):
        super(SkeletonDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.num_joints = num_joints

        # Chuỗi các khối Conv2d (kernel_size=1) giải nén kênh đặc trưng
        # Toàn bộ quá trình duy trì kích thước không-thời gian (1, V)
        self.deconv = nn.Sequential(
            # Khối 1: 256 -> 128
            nn.Conv2d(latent_dim, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Khối 2: 128 -> 64
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Khối 3: Chiếu về 3 tọa độ không gian (x, y, z)
            nn.Conv2d(64, out_channels, kernel_size=1, bias=True)
        )

    def forward(self, context):
        if context.dim() == 3:
            # (B, latent_dim, V) -> (B, latent_dim, 1, V)
            x = context.unsqueeze(2)
        elif context.dim() == 2:
            # (B, latent_dim) -> (B, latent_dim, 1, num_joints)
            x = context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.num_joints).contiguous()
        elif context.dim() == 4:
            # Đảm bảo trục thời gian đúng T = 1
            x = context
        else:
            raise ValueError(f"Kích thước context không hợp lệ: {context.shape}")

        # Cho tensor đi qua các khối Conv2d (giữ nguyên T = 1)
        reconstructed = self.deconv(x)  # Kích thước: (B, 3, 1, V)
        return reconstructed