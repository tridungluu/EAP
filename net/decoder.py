# net/decoder.py
import torch
import torch.nn as nn

class SkeletonDecoder(nn.Module):
    """
    Decoder nhận context đã nén thời gian (B, C_h=256, 1, V) hoặc (B, C_h=256, V)
    và tái tạo trực tiếp tọa độ không gian 3D với T = 1: (B, 3, 1, V).
    """
    def __init__(self, latent_dim=256, out_channels=3):
        super(SkeletonDecoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.Conv2d(latent_dim, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, context):
        # Đảm bảo context có 4 chiều: (B, C_h, 1, V)
        if context.dim() == 3:
            # (B, C_h, V) -> (B, C_h, 1, V)
            x = context.unsqueeze(2)
        else:
            x = context
            
        # Tái tạo tọa độ: (B, 3, 1, V)
        reconstructed = self.deconv(x)
        return reconstructed