import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonDecoder(nn.Module):
    """
    Decoder to reconstruct skeleton sequence (C_in, T, V) from latent feature map (C_lat, T_lat, V).
    Used in Teacher (Section 3.3, Fig. 2e) and transferred to Student.
    """
    def __init__(self, in_channels=256, out_channels=3, kernel_size=(5, 1)):
        super().__init__()
        # Upsample temporal dimension if latent is downsampled (e.g. T/4 -> T)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, feat, target_shape):
        """
        feat: (B, C_lat, T_lat, V)
        target_shape: (B, C_out, T_target, V)
        """
        out = self.deconv(feat)
        # Match target shape exactly
        if out.shape[2:] != target_shape[2:]:
            out = F.interpolate(out, size=target_shape[2:], mode='bilinear', align_corners=False)
        return out
