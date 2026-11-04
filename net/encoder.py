# net/encoder.py
import torch
import torch.nn as nn
from .hyper_sa import HyperSABlock

class TeacherEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, num_joints=48, num_layers=3):
        super(TeacherEncoder, self).__init__()
        layers = []
        c_in = in_channels
        c_out = 64
        for i in range(num_layers):
            layers.append(HyperSABlock(c_in, c_out, num_joints=num_joints))
            c_in = c_out
            c_out = latent_dim if i == num_layers - 2 else c_out
            
        self.encoder_blocks = nn.ModuleList(layers)

    def forward(self, x, return_latent=False):
        out = x
        for block in self.encoder_blocks:
            out = block(out)
            
        latent_tensor = out  # (B, 256, T, V)
        # Nén theo trục thời gian T nhưng GIỮ NGUYÊN shape 4D (T=1, V=48)
        context = torch.mean(latent_tensor, dim=2, keepdim=True)  # (B, 256, 1, V)
        
        if return_latent:
            return latent_tensor, context
        return context


class StudentEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, num_joints=48, num_layers=2):
        super(StudentEncoder, self).__init__()
        layers = []
        c_in = in_channels
        c_out = 128
        for i in range(num_layers):
            layers.append(HyperSABlock(c_in, c_out, num_joints=num_joints))
            c_in = c_out
            c_out = latent_dim
            
        self.encoder_blocks = nn.ModuleList(layers)

    def forward(self, x, return_latent=False):
        out = x
        for block in self.encoder_blocks:
            out = block(out)
            
        latent_tensor = out  # (B, 256, T, V)
        # Nén theo trục thời gian T nhưng GIỮ NGUYÊN shape 4D (T=1, V=48)
        context = torch.mean(latent_tensor, dim=2, keepdim=True)  # (B, 256, 1, V)
        
        if return_latent:
            return latent_tensor, context
        return context


class ActionClassifier(nn.Module):
    def __init__(self, latent_dim=256, num_joints=48, num_classes=16):
        super(ActionClassifier, self).__init__()
        # Flatten (C_h * V) -> num_classes
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim * num_joints, num_classes)
        )

    def forward(self, context):
        # context có thể là (B, 256, V) hoặc (B, 256, 1, V)
        flat_context = context.flatten(start_dim=1)  # Đều ra phẳng (B, 256 * 48)
        return self.fc(flat_context)