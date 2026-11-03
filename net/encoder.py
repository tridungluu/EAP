# net/encoder.py
import torch
import torch.nn as nn
from .hyper_sa import HyperSABlock

class TeacherEncoder(nn.Module):
    """
    Teacher Encoder: Xếp chồng 3 khối HyperSABlock.
    Input: (B, C=3, T=64, V=48)
    Output:
      - latent_tensor: (B, C_h=256, T, V)
      - context: (B, C_h=256, V=48) sau khi average pooling chỉ theo chiều T
    """
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

    def forward(self, x):
        out = x
        for block in self.encoder_blocks:
            out = block(out)
            
        latent_tensor = out  # (B, 256, T, V)
        # CHỈ nén theo chiều thời gian T (dim=2), GIỮ NGUYÊN chiều khớp V (dim=3)
        context = torch.mean(latent_tensor, dim=2)  # (B, 256, V)
        return latent_tensor, context


class StudentEncoder(nn.Module):
    """
    Student Encoder: Xếp chồng 2 khối HyperSABlock.
    Nhận chuỗi quan sát dở dang (20% - 80% frames).
    Output:
      - latent_tensor: (B, C_h=256, T, V)
      - context: (B, C_h=256, V=48)
    """
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

    def forward(self, x):
        out = x
        for block in self.encoder_blocks:
            out = block(out)
            
        latent_tensor = out  # (B, 256, T, V)
        # CHỈ nén theo chiều thời gian T (dim=2)
        context = torch.mean(latent_tensor, dim=2)  # (B, 256, V)
        return latent_tensor, context


class ActionClassifier(nn.Module):
    """
    Classifier nhận Context (B, C_h, V).
    Làm phẳng (Flatten) hoặc pooling trước khi đưa qua Linear layer.
    """
    def __init__(self, latent_dim=256, num_joints=48, num_classes=16):
        super(ActionClassifier, self).__init__()
        # Flatten (C_h * V) -> num_classes hoặc Linear trên toàn bộ đặc trưng
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim * num_joints, num_classes)
        )

    def forward(self, context):
        # context shape: (B, C_h=256, V=48)
        flat_context = context.flatten(start_dim=1)  # (B, 256 * 48)
        return self.fc(flat_context)