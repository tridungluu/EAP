# net/dcl_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCLDisentangleModule(nn.Module):
    def __init__(self, latent_dim=256, proj_dim=128):
        super(DCLDisentangleModule, self).__init__()
        
        # Nhánh Không gian (Spatial Branch)
        self.spatial_proj = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # Nhánh Thời gian (Temporal Branch)
        self.temporal_proj = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, latent_tensor):
        # Kiểm tra và đảm bảo kích thước 4D: (B, C, T, V)
        if latent_tensor.dim() != 4:
            raise ValueError(f"DCLDisentangleModule yêu cầu tensor 4D (B, C, T, V), nhận được: {latent_tensor.shape}")
            
        B, C, T, V = latent_tensor.shape
        
        # 1. Nhánh Không gian (Spatial):
        # Gộp chiều thời gian T (giữ lại cấu hình khung xương V)
        spatial_feat = torch.mean(latent_tensor, dim=2)        # (B, C, V)
        spatial_vec = torch.mean(spatial_feat, dim=2)         # (B, C)
        C_s = self.spatial_proj(spatial_vec)
        C_s = F.normalize(C_s, p=2, dim=-1)                   # (B, proj_dim)
        
        # 2. Nhánh Thời gian (Temporal):
        # Gộp chiều khớp V (giữ lại tiến trình động học theo thời gian T)
        temporal_feat = torch.mean(latent_tensor, dim=3)       # (B, C, T)
        temporal_vec = torch.mean(temporal_feat, dim=2)        # (B, C)
        C_t = self.temporal_proj(temporal_vec)
        C_t = F.normalize(C_t, p=2, dim=-1)                   # (B, proj_dim)
        
        return C_s, C_t