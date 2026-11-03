# net/dcl_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCLDisentangleModule(nn.Module):
    """
    Phân tách biểu diễn thành 2 nhánh:
      - Nhánh Không gian (Spatial Branch Cs)
      - Nhánh Thời gian (Temporal Branch Ct)
    """
    def __init__(self, latent_dim=256, proj_dim=128):
        super(DCLDisentangleModule, self).__init__()
        # Nhánh Không gian: Nén chiều T, giữ thông tin V
        self.spatial_proj = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )
        
        # Nhánh Thời gian: Nén chiều V, giữ thông tin T
        self.temporal_proj = nn.Sequential(
            nn.Linear(latent_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, latent_tensor):
        # latent_tensor shape: (B, C_h=256, T, V)
        B, C, T, V = latent_tensor.shape
        
        # 1. Nhánh Spatial: pooling theo T, sau đó pooling theo V
        spatial_feat = torch.mean(latent_tensor, dim=2)      # (B, C, V)
        spatial_vec = torch.mean(spatial_feat, dim=2)       # (B, C)
        C_s = F.normalize(self.spatial_proj(spatial_vec), dim=-1) # (B, proj_dim)
        
        # 2. Nhánh Temporal: pooling theo V, sau đó pooling theo T
        temporal_feat = torch.mean(latent_tensor, dim=3)     # (B, C, T)
        temporal_vec = torch.mean(temporal_feat, dim=2)      # (B, C)
        C_t = F.normalize(self.temporal_proj(temporal_vec), dim=-1) # (B, proj_dim)
        
        return C_s, C_t