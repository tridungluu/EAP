# net/hyper_sa.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HypergraphSelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=48, num_hyperedges=16):
        super(HypergraphSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        self.num_hyperedges = num_hyperedges
        
        # Chiếu tuyến tính cho Q, K, V
        self.q_proj = nn.Linear(in_channels, out_channels)
        self.k_proj = nn.Linear(in_channels, out_channels)
        self.v_proj = nn.Linear(in_channels, out_channels)
        
        # Ma trận liên thuộc học được (Learnable Incidence Matrix H: V x E)
        self.H = nn.Parameter(torch.randn(num_joints, num_hyperedges) * 0.02)
        
        # Batch Normalization & Residual Connection
        self.bn = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x):
        # x shape: (B, C, T, V)
        B, C, T, V = x.shape
        residual = self.residual_proj(x)
        
        # Chuyển đổi định dạng để đưa vào lớp Linear: (B*T, V, C)
        x_perm = x.permute(0, 2, 3, 1).reshape(B * T, V, C)
        
        Q = self.q_proj(x_perm)      # (B*T, V, C')
        K = self.k_proj(x_perm)      # (B*T, V, C')
        V_val = self.v_proj(x_perm)  # (B*T, V, C')
        
        # 1. Tự chú ý giữa các khớp xương (Node-to-Node Attention)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.out_channels ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)         # (B*T, V, V)
        node_feats = torch.matmul(attn_weights, V_val)        # (B*T, V, C')
        
        # 2. Truyền tin qua Siêu đồ thị: Nodes -> Hyperedges -> Nodes
        # Chuẩn hóa mềm theo hai hướng:
        H_edge = F.softmax(self.H, dim=0)                     # (V, E): Node -> Hyperedge
        H_node = F.softmax(self.H, dim=1)                     # (V, E): Hyperedge -> Node
        
        hyperedge_feats = torch.matmul(H_edge.T, node_feats)  # (B*T, E, C')
        out_nodes = torch.matmul(H_node, hyperedge_feats)     # (B*T, V, C')
        
        # Thêm kết nối tắt nội bộ để bảo toàn đặc trưng nút
        out_nodes = out_nodes + node_feats
        
        # Đưa về Tensor 4D gốc: (B, C', T, V)
        out = out_nodes.view(B, T, V, self.out_channels).permute(0, 3, 1, 2)
        out = F.relu(self.bn(out) + residual)
        return out
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, padding=4):
        super(TemporalConv, self).__init__()
        self.t_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = self.t_conv(x)
        out = F.relu(self.bn(out) + res)
        return out


class HyperSABlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints=48, kernel_size=9):
        super(HyperSABlock, self).__init__()
        self.hyper_sa = HypergraphSelfAttention(in_channels, out_channels, num_joints=num_joints)
        self.temporal_conv = TemporalConv(out_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.hyper_sa(x)
        x = self.temporal_conv(x)
        return x