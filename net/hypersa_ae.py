"""
HyperSA backbone (Zhou et al., "Hypergraph Transformer for Skeleton-based Action Recognition",
arXiv:2211.09590), viet lai tu cac phuong trinh trong bai (Eq. 4-9, Eq. 10-12, Fig. 2).
Cung giao dien voi net/ctr_ae.py: encoder tra ve (logits, latent, mid), co out_channels va build_decoder().
"""
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class

from .ctr_gcn import MultiScale_TemporalConv, unit_tcn, bn_init


def build_spd(graph, num_point, hop_edges=None):
    """Shortest Path Distance giua moi cap khop (cho k-Hop RPE, Sec. 4.2)."""
    adj, src = None, None
    if hop_edges is not None:
        adj = np.zeros((num_point, num_point), dtype=bool)
        for i, j in hop_edges:
            adj[i, j] = adj[j, i] = True
        src = 'hop_edges (yaml)'
    if adj is None and hasattr(graph, 'A_binary'):
        adj = np.asarray(graph.A_binary) > 0
        src = 'graph.A_binary'
    if adj is None and hasattr(graph, 'neighbor_link'):
        links = [tuple(l) for l in graph.neighbor_link]
        off = 1 if max(max(l) for l in links) >= num_point else 0   # 1-indexed -> 0-indexed
        adj = np.zeros((num_point, num_point), dtype=bool)
        for i, j in links:
            adj[i - off, j - off] = adj[j - off, i - off] = True
        src = 'graph.neighbor_link'
    if adj is None:
        A = np.asarray(graph.A)
        adj = A[1:].sum(0) > 0
        src = 'FALLBACK graph.A[1:] (co the sai neu A la hop-k)'
        warnings.warn('build_spd: dung fallback tu graph.A. Hay dat hop_edges trong YAML '
                      'neu SPD in ra khong dung.')
    assert adj.shape == (num_point, num_point), (adj.shape, num_point)

    adj = adj | adj.T
    np.fill_diagonal(adj, False)
    dist = np.full((num_point, num_point), np.inf)
    dist[adj] = 1.0
    np.fill_diagonal(dist, 0.0)
    for k in range(num_point):                       # Floyd-Warshall
        dist = np.minimum(dist, dist[:, k:k + 1] + dist[k:k + 1, :])
    finite = np.isfinite(dist)
    dist[~finite] = dist[finite].max() + 1.0         # khop khong noi duoc voi nhau
    return dist.astype(np.int64), src


class HyperSA(nn.Module):
    """
    Hypergraph Self-Attention tren (N, C_in, T, V) -> (N, C_out, T, V); moi frame la mot tap V khop.
      E     = D_e^-1 H^T X W_e          (Eq. 4)
      E_aug = H D_e^-1 H^T X W_e        (Eq. 5)
      A_ij  = q_i.k_j + q_i.E_aug,j + q_i.R_phi(i,j) + u.E_aug,j   (Eq. 6)
      y_i   = sum_j (softmax(A)_ij + B_ij) v_j                      (Eq. 7)
    H la ma tran phan hoach (V x |E|): hoc duoc qua softmax (Eq. 8) hoac co dinh theo `groups`.
    """

    def __init__(self, in_ch, out_ch, num_point, spd, heads=8, num_edges=5, groups=None):
        super().__init__()
        assert out_ch % heads == 0, 'out_ch phai chia het cho heads'
        self.h, self.d = heads, out_ch // heads
        self.scale = self.d ** -0.5
        V = num_point

        self.qkv = nn.Linear(in_ch, 3 * out_ch, bias=False)
        self.We = nn.Linear(in_ch, out_ch, bias=False)            # W_e trong Eq. 4
        self.proj = nn.Linear(out_ch, out_ch)

        spd = torch.as_tensor(spd, dtype=torch.long)
        self.register_buffer('spd', spd)
        K = int(spd.max()) + 1
        self.R = nn.Parameter(torch.randn(K, heads, self.d) * 0.02)       # bang k-Hop RPE
        self.u = nn.Parameter(torch.randn(heads, self.d) * 0.02)          # khoa tinh u (Eq. 6d)
        self.B = nn.Parameter(torch.zeros(heads, V, V))                   # Relational Bias (Eq. 7)

        if groups is None:
            self.part_logits = nn.Parameter(torch.randn(V, num_edges) * 0.1)   # partition hoc duoc
            self.register_buffer('H_fixed', None)
        else:
            groups = torch.as_tensor(groups, dtype=torch.long)
            self.part_logits = None
            self.register_buffer('H_fixed', F.one_hot(groups, int(groups.max()) + 1).float())

    def partition(self):
        if self.part_logits is not None:
            return F.softmax(self.part_logits, dim=1)     # Eq. 8: softmax tren cac hyperedge cua moi khop
        return self.H_fixed

    def forward(self, x):
        N, C, T, V = x.shape
        t = x.permute(0, 2, 3, 1).reshape(N * T, V, C)
        B_ = t.shape[0]

        q, k, v = self.qkv(t).chunk(3, dim=-1)
        H = self.partition()                                          # (V, E)
        De = H.sum(0).clamp(min=1e-6)                                 # bac cua hyperedge
        xw = self.We(t)                                               # (B, V, out)
        E = torch.einsum('ve,bvc->bec', H, xw) / De[None, :, None]    # Eq. 4
        E_aug = torch.einsum('ve,bec->bvc', H, E)                     # Eq. 5

        def split(z):
            return z.reshape(B_, V, self.h, self.d).permute(0, 2, 1, 3)   # (B, h, V, d)
        q, k, v, Ea = split(q), split(k), split(v), split(E_aug)

        ab = torch.einsum('bhid,bhjd->bhij', q, k + Ea)               # (a) + (b)
        R = self.R[self.spd]                                          # (V, V, h, d)
        c = torch.einsum('bhid,ijhd->bhij', q, R)                     # (c) k-Hop RPE
        d_ = torch.einsum('hd,bhjd->bhj', self.u, Ea).unsqueeze(2)    # (d) attentive bias
        attn = ((ab + c + d_) * self.scale).softmax(dim=-1) + self.B  # Eq. 7

        y = torch.einsum('bhij,bhjd->bhid', attn, v)
        y = y.permute(0, 2, 1, 3).reshape(B_, V, self.h * self.d)
        y = self.proj(y)
        return y.reshape(N, T, V, -1).permute(0, 3, 1, 2).contiguous()


def _ln(ln, x):
    return ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class HyperSABlock(nn.Module):
    """Mot khoi Hyperformer: LN -> HyperSA (+res) -> ReLU -> LN -> MS-TC (+res) -> ReLU (Eq. 10-12)."""

    def __init__(self, in_ch, out_ch, num_point, spd, stride=1, residual=True,
                 heads=8, num_edges=5, groups=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(in_ch)
        self.sa = HyperSA(in_ch, out_ch, num_point, spd, heads, num_edges, groups)
        if not residual:
            self.res1 = lambda x: 0
        elif in_ch == out_ch:
            self.res1 = lambda x: x
        else:
            self.res1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch))

        self.ln2 = nn.LayerNorm(out_ch)
        self.tcn = MultiScale_TemporalConv(out_ch, out_ch, kernel_size=5, stride=stride,
                                           dilations=[1, 2], residual=False)
        if not residual:
            self.res2 = lambda x: 0
        elif stride == 1:
            self.res2 = lambda x: x
        else:
            self.res2 = unit_tcn(out_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x):
        y = F.relu(self.sa(_ln(self.ln1, x)) + self.res1(x))
        z = F.relu(self.tcn(_ln(self.ln2, y)) + self.res2(y))
        return z


class HyperSAEncoder(nn.Module):
    """
    Cung giao dien voi CTREncoder: forward -> (logits, feat_map, mid_feat).
    Teacher: channels=(64,128,256), strides=(1,2,2) (3 khoi). Student: channels=(128,256), strides=(2,2).
    """

    def __init__(self, num_class, num_point, num_person, graph, graph_args,
                 in_channels=3, drop_out=0.0,
                 channels=(64, 128, 256), strides=(1, 2, 2),
                 heads=8, num_hyperedges=5, partition_groups=None, hop_edges=None, **kwargs):
        super().__init__()
        channels, strides = tuple(channels), tuple(strides)
        assert len(channels) == len(strides)
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        spd, src = build_spd(self.graph, num_point, hop_edges)
        print('[HyperSA] SPD source: {} | max SPD = {}'.format(src, int(spd.max())))
        self.spd_np = spd
        self.num_point = num_point
        self.heads = heads
        self.groups = partition_groups
        self.num_edges = (max(partition_groups) + 1) if partition_groups is not None else num_hyperedges

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        blocks, c_in = [], in_channels
        for c, s in zip(channels, strides):
            blocks.append(HyperSABlock(c_in, c, num_point, spd, stride=s, residual=True,
                                       heads=heads, num_edges=self.num_edges, groups=partition_groups))
            c_in = c
        self.blocks = nn.ModuleList(blocks)
        self.mid_idx = 1 if len(blocks) >= 3 else 0
        self.out_channels = channels[-1]

        self.drop_out = nn.Dropout(drop_out) if drop_out else nn.Identity()
        self.fc = nn.Linear(channels[-1], num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)

    def build_decoder(self, out_channels=3, hidden=(128, 64)):
        return HyperSADecoder(self.out_channels, out_channels, self.num_point, self.spd_np,
                              hidden, self.heads, self.num_edges, self.groups)

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(-1)
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        mid = None
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.mid_idx:
                mid = x
        feat = x
        c = feat.size(1)
        f = feat.view(N, M, c, -1).mean(3).mean(1)
        out = self.fc(self.drop_out(f))
        return out, feat, mid


class HyperSADecoder(nn.Module):
    """Decoder 2 khoi HyperSA + 2D Avg Pooling + Conv2d (Fig. 2e). Context (N,C,t_len,V) -> (N,3,T,V)."""

    def __init__(self, in_channels, out_channels, num_point, spd, hidden=(128, 64),
                 heads=8, num_edges=5, groups=None):
        super().__init__()
        self.b1 = HyperSABlock(in_channels, hidden[0], num_point, spd, 1, True, heads, num_edges, groups)
        self.b2 = HyperSABlock(hidden[0], hidden[1], num_point, spd, 1, True, heads, num_edges, groups)
        self.out = nn.Conv2d(hidden[1], out_channels, kernel_size=1)

    def forward(self, z, target_shape):
        T = int(target_shape[2])
        V = z.shape[-1]
        x = self.b1(z)
        t_mid = max(z.shape[2], T // 2)
        x = F.interpolate(x, size=(t_mid, V), mode='bilinear', align_corners=False)
        x = self.b2(x)
        x = F.interpolate(x, size=(T, V), mode='bilinear', align_corners=False)
        x = F.adaptive_avg_pool2d(x, (T, V))
        return self.out(x)