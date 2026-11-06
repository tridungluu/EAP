import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class

# Giả định net/ctr_gcn.py là bản CTR-GCN chuẩn: có TCN_GCN_unit và bn_init
from .ctr_gcn import TCN_GCN_unit, bn_init


class CTREncoder(nn.Module):
    """
    Encoder CTR-GCN rút gọn, trả về (logits, feat_map, mid_feat) giống net.ctr_gcn.Model.

    Teacher: channels=(64, 128, 256), strides=(1, 2, 2) -> 3 khối CTR-GCN (TCN_GCN_unit)
    Student: channels=(128, 256),     strides=(2, 2)    -> 2 khối CTR-GCN (TCN_GCN_unit)

    Cả hai đều ra feat (N*M, 256, T/4, V) nên context của teacher và student
    nằm cùng một không gian.
    """

    def __init__(self, num_class, num_point, num_person, graph, graph_args,
                 in_channels=3, drop_out=0.0,
                 channels=(64, 128, 256), strides=(1, 2, 2), **kwargs):
        super().__init__()
        channels, strides = tuple(channels), tuple(strides)
        assert len(channels) == len(strides)

        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = self.graph.A
        self.A_np = A              # decoder dùng lại cùng adjacency
        self.num_point = num_point

        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        blocks, c_in = [], in_channels
        for i, (c, s) in enumerate(zip(channels, strides)):
            blocks.append(TCN_GCN_unit(c_in, c, A, stride=s, residual=(i > 0)))
            c_in = c
        self.blocks = nn.ModuleList(blocks)
        self.mid_idx = 1 if len(blocks) >= 3 else 0
        self.out_channels = channels[-1]

        self.drop_out = nn.Dropout(drop_out) if drop_out else nn.Identity()
        self.fc = nn.Linear(channels[-1], num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)

    def build_decoder(self, out_channels=3, hidden=(128, 64)):
        # Decoder cung ho voi encoder (de doi backbone chi bang YAML)
        return CTRDecoder(self.out_channels, out_channels, self.A_np, hidden)

    def forward(self, x):
        # x: (N, C, T, V, M) hoặc (N, C, T, V)
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
        feat = x                                   # (N*M, 256, T/4, V)

        c = feat.size(1)
        f = feat.view(N, M, c, -1).mean(3).mean(1)
        out = self.fc(self.drop_out(f))
        return out, feat, mid


class CTRDecoder(nn.Module):
    """
    Decoder 2 khối CTR-GCN (Fig. 2e).
    Context (N, 256, t_len, V) -> (N, 3, T, V). Giữ nguyên chiều V để STA tính từng khớp.
    """

    def __init__(self, in_channels, out_channels, A, hidden=(128, 64)):
        super().__init__()
        self.b1 = TCN_GCN_unit(in_channels, hidden[0], A, stride=1, residual=True)
        self.b2 = TCN_GCN_unit(hidden[0], hidden[1], A, stride=1, residual=True)
        self.out = nn.Conv2d(hidden[1], out_channels, kernel_size=1)

    def forward(self, z, target_shape):
        # target_shape: (N, C, T, V) -> chỉ cần T
        T = int(target_shape[2])
        V = z.shape[-1]
        x = self.b1(z)
        t_mid = max(z.shape[2], T // 2)
        x = F.interpolate(x, size=(t_mid, V), mode='bilinear', align_corners=False)
        x = self.b2(x)
        x = F.interpolate(x, size=(T, V), mode='bilinear', align_corners=False)
        # Fig. 2e: 2D Avg Pooling -> Conv2d. O day pooling dam bao dung kich thuoc (T, V) dau ra.
        x = F.adaptive_avg_pool2d(x, (T, V))
        return self.out(x)