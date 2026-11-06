import torch
import torch.nn as nn
from torchlight import import_class
import torch.nn.functional as F
import torch.fft as fft

class JTAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2D = nn.Sequential(
                    nn.Conv2d(
                        in_channels=2, 
                        out_channels=1, 
                        kernel_size=7,
                        padding=3,
                        stride=1,
                        dilation=1),
                )

    def forward(self, x): # x: (N, 256, T/4, V) - (32, 256, 24, 48)
        # avg pool theo C
        x_avg = torch.mean(x, dim=1, keepdim=True) #(N,1,T,V)
        
        # max pool theo C
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        
        concat = torch.cat((x_max, x_avg), dim=1)
        
        # attention map
        attn = self.conv2D(concat)
        attn = torch.sigmoid(attn)
        return attn #(N,1,T,V)


class CAM(nn.Module):
    def __init__(self, C, reduction):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(C, C // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(C // reduction, C, bias=True)
        )

    def forward(self, x):  # x: (N, C, T, V)

        # global pooling
        x_max = F.adaptive_max_pool2d(x, 1)
        x_avg = F.adaptive_avg_pool2d(x, 1)

        N, C, _, _ = x.shape

        # MLP
        max_out = self.mlp(x_max.view(N, C)).view(N, C, 1, 1)
        avg_out = self.mlp(x_avg.view(N, C)).view(N, C, 1, 1)

        # attention
        attn = torch.sigmoid(max_out + avg_out)
        return attn
        

class CompactBilinearPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim

        for i in range(2):
            h = torch.randint(0, self.output_dim, size=(input_dim,))
            s = (torch.randint(0, 2, (input_dim,)) * 2 - 1).float()
            self.register_buffer(f'h{i}',h)
            self.register_buffer(f's{i}',s)

    def count_sketch(self, x, h, s):
        batch_size = x.size(0)
        y = x.new_zeros(batch_size, self.output_dim)
        h_expand = h.view(1,-1).expand(batch_size, -1)
        y.scatter_add_(1, h_expand, x * s)
        return y

    def forward(self, x1, x2):
        x1_hash = self.count_sketch(x1, self.h0, self.s0)
        x2_hash = self.count_sketch(x2, self.h1, self.s1)
        x1_fft = fft.rfft(x1_hash, dim=-1)
        x2_fft = fft.rfft(x2_hash, dim=-1)
        x = x1_fft * x2_fft
        x = fft.irfft(x, n= self.output_dim, dim=-1)
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-8) # Power norm
        x = F.normalize(x, p=2, dim=-1)   # L2 norm
        return x
        
        
class CrossAttention(nn.Module):
    def __init__(self, d_q, d_kv, d_model, d_out=None):
        super().__init__()
        if d_out is None:
            d_out = d_model
        self.scale = d_model ** 0.5
        # projection
        self.q_proj = nn.Linear(d_q, d_model, bias=False)
        self.k_proj = nn.Linear(d_kv, d_model, bias=False)
        self.v_proj = nn.Linear(d_kv, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_out)
        self.attn_drop = nn.Dropout(0.1)

    def forward(self, q, kv):
        """
        q:  (N, Lq, d_q)
        kv: (N, Lk, d_kv)
        """
        Q = self.q_proj(q)   # (N, Lq, d)
        K = self.k_proj(kv)  # (N, Lk, d)
        V = self.v_proj(kv)  # (N, Lk, d)
        # attention
        attn = (Q @ K.transpose(-2, -1)) / self.scale   # (N, Lq, Lk)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ V   # (N, Lq, d)
        out = self.out_proj(out)
        return out
        
        
class FiLM(nn.Module):
    def __init__(self, text_dim, feat_dim):
        super().__init__()
        self.gamma = nn.Linear(text_dim, feat_dim)
        self.beta  = nn.Linear(text_dim, feat_dim)

    def forward(self, x, text):
        # x: (N, C, T, V)
        # text: (N, text_dim)

        gamma = self.gamma(text).unsqueeze(-1).unsqueeze(-1)  # (N,C,1,1)
        beta  = self.beta(text).unsqueeze(-1).unsqueeze(-1)

        return gamma * x + beta
    


