import torch
import torch.nn as nn


class STALoss(nn.Module):
    """
    Spatial-Temporal Angular (STA) Loss, Section 3.2 (Eq. 9, 10, 11).

    So sánh cos của góc giữa vector (khớp gốc a -> khớp b) ở hai frame cách nhau
    lambda (= stride) giữa chuỗi thật và chuỗi tái tạo.
    """

    def __init__(self, anchor_joint=0, stride=2, eps=1e-8):
        super().__init__()
        self.anchor_joint = anchor_joint
        self.stride = stride
        self.eps = eps

    def compute_joint_cosines(self, x):
        """
        x: (B, C, T, V) hoặc (B, C, T, V, M)
        return: (B, num_pairs, V - 1)
        """
        if x.dim() == 5:
            x = x.mean(dim=-1)                                   # (B, C, T, V)

        B, C, T, V = x.shape
        lam = self.stride
        if T <= lam:
            return torch.zeros(B, 1, V - 1, device=x.device, dtype=x.dtype)

        root = x[:, :, :, self.anchor_joint:self.anchor_joint + 1]
        disp = x - root                                          # (B, C, T, V)

        non_anchor_idx = [i for i in range(V) if i != self.anchor_joint]
        disp = disp[:, :, :, non_anchor_idx]                     # (B, C, T, V-1)

        # Các cặp frame không chồng lấp (n_t, n_t + lambda)
        t_starts = torch.arange(0, T - lam, lam, device=x.device)
        if len(t_starts) == 0:
            t_starts = torch.tensor([0], device=x.device)
            lam = min(1, T - 1)

        d0 = disp[:, :, t_starts, :]                             # (B, C, P, V-1)
        d1 = disp[:, :, t_starts + lam, :]

        dot = (d0 * d1).sum(dim=1)                               # (B, P, V-1)
        n0 = torch.sqrt((d0 ** 2).sum(dim=1) + self.eps)         # tránh NaN khi vector = 0
        n1 = torch.sqrt((d1 ** 2).sum(dim=1) + self.eps)

        return torch.clamp(dot / (n0 * n1), -1.0, 1.0)

    def forward(self, input_skel, recon_skel):
        """
        Returns:
            loss: scalar, trung bình batch (Eq. 11)
            sample_sta: (B,) hệ số S_STA theo từng mẫu (đã detach)
        """
        cos_real = self.compute_joint_cosines(input_skel)
        cos_recon = self.compute_joint_cosines(recon_skel)

        diff = torch.abs(cos_real - cos_recon)                   # (B, P, V-1)
        sample_sta = diff.mean(dim=[-2, -1])                     # (B,)
        return sample_sta.mean(), sample_sta.detach()