import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTemporalDisentanglement(nn.Module):
    """
    Fig. 1. Input latent (B, C, T, V) -> C_s, C_t, mỗi cái (B, ch). ch phải chia hết cho V.

    Spatial : pool theo T -> BN -> FC (C -> ch/V) -> flatten
    Temporal: pool theo C -> BN -> FC (T -> ch/V) -> flatten
    (bài bỏ conv, chỉ giữ FC)
    """

    def __init__(self, in_channels, num_joints, ch=384, t_len=24):
        super().__init__()
        assert ch % num_joints == 0, "ch phải chia hết cho số khớp V"
        self.per_joint = ch // num_joints
        self.unified_ch = ch
        self.t_len = t_len

        self.spatial_bn = nn.BatchNorm2d(in_channels)
        self.spatial_fc = nn.Conv2d(in_channels, self.per_joint, kernel_size=1)

        self.temporal_bn = nn.BatchNorm2d(1)
        self.temporal_fc = nn.Linear(t_len, self.per_joint)

    def forward(self, x):
        B, C, T, V = x.shape

        xs = x.mean(dim=2, keepdim=True)                       # (B, C, 1, V)
        xs = self.spatial_fc(self.spatial_bn(xs))              # (B, ch/V, 1, V)
        C_s = xs.flatten(1)                                    # (B, ch)

        xt = x.mean(dim=1, keepdim=True)                       # (B, 1, T, V)
        if T != self.t_len:
            xt = F.adaptive_avg_pool2d(xt, (self.t_len, V))
        xt = self.temporal_bn(xt).squeeze(1)                   # (B, t_len, V)
        xt = self.temporal_fc(xt.transpose(1, 2))              # (B, V, ch/V)
        C_t = xt.flatten(1)                                    # (B, ch)
        return C_s, C_t


class DSTCLoss(nn.Module):
    """Dynamic Spatial-Temporal Contrastive Loss, Eq. (1)-(8)."""

    def __init__(self, num_classes, feat_dim, temperature=0.125, momentum=0.9):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.tau = temperature
        self.rho = momentum
        for d in ("spatial", "temporal"):
            self.register_buffer(f"G_{d}", torch.zeros(num_classes, feat_dim))
            self.register_buffer(f"G_init_{d}", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def _update(self, z, labels, is_tp, G, G_init):
        """Eq. (2): EMA của prototype lớp k từ các mẫu True Positive."""
        oh = F.one_hot(labels, self.num_classes).float() * is_tp.float().unsqueeze(1)  # (B, K)
        cnt = oh.sum(0)                                                                  # (K,)
        has = cnt > 0
        mean = F.normalize((oh.t() @ z) / cnt.clamp(min=1).unsqueeze(1), dim=-1)
        ema = (1.0 - self.rho) * mean + self.rho * G
        new = F.normalize(torch.where(G_init.unsqueeze(1), ema, mean), dim=-1)
        G.copy_(torch.where(has.unsqueeze(1), new, G))
        G_init.copy_(G_init | has)

    def _dcl(self, z, labels, preds, probs, s_sta, domain):
        K, tau = self.num_classes, self.tau
        z = F.normalize(z, dim=-1)
        is_tp = preds == labels

        G = getattr(self, f"G_{domain}")
        G_init = getattr(self, f"G_init_{domain}")

        # Similarity voi prototype LICH SU (truoc khi cap nhat bang batch hien tai),
        # de mau moi khong tu "ro ri" vao prototype cua chinh no.
        sim = z @ F.normalize(G, dim=-1).t()                    # dis(z_i, G_l): (B, K)

        # FN của lớp k: label == k, pred != k. FP của lớp k: pred == k, label != k  (Eq. 4)
        oh_label = F.one_hot(labels, K)
        notp = (~is_tp).float().unsqueeze(1)
        fn_m = oh_label.float() * notp
        fp_m = F.one_hot(preds, K).float() * notp
        n_fn, n_fp = fn_m.sum(0), fp_m.sum(0)                   # (K,)
        F_N = (fn_m.t() @ z) / n_fn.clamp(min=1).unsqueeze(1)   # (K, D)
        F_P = (fp_m.t() @ z) / n_fp.clamp(min=1).unsqueeze(1)

        # Eq. (5): D_N, D_P có gradient (kéo FN lại gần / đẩy FP ra xa mẫu tin cậy)
        D_N = F.cosine_similarity(F_N[labels], z, dim=-1)
        D_P = F.cosine_similarity(F_P[labels], z, dim=-1)

        # Eq. (6), (7): S_STA là hệ số co giãn, không truyền gradient
        s_sta = s_sta.detach()
        phi = (1.0 - D_N) * s_sta * (is_tp & (n_fn[labels] > 0)).float()
        psi = (1.0 + D_P) * s_sta * (is_tp & (n_fp[labels] > 0)).float()

        p_ik = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        pos = sim.gather(1, labels.unsqueeze(1)).squeeze(1) / tau
        log_neg = torch.logsumexp(
            (sim / tau).masked_fill(oh_label.bool(), float("-inf")), dim=1)   # log sum_{l!=k} e^{.}

        # Eq. (8): -log(e^a / (e^a + O)) = softplus(log O - a)
        loss = F.softplus(log_neg - (pos - (1.0 - p_ik) * phi)) \
             + F.softplus(log_neg - (pos - (1.0 - p_ik) * psi))
        # Eq. (2): cap nhat EMA SAU khi tinh loss
        if self.training:
            self._update(z.detach(), labels, is_tp, G, G_init)
        return loss.mean()

    def forward(self, C_s, C_t, labels, logits, S_STA):
        """
        logits: (B, K) logits dùng để quyết định TP/FP/FN và p_ik (không truyền gradient).
        Eq. (1): L_DSTCL = DCL(C_s) + DCL(C_t)
        """
        logits = logits.detach()
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        return (self._dcl(C_s, labels, preds, probs, S_STA, "spatial")
                + self._dcl(C_t, labels, preds, probs, S_STA, "temporal"))