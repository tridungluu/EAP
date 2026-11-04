# processor/distillers_eap.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionLoss(nn.Module):
    """
    Reconstruction Loss giữa khung xương tái tạo (B, 3, 1, V)
    và Ground-Truth skeleton gốc sau khi đã nén theo chiều T: (B, 3, 1, V).
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, pred_skeleton, gt_skeleton):
        # gt_skeleton gốc: (B, 3, T, V)
        gt_target = torch.mean(gt_skeleton, dim=2, keepdim=True)  # (B, 3, 1, V)
        return self.loss_fn(pred_skeleton, gt_target)


class STALoss(nn.Module):
    def __init__(self, time_step=2, root_joint=0, eps=1e-7):
        super(STALoss, self).__init__()
        self.time_step = time_step
        self.root_joint = root_joint
        self.eps = eps

    def compute_angles(self, skeleton):
        # skeleton: (B, 3, T, V)
        # 1. Lấy mỏ neo tại Cổ tay (index 0): (B, 3, T, 1)
        root = skeleton[:, :, :, self.root_joint:self.root_joint+1]
        
        # 2. Vector tương đối từ Cổ tay tới các khớp ngón tay / khớp còn lại
        rel_vec = skeleton - root  # (B, 3, T, V)
        
        # 3. Chuyển động góc qua time_step (lambda)
        v1 = rel_vec[:, :, :-self.time_step, :]
        v2 = rel_vec[:, :, self.time_step:, :]
        
        dot = torch.sum(v1 * v2, dim=1)
        norm_v1 = torch.norm(v1, dim=1) + self.eps
        norm_v2 = torch.norm(v2, dim=1) + self.eps
        cos_sim = torch.clamp(dot / (norm_v1 * norm_v2), -1.0 + self.eps, 1.0 - self.eps)
        return torch.acos(cos_sim)

    def forward(self, pred_skeleton, gt_skeleton):
        if pred_skeleton.shape[2] == 1:
            gt_target = torch.mean(gt_skeleton, dim=2, keepdim=True)
            loss_sta = F.l1_loss(pred_skeleton, gt_target)
        else:
            pred_angles = self.compute_angles(pred_skeleton)
            gt_angles = self.compute_angles(gt_skeleton)
            loss_sta = F.l1_loss(pred_angles, gt_angles)
            
        with torch.no_grad():
            s_sta = torch.sigmoid(loss_sta).item()
            
        return loss_sta, s_sta


class DCLLoss(nn.Module):
    def __init__(self, num_classes=16, temperature=0.125, momentum=0.9):
        super(DCLLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.momentum = momentum
        
        # Biểu diễn toàn cục G_k cho từng nhánh
        self.register_buffer("G_spatial", torch.zeros(num_classes, 128))
        self.register_buffer("G_temporal", torch.zeros(num_classes, 128))
        
        # Tách riêng 2 cờ khởi tạo cho 2 nhánh tránh xung đột
        self.register_buffer("init_spatial", torch.zeros(num_classes, dtype=torch.bool))
        self.register_buffer("init_temporal", torch.zeros(num_classes, dtype=torch.bool))

    def _dcl_branch(self, feats, logits, labels, G_bank, init_flags, s_sta):
        feats = F.normalize(feats, dim=-1)
        preds = torch.argmax(logits, dim=-1)
        total_loss = torch.tensor(0.0, device=feats.device)
        valid_classes = 0

        for k in range(self.num_classes):
            mask_gt = (labels == k)
            mask_pred = (preds == k)

            tp_idx = torch.where(mask_gt & mask_pred)[0]
            fn_idx = torch.where(mask_gt & (~mask_pred))[0]
            fp_idx = torch.where((~mask_gt) & mask_pred)[0]

            if len(tp_idx) == 0:
                continue

            # Mẫu tự tin (TP)
            tp_feats = feats[tp_idx]
            mean_tp = torch.mean(tp_feats, dim=0)

            # Cập nhật biểu diễn toàn cục G_k bằng EMA
            if not init_flags[k]:
                G_bank[k] = mean_tp.detach()
                init_flags[k] = True
            else:
                G_bank[k] = self.momentum * G_bank[k] + (1 - self.momentum) * mean_tp.detach()
            G_k = F.normalize(G_bank[k].unsqueeze(0), dim=-1) # (1, 128)

            # 1. Tính D_N (FN độ tương đồng với TP)
            if len(fn_idx) > 0:
                mean_fn = F.normalize(torch.mean(feats[fn_idx], dim=0, keepdim=True), dim=-1)
                d_n = torch.clamp(torch.sum(mean_fn * G_k), -1.0, 1.0).item()
            else:
                d_n = 0.5
            phi = float(torch.clamp(torch.tensor((1.0 - d_n) * s_sta), 0.0, 5.0))

            # 2. Tính D_P (FP độ tương đồng với TP)
            if len(fp_idx) > 0:
                mean_fp = F.normalize(torch.mean(feats[fp_idx], dim=0, keepdim=True), dim=-1)
                d_p = torch.clamp(torch.sum(mean_fp * G_k), -1.0, 1.0).item()
            else:
                d_p = 0.5
            psi = float(torch.clamp(torch.tensor((1.0 + d_p) * s_sta), 0.0, 5.0))

            # Tính độ tương đồng cosin an toàn
            pos_sim = torch.clamp(torch.sum(tp_feats * G_k, dim=-1), -1.0, 1.0) / self.temperature
            pos_sim = torch.clamp(pos_sim, max=25.0)
            exp_pos = torch.exp(pos_sim) * (1.0 + phi)

            # Phạt các mẫu âm tính
            neg_mask = (labels != k)
            if torch.any(neg_mask):
                neg_feats = feats[neg_mask]
                neg_sim = torch.clamp(torch.matmul(tp_feats, neg_feats.T), -1.0, 1.0) / self.temperature
                neg_sim = torch.clamp(neg_sim, max=25.0)
                exp_neg = torch.sum(torch.exp(neg_sim), dim=-1) * (1.0 + psi)
            else:
                exp_neg = torch.zeros_like(exp_pos)

            denom = exp_pos + exp_neg + 1e-8
            loss_k = -torch.mean(torch.log(torch.clamp(exp_pos / denom, min=1e-8, max=1.0)))
            if not torch.isnan(loss_k) and not torch.isinf(loss_k):
                total_loss += loss_k
                valid_classes += 1

        if valid_classes > 0:
            return total_loss / valid_classes
        return total_loss

    def forward(self, C_s, C_t, logits, labels, s_sta):
        loss_s = self._dcl_branch(C_s, logits, labels, self.G_spatial, self.init_spatial, s_sta)
        loss_t = self._dcl_branch(C_t, logits, labels, self.G_temporal, self.init_temporal, s_sta)
        return loss_s + loss_t


class ContextDistillationLoss(nn.Module):
    """
    Context Distillation Loss (L_CD) cho Student.
    So sánh context giữa Student và Teacher: (B, 256, 48)
    L_CD = (1 / N) * sum(|v_i - v_hat_i|)
    """
    def __init__(self):
        super(ContextDistillationLoss, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, student_context, teacher_context):
        return self.mae(student_context, teacher_context.detach())