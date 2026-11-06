import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDistiller(nn.Module):
    """Base class for all distillers"""
    def __init__(self, weight=1.0):
        super().__init__()
        self.weight = weight

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class CosineDistiller(BaseDistiller):
    """Cosine similarity distillation between student and teacher"""
    def __init__(self, weight=1.0):
        super().__init__(weight)

    def forward(self, skel_feat_student, skel_feat_teacher):
        """
        Args:
            skel_feat_student: (batch_size, feat_dim)
            skel_feat_teacher: (batch_size, feat_dim)
        Returns:
            loss: scalar
        """
        loss = 1 - F.cosine_similarity(skel_feat_student, skel_feat_teacher, dim=-1).mean()
        return self.weight * loss


class AlignDistiller(BaseDistiller):
    """Alignment distillation with text-guided weighting"""
    def __init__(self, weight=1.0):
        super().__init__(weight)

    def forward(self, skel_feat_student, text_feat_teacher, skel_feat_teacher):
        """
        Args:
            skel_feat_student: (batch_size, feat_dim)
            text_feat_teacher: (batch_size, feat_dim)
            skel_feat_teacher: (batch_size, feat_dim)
        Returns:
            loss: scalar
        """
        sim_align = F.cosine_similarity(skel_feat_student, text_feat_teacher, dim=-1)
        weight_align = torch.relu(sim_align).unsqueeze(-1)  # (batch_size, 1)
        fused_skel_feat_student = weight_align * skel_feat_student
        loss = 1 - F.cosine_similarity(fused_skel_feat_student, skel_feat_teacher, dim=-1).mean()
        return self.weight * loss


class SpatialFeatureDistiller(BaseDistiller):
    """Mid-layer spatial feature distillation using MSE loss"""
    def __init__(self, weight=1.0):
        super().__init__(weight)

    def forward(self, P_mid, mid_feat):
        """
        Args:
            P_mid: (batch_size, C, T, V) - student mid features
            mid_feat: (batch_size, C, T, V) - teacher mid features
        Returns:
            loss: scalar
        """
        loss = F.mse_loss(P_mid, mid_feat)
        return self.weight * loss


class ContextDistiller(BaseDistiller):
    """
    Context Distillation Loss (L_CD) via MAE (L1 loss), Eq. (15) của Ranasingha et al. (2026):
    L_CD = (1 / N) * sum |v_i - v_hat_i|
    v_i, v_hat_i: biểu diễn context của teacher và student.
    """
    def __init__(self, weight=50.0):
        super().__init__(weight)

    def forward(self, v_student, v_teacher):
        loss = F.l1_loss(v_student, v_teacher)
        return self.weight * loss


class SequenceReconstructionDistiller(BaseDistiller):
    """
    Sequence Reconstruction Loss (L_Recon) via MSE, Eq. (14) của Ranasingha et al. (2026):
    L_Recon = (1 / N) * sum (y_i - y_hat_i)^2
    """
    def __init__(self, weight=1.0):
        super().__init__(weight)

    def forward(self, recon_student, target_skel):
        # Chi tinh tren cac frame da quan sat: frame toan 0 la padding cua Partial Sequence
        valid = (target_skel.abs().sum(dim=(1, 3)) > 0).float()            # (B, T)
        se = ((recon_student - target_skel) ** 2).mean(dim=(1, 3))         # (B, T)
        loss = (se * valid).sum() / valid.sum().clamp(min=1.0)
        return self.weight * loss


class KDDistiller(BaseDistiller):
    def __init__(self, weight=1.0):
        super().__init__(weight)

    def forward(self, sim_s2t, sim_s2t_teacher, temp):
        soft_targets_teacher = F.softmax(sim_s2t_teacher / temp, dim=-1)
        soft_prob_student = F.log_softmax(sim_s2t / temp, dim=-1)
        loss = torch.sum(soft_targets_teacher * (soft_targets_teacher.log() - soft_prob_student)) / soft_prob_student.size()[0]

        return self.weight * loss


class HardKDDistiller(BaseDistiller):
    def __init__(self, weight=1.0):
        super().__init__(weight)

    def forward(self, sim_s2t, sim_s2t_teacher):
        loss = F.cross_entropy(sim_s2t, sim_s2t_teacher.argmax(dim=1))  # for each skel, find highest match text
        return self.weight * loss


class DKDDistiller(BaseDistiller):
    def __init__(self, weight=1.0):
        super().__init__(weight)
        self.EPS = 1e-8

    def forward(self, sim_s2t, sim_s2t_teacher, groundtruth, temp):
        p_teacher = F.softmax(sim_s2t_teacher / temp, dim=-1)
        p_student = F.softmax(sim_s2t / temp, dim=-1)

        target_mask = torch.zeros_like(p_teacher).scatter_(
            1,
            groundtruth.unsqueeze(1).to(torch.int64),
            1
        )

        log_p_student_target = torch.log(p_student + self.EPS)
        log_p_teacher_target = torch.log(p_teacher + self.EPS)

        tckd = F.kl_div(
            log_p_student_target * target_mask,
            log_p_teacher_target * target_mask,
            reduction='sum',
            log_target=True
        ) * (temp ** 2) / groundtruth.size(0)

        non_target_mask = 1 - target_mask

        log_p_student_nontarget = torch.log(p_student + self.EPS)
        log_p_teacher_nontarget = torch.log(p_teacher + self.EPS)

        nckd = F.kl_div(
            log_p_student_nontarget * non_target_mask,
            log_p_teacher_nontarget * non_target_mask,
            reduction='sum',
            log_target=True
        ) * (temp ** 2) / groundtruth.size(0)

        loss = tckd + nckd
        return self.weight * loss


class CTKDDistiller(BaseDistiller):
    def __init__(self, weight=1.0):
        super().__init__(weight)
        self.EPS = 1e-8
        self.tau = 0.2
        self.tau_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('base_tau', torch.tensor(self.tau))

    def forward(self, sim_s2t, sim_s2t_teacher, temp):
        eps = self.EPS
        max_logit = 20.0

        logit_gap = torch.norm(sim_s2t - sim_s2t_teacher, p=2, dim=1).mean()
        scaled_gap = logit_gap / (self.tau_scale.abs() + eps)
        dynamic_T = 1.0 + self.base_tau * torch.sigmoid(scaled_gap)
        dynamic_T = torch.clamp(dynamic_T, min=0.5, max=5.0)

        student_logits = sim_s2t / (dynamic_T + eps)
        teacher_logits = sim_s2t_teacher / (dynamic_T + eps)

        student_logits = torch.clamp(student_logits, min=-max_logit, max=max_logit)
        teacher_logits = torch.clamp(teacher_logits, min=-max_logit, max=max_logit)

        student_log_probs = F.log_softmax(student_logits, dim=1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=1)

        valid_mask = (teacher_log_probs.isfinite() & student_log_probs.isfinite()).all(dim=1)
        if valid_mask.sum() == 0:
            distillation_loss = torch.tensor(1e-5, device=sim_s2t.device)
        else:
            distillation_loss = F.kl_div(
                student_log_probs[valid_mask],
                teacher_log_probs[valid_mask],
                reduction='batchmean',
                log_target=True
            ) * (dynamic_T ** 2)
        loss = torch.clamp(distillation_loss, max=1e4)
        return self.weight * loss


class FeatureDistillationManager(nn.Module):
    """Manager for feature distillers"""
    def __init__(self, distill_config=None):
        """
        Args:
            distill_config: dict with weights for each type
                Example: {'cosine': 1.0, 'align': 0.5, 'spatial': 1.0,
                          'context_distill': 50.0, 'recon': 1.0}
                Types with weight > 0 will be enabled, weight <= 0 will be disabled
        """
        super().__init__()
        self.distillers = nn.ModuleDict()

        if distill_config is None:
            distill_config = {}

        if distill_config.get('cosine', 0) > 0:
            self.distillers['cosine'] = CosineDistiller(weight=distill_config['cosine'])

        if distill_config.get('align', 0) > 0:
            self.distillers['align'] = AlignDistiller(weight=distill_config['align'])

        if distill_config.get('spatial', 0) > 0:
            self.distillers['spatial'] = SpatialFeatureDistiller(weight=distill_config['spatial'])

        # [Ranasingha et al., 2026] Context Distillation & Reconstruction
        if distill_config.get('context_distill', 0) > 0:
            self.distillers['context_distill'] = ContextDistiller(weight=distill_config['context_distill'])

        if distill_config.get('recon', 0) > 0:
            self.distillers['recon'] = SequenceReconstructionDistiller(weight=distill_config['recon'])

    def forward(self, skel_feat_student=None, skel_feat_teacher=None,
                text_feat_teacher=None, P_mid=None, mid_feat=None,
                recon_student=None, target_skel=None,
                ctx_student=None, ctx_teacher=None):
        """
        Args:
            skel_feat_student: context của student
            skel_feat_teacher: context của teacher
            text_feat_teacher: text feature của teacher (cho align)
            P_mid, mid_feat: feature tầng giữa của student / teacher (cho spatial)
            recon_student: chuỗi student tái tạo, (B, C, T, V)
            target_skel: chuỗi đích, (B, C, T, V)
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        total_loss = 0.0
        loss_dict = {}

        if 'cosine' in self.distillers:
            loss = self.distillers['cosine'](skel_feat_student, skel_feat_teacher)
            total_loss += loss
            loss_dict['loss_cosine'] = loss.detach()

        if 'align' in self.distillers:
            if text_feat_teacher is None:
                raise ValueError("text_feat_teacher required for align distillation")
            loss = self.distillers['align'](skel_feat_student, text_feat_teacher, skel_feat_teacher)
            total_loss += loss
            loss_dict['loss_align'] = loss.detach()

        if 'spatial' in self.distillers:
            if P_mid is None or mid_feat is None:
                raise ValueError("P_mid and mid_feat required for spatial distillation")
            loss = self.distillers['spatial'](P_mid, mid_feat)
            total_loss += loss
            loss_dict['loss_spatial'] = loss.detach()

        if 'context_distill' in self.distillers:
            if ctx_student is None or ctx_teacher is None:
                raise ValueError("ctx_student and ctx_teacher required for context distillation")
            loss = self.distillers['context_distill'](ctx_student, ctx_teacher)
            total_loss += loss
            loss_dict['loss_context_distill'] = loss.detach()

        if 'recon' in self.distillers:
            if recon_student is None or target_skel is None:
                raise ValueError("recon_student and target_skel required for recon loss")
            loss = self.distillers['recon'](recon_student, target_skel)
            total_loss += loss
            loss_dict['loss_recon'] = loss.detach()

        return total_loss, loss_dict


class LogitDistillationManager(nn.Module):
    """Manager for logit-based distillers (KD, HardKD, DKD, CTKD)"""

    def __init__(self, distill_config=None):
        super().__init__()
        self.distillers = nn.ModuleDict()

        if distill_config is None:
            distill_config = {}

        if distill_config.get('kd', 0) > 0:
            self.distillers['kd'] = KDDistiller(weight=distill_config['kd'])

        if distill_config.get('hardkd', 0) > 0:
            self.distillers['hardkd'] = HardKDDistiller(weight=distill_config['hardkd'])

        if distill_config.get('dkd', 0) > 0:
            self.distillers['dkd'] = DKDDistiller(weight=distill_config['dkd'])

        if distill_config.get('ctkd', 0) > 0:
            self.distillers['ctkd'] = CTKDDistiller(weight=distill_config['ctkd'])

    def forward(self, sim_s2t, sim_s2t_teacher, groundtruth, temp):
        """
        Args:
            sim_s2t:         (B, B) student similarity matrix
            sim_s2t_teacher: (B, B) teacher similarity matrix
            groundtruth:     (B,)   arange labels
            temp:            float  temperature
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses
        """
        total_loss = 0.0
        loss_dict = {}

        if 'kd' in self.distillers:
            loss = self.distillers['kd'](sim_s2t, sim_s2t_teacher, temp)
            total_loss += loss
            loss_dict['loss_kd'] = loss.detach()

        if 'hardkd' in self.distillers:
            loss = self.distillers['hardkd'](sim_s2t, sim_s2t_teacher)
            total_loss += loss
            loss_dict['loss_hardkd'] = loss.detach()

        if 'dkd' in self.distillers:
            if groundtruth is None:
                raise ValueError("groundtruth required for DKD distillation")
            loss = self.distillers['dkd'](sim_s2t, sim_s2t_teacher, groundtruth, temp)
            total_loss += loss
            loss_dict['loss_dkd'] = loss.detach()

        if 'ctkd' in self.distillers:
            loss = self.distillers['ctkd'](sim_s2t, sim_s2t_teacher, temp)
            total_loss += loss
            loss_dict['loss_ctkd'] = loss.detach()

        return total_loss, loss_dict