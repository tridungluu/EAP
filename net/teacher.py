import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class

from .utils.dstcl_loss import SpatialTemporalDisentanglement, DSTCLoss
from .utils.sta_loss import STALoss


class TeacherModel(nn.Module):
    """
    Teacher (Ranasingha et al., 2026), backbone CTR-GCN.
      Encoder (3 khoi) -> Latent Vector -> 2D Avg Pooling -> Context
      Context -> FC  -> logits (CE loss)        [Fig. 2d, FC nam trong encoder.fc]
      Context -> Decoder (2 khoi) -> chuoi tai tao (STA loss)   [Fig. 2e]
      Latent Vector -> Disentangle -> (C_s, C_t) -> DSTCL        [Fig. 1, 2c]
    """

    def __init__(self, base_encoder, num_class, num_point, num_person, graph, graph_args,
                 drop_out, root_joint=0, sta_stride=2, tau=0.125, rho=0.9,
                 ctx_ch=384, context_t_len=1, disentangle_t_len=24, **kwargs):
        super().__init__()
        base_encoder = import_class(base_encoder)
        # kwargs co the chua channels / strides cho CTREncoder
        self.skel_encoder = base_encoder(num_class=num_class, num_point=num_point,
                                         num_person=num_person, graph=graph,
                                         graph_args=graph_args, drop_out=drop_out, **kwargs)

        # Fig. 2d: Context -> FC -> CE. FC dat o day, fc cua encoder bo di.
        self.skel_encoder.fc = nn.Identity()
        self.drop = nn.Dropout(drop_out) if drop_out else nn.Identity()
        feat_ch = self.skel_encoder.out_channels
        self.classifier = nn.Linear(feat_ch, num_class)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_class))

        self.num_point = num_point
        self.context_t_len = context_t_len
        self.disentangle_t_len = disentangle_t_len
        self.decoder = self.skel_encoder.build_decoder()
        self.disentangle = SpatialTemporalDisentanglement(feat_ch, num_point, ch=ctx_ch, t_len=disentangle_t_len)
        self.dstcl_loss = DSTCLoss(num_class, ctx_ch, temperature=tau, momentum=rho)
        self.sta_loss = STALoss(anchor_joint=root_joint, stride=sta_stride)

    def forward(self, x):
        """
        Tra ve (output, context, recon_skel, (C_s, C_t)).
          output: (N, num_class) logits tu FC tren Context (pool trung binh theo T, V)
          context: (N, 256, context_t_len, V)
          recon_skel, C_s, C_t chi co khi model.train(); o eval la None.
        """
        N = x.shape[0]
        M = x.shape[-1] if x.dim() == 5 else 1

        _, feat, _ = self.skel_encoder(x)                    # Latent Vector (N*M, 256, T/4, V)
        _, C, T, V = feat.shape
        prepool_context = feat if M == 1 else feat.view(N, M, C, T, V).mean(dim=1)

        # Nhanh classifier/decoder: pre-pooling feature -> 2D Avg Pooling -> context (T=1).
        context = F.adaptive_avg_pool2d(prepool_context, (self.context_t_len, V))

        # Fig. 2d: Context -> FC
        output = self.classifier(self.drop(context.mean(dim=(2, 3))))

        if self.training:
            target_shape = x.shape[:4] if x.dim() == 5 else x.shape
            recon_skel = self.decoder(context, target_shape)         # Fig. 2e
            C_s, C_t = self.disentangle(prepool_context)          # DCL branch uses feature BEFORE 2D Avg Pooling
        else:
            recon_skel, C_s, C_t = None, None, None

        return output, context, recon_skel, (C_s, C_t)