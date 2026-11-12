import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class



class StudentModel(nn.Module):
    """
    Student: encoder CTR-GCN 2 khoi -> 2D Avg Pooling -> Context -> (FC classifier, decoder).
    Classifier giong teacher (FC tren Context pool). Decoder khoi tao tu teacher roi fine-tune.
    Inference chi dung encoder + classifier.
    Tra ve (output, context, recon).
    """

    def __init__(self, base_encoder, num_class, num_point, num_person, graph, graph_args,
                 drop_out, feature_distillation=None,
                 enc_channels=(128, 256), enc_strides=(2, 2), context_t_len=1, **kwargs):
        super().__init__()
        enc = import_class(base_encoder)
        self.skel_encoder = enc(num_class=num_class, num_point=num_point, num_person=num_person,
                                graph=graph, graph_args=graph_args, drop_out=drop_out,
                                channels=tuple(enc_channels), strides=tuple(enc_strides),
                                **kwargs)
        # Classifier giong teacher: Context -> FC (fc cua encoder bo di)
        self.skel_encoder.fc = nn.Identity()
        self.drop = nn.Dropout(drop_out) if drop_out else nn.Identity()
        self.classifier = nn.Linear(self.skel_encoder.out_channels, num_class)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2.0 / num_class))

        self.context_t_len = context_t_len        # phai bang context_t_len cua teacher
        self.decoder = self.skel_encoder.build_decoder()

    def init_decoder_from_teacher(self, teacher):
        """Bai bao: fine-tune decoder pre-trained cua teacher lam decoder cua student."""
        self.decoder.load_state_dict(teacher.decoder.state_dict())

    @staticmethod
    def _to_ctr_input(x):
        # (B, T, V, C) -> (B, C, T, V, 1); (B, C, T, V) -> (B, C, T, V, 1); 5 chieu giu nguyen
        if x.dim() == 4:
            if x.shape[-1] == 3 and x.shape[1] != 3:
                x = x.permute(0, 3, 1, 2)
            x = x.unsqueeze(-1)
        return x.contiguous()

    def forward(self, x, target_shape=None):
        x = self._to_ctr_input(x)
        N, _, T, V, M = x.shape
        _, feat, _ = self.skel_encoder(x)                    # Latent (N*M, 256, T/4, V)
        if M > 1:
            feat = feat.view(N, M, *feat.shape[1:]).mean(1)

        # Classifier/decoder branch: encoder feature -> 2D Avg Pooling -> context (T=1).
        context = F.adaptive_avg_pool2d(feat, (self.context_t_len, V))
        output = self.classifier(self.drop(context.mean(dim=(2, 3))))

        recon = None
        if self.training or target_shape is not None:
            t_out = int(target_shape[2]) if target_shape is not None else T
            recon = self.decoder(context, (N, 3, t_out, V))      # (N, 3, T, V)

        return output, context, recon