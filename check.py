# test_losses.py (thử nghiệm giả lập có mẫu đoán đúng)
import torch
from processor.distillers_eap import DCLLoss

dcl_criterion = DCLLoss(num_classes=16)

# Giả lập 4 mẫu với nhãn [0, 1, 2, 0]
labels = torch.tensor([0, 1, 2, 0])

# Ép logits dự đoán đúng nhãn để sinh ra các mẫu TP:
logits = torch.zeros(4, 16)
logits[0, 0] = 5.0  # Mẫu 0 đoán đúng lớp 0 (TP)
logits[1, 1] = 5.0  # Mẫu 1 đoán đúng lớp 1 (TP)
logits[2, 3] = 5.0  # Mẫu 2 nhãn 2 nhưng đoán nhầm thành 3 (FN của lớp 2, FP của lớp 3)
logits[3, 0] = 5.0  # Mẫu 3 đoán đúng lớp 0 (TP)

C_s = torch.randn(4, 128)
C_t = torch.randn(4, 128)
s_sta = 0.6848

loss_dcl = dcl_criterion(C_s, C_t, logits, labels, s_sta)
print(f"✅ DCL Loss khi có mẫu TP: {loss_dcl.item():.4f}")