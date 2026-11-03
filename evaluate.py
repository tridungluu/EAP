import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from net.encoder import StudentEncoder, ActionClassifier
from feeder.feeder_eap import FeederEAP

def evaluate_single_ratio(
    ratio,
    val_data_path,
    val_label_path,
    device,
    in_channels=3,
    latent_dim=256,
    num_joints=48,
    num_classes=16,
    batch_size=64,
):
    ckpt_path = f"work_dir/student_ratio_{int(ratio*100)}/student_best.pth"
    if not os.path.exists(ckpt_path):
        print(f"⚠️  Chưa tìm thấy checkpoint mốc {int(ratio*100)}% ({ckpt_path}). Bỏ qua.")
        return None, None, None

    # Khởi tạo mô hình gọn nhẹ chỉ gồm Encoder + Classifier
    student_enc = StudentEncoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
        num_joints=num_joints,
        num_layers=2,
    ).to(device)

    classifier = ActionClassifier(
        latent_dim=latent_dim,
        num_joints=num_joints,
        num_classes=num_classes,
    ).to(device)

    # Nạp checkpoint đã huấn luyện
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    student_enc.load_state_dict(ckpt["student_enc"])
    classifier.load_state_dict(ckpt["classifier"])

    student_enc.eval()
    classifier.eval()

    # Dùng đúng FeederEAP chuẩn 96 frames
    val_dataset = FeederEAP(val_data_path, val_label_path, ratio=ratio, target_time=96)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_gts = []

    with torch.no_grad():
        for partial_data, _, labels in val_loader:
            partial_data = partial_data.to(device)
            labels = labels.to(device)

            # Suy luận nhanh chỉ qua Student Encoder & Classifier
            out = student_enc(partial_data)
            v_student = out[1] if isinstance(out, (tuple, list)) else out

            logits = classifier(v_student)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    acc = np.mean(all_preds == all_gts) * 100.0

    return acc, all_preds, all_gts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("🚀 ĐÁNH GIÁ MÔ HÌNH NHẬN DẠNG SỚM (EAP) TRÊN CHUẨN 96 FRAMES")
    print(f"Thiết bị tính toán: {device}")
    print("=" * 60 + "\n")

    val_data_path = r"datacobot_v2/xsub/val_position.npy"
    val_label_path = r"datacobot_v2/xsub/val_label.pkl"
    os.makedirs("results", exist_ok=True)

    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    evaluated_ratios = []
    accuracies = []
    preds_dict = {}

    print(f"{'Tỷ lệ quan sát':<20} | {'Độ chính xác (Val Acc)':<25}")
    print("-" * 50)

    for r in ratios:
        acc, preds, gts = evaluate_single_ratio(r, val_data_path, val_label_path, device)
        if acc is not None:
            evaluated_ratios.append(r)
            accuracies.append(acc)
            preds_dict[r] = (preds, gts)
            print(f"{f'{int(r*100)}% frames':<20} | {acc:.2f}%")

    print("-" * 50)

    if len(accuracies) == 0:
        print("❌ Không tìm thấy checkpoint nào. Vui lòng kiểm tra lại work_dir!")
        return

    # Tính chỉ số AUC (tương thích NumPy 1.x và 2.x)
    if len(evaluated_ratios) >= 2:
        x = np.array(evaluated_ratios)
        y = np.array(accuracies)
        trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc_score = trapz_func(y, x) / (x[-1] - x[0])
        print(f"\n🏆 CHỈ SỐ AUC (Area Under Curve): {auc_score:.2f}%\n")
    else:
        auc_score = accuracies[0]
        print(f"\nℹ️ Có ít hơn 2 mốc quan sát, chưa đủ điểm dữ liệu tính AUC.\n")

    # Xuất biểu đồ đường cong nhận dạng sớm EAP
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(
        [r * 100 for r in evaluated_ratios],
        accuracies,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label=f"Student (AUC: {auc_score:.2f}%)",
    )
    plt.axhline(y=89.10, color="crimson", linestyle="--", label="Teacher (Full 100%: 89.10%)")
    plt.axhline(y=6.25, color="gray", linestyle=":", label="Random Guess (16 classes: 6.25%)")

    plt.title("Early Action Prediction Curve (CoboGesture - 96 Frames)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Observation Ratio (%)", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.xticks([20, 40, 60, 80, 100])
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)

    curve_path = "results/eap_curve.png"
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    print(f"📊 Đã lưu biểu đồ EAP Curve tại: {curve_path}")

    # Xuất Confusion Matrix tại mốc 20%
    if 0.2 in preds_dict:
        preds_20, gts_20 = preds_dict[0.2]
        cm = confusion_matrix(gts_20, preds_20, normalize="true") * 100
        plt.figure(figsize=(10, 8), dpi=300)
        sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
        plt.title("Confusion Matrix @ 20% Observation (Student)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Predicted Class", fontsize=11)
        plt.ylabel("True Class", fontsize=11)
        cm_path = "results/confusion_matrix_ratio_20.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        print(f"📊 Đã lưu Confusion Matrix (20%) tại: {cm_path}")

    print("\n🎉 HOÀN TẤT ĐÁNH GIÁ THÀNH CÔNG!")

if __name__ == "__main__":
    main()