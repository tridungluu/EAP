import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Đảm bảo nhận diện đúng thư mục gốc dự án
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Nạp kiến trúc Student Encoder và Classifier từ thư mục net của bạn
from net.encoder import StudentEncoder, ActionClassifier


def load_val_data(data_dir=r"D:\Thesis2\datacobot_v2\xsub"):
    """
    Nạp tập dữ liệu kiểm thử Validation của CoboGesture.
    Hỗ trợ xử lý an toàn mảng 4D hoặc 5D.
    """
    pos_path = os.path.join(data_dir, "val_position.npy")
    lbl_path = os.path.join(data_dir, "val_label.pkl")

    if not os.path.exists(pos_path) or not os.path.exists(lbl_path):
        raise FileNotFoundError(f"Không tìm thấy tập dữ liệu kiểm thử tại: {data_dir}")

    positions = np.load(pos_path)  # Kỳ vọng dạng: (N, C, T, V) hoặc (N, C, T, V, M)

    if positions.ndim == 5:
        # Nếu có chiều người M (nhân vật), lấy chủ thể đầu tiên
        positions = positions[..., 0]

    if positions.ndim != 4:
        raise ValueError(f"Dữ liệu cần có dạng (N, C, T, V), nhưng nhận được shape: {positions.shape}")

    with open(lbl_path, "rb") as f:
        labels = pickle.load(f)

    if isinstance(labels, tuple):
        labels = labels[1]
    labels = np.asarray(labels, dtype=np.int64)

    return positions, labels


def evaluate_single_ratio(
    ratio,
    val_positions,
    val_labels,
    device,
    in_channels=3,
    latent_dim=256,
    num_joints=48,
    num_classes=16,
    batch_size=64,
):
    """
    Suy luận thời gian thực cho 1 mốc tỷ lệ quan sát (ratio):
    Chỉ dùng Student Encoder + Classifier (đã lược bỏ Decoder và Teacher).
    """
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

    # Nạp trọng số checkpoint đã train
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    student_enc.load_state_dict(ckpt["student_enc"])
    classifier.load_state_dict(ckpt["classifier"])

    student_enc.eval()
    classifier.eval()

    N, C, T, V = val_positions.shape
    cutoff_T = max(1, int(T * ratio))

    # Cắt chuỗi khung hình theo đúng tỷ lệ ratio quan sát
    partial_pos = np.copy(val_positions)
    partial_pos[:, :, cutoff_T:, :] = 0.0

    all_preds = []
    total_samples = len(val_labels)

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_x = torch.tensor(
                partial_pos[i : i + batch_size], dtype=torch.float32
            ).to(device)

            # Suy luận: Chỉ qua Student Encoder để lấy context vector, rồi qua Classifier
            out = student_enc(batch_x)
            v_student = out[1] if isinstance(out, (tuple, list)) else out

            logits = classifier(v_student)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    all_preds = np.array(all_preds)
    acc = np.mean(all_preds == val_labels) * 100.0

    return acc, all_preds, val_labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("🚀 ĐÁNH GIÁ MÔ HÌNH NHẬN DẠNG SỚM (EAP) & TÍNH TOÁN AUC")
    print(f"Thiết bị tính toán: {device}")
    print("=" * 60 + "\n")

    os.makedirs("results", exist_ok=True)

    # 1. Nạp dữ liệu Validation
    try:
        val_positions, val_labels = load_val_data()
        print(f"✅ Đã nạp tập Validation: {val_positions.shape[0]} mẫu, {val_positions.shape[2]} frames, {val_positions.shape[3]} joints.\n")
    except Exception as e:
        print(f"❌ Lỗi khi nạp dữ liệu: {e}")
        return

    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    evaluated_ratios = []
    accuracies = []
    preds_dict = {}

    print(f"{'Tỷ lệ quan sát':<20} | {'Độ chính xác (Val Acc)':<25}")
    print("-" * 50)

    for r in ratios:
        acc, preds, gts = evaluate_single_ratio(r, val_positions, val_labels, device)
        if acc is not None:
            evaluated_ratios.append(r)
            accuracies.append(acc)
            preds_dict[r] = (preds, gts)
            print(f"{f'{int(r*100)}% frames':<20} | {acc:.2f}%")

    print("-" * 50)

    if len(accuracies) == 0:
        print("❌ Không tìm thấy checkpoint nào trong work_dir/student_ratio_*/. Hãy kiểm tra lại đường dẫn!")
        return

    # 2. Tính diện tích dưới đường cong (AUC) tương thích NumPy 1.x và 2.x
    if len(evaluated_ratios) >= 2:
        x = np.array(evaluated_ratios)
        y = np.array(accuracies)
        # Tự động chọn hàm phù hợp cho cả NumPy mới (>=2.0) và cũ
        trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc_score = trapz_func(y, x) / (x[-1] - x[0])
        print(f"\n🏆 CHỈ SỐ AUC (Area Under Curve): {auc_score:.2f}%\n")
    else:
        auc_score = accuracies[0]
        print(f"\nℹ️ Chỉ có 1 mốc quan sát, chưa đủ điểm dữ liệu để tính AUC hoàn chỉnh.\n")

    # 3. Xuất biểu đồ đường cong nhận dạng sớm (EAP Curve)
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(
        [r * 100 for r in evaluated_ratios],
        accuracies,
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        label=f"Student (AUC: {auc_score:.2f}%)",
    )

    # Đường mốc Teacher (89.10%) và Random Guess (16 lớp = 6.25%)
    plt.axhline(
        y=89.10,
        color="crimson",
        linestyle="--",
        label="Teacher (Full 100% Sequence: 89.10%)",
    )
    plt.axhline(
        y=6.25,
        color="gray",
        linestyle=":",
        label="Random Guess (16 classes: 6.25%)",
    )

    plt.title("Early Action Prediction Curve (CoboGesture)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Observation Ratio (%)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xticks([20, 40, 60, 80, 100])
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)

    curve_path = "results/eap_curve.png"
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    print(f"📊 Đã lưu biểu đồ EAP Curve tại: {curve_path}")

    # 4. Xuất Ma trận nhầm lẫn (Confusion Matrix) cho mốc sớm 20%
    if 0.2 in preds_dict:
        preds_20, gts_20 = preds_dict[0.2]
        cm = confusion_matrix(gts_20, preds_20, normalize="true") * 100

        plt.figure(figsize=(10, 8), dpi=300)
        sns.heatmap(cm, annot=False, cmap="Blues", cbar=True)
        plt.title("Confusion Matrix @ 20% Observation Ratio (Student)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("True Class", fontsize=12)

        cm_path = "results/confusion_matrix_ratio_20.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        print(f"📊 Đã lưu Confusion Matrix (20%) tại: {cm_path}")

    print("\n🎉 HOÀN TẤT ĐÁNH GIÁ VÀ XUẤT KẾT QUẢ THÀNH CÔNG!")


if __name__ == "__main__":
    main()