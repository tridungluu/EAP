import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import các kiến trúc mạng
from net.encoder import StudentEncoder, ActionClassifier
# Import Feeder nạp dữ liệu kiểm thử
from feeder.feeder_eap import FeederEAP


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá mô hình EAP Student trên tập CoboGesture qua các tỷ lệ quan sát")
    parser.add_argument("--val_data", type=str, default="datacobot_v2/xsub/val_position.npy", help="Đường dẫn dữ liệu kiểm thử")
    parser.add_argument("--val_label", type=str, default="datacobot_v2/xsub/val_label.pkl", help="Đường dẫn nhãn kiểm thử")
    parser.add_argument("--work_dir", type=str, default="work_dir", help="Thư mục chứa các checkpoint huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--target_time", type=int, default=96, help="Số khung hình chuẩn hóa")
    parser.add_argument("--num_joints", type=int, default=48, help="Số lượng khớp xương")
    parser.add_argument("--num_classes", type=int, default=16, help="Số lượng lớp cử chỉ")
    parser.add_argument("--latent_dim", type=int, default=256, help="Kích thước vector đặc trưng ẩn")
    return parser.parse_args()


def evaluate_single_ratio(args, ratio, device):
    """
    Hàm đánh giá độ chính xác tại một tỷ lệ quan sát (ratio) cụ thể.
    """
    ratio_percent = int(ratio * 100)
    ckpt_path = os.path.join(args.work_dir, f"student_ratio_{ratio_percent}", "student_best.pth")

    if not os.path.exists(ckpt_path):
        print(f"⚠️  [Bỏ qua {ratio_percent}%] Không tìm thấy file checkpoint: {ckpt_path}")
        return None

    # 1. Khởi tạo Feeder với tỷ lệ ratio tương ứng
    val_dataset = FeederEAP(
        args.val_data,
        args.val_label,
        ratio=ratio,
        target_time=args.target_time
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 2. Khởi tạo kiến trúc Student Encoder và Classifier
    student_enc = StudentEncoder(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_joints=args.num_joints,
        num_layers=2
    ).to(device)

    classifier = ActionClassifier(
        latent_dim=args.latent_dim,
        num_joints=args.num_joints,
        num_classes=args.num_classes
    ).to(device)

    # 3. Nạp trọng số checkpoint đã huấn luyện
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    student_enc.load_state_dict(checkpoint["student_enc"])
    classifier.load_state_dict(checkpoint["classifier"])

    student_enc.eval()
    classifier.eval()

    val_correct = 0
    val_total = 0

    # 4. Tiến hành suy luận trên tập kiểm thử
    with torch.no_grad():
        for partial_data, _, labels in val_loader:
            partial_data = partial_data.to(device)
            labels = labels.to(device)

            out_s = student_enc(partial_data)
            student_context = out_s[1] if isinstance(out_s, (tuple, list)) else out_s
            logits = classifier(student_context)

            preds = torch.argmax(logits, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
    return acc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("📊 BẮT ĐẦU ĐÁNH GIÁ TỔNG HỢP EARLY ACTION PREDICTION (EAP)")
    print(f"Thiết bị: {device} | Chuẩn độ dài: {args.target_time} frames")
    print(f"Dữ liệu kiểm thử: {args.val_data}")
    print("=" * 65 + "\n")

    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = {}

    for r in ratios:
        acc = evaluate_single_ratio(args, r, device)
        if acc is not None:
            results[r] = acc
            print(f"✅ Tỷ lệ {int(r * 100):3d}% video: Top-1 Acc = {acc:.2f}%")

    # In bảng tổng hợp kết quả
    print("\n" + "=" * 65)
    print("📈 BẢNG TỔNG KẾT HIỆU NĂNG DỰ ĐOÁN SỚM (EARLY ACTION PREDICTION)")
    print("=" * 65)
    print(f"{'Tỷ lệ quan sát (%)':<25} | {'Độ chính xác Top-1 (%)':<25}")
    print("-" * 53)

    for r in ratios:
        if r in results:
            print(f"{int(r * 100):<25} | {results[r]:.2f}%")
        else:
            print(f"{int(r * 100):<25} | N/A (Chưa huấn luyện)")

    print("-" * 53)
    if len(results) > 0:
        avg_acc = sum(results.values()) / len(results)
        print(f"{'Độ chính xác trung bình (Avg)':<25} | {avg_acc:.2f}%")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()