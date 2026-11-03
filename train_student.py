import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Thêm đường dẫn thư mục gốc dự án
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from net.encoder import TeacherEncoder, StudentEncoder, ActionClassifier
from net.decoder import SkeletonDecoder
from feeder.feeder_eap import FeederEAP


def parse_args():
    parser = argparse.ArgumentParser(description="Huấn luyện Student Model (EAP) trên tập CoboGesture 96 frames")
    parser.add_argument("--ratio", type=float, default=0.2, help="Tỷ lệ quan sát (0.2, 0.4, 0.6, 0.8, 1.0)")
    parser.add_argument("--epochs", type=int, default=100, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=64, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tốc độ học khởi tạo")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Hệ số suy giảm trọng số (Weight decay)")
    parser.add_argument("--step_size", type=int, default=40, help="Chu kỳ giảm learning rate")
    parser.add_argument("--gamma", type=float, default=0.1, help="Tỷ lệ giảm learning rate")

    # Trọng số các hàm loss (theo bài báo EAP)
    parser.add_argument("--w_ce", type=float, default=1.0, help="Trọng số Cross Entropy Loss")
    parser.add_argument("--w_cd", type=float, default=50.0, help="Trọng số Context Distillation Loss")
    parser.add_argument("--w_recon", type=float, default=1.0, help="Trọng số Reconstruction Loss")

    # Đường dẫn dữ liệu và thư mục lưu checkpoint
    parser.add_argument("--train_data", type=str, default="datacobot_v2/xsub/train_position.npy")
    parser.add_argument("--train_label", type=str, default="datacobot_v2/xsub/train_label.pkl")
    parser.add_argument("--val_data", type=str, default="datacobot_v2/xsub/val_position.npy")
    parser.add_argument("--val_label", type=str, default="datacobot_v2/xsub/val_label.pkl")
    parser.add_argument("--teacher_ckpt", type=str, default="work_dir/teacher_stage1/teacher_best.pth")
    parser.add_argument("--work_dir", type=str, default="work_dir")
    parser.add_argument("--target_time", type=int, default=96, help="Số khung hình chuẩn hóa (CoboGesture)")

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join(args.work_dir, f"student_ratio_{int(args.ratio * 100)}")
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 65)
    print(f"🚀 HUẤN LUYỆN STUDENT NETWORK (EAP) - TỶ LỆ QUAN SÁT: {int(args.ratio * 100)}%")
    print(f"Thiết bị: {device} | Độ dài chuỗi: {args.target_time} frames")
    print(f"Thư mục lưu mô hình: {save_dir}")
    print("=" * 65 + "\n")

    # 1. Khởi tạo DataLoaders bằng FeederEAP chuẩn 96 frames
    train_dataset = FeederEAP(args.train_data, args.train_label, ratio=args.ratio, target_time=args.target_time)
    val_dataset = FeederEAP(args.val_data, args.val_label, ratio=args.ratio, target_time=args.target_time)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Khởi tạo Teacher Model (Đóng băng hoàn toàn tham số)
    teacher_enc = TeacherEncoder(in_channels=3, latent_dim=256, num_joints=48, num_layers=3).to(device)
    if os.path.exists(args.teacher_ckpt):
        teacher_checkpoint = torch.load(args.teacher_ckpt, map_location=device, weights_only=True)
        teacher_enc.load_state_dict(teacher_checkpoint["teacher_enc"])
        print(f"✅ Đã nạp Teacher Checkpoint từ: {args.teacher_ckpt}")
    else:
        raise FileNotFoundError(f"Không tìm thấy checkpoint Teacher tại {args.teacher_ckpt}. Hãy huấn luyện Teacher trước!")

    teacher_enc.eval()
    for param in teacher_enc.parameters():
        param.requires_grad = False

    # 3. Khởi tạo Student Models (Encoder, Classifier, Decoder giữ T = 1)
    student_enc = StudentEncoder(in_channels=3, latent_dim=256, num_joints=48, num_layers=2).to(device)
    classifier = ActionClassifier(latent_dim=256, num_joints=48, num_classes=16).to(device)
    decoder = SkeletonDecoder(latent_dim=256, out_channels=3, num_joints=48).to(device)

    # 4. Định nghĩa Optimizer, Scheduler và các hàm Loss
    parameters = list(student_enc.parameters()) + list(classifier.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()

    best_val_acc = 0.0

    # 5. Vòng lặp huấn luyện chính
    for epoch in range(1, args.epochs + 1):
        student_enc.train()
        classifier.train()
        decoder.train()

        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for partial_data, full_data, labels in train_loader:
            partial_data = partial_data.to(device)
            full_data = full_data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # (a) Teacher nhận chuỗi ĐẦY ĐỦ 100% video (96 frames) để trích xuất ngữ cảnh v_i
            with torch.no_grad():
                out_t = teacher_enc(full_data)
                teacher_context = out_t if not isinstance(out_t, (tuple, list)) else out_t

            # (b) Student chỉ nhận chuỗi DỞ DANG (partial_data) để trích xuất ngữ cảnh \hat{v}_i
            out_s = student_enc(partial_data)
            student_context = out_s if not isinstance(out_s, (tuple, list)) else out_s

            logits = classifier(student_context)
            # Decoder tái tạo tư thế với T = 1: (B, 3, 1, 48)
            recon_motion = decoder(student_context)

            # (c) Tính toán các hàm Loss:
            # 1. Loss CE: Phân loại cử chỉ
            loss_ce = criterion_ce(logits, labels)

            # 2. Loss CD: Context Distillation Loss (L1/MAE ép context Student tiệm cận Teacher)
            loss_cd = torch.mean(torch.abs(student_context - teacher_context))

            # 3. Loss Recon: Đồng bộ hóa trên không gian T = 1 (Tư thế trung bình qua 96 frames)
            target_pose = full_data.mean(dim=2, keepdim=True)  # (B, 3, 1, 48)
            loss_recon = criterion_recon(recon_motion, target_pose)

            # Tổng hợp hàm mất mát
            loss = (args.w_ce * loss_ce) + (args.w_cd * loss_cd) + (args.w_recon * loss_recon)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * partial_data.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        train_loss = total_loss / train_total
        train_acc = (train_correct / train_total) * 100.0

        # 6. Đánh giá kiểm thử trên tập Validation
        student_enc.eval()
        classifier.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for partial_data, _, labels in val_loader:
                partial_data = partial_data.to(device)
                labels = labels.to(device)

                out_s = student_enc(partial_data)
                student_context = out_s if not isinstance(out_s, (tuple, list)) else out_s
                logits = classifier(student_context)

                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100.0

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)"
        )

        # Lưu checkpoint khi đạt độ chính xác kiểm thử cao nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_save_path = os.path.join(save_dir, "student_best.pth")
            torch.save({
                "epoch": epoch,
                "student_enc": student_enc.state_dict(),
                "classifier": classifier.state_dict(),
                "decoder": decoder.state_dict(),
                "best_acc": best_val_acc,
                "ratio": args.ratio
            }, ckpt_save_path)

    print("\n" + "=" * 65)
    print(f"🎉 Hoàn thành huấn luyện Student ({int(args.ratio * 100)}%)! Best Val Acc: {best_val_acc:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()