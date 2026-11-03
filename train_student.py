# train_student.py
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from feeder.feeder_eap import FeederEAP
from net.encoder import TeacherEncoder, StudentEncoder, ActionClassifier
from net.decoder import SkeletonDecoder
from processor.distillers_eap import ContextDistillationLoss, ReconstructionLoss

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Student EAP với Context Distillation")
    parser.add_argument('--ratio', type=float, default=0.2, help='Tỷ lệ quan sát chuỗi dở dang (vd: 0.2, 0.4, 0.6, 0.8)')
    parser.add_argument('--epochs', type=int, default=100, help='Số epoch huấn luyện')
    parser.add_argument('--batch_size', type=int, default=32, help='Kích thước batch')
    parser.add_argument('--lr', type=float, default=0.001, help='Tốc độ học khởi tạo')
    parser.add_argument('--teacher_ckpt', type=str, default='work_dir/teacher/teacher_best.pth', help='Đường dẫn checkpoint Teacher')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print(f"🔥 HUẤN LUYỆN STUDENT EAP | TỶ LỆ QUAN SÁT (RATIO): {int(args.ratio * 100)}%")
    print(f"   Thiết bị: {device}")
    print("=" * 60)

    num_joints = 48
    num_classes = 16
    latent_dim = 256

    # Trọng số loss theo bài báo
    w_ce = 1.0
    w_cd = 50.0
    w_recon = 1.0

    work_dir = f"work_dir/student_ratio_{int(args.ratio * 100)}"
    os.makedirs(work_dir, exist_ok=True)

    # 1. Nạp dữ liệu với tỷ lệ quan sát args.ratio
    train_data_path = "datacobot_v2/xsub/train_position.npy"
    train_label_path = "datacobot_v2/xsub/train_label.pkl"
    val_data_path = "datacobot_v2/xsub/val_position.npy"
    val_label_path = "datacobot_v2/xsub/val_label.pkl"

    print(f"\n📂 Nạp dữ liệu với chuỗi dở dang {int(args.ratio * 100)}%...")
    train_dataset = FeederEAP(train_data_path, train_label_path, ratio=args.ratio, target_time=64)
    val_dataset = FeederEAP(val_data_path, val_label_path, ratio=args.ratio, target_time=64)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Khởi tạo Teacher (Load trọng số đã train và ĐÓNG BĂNG)
    print(f"\n🔒 Đang tải checkpoint Teacher từ: {args.teacher_ckpt}")
    teacher_enc = TeacherEncoder(in_channels=3, latent_dim=latent_dim, num_joints=num_joints, num_layers=3).to(device)
    
    if os.path.exists(args.teacher_ckpt):
        ckpt = torch.load(args.teacher_ckpt, map_location=device)
        teacher_enc.load_state_dict(ckpt['teacher_enc'])
        print(f"   Đã nạp thành công Teacher (Val Acc gốc: {ckpt.get('val_acc', 0.0):.2f}%)")
    else:
        raise FileNotFoundError(f"Không tìm thấy checkpoint Teacher tại {args.teacher_ckpt}! Hãy chờ train_teacher.py chạy xong trước.")

    teacher_enc.eval()
    for param in teacher_enc.parameters():
        param.requires_grad = False

    # 3. Khởi tạo Student Modules
    student_enc = StudentEncoder(in_channels=3, latent_dim=latent_dim, num_joints=num_joints, num_layers=2).to(device)
    student_classifier = ActionClassifier(latent_dim=latent_dim, num_joints=num_joints, num_classes=num_classes).to(device)
    student_decoder = SkeletonDecoder(latent_dim=latent_dim, out_channels=3).to(device)

    # 4. Định nghĩa Loss & Optimizer
    ce_loss_fn = nn.CrossEntropyLoss().to(device)
    cd_loss_fn = ContextDistillationLoss().to(device)
    recon_loss_fn = ReconstructionLoss().to(device)

    student_params = list(student_enc.parameters()) + list(student_classifier.parameters()) + list(student_decoder.parameters())
    optimizer = optim.Adam(student_params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = 0.0

    print(f"\n🚀 Bắt đầu huấn luyện Student ({int(args.ratio * 100)}% frames)...")
    for epoch in range(1, args.epochs + 1):
        student_enc.train()
        student_classifier.train()
        student_decoder.train()

        total_loss, correct_train, total_samples = 0.0, 0, 0
        start_time = time.time()

        for batch_idx, (partial_data, labels, _) in enumerate(train_loader):
            partial_data = partial_data.to(device) # Chuỗi quan sát dở dang
            labels = labels.to(device)

            optimizer.zero_grad()

            # Lấy ngữ cảnh Teacher (chuỗi đầy đủ làm mẫu - forward no_grad)
            with torch.no_grad():
                _, teacher_context = teacher_enc(partial_data)

            # Student forward
            _, student_context = student_enc(partial_data)
            logits = student_classifier(student_context)
            reconstructed = student_decoder(student_context)

            # Tính các hàm loss của Student
            loss_ce = ce_loss_fn(logits, labels)
            loss_cd = cd_loss_fn(student_context, teacher_context)
            loss_recon = recon_loss_fn(reconstructed, partial_data)

            loss_total = w_ce * loss_ce + w_cd * loss_cd + w_recon * loss_recon

            loss_total.backward()
            optimizer.step()

            total_loss += loss_total.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == labels).sum().item()
            total_samples += labels.size(0)

        scheduler.step()

        train_acc = (correct_train / total_samples) * 100.0
        train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time

        # Đánh giá Student trên tập kiểm thử
        student_enc.eval()
        student_classifier.eval()
        correct_val, total_val = 0, 0

        with torch.no_grad():
            for partial_data, labels, _ in val_loader:
                partial_data = partial_data.to(device)
                labels = labels.to(device)

                _, student_context = student_enc(partial_data)
                logits = student_classifier(student_context)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = (correct_val / total_val) * 100.0 if total_val > 0 else 0.0

        print(f"Epoch [{epoch:03d}/{args.epochs:03d}] | Thời gian: {epoch_time:.1f}s | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(work_dir, "student_best.pth")
            torch.save({
                'epoch': epoch,
                'student_enc': student_enc.state_dict(),
                'classifier': student_classifier.state_dict(),
                'val_acc': val_acc,
                'ratio': args.ratio
            }, save_path)
            print(f" ⭐ Lưu Student ({int(args.ratio*100)}%) tốt nhất: {save_path} (Val Acc: {best_val_acc:.2f}%)")

    print(f"\n🎉 Hoàn thành huấn luyện Student ({int(args.ratio * 100)}%)! Best Val Acc: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main()