# train_teacher.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from feeder.feeder_eap import FeederEAP
from net.encoder import TeacherEncoder, ActionClassifier
from net.decoder import SkeletonDecoder
from net.dcl_module import DCLDisentangleModule
from processor.distillers_eap import STALoss, DCLLoss

def main():
    # 1. Cấu hình thiết bị & Tham số huấn luyện
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Đang sử dụng thiết bị: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    num_joints = 48
    num_classes = 16
    latent_dim = 256
    batch_size = 32
    num_epochs = 100
    lr = 0.001

    # Trọng số các hàm mất mát theo bài báo
    w_ce = 1.0
    w_sta = 1.0
    w_cl = 0.1

    work_dir = "work_dir/teacher"
    os.makedirs(work_dir, exist_ok=True)

    # 2. Khởi tạo DataLoaders (Teacher quan sát 100% video: ratio=1.0, chuẩn 96 frames)
    train_data_path = "datacobot_v2/xsub/train_position.npy"
    train_label_path = "datacobot_v2/xsub/train_label.pkl"
    val_data_path = "datacobot_v2/xsub/val_position.npy"
    val_label_path = "datacobot_v2/xsub/val_label.pkl"

    print("\n📂 Nạp dữ liệu huấn luyện & kiểm thử...")
    train_dataset = FeederEAP(train_data_path, train_label_path, ratio=1.0, target_time=96)
    val_dataset = FeederEAP(val_data_path, val_label_path, ratio=1.0, target_time=96)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # 3. Khởi tạo các module mạng
    teacher_enc = TeacherEncoder(
        in_channels=3,
        latent_dim=latent_dim,
        num_joints=num_joints,
        num_layers=3
    ).to(device)

    classifier = ActionClassifier(
        latent_dim=latent_dim,
        num_joints=num_joints,
        num_classes=num_classes
    ).to(device)

    decoder = SkeletonDecoder(
        latent_dim=latent_dim,
        out_channels=3,
        num_joints=num_joints
    ).to(device)

    dcl_mod = DCLDisentangleModule(
        latent_dim=latent_dim,
        proj_dim=128
    ).to(device)

    # 4. Định nghĩa hàm Loss & Optimizer
    ce_loss_fn = nn.CrossEntropyLoss().to(device)
    sta_loss_fn = STALoss(time_step=2, root_joint=0).to(device)
    dcl_loss_fn = DCLLoss(num_classes=num_classes, temperature=0.125, momentum=0.9).to(device)

    all_params = (
        list(teacher_enc.parameters()) +
        list(classifier.parameters()) +
        list(decoder.parameters()) +
        list(dcl_mod.parameters())
    )
    optimizer = optim.Adam(all_params, lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_acc = 0.0

    print("\n🚀 Bắt đầu quá trình huấn luyện Teacher...")
    for epoch in range(1, num_epochs + 1):
        teacher_enc.train()
        classifier.train()
        decoder.train()
        dcl_mod.train()

        total_loss, correct_train, total_samples = 0.0, 0, 0
        start_time = time.time()

        for batch_idx, (_, full_data, labels) in enumerate(train_loader):
            data = full_data.to(device)      # (B, 3, 96, 48) - Teacher dùng chuỗi hoàn chỉnh 96 frames
            labels = labels.to(device)        # (B,)

            optimizer.zero_grad()

            # Forward qua mạng (Xử lý an toàn tuple trả về từ TeacherEncoder)
            out_enc = teacher_enc(data)
            if isinstance(out_enc, tuple):
                latent_tensor, context = out_enc
            else:
                context = out_enc
                latent_tensor = out_enc

            logits = classifier(context)                     # (B, 16)
            reconstructed = decoder(context)                 # (B, 3, 1, 48)
            C_s, C_t = dcl_mod(latent_tensor)                # (B, 128), (B, 128)

            # Tính các hàm loss theo bài báo
            loss_ce = ce_loss_fn(logits, labels)
            loss_sta, s_sta = sta_loss_fn(reconstructed, data)
            loss_dcl = dcl_loss_fn(C_s, C_t, logits, labels, s_sta)

            loss_total = w_ce * loss_ce + w_sta * loss_sta + w_cl * loss_dcl

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

        # 5. Đánh giá trên tập Validation
        teacher_enc.eval()
        classifier.eval()
        correct_val, total_val = 0, 0

        with torch.no_grad():
            for _, full_data, labels in val_loader:
                data = full_data.to(device)
                labels = labels.to(device)

                out_val = teacher_enc(data)
                context = out_val if isinstance(out_val, tuple) else out_val

                logits = classifier(context)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = (correct_val / total_val) * 100.0 if total_val > 0 else 0.0

        print(
            f"Epoch [{epoch:03d}/{num_epochs:03d}] | Thời gian: {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        # Lưu checkpoint khi đạt độ chính xác cao nhất
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(work_dir, "teacher_best.pth")
            torch.save({
                'epoch': epoch,
                'teacher_enc': teacher_enc.state_dict(),
                'classifier': classifier.state_dict(),
                'decoder': decoder.state_dict(),
                'val_acc': val_acc
            }, save_path)
            print(f" ⭐ Đã lưu checkpoint tốt nhất mới tại: {save_path} (Val Acc: {best_acc:.2f}%)")

    print(f"\n🎉 Huấn luyện Teacher hoàn tất! Độ chính xác cao nhất: {best_acc:.2f}%")


if __name__ == '__main__':
    main()