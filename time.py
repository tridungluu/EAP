import argparse
import pickle
import time

import numpy as np
import torch

from torchlight import import_class


def load_data(data_path, label_path, mmap=True):
    if mmap:
        data = np.load(data_path, mmap_mode="r")
    else:
        data = np.load(data_path)

    with open(label_path, "rb") as f:
        sample_names, labels = pickle.load(f)

    labels = np.array(labels)
    return data, labels, sample_names


def load_model(model_path, device):
    """
    Load standard ST-GCN classifier
    """
    STGCN = import_class("net.st_gcn.Model")

    model = STGCN(
        in_channels=3,
        num_class=20,
        graph_args={"layout": "cobot", "strategy": "distance"},
        edge_importance_weighting=True,
    )

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def measure_inference_time(
    data,
    model,
    device,
    batch_size=1,
    warmup=True,
):
    num_samples = data.shape[0]
    times = []

    # ---------- warm-up ----------
    if warmup:
        with torch.no_grad():
            for i in range(0, min(num_samples, 10 * batch_size), batch_size):
                batch = data[i : i + batch_size]
                batch = torch.from_numpy(batch).float().to(device, non_blocking=True)
                _ = model(batch)
                if device.type == "cuda":
                    torch.cuda.synchronize()

    # ---------- timed ----------
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = data[i : i + batch_size]
            batch = torch.from_numpy(batch).float().to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.time()
            _ = model(batch)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.time()
            times.append(end - start)

    avg_time_per_batch = float(np.mean(times))
    avg_time_per_sample = avg_time_per_batch / float(batch_size)
    return avg_time_per_sample


def main():
    parser = argparse.ArgumentParser(
        description="Measure ST-GCN inference time per sample"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data_ori/xsub/val_position.npy",
        help="Skeleton data (.npy)",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="data_ori/xsub/val_label.pkl",
        help="Labels (.pkl)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="work_dir/recognition/cobot/ST_GCN/epoch180_model.pt",
        help="Trained ST-GCN checkpoint",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--no_warmup",
        action="store_true",
        help="Disable warm-up",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    data, labels, _ = load_data(args.data_path, args.label_path, mmap=True)
    print(f"#samples: {len(labels)}, data shape: {data.shape}")

    print("Loading model...")
    model = load_model(args.model_path, device)

    print("Measuring inference time...")
    t = measure_inference_time(
        data=data,
        model=model,
        device=device,
        batch_size=args.batch_size,
        warmup=not args.no_warmup,
    )

    print(f"Average inference time per sample: {t:.6f} s")
    print(f"Average inference time per sample: {t * 1000:.3f} ms")


if __name__ == "__main__":
    main()
