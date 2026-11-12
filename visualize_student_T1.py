#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchlight import import_class


def parse_args():
    parser = argparse.ArgumentParser(
        description="T=1 Student evaluation and feature visualization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/cobot/test_student.yaml",
        help="Student test YAML config",
    )
    parser.add_argument(
        "--model_type",
        choices=["auto", "student", "teacher"],
        default="auto",
        help="Model pipeline. Auto detects TeacherModel/StudentModel from the YAML.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default depends on model type)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional maximum number of samples; uses the first N samples",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="For example: cuda:0 or cpu")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def resolve_path(path):
    if path is None:
        return None
    return str(Path(path).expanduser())


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def choose_device(config, cli_device):
    if cli_device:
        return torch.device(cli_device)

    use_gpu = config.get("use_gpu", True)
    devices = config.get("device", [0])
    if isinstance(devices, int):
        devices = [devices]

    if use_gpu and torch.cuda.is_available():
        index = int(devices[0]) if devices else 0
        return torch.device(f"cuda:{index}")

    return torch.device("cpu")


def build_model(config, device):
    model_class = import_class(config["model"])
    model_args = dict(config.get("model_args", {}))

    model = model_class(**model_args)

    weights_path = resolve_path(config.get("weights"))
    if not weights_path:
        raise ValueError("The YAML config must specify 'weights'.")

    checkpoint = torch.load(weights_path, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError(
            f"Checkpoint {weights_path} does not contain a 'model' state_dict."
        )

    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    print(f"Loaded model: {config['model']}")
    print(f"Checkpoint:  {weights_path}")
    print(f"Device:      {device}")
    print(f"Parameters:  {sum(p.numel() for p in model.parameters()) / 1e6:.3f} M")
    return model


def build_dataset(config):
    feeder_class = import_class(config["feeder"])
    feeder_args = dict(config["test_feeder_args"])
    dataset = feeder_class(**feeder_args)
    return dataset, feeder_args


def detect_model_type(config, requested_type="auto"):
    if requested_type != "auto":
        return requested_type

    model_name = str(config.get("model", "")).lower()
    if "teacher" in model_name:
        return "teacher"
    if "student" in model_name:
        return "student"

    raise ValueError(
        "Cannot detect model type from config['model']. "
        "Please pass --model_type teacher or --model_type student."
    )


def extract_logits_and_features(
    model, loader, device, model_type, max_samples=None
):
    all_logits, all_features, all_labels = [], [], []
    processed = 0

    with torch.no_grad():
        for batch in loader:
            # Feeder returns (data_teacher, data_student, label, text).
            data = batch[0] if model_type == "teacher" else batch[1]
            labels = batch[2]

            if max_samples is not None:
                remaining = max_samples - processed
                if remaining <= 0:
                    break
                if len(labels) > remaining:
                    data = data[:remaining]
                    labels = labels[:remaining]

            data = data.float().to(device)
            labels_np = labels.detach().cpu().numpy().astype(np.int64)

            if model_type == "teacher":
                # Teacher: (logits, context, recon_skel, (C_s, C_t)).
                logits, context, _, _ = model(data)
            else:
                # Student: (logits, context, reconstruction).
                logits, context, _ = model(data)

            # Context: (B, C, context_T, V) -> feature: (B, C).
            feature_vectors = context.mean(dim=(2, 3))

            all_logits.append(logits.detach().cpu().numpy())
            all_features.append(feature_vectors.detach().cpu().numpy())
            all_labels.append(labels_np)

            processed += len(labels_np)
            if processed % 500 == 0:
                print(f"Processed {processed} samples...")

    if not all_labels:
        raise RuntimeError("No samples were processed. Check the dataset/config.")

    logits = np.concatenate(all_logits, axis=0)
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print(f"Model type:     {model_type}")
    print(f"Logits shape:   {logits.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape:   {labels.shape}")
    return logits, features, labels



def calculate_metrics(logits, features, labels, num_classes):
    predictions = logits.argmax(axis=1)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    confidence = probabilities.max(axis=1)

    accuracy = accuracy_score(labels, predictions)
    top_k = min(5, logits.shape[1])
    top_indices = np.argsort(logits, axis=1)[:, -top_k:]
    top5 = np.mean([label in row for label, row in zip(labels, top_indices)])

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(num_classes),
        average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels, predictions, labels=np.arange(num_classes),
            average="weighted", zero_division=0
        )
    )

    per_precision, per_recall, per_f1, per_support = (
        precision_recall_fscore_support(
            labels, predictions, labels=np.arange(num_classes),
            average=None, zero_division=0
        )
    )

    cm = confusion_matrix(labels, predictions, labels=np.arange(num_classes))

    nmi = None
    unique_labels = np.unique(labels)
    n_clusters = min(num_classes, len(features))
    if len(features) >= 2 and n_clusters >= 2:
        cluster_labels = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10
        ).fit_predict(features)
        nmi = float(normalized_mutual_info_score(labels, cluster_labels))

    metrics = {
        "num_samples": int(len(labels)),
        "num_classes_configured": int(num_classes),
        "num_classes_present": int(len(unique_labels)),
        "feature_dim": int(features.shape[1]),
        "accuracy": float(accuracy),
        "top5_accuracy": float(top5),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "nmi": nmi,
    }

    per_class = []
    for class_id in range(num_classes):
        per_class.append({
            "class_id": class_id,
            "precision": float(per_precision[class_id]),
            "recall": float(per_recall[class_id]),
            "f1_score": float(per_f1[class_id]),
            "support": int(per_support[class_id]),
        })

    return predictions, confidence, cm, metrics, per_class


def get_class_info(dataset, num_classes):
    """
    Trích xuất tên đầy đủ, tạo ID và tên viết tắt (ví dụ: '00-DW' cho Drink Water).
    """
    names = getattr(dataset, "text_list", None)
    if not names or len(names) < num_classes:
        names = [f"Class {i}" for i in range(num_classes)]
    
    class_info = []
    for i in range(num_classes):
        full_name = str(names[i])
        words = full_name.replace("/", " ").replace("-", " ").replace("_", " ").split()
        if len(words) == 1:
            abbr = words[0][:3].upper()
        else:
            abbr = "".join([w[0].upper() for w in words if w])[:3]
            
        short_name = f"{i:02d}-{abbr}"
        class_info.append({
            "id": i,
            "short_name": short_name,
            "full_name": full_name
        })
    return class_info


def plot_confusion_matrix(cm, class_info, output_dir):
    short_names = [info["short_name"] for info in class_info]
    fig_size = max(10, min(18, len(short_names) * 0.85))
    
    # 1. Ma trận nhầm lẫn (Raw Count)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm, annot=(len(short_names) <= 20), fmt="d", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names, cbar=True, ax=ax
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("T=1 Confusion Matrix (Raw Count)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    raw_path = os.path.join(output_dir, "confusion_matrix_raw.png")
    fig.savefig(raw_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {raw_path}")

    # 2. Ma trận nhầm lẫn (Normalized - Phần trăm)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm_normalized, annot=(len(short_names) <= 20), fmt=".1%", cmap="Blues",
        xticklabels=short_names, yticklabels=short_names, cbar=True, ax=ax
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("T=1 Confusion Matrix (Normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    norm_path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    fig.savefig(norm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {norm_path}")


def plot_feature_correlation(features, output_path, sample_size=100):
    sample_size = min(sample_size, features.shape[1])
    selected = features[:, :sample_size]

    corr = np.corrcoef(selected, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(
        corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto"
    )
    ax.set_title(f"Feature Correlation Matrix (First {sample_size} dimensions)", pad=15)
    ax.set_xlabel("Feature dimension", labelpad=10)
    ax.set_ylabel("Feature dimension", labelpad=10)
    
    # Chia khoảng trục (ticks) đều và chi tiết hơn
    ticks = np.linspace(0, sample_size - 1, num=min(11, sample_size), dtype=int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)
    
    # Label cho colorbar
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Pearson Correlation Coefficient", rotation=270, labelpad=20)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_tsne(features, labels, class_info, nmi, output_path,
              perplexity=30.0, random_state=42):
    n_samples = len(features)
    if n_samples < 3:
        print("Skipping t-SNE: at least 3 samples are required.")
        return

    perplexity = min(float(perplexity), float(n_samples - 1))
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca",
        "learning_rate": "auto",
        "random_state": random_state,
        "verbose": 1,
    }
    tsne_parameters = inspect.signature(TSNE).parameters
    if "max_iter" in tsne_parameters:
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000

    print(f"Computing t-SNE for {n_samples} samples (perplexity={perplexity})...")
    embedding = TSNE(**kwargs).fit_transform(features)

    num_classes = len(class_info)
    # Cố định colormap để màu của các class không bị xê dịch khi batch thiếu sample
    cmap = plt.get_cmap("tab20", num_classes)

    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        cmap=cmap,
        vmin=-0.5,
        vmax=num_classes - 0.5,
        s=24,
        alpha=0.85,
    )
    
    # Tạo legend tường minh, ép hiển thị đủ tất cả các lớp 
    legend_handles = []
    legend_labels = []
    for class_id in range(num_classes):
        color = cmap(class_id / (num_classes - 1) if num_classes > 1 else 0)
        handle = matplotlib.lines.Line2D(
            [0], [0], marker='o', color='w', markerfacecolor=color, markersize=8
        )
        legend_handles.append(handle)
        legend_labels.append(class_info[class_id]["short_name"])

    ax.legend(
        legend_handles,
        legend_labels,
        title="Action classes",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5), # Đẩy legend ra ngoài viền data
        fontsize=9,
        ncol=1 if num_classes <= 10 else 2
    )
    
    if nmi is not None:
        ax.text(
            0.98, 0.98, f"NMI: {nmi:.4f}",
            transform=ax.transAxes, ha="right", va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    ax.set_title("T=1 Feature t-SNE")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_per_class_f1(per_class, class_info, output_path):
    # Trích xuất và map với short_name
    merged_metrics = []
    for pc in per_class:
        cid = pc["class_id"]
        merged_metrics.append({
            "short_name": class_info[cid]["short_name"],
            "f1_score": pc["f1_score"]
        })
        
    # Sắp xếp tăng dần theo F1 score
    merged_metrics.sort(key=lambda x: x["f1_score"])
    
    names = [x["short_name"] for x in merged_metrics]
    f1s = [x["f1_score"] for x in merged_metrics]
    
    fig_height = max(6, len(names) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_pos = np.arange(len(names))
    
    bars = ax.barh(y_pos, f1s, align='center', color='skyblue', edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('F1 Score')
    ax.set_title('Per-class F1 Score (Weakest to Strongest)')
    
    # Hiển thị số liệu F1 ở cuối thanh bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f"{width:.4f}", va='center', ha='left', fontsize=9)
        
    ax.set_xlim(0, 1.15) 
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def export_class_mapping(class_info, output_path):
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Class ID", "Short Name", "Full Name"])
        for info in class_info:
            writer.writerow([info["id"], info["short_name"], info["full_name"]])
    print(f"Saved: {output_path}")


def save_reports(labels, predictions, confidence, metrics, per_class,
                 class_info, output_dir):
    short_names = [info["short_name"] for info in class_info]
    
    report_path = os.path.join(output_dir, "classification_report.txt")
    report = classification_report(
        labels,
        predictions,
        labels=np.arange(len(class_info)),
        target_names=short_names,
        zero_division=0,
        digits=4,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("T=1 Classification Report\n")
        f.write("=" * 60 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write(report)
    print(f"Saved: {report_path}")

    per_class_path = os.path.join(output_dir, "per_class_metrics.csv")
    with open(per_class_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "short_name", "full_name", "precision",
                        "recall", "f1_score", "support"],
        )
        writer.writeheader()
        for row in per_class:
            cid = row["class_id"]
            out_row = dict(row)
            out_row["short_name"] = class_info[cid]["short_name"]
            out_row["full_name"] = class_info[cid]["full_name"]
            writer.writerow(out_row)
    print(f"Saved: {per_class_path}")

    predictions_path = os.path.join(output_dir, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_index", "true_label", "true_class",
            "predicted_label", "predicted_class", "confidence", "correct"
        ])
        for i, (true_label, pred_label, conf) in enumerate(
            zip(labels, predictions, confidence)
        ):
            writer.writerow([
                i,
                int(true_label),
                short_names[int(true_label)],
                int(pred_label),
                short_names[int(pred_label)],
                f"{float(conf):.6f}",
                int(true_label == pred_label),
            ])
    print(f"Saved: {predictions_path}")


def main():
    args = parse_args()
    config = load_config(args.config)

    model_type = detect_model_type(config, args.model_type)
    device = choose_device(config, args.device)
    model = build_model(config, device)
    dataset, feeder_args = build_dataset(config)

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max_samples must be positive.")

    batch_size = args.batch_size or config.get("test_batch_size", 32)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    obs_ratio = float(feeder_args.get("obs_ratio", 1.0))
    ratio_tag = f"r{int(round(obs_ratio * 100)):02d}"
    if model_type == "teacher":
        default_output_dir = os.path.join("visualization_results", "teacher_T1")
    else:
        default_output_dir = os.path.join(
            "visualization_results", f"student_T1_{ratio_tag}"
        )
    output_dir = args.output_dir or default_output_dir
    os.makedirs(output_dir, exist_ok=True)

    if model_type == "student":
        print(f"Observation ratio: {obs_ratio:.2f}")
    else:
        print("Input pipeline:    full skeleton sequence (Teacher)")
    print(f"Dataset samples:   {len(dataset)}")
    print(f"Output directory:  {output_dir}")

    logits, features, labels = extract_logits_and_features(
        model, loader, device, model_type=model_type,
        max_samples=args.max_samples
    )

    num_classes = int(config["model_args"].get("num_class", 16))
    if logits.shape[1] != num_classes:
        raise ValueError(
            f"Model outputs {logits.shape[1]} classes, but config says "
            f"num_class={num_classes}."
        )

    predictions, confidence, cm, metrics, per_class = calculate_metrics(
        logits, features, labels, num_classes
    )
    
    # 5. Extract class mappings
    class_info = get_class_info(dataset, num_classes)

    # 1. Confusion Matrix (Raw & Normalized)
    plot_confusion_matrix(cm, class_info, output_dir)
    
    # 3. Feature Correlation
    plot_feature_correlation(
        features, os.path.join(output_dir, "feature_correlation.png")
    )
    
    # 2. t-SNE legend và colormap
    plot_tsne(
        features,
        labels,
        class_info,
        metrics["nmi"],
        os.path.join(output_dir, "tsne_visualization.png"),
        perplexity=args.tsne_perplexity,
        random_state=args.random_state,
    )
    
    # 4. Per-class F1 
    plot_per_class_f1(
        per_class, class_info, os.path.join(output_dir, "per_class_f1.png")
    )
    
    # Xuất class mapping
    export_class_mapping(
        class_info, os.path.join(output_dir, "class_mapping.csv")
    )
    
    # Cập nhật reports
    save_reports(
        labels, predictions, confidence, metrics, per_class,
        class_info, output_dir
    )

    metrics["model_type"] = model_type
    if model_type == "student":
        metrics["observation_ratio"] = obs_ratio
    metrics["checkpoint"] = str(config.get("weights"))
    metrics["config"] = args.config
    metrics["evaluation_split"] = (
        "validation" if "val" in str(feeder_args.get("data_path", "")).lower()
        or "val" in str(feeder_args.get("label_path", "")).lower()
        else "configured evaluation split"
    )

    json_path = os.path.join(output_dir, "analysis_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"{model_type.upper()} T=1 VISUALIZATION AND EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Accuracy:       {metrics['accuracy'] * 100:.2f}%")
    print(f"Top-5 Accuracy: {metrics['top5_accuracy'] * 100:.2f}%")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:    {metrics['weighted_f1']:.4f}")
    if metrics["nmi"] is not None:
        print(f"NMI:            {metrics['nmi']:.4f}")
    print(f"Samples:        {metrics['num_samples']}")
    print(f"Features:       {metrics['feature_dim']} dimensions")
    print(f"Saved to:       {output_dir}")


if __name__ == "__main__":
    main()