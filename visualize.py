import pickle
import numpy as np
import matplotlib.pyplot as plt

# ============ CẤU HÌNH ============
POSITION_PATH = "D:\\Thesis2\\datacobot_v2\\xsub\\val_position.npy"
LABEL_PATH = "D:\\Thesis2\\datacobot_v2\\xsub\\val_label.pkl"
CLASS_ID = 10               # nhãn 0-15 (đúng như trong file)
SAMPLE_INDEX = 0           # sample thứ mấy trong class này
N_SNAPSHOTS = 5
LAMBDA = 2                 # khoảng cách giữa 2 frame so sánh góc
FIX_HAND_SLOTS = True      # sửa frame mà bàn tay bị lưu nhầm slot A/B (gán lại theo cổ tay pose gần nhất)
EXCLUDE_STATIC_PAIRS = True  # bỏ thêm các cặp frame giống hệt nhau (phòng khi dữ liệu đã bị forward-fill)
HIST_MAX_DEG = 60          # trục x của histogram; giá trị lớn hơn bị cắt khỏi hình nhưng vẫn được đếm/in ra
OUT_PNG = "cobot_skeleton_histogram.png"

WRIST_POSE_A, WRIST_POSE_B = 47, 46   # cổ tay thân trên ứng với bàn tay A / B

HAND_NAMES = ["Wrist",
              "ThumbCMC", "ThumbMCP", "ThumbIP", "ThumbTip",
              "IndexMCP", "IndexPIP", "IndexDIP", "IndexTip",
              "MiddleMCP", "MiddlePIP", "MiddleDIP", "MiddleTip",
              "RingMCP", "RingPIP", "RingDIP", "RingTip",
              "PinkyMCP", "PinkyPIP", "PinkyDIP", "PinkyTip"]
JOINT_NAMES = ([f"A-{n}" for n in HAND_NAMES] + [f"B-{n}" for n in HAND_NAMES] +
               ["ShoulderB", "ShoulderA", "ElbowB", "ElbowA", "WristB", "WristA"])

HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
              (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
              (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
BONES = (HAND_BONES + [(a + 21, b + 21) for a, b in HAND_BONES] +
         [(42, 43), (42, 44), (44, 46), (43, 45), (45, 47)])
GROUP_COLORS = {"A": "#C44E52", "B": "#4C72B0", "pose": "#55A868"}
# ===================================


def group_of(v):
    return "A" if v < 21 else ("B" if v < 42 else "pose")


def load_data():
    pos = np.load(POSITION_PATH)                     # (N, C, T, V, M)
    with open(LABEL_PATH, "rb") as f:
        names, labels = pickle.load(f)
    return pos, list(names), np.array(labels)


def get_sequence(pos, i):
    return pos[i, :, :, :, 0].transpose(1, 2, 0).astype(np.float64)   # (T, V, C)


def valid_mask(seq):
    """Khớp bị mất (bàn tay không được nhận diện) được lưu là (0, 0, 0)."""
    return np.any(seq != 0, axis=-1)                  # (T, V)


def fix_hand_slots(seq):
    """Mỗi frame, gán bàn tay đã nhận diện về đúng slot theo cổ tay thân trên gần nhất
    (A <-> khớp 47, B <-> khớp 46). Trả về (chuỗi đã sửa, số frame bị sửa)."""
    out, valid, n_fixed = seq.copy(), valid_mask(seq), 0
    for t in range(seq.shape[0]):
        wa, wb = seq[t, WRIST_POSE_A, :2], seq[t, WRIST_POSE_B, :2]
        dist = lambda r, w: np.linalg.norm(seq[t, r, :2] - w)
        present = [r for r in (0, 21) if valid[t, r]]
        if len(present) == 2:
            if dist(0, wb) + dist(21, wa) < dist(0, wa) + dist(21, wb):
                out[t, 0:21], out[t, 21:42] = seq[t, 21:42], seq[t, 0:21]
                n_fixed += 1
        elif len(present) == 1:
            r = present[0]
            target = 0 if dist(r, wa) <= dist(r, wb) else 21
            if target != r:
                out[t, target:target + 21], out[t, r:r + 21] = seq[t, r:r + 21], 0
                n_fixed += 1
    return out, n_fixed


def group_anchors(seq):
    """Mỗi nhóm có gốc riêng: cổ tay cho từng bàn tay, trung điểm 2 vai cho thân trên."""
    anc = np.empty_like(seq)
    anc[:, 0:21] = seq[:, 0:1]
    anc[:, 21:42] = seq[:, 21:22]
    anc[:, 42:48] = ((seq[:, 42] + seq[:, 43]) / 2)[:, None]
    return anc


def angle_timeseries(seq, lam, exclude_static):
    anc = group_anchors(seq)
    ab = seq - anc
    a, b = ab[:-lam], ab[lam:]
    dot = np.sum(a * b, axis=-1)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = dot / (na * nb)
    cos = np.where((na < 1e-8) | (nb < 1e-8), np.nan, np.clip(cos, -1, 1))
    theta = np.degrees(np.arccos(cos))

    valid, valid_anchor = valid_mask(seq), valid_mask(anc)
    ok = valid[:-lam] & valid[lam:] & valid_anchor[:-lam] & valid_anchor[lam:]
    theta = np.where(ok, theta, np.nan)               # loại cặp frame có khớp/gốc bị mất
    if exclude_static:
        theta = np.where(np.all(a == b, axis=-1), np.nan, theta)
    return theta                                      # (T-lam, V)


def draw_body(ax, pose, radius, title):
    ok = np.any(pose != 0, axis=-1)
    for a, b in BONES:
        if ok[a] and ok[b]:
            ax.plot(pose[[a, b], 0], pose[[a, b], 1], c=GROUP_COLORS[group_of(a)], linewidth=1.2)
    ax.scatter(pose[ok, 0], pose[ok, 1], c=[GROUP_COLORS[group_of(v)] for v in np.where(ok)[0]], s=8)
    pts = pose[ok, :2]
    c = (pts.max(axis=0) + pts.min(axis=0)) / 2
    ax.set_xlim(c[0] - radius, c[0] + radius)
    ax.set_ylim(c[1] + radius, c[1] - radius)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)


def main():
    pos, names, labels = load_data()
    idx_in_class = np.where(labels == CLASS_ID)[0]
    print(f"Class {CLASS_ID}: {len(idx_in_class)} sample trong tập này")
    i = idx_in_class[min(SAMPLE_INDEX, len(idx_in_class) - 1)]
    seq = get_sequence(pos, i)                        # (T, V, C)
    T = seq.shape[0]

    if FIX_HAND_SLOTS:
        seq, n_fixed = fix_hand_slots(seq)
        print(f"Đã gán lại slot bàn tay ở {n_fixed}/{T} frame (bàn tay bị lưu nhầm slot A/B)")

    theta = angle_timeseries(seq, LAMBDA, EXCLUDE_STATIC_PAIRS)
    joint_ids = [v for v in range(48) if v not in (0, 21)]   # 2 cổ tay là gốc -> bỏ
    dropped = np.isnan(theta[:, joint_ids]).mean() * 100
    print(f"Sample: {names[i]} | {T} frame | {dropped:.1f}% phép đo góc bị loại (tay bị mất / frame lặp)")

    # Xếp hạng theo MEDIAN: không bị kéo lệch bởi vài giá trị góc rất lớn
    stats = {j: (np.nanmedian(theta[:, j]), np.nanmean(theta[:, j]))
             for j in joint_ids if np.isfinite(theta[:, j]).any()}
    ranked = sorted(stats.items(), key=lambda kv: kv[1][0], reverse=True)
    print("\nTop 5 khớp XOAY NHIỀU NHẤT (xếp theo median):")
    for j, (med, mean) in ranked[:5]:
        print(f"  {JOINT_NAMES[j]:16s} median={med:.2f}°  mean={mean:.2f}°")
    print("Top 5 khớp XOAY ÍT NHẤT:")
    for j, (med, mean) in ranked[-5:]:
        print(f"  {JOINT_NAMES[j]:16s} median={med:.2f}°  mean={mean:.2f}°")

    # ---- Vẽ ----
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 3.4], hspace=0.15,
                          left=0.03, right=0.99, top=0.95, bottom=0.02)
    gs_top = gs[0].subgridspec(1, N_SNAPSHOTS, wspace=0.15)

    vmask = valid_mask(seq)
    def extent(t):
        p = seq[t][vmask[t], :2]
        return (p.max(axis=0) - p.min(axis=0)).max()
    radius = max(extent(t) for t in range(T)) / 2 * 1.15
    for k, t in enumerate(np.linspace(0, T - 1, N_SNAPSHOTS).astype(int)):
        ax0 = fig.add_subplot(gs_top[0, k])
        draw_body(ax0, seq[t], radius, f"frame {t}/{T-1}")

    n_cols = 6
    n_rows = int(np.ceil(len(joint_ids) / n_cols))
    gs_bot = gs[1].subgridspec(n_rows, n_cols, hspace=0.95, wspace=0.4)
    finite = theta[:, joint_ids][np.isfinite(theta[:, joint_ids])]
    n_over = int((finite > HIST_MAX_DEG).sum())
    print(f"{n_over}/{finite.size} giá trị góc > {HIST_MAX_DEG}° (bị cắt khỏi trục x của histogram)")
    for k, j in enumerate(joint_ids):
        ax = fig.add_subplot(gs_bot[divmod(k, n_cols)])
        vals = theta[:, j][np.isfinite(theta[:, j])]
        if vals.size == 0:
            ax.set_title(f"{JOINT_NAMES[j]} (không có dữ liệu)", fontsize=7, pad=3)
            ax.axis("off")
            continue
        ax.hist(vals, bins=20, range=(0, HIST_MAX_DEG), color=GROUP_COLORS[group_of(j)],
                edgecolor="black", alpha=0.85)
        ax.axvline(np.median(vals), color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{JOINT_NAMES[j]} (med {np.median(vals):.1f}°)", fontsize=7, pad=3)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"COBOT_v2 - class {CLASS_ID} - {names[i]}  ({len(idx_in_class)} sample trong class, λ={LAMBDA})",
                 fontsize=14, y=0.985)
    fig.savefig(OUT_PNG, dpi=110)
    plt.show()
    print(f"\nĐã lưu hình: {OUT_PNG}")


if __name__ == "__main__":
    main()