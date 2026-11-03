import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  

PKL_PATH = "ntu60_3d.pkl"
ANCHOR_JOINT_IDX = 0      
LAMBDA = 2
ACTION_ID = 23   
SAMPLE_INDEX = 0           
N_SNAPSHOTS = 4

JOINT_NAMES = [
    "SpineBase(anchor)", "SpineMid", "Neck", "Head",
    "ShoulderL", "ElbowL", "WristL", "HandL",
    "ShoulderR", "ElbowR", "WristR", "HandR",
    "HipL", "KneeL", "AnkleL", "FootL",
    "HipR", "KneeR", "AnkleR", "FootR",
    "SpineShoulder", "HandTipL", "ThumbL", "HandTipR", "ThumbR",
]

BONES = [
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14),
    (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), (22, 7), (23, 24), (24, 11),
]

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_keypoint_array(sample):
    kp = np.array(sample["keypoint"], dtype=np.float64)
    return kp[0] if kp.ndim == 4 else kp


def find_samples_by_action(annotations, action_id_1based):
    target_label = action_id_1based - 1
    return [s for s in annotations if s.get("label") == target_label]


def compute_angle_timeseries(keypoint_seq, anchor_idx, lam):
    T = keypoint_seq.shape[0]
    anchor = keypoint_seq[:, anchor_idx:anchor_idx + 1, :]
    ab = keypoint_seq - anchor
    ab_t, ab_t2 = ab[: T - lam], ab[lam:]
    dot = np.sum(ab_t * ab_t2, axis=-1)
    norm_t = np.linalg.norm(ab_t, axis=-1)
    norm_t2 = np.linalg.norm(ab_t2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = dot / (norm_t * norm_t2)
    cos_theta = np.where((norm_t < 1e-8) | (norm_t2 < 1e-8), np.nan, cos_theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))   


def draw_skeleton(ax, pose, title=""):
    ax.clear()
    x, y, z = pose[:, 0], pose[:, 1], pose[:, 2]
    ax.scatter(x, z, y, c="crimson", s=22)
    for a, b in BONES:
        ax.plot([pose[a, 0], pose[b, 0]], [pose[a, 2], pose[b, 2]], [pose[a, 1], pose[b, 1]],
                 c="#4C72B0", linewidth=2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=6)


def fixed_cube_limits(kp_seq):
    mins, maxs = kp_seq.min(axis=(0, 1)), kp_seq.max(axis=(0, 1))
    center = (mins + maxs) / 2
    radius = (maxs - mins).max() / 2 * 0.7
    return center, radius


def apply_cube_limits(ax, center, radius):
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[2] - radius, center[2] + radius)
    ax.set_zlim(center[1] - radius, center[1] + radius)


def main():
    data = load_pkl(PKL_PATH)
    annotations = data["annotations"] if isinstance(data, dict) and "annotations" in data else data

    samples = find_samples_by_action(annotations, ACTION_ID)
    sample = samples[min(SAMPLE_INDEX, len(samples) - 1)]
    kp_seq = get_keypoint_array(sample)
    T = kp_seq.shape[0]

    theta = compute_angle_timeseries(kp_seq, ANCHOR_JOINT_IDX, LAMBDA)   
    joint_ids = [v for v in range(theta.shape[1]) if v != ANCHOR_JOINT_IDX]

    # Xếp hạng khớp theo mean, in ra top 5 xoay nhiều nhất / ít nhất
    means = {j: np.nanmean(theta[:, j]) for j in joint_ids}
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    print(f"Sample: {sample.get('frame_dir')} | Số frame: {T}\n")
    print("Top 5 khớp XOAY NHIỀU NHẤT (mean cao nhất):")
    for j, m in ranked[:5]:
        print(f"  {JOINT_NAMES[j]:15s}  mean={m:.2f}°")
    print("\nTop 5 khớp XOAY ÍT NHẤT (mean thấp nhất):")
    for j, m in ranked[-5:]:
        print(f"  {JOINT_NAMES[j]:15s}  mean={m:.2f}°")

    # ---- Vẽ ----
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 2.5],
                           hspace=0.25, left=0.03, right=0.99, top=0.9, bottom=0.03)

    gs_top = gs[0].subgridspec(1, N_SNAPSHOTS, wspace=0.05)
    center, radius = fixed_cube_limits(kp_seq)
    snap_idxs = np.linspace(0, T - 1, N_SNAPSHOTS).astype(int)
    for i, t in enumerate(snap_idxs):
        ax = fig.add_subplot(gs_top[0, i], projection="3d")
        draw_skeleton(ax, kp_seq[t], title=f"frame {t}/{T-1}")
        apply_cube_limits(ax, center, radius)

    # Hàng dưới: lưới 24 histogram, mỗi khớp 1 ô
    n_cols, n_rows = 6, 4
    gs_bottom = gs[1].subgridspec(n_rows, n_cols, hspace=0.9, wspace=0.4)
    x_max = np.nanpercentile(theta[:, joint_ids], 99)

    for idx, j in enumerate(joint_ids):
        r, c = divmod(idx, n_cols)
        ax = fig.add_subplot(gs_bottom[r, c])
        vals = theta[:, j]
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=20, range=(0, x_max), color="#4C72B0", edgecolor="black", alpha=0.85)
        ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{JOINT_NAMES[j]} ({vals.mean():.1f}°)", fontsize=7.5, pad=3)
        ax.tick_params(labelsize=6)

    fig.suptitle(f"A{ACTION_ID:03d} - sample {sample.get('frame_dir')}  "
                 f"({len(samples)} sample)",
                 fontsize=14, y=0.97)
    fig.savefig("test.png", dpi=150)
    plt.show()
    print("\nĐã lưu hình: test.png")


if __name__ == "__main__":
    main()