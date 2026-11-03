import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

DATA_DIR = r"D:\Thesis2\datacobot_v2\xsub"   # change to your own path (e.g. Drive / local)
T, V = 96, 48
HAND_A = slice(0, 21)
HAND_B = slice(21, 42)
ANCHOR = (42, 43)                      # STA anchor = midpoint of the two shoulders

# ---------------------------------------------------------------- skeleton graph
_HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4),
               (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16),
               (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
EDGES = (_HAND_EDGES
         + [(a + 21, b + 21) for a, b in _HAND_EDGES]
         + [(42, 43), (42, 44), (44, 46), (43, 45), (45, 47)]   # shoulders / arms
         + [(0, 47), (21, 46)])                                  # hand wrist <-> pose wrist


def adjacency(self_loop=True):
    A = np.zeros((V, V), dtype=np.float32)
    for i, j in EDGES:
        A[i, j] = A[j, i] = 1
    if self_loop:
        A += np.eye(V, dtype=np.float32)
    return A


# ---------------------------------------------------------------- loading
def load_split(split, data_dir=DATA_DIR):
    """split in {'train', 'val'} -> X (N,3,T,V) float32, y (N,) int64"""
    X = np.load(f"{data_dir}/{split}_position.npy")[..., 0]          # drop M
    with open(f"{data_dir}/{split}_label.pkl", "rb") as f:
        _names, labels = pickle.load(f)
    return X.astype(np.float32), np.asarray(labels, dtype=np.int64)


# ---------------------------------------------------------------- missing hands
def fill_missing_hands_causal(X):
    X = X.copy()
    N, _, Tn, _ = X.shape
    obs = np.zeros((N, 2, Tn), dtype=np.float32)
    t_idx = np.arange(Tn)[None, :]                                   # (1,T)
    for hi, (sl, pose_wrist) in enumerate([(HAND_A, 47), (HAND_B, 46)]):
        hand = X[:, :, :, sl]                                        # (N,3,T,21)
        vis = np.abs(hand).sum(axis=(1, 3)) > 0                      # (N,T)
        obs[:, hi] = vis
        last = np.maximum.accumulate(np.where(vis, t_idx, -1), axis=1)   # (N,T)
        has_past = last >= 0
        gather = np.broadcast_to(np.clip(last, 0, None)[:, None, :, None], hand.shape)
        ffill = np.take_along_axis(hand, gather, axis=2)             # last visible pose
        wrist = np.broadcast_to(X[:, :, :, pose_wrist][..., None], hand.shape)
        X[:, :, :, sl] = np.where(has_past[:, None, :, None], ffill, wrist)
    return X, obs


# ---------------------------------------------------------------- partial observation
def truncate(x, ratio):
    """Keep the first `ratio` of the frames. x: (..., T, V) with T on dim -2."""
    t = max(2, int(np.ceil(ratio * x.shape[-2])))
    return x[..., :t, :]


# ---------------------------------------------------------------- dataset
class CobotEAP(Dataset):
    """Returns the FULL sequence; cut it per batch with `truncate()`.
    (Teacher needs the full sequence, student needs a prefix of the same one.)"""

    def __init__(self, split, data_dir=DATA_DIR, fill=True, return_mask=False):
        self.return_mask = return_mask
        X, y = load_split(split, data_dir)
        self.obs = None
        if fill:
            X, self.obs = fill_missing_hands_causal(X)
        self.X, self.y = torch.from_numpy(X), torch.from_numpy(y)
        if self.obs is not None:
            self.obs = torch.from_numpy(self.obs)          # (N,2,T), analysis only

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.return_mask and self.obs is not None:
            return self.X[i], self.y[i], self.obs[i]
        return self.X[i], self.y[i]          # (3,T,V), scalar


if __name__ == "__main__":
    tr, va = CobotEAP("train"), CobotEAP("val")
    x, y = tr[0]
    print("train", len(tr), "val", len(va), "| sample", tuple(x.shape), "label", int(y))
    print("zeros left in hands after fill:",
          int((tr.X[:, :, :, :42].abs().sum(1) == 0).sum()))
    print("NaN:", bool(torch.isnan(tr.X).any()))
    print("prefix 20% ->", tuple(truncate(x, 0.2).shape))
    A = adjacency()
    print("graph:", A.shape, "edges:", len(EDGES), "connected components check:",
          bool((np.linalg.matrix_power(A, V) > 0).all()))