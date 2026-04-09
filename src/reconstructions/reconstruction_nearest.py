import numpy as np

from debug.timer import time_method

@time_method(threshold=0.05)
def nearest_reconstruction(y, idx):
    y = np.asarray(y)
    idx = np.asarray(idx, dtype=int)

    if idx.size == 0:
        return np.zeros_like(y)
    if idx.size == 1:
        out = np.empty_like(y, dtype=np.float64)
        out.fill(y[idx[0]])
        return out

    idx = np.clip(idx, 0, len(y) - 1)
    idx = np.unique(idx)

    x = np.arange(len(y))
    midpoints = (idx[:-1] + idx[1:]) / 2.0
    bins = np.digitize(x, midpoints)

    return y[idx[bins]]
