import numpy as np

from debug.timer import time_method

@time_method(threshold=0.05)
def linear_reconstruction(y, idx):
    y = np.asarray(y)
    idx = np.asarray(idx, dtype=int)

    if idx.size == 0:
        return np.zeros_like(y)
    if idx.size == 1:
        out = np.empty_like(y, dtype=np.float64)
        out.fill(y[idx[0]])
        return out

    x = np.arange(len(y))
    return np.interp(x, idx, y[idx])