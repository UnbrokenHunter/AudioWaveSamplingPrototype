import numpy as np
from scipy.interpolate import CubicSpline

from debug.timer import time_method

@time_method(threshold=0.05)
def cubic_reconstruction(y, idx):
    y = np.asarray(y)
    idx = np.asarray(idx, dtype=int)

    if idx.size == 0:
        return np.zeros_like(y)
    if idx.size == 1:
        out = np.empty_like(y, dtype=np.float64)
        out.fill(y[idx[0]])
        return out
    if idx.size == 2:
        return np.interp(np.arange(len(y)), idx, y[idx])

    idx = np.clip(idx, 0, len(y) - 1)
    idx = np.unique(idx)

    if idx.size == 2:
        return np.interp(np.arange(len(y)), idx, y[idx])

    cs = CubicSpline(idx, y[idx], bc_type="natural")
    x = np.arange(len(y))
    return cs(x)