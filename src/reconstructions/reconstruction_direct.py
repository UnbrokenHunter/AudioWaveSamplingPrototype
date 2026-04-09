import numpy as np

from debug.timer import time_method

@time_method(threshold=0.05)
def direct_reconstruction(y, idx):
    """Spikes at sample locations."""
    out = np.zeros_like(y)
    out[idx] = y[idx]
    return out

@time_method(threshold=0.05)
def subtract_direct_reconstruction(y, idx):
    """Spikes at sample locations."""
    direct = direct_reconstruction(y, idx)
    subtracted = y - direct
    return subtracted