import numpy as np

def direct_reconstruction(y, idx):
    """Spikes at sample locations."""
    out = np.zeros_like(y)
    out[idx] = y[idx]
    return out
