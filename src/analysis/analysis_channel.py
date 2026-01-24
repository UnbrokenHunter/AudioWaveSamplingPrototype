import numpy as np
from .analysis_reshaping import to_channel_last

def get_channel(y, ch=0):
    """Extract a single channel as 1D."""
    y = to_channel_last(y)
    ch = int(np.clip(ch, 0, y.shape[1] - 1))
    return y[:, ch]

def to_mono(y, method="mean"):
    """
    Stereo -> mono.
    method: 'mean' | 'left' | 'right'
    """
    y = to_channel_last(y)
    if y.shape[1] == 1:
        return y[:, 0]
    if method == "left":
        return y[:, 0]
    if method == "right":
        return y[:, 1]
    return np.mean(y, axis=1)
