import numpy as np
from .analysis_reshaping import num_samples

def time_to_index(t, sr, clamp=None, n=None):
    """
    Seconds -> sample index.
    clamp: None | 'clip' (requires n) to keep index in [0, n-1]
    """
    i = int(round(float(t) * float(sr)))
    if clamp == "clip":
        if n is None:
            raise ValueError("n must be provided when clamp='clip'")
        i = int(np.clip(i, 0, n - 1))
    return i

def index_to_time(i, sr):
    """Sample index -> seconds."""
    return float(i) / float(sr)

def duration_sec(y, sr):
    """Duration in seconds."""
    return num_samples(y) / float(sr)

def get_sample_at_time(y, sr, t, default=0.0):
    """Safe sample at time t; returns default if out of bounds."""
    y = np.asarray(y)
    i = time_to_index(t, sr)
    if i < 0 or i >= num_samples(y):
        return default
    return y[i]
