import numpy as np
from .analysis_reshaping import clip

def rms(y):
    """Root-mean-square amplitude."""
    y = np.asarray(y)
    return np.sqrt(np.mean(y**2))

def peak(y):
    """Peak absolute amplitude."""
    y = np.asarray(y)
    return np.max(np.abs(y))

def dbfs_rms(y, eps=1e-12):
    """RMS in dBFS."""
    return 20.0 * np.log10(rms(y) + eps)

def dbfs_peak(y, eps=1e-12):
    """Peak in dBFS."""
    return 20.0 * np.log10(peak(y) + eps)

def normalize(y, peak_level=1.0):
    """Scale so peak(abs) == peak_level."""
    y = np.asarray(y)
    p = peak(y)
    if p == 0:
        return y
    return y * (peak_level / p)

def soft_clip_tanh(y, drive=1.0):
    """
    Gentle soft clip using tanh. drive > 1 increases saturation.
    Output stays in (-1, 1).
    """
    y = np.asarray(y, dtype=np.float64)
    return np.tanh(drive * y)

def safe_for_playback(y, lo=-1.0, hi=1.0, dtype=np.float32):
    """Clip and cast to a good playback dtype."""
    y = clip(y, lo, hi)
    return np.asarray(y, dtype=dtype)
