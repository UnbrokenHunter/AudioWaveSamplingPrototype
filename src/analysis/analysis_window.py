import numpy as np
from .analysis_time import time_to_index
from .analysis_reshaping import num_samples, pad_to_length

def slice_time(y, sr, t0, t1):
    """Slice y between times t0 and t1 (seconds)."""
    y = np.asarray(y)
    n = num_samples(y)
    i0 = time_to_index(t0, sr, clamp="clip", n=n)
    i1 = time_to_index(t1, sr, clamp="clip", n=n)
    if i1 < i0:
        i0, i1 = i1, i0
    return y[i0:i1]

def center_window(y, sr, t_center, width_sec):
    """Window centered at t_center with duration width_sec."""
    half = width_sec / 2.0
    return slice_time(y, sr, t_center - half, t_center + half)

def hann(n):
    """Hann window."""
    return np.hanning(int(n))

def fade_in(y, sr, duration_sec):
    """Linear fade-in."""
    y = np.asarray(y).copy()
    n = int(round(duration_sec * sr))
    n = max(0, min(n, len(y)))
    if n == 0:
        return y
    y[:n] *= np.linspace(0.0, 1.0, n)
    return y

def fade_out(y, sr, duration_sec):
    """Linear fade-out."""
    y = np.asarray(y).copy()
    n = int(round(duration_sec * sr))
    n = max(0, min(n, len(y)))
    if n == 0:
        return y
    y[-n:] *= np.linspace(1.0, 0.0, n)
    return y

def frame_signal(y, frame_size, hop):
    """
    Split a 1D signal into overlapping frames.
    Returns shape (n_frames, frame_size).
    """
    y = np.asarray(y)
    frame_size = int(frame_size)
    hop = int(hop)
    if frame_size <= 0 or hop <= 0:
        raise ValueError("frame_size and hop must be > 0")

    if len(y) < frame_size:
        y = pad_to_length(y, frame_size, value=0.0)

    n_frames = 1 + (len(y) - frame_size) // hop
    frames = np.empty((n_frames, frame_size), dtype=np.float64)

    for i in range(n_frames):
        start = i * hop
        frames[i] = y[start:start + frame_size]
    return frames
