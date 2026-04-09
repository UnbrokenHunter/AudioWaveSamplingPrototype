import numpy as np
from scipy.signal import lfilter

from debug.timer import time_method

@time_method(0.05)
def zoh_hold_from_indices(y, idx):
    y = np.asarray(y)
    n = len(y)

    idx = np.asarray(idx, dtype=int)
    idx = np.clip(idx, 0, n - 1)
    idx = np.unique(idx)

    if idx.size == 0:
        return np.zeros_like(y)

    lengths = np.diff(np.r_[idx, n])
    return np.repeat(y[idx], lengths)

@time_method(threshold=0.05)
def one_pole_lowpass(x, sr, cutoff_hz):
    x = np.asarray(x, dtype=np.float64)

    if cutoff_hz <= 0:
        return np.zeros_like(x)

    dt = 1.0 / float(sr)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)

    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    return lfilter(b, a, x)

@time_method(threshold=0.05)
def dac_reconstruction(y, idx, sr, fs, cutoff_hz=None, poles=2):
    """
    "Genuine-ish" DAC reconstruction:
      1) ZOH hold at sample update points (idx)
      2) Analog reconstruction LPF (modeled as cascaded 1-pole filters)

    y: original signal at sample rate sr
    fs: sampling frequency you chose (Hz)
    cutoff_hz:
      - default: 0.45*fs (but never above 0.49*Nyquist of sr)
    poles:
      - number of cascaded 1-pole filters (2–4 feels more "DAC-like")
    """
    zoh = zoh_hold_from_indices(y, idx)

    nyquist = 0.5 * float(sr)
    if cutoff_hz is None:
        cutoff_hz = 0.45 * float(fs)
    cutoff_hz = min(float(cutoff_hz), 0.49 * nyquist)

    out = zoh.astype(np.float64, copy=False)
    for _ in range(max(1, int(poles))):
        out = one_pole_lowpass(out, sr, cutoff_hz)

    return out
