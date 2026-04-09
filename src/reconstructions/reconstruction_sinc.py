import numpy as np
from scipy.signal import butter, sosfiltfilt

from debug.timer import time_method

@time_method(0.05)
def sinc_reconstruction(y, sr, Fs, taps=64, block_size=2048):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    k = np.arange(int(np.ceil((n / sr) * Fs)))
    sample_idx = np.round(k * sr / Fs).astype(int)
    sample_idx = np.clip(sample_idx, 0, n - 1)
    sample_idx = np.unique(sample_idx)
    s = y[sample_idx]

    out = np.empty(n, dtype=np.float64)
    offsets = np.arange(-taps, taps + 1)
    win = np.hanning(2 * taps + 1)

    for start in range(0, n, block_size):
        stop = min(n, start + block_size)

        i = np.arange(start, stop)
        tFs = i * (Fs / sr)                 # output time in sample-domain
        k0 = np.rint(tFs).astype(int)

        kk = k0[:, None] + offsets[None, :]
        kk = np.clip(kk, 0, len(s) - 1)

        xarg = tFs[:, None] - kk
        out[start:stop] = np.sum(s[kk] * np.sinc(xarg) * win[None, :], axis=1)

    return out

@time_method(0.05)
def lowpass_for_sampling(y, sr, Fs, transition_ratio=0.90, order=8):
    """
    Anti-alias filter before sampling at Fs.
    Keeps content below about transition_ratio * Fs/2.
    """
    y = np.asarray(y, dtype=np.float64)

    nyq_hi = 0.5 * float(sr)
    cutoff = 0.5 * float(Fs) * float(transition_ratio)

    if cutoff <= 0:
        return np.zeros_like(y)

    cutoff = min(cutoff, 0.999 * nyq_hi)
    sos = butter(order, cutoff / nyq_hi, btype="low", output="sos")
    return sosfiltfilt(sos, y)

@time_method(0.05)
def sinc_reconstruction_reference(y, sr, Fs):
    """
    Theory-oriented sinc reconstruction:
      1) anti-alias
      2) exact uniform sampling
      3) full finite sinc interpolation
    """
    y = np.asarray(y, dtype=np.float64)
    y_lp = lowpass_for_sampling(y, sr, Fs)
    out = sinc_reconstruction(y_lp, sr, Fs)
    return out