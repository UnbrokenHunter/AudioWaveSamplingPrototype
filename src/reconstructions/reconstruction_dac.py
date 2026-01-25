import numpy as np

def zoh_hold_from_indices(y, idx):
    """
    Zero-Order Hold (sample-and-hold) reconstruction at the original sample rate.
    idx: sample indices (ascending) where the DAC updates its held value.
    """
    y = np.asarray(y)
    n = len(y)

    idx = np.asarray(idx, dtype=int)
    idx = np.clip(idx, 0, n - 1)
    idx = np.unique(idx)
    if idx.size == 0:
        return np.zeros_like(y)

    out = np.zeros_like(y)
    # Hold each sample value until next sample index
    for a, b in zip(idx[:-1], idx[1:]):
        out[a:b] = y[a]
    out[idx[-1]:] = y[idx[-1]]
    return out


def one_pole_lowpass(x, sr, cutoff_hz):
    """
    1st-order RC low-pass filter (analog-ish).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros_like(x)

    cutoff_hz = float(cutoff_hz)
    if cutoff_hz <= 0:
        return y

    dt = 1.0 / float(sr)
    rc = 1.0 / (2.0 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)

    y[0] = x[0]
    for n in range(1, len(x)):
        y[n] = y[n - 1] + alpha * (x[n] - y[n - 1])
    return y


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
