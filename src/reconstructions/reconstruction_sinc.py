import numpy as np

def sinc_reconstruction(y, sr, Fs, taps=64):
    """
    Approximate ideal sinc reconstruction from samples taken at Fs,
    reconstructing at sr. Uses a windowed sinc kernel with 'taps'.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)

    # sample indices and values
    k = np.arange(int(np.ceil((n / sr) * Fs)))
    sample_idx = np.round(k * sr / Fs).astype(int)
    sample_idx = np.clip(sample_idx, 0, n - 1)
    sample_idx = np.unique(sample_idx)
    s = y[sample_idx]

    # output positions
    t = np.arange(n) / sr  # seconds

    out = np.zeros(n, dtype=np.float64)

    # window (Hann)
    win = np.hanning(2 * taps + 1)

    # for each output sample, sum nearby sinc contributions
    # (not the fastest, but straightforward)
    for i in range(n):
        ti = t[i]
        # nearest sample index in k-domain
        k0 = int(round(ti * Fs))

        k_start = max(0, k0 - taps)
        k_end   = min(len(s) - 1, k0 + taps)

        kk = np.arange(k_start, k_end + 1)
        # time difference from each sample
        tau = ti - (kk / Fs)
        # sinc argument
        xarg = Fs * tau
        # window slice aligned to kk range
        w = win[(kk - (k0 - taps))]

        out[i] = np.sum(s[kk] * np.sinc(xarg) * w)

    return out
