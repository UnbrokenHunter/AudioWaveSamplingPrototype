import numpy as np

def num_samples(y):
    """Number of samples (works for mono or channel-last arrays)."""
    y = np.asarray(y)
    return y.shape[0] if y.ndim >= 1 else 0

def to_channel_last(y):
    """
    Normalize to shape (n_samples, channels).
    Accepts:
      (n,) mono
      (n, c) channel-last
      (c, n) channel-first (common when mono=False in some libs)
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return y[:, None]
    if y.shape[0] in (1, 2) and y.shape[0] < y.shape[1]:
        return y.T
    return y

def clip(y, lo=-1.0, hi=1.0):
    """Hard clip to a range (useful before playback)."""
    return np.clip(np.asarray(y), lo, hi)

def trim_to_length(y, n, from_end=False):
    """Trim to exactly n samples."""
    y = np.asarray(y)
    if num_samples(y) <= n:
        return y
    return y[-n:] if from_end else y[:n]

def pad_to_length(y, n, mode="constant", value=0.0, from_end=True):
    """
    Pad to exactly n samples.
    mode: np.pad mode ('constant', 'reflect', etc.)
    """
    y = np.asarray(y)
    cur = num_samples(y)
    if cur >= n:
        return y

    pad_amt = n - cur
    if from_end:
        pad_width = ((0, pad_amt),) + ((0, 0),) * (y.ndim - 1)
    else:
        pad_width = ((pad_amt, 0),) + ((0, 0),) * (y.ndim - 1)

    if mode == "constant":
        return np.pad(y, pad_width, mode=mode, constant_values=value)
    return np.pad(y, pad_width, mode=mode)

def match_lengths(signals, mode="trim"):
    """
    Make a list of 1D arrays match length.
    mode: 'trim' -> trim all to min length
          'pad'  -> pad all to max length with zeros
    """
    ys = [np.asarray(s) for s in signals]
    lens = [num_samples(s) for s in ys]

    if mode == "trim":
        n = min(lens)
        return [trim_to_length(s, n, from_end=False) for s in ys]
    if mode == "pad":
        n = max(lens)
        return [pad_to_length(s, n, value=0.0, from_end=True) for s in ys]

    raise ValueError("mode must be 'trim' or 'pad'")
