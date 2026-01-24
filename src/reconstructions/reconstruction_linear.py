import numpy as np

def linear_reconstruction(y, idx):
    """Piecewise linear interpolation between sample points."""
    out = np.zeros_like(y)

    if len(idx) == 0:
        return out
    if len(idx) == 1:
        out[idx[0]] = y[idx[0]]
        return out

    # Fill each segment between consecutive sampled points
    for a, b in zip(idx[:-1], idx[1:]):
        ya, yb = y[a], y[b]
        span = b - a
        if span <= 0:
            continue
        t = np.arange(span + 1) / span  # 0..1
        out[a:b + 1] = ya + (yb - ya) * t

    # (optional) hold ends instead of zero
    out[:idx[0]] = y[idx[0]]
    out[idx[-1]:] = y[idx[-1]]
    return out
