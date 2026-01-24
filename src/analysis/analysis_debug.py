import numpy as np
from .analysis_reshaping import num_samples
from .analysis_level import rms, peak, dbfs_rms, dbfs_peak

def describe(y, sr=None):
    """Quick summary dict for a signal."""
    y = np.asarray(y)
    info = {
        "shape": y.shape,
        "dtype": str(y.dtype),
        "num_samples": int(num_samples(y)),
        "peak": float(peak(y)),
        "rms": float(rms(y)),
        "dbfs_peak": float(dbfs_peak(y)),
        "dbfs_rms": float(dbfs_rms(y)),
    }
    if sr is not None:
        info["sr"] = int(sr)
        info["duration_sec"] = float(num_samples(y) / float(sr))
    return info
