import numpy as np

from .analysis_channel import to_mono
from .analysis_reshaping import pad_to_length


# ---------------------------
# Utility
# ---------------------------

def mag_to_db(mag, ref=1.0, amin=1e-12):
    """Linear magnitude → dB."""
    mag = np.asarray(mag)
    return 20.0 * np.log10(np.maximum(mag, amin) / float(ref))


# ---------------------------
# One-shot FFT / spectrum
# ---------------------------

def rfft_spectrum(y, sr, n_fft=None, window="hann", center=True):
    """
    Compute a magnitude spectrum using rFFT.

    Returns:
        freqs_hz, mag_linear, phase_rad
    """
    x = np.asarray(y, dtype=np.float64)
    if x.ndim != 1:
        x = to_mono(x)

    if n_fft is None:
        n_fft = len(x)
    n_fft = int(n_fft)

    if len(x) < n_fft:
        x = pad_to_length(x, n_fft, value=0.0)

    if center:
        start = max(0, (len(x) - n_fft) // 2)
        x = x[start:start + n_fft]
    else:
        x = x[:n_fft]

    if window == "hann":
        w = np.hanning(n_fft)
    elif window is None:
        w = np.ones(n_fft)
    else:
        raise ValueError("window must be 'hann' or None")

    X = np.fft.rfft(x * w)
    mag = np.abs(X)
    phase = np.angle(X)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    return freqs, mag, phase


def dominant_freq(y, sr, n_fft=4096):
    """Return (frequency_hz, magnitude) of strongest FFT bin."""
    freqs, mag, _ = rfft_spectrum(y, sr, n_fft=n_fft, window="hann", center=True)
    k = int(np.argmax(mag))
    return float(freqs[k]), float(mag[k])


# ---------------------------
# STFT
# ---------------------------

def stft(y, n_fft=1024, hop=256, window="hann", center=True):
    """
    Short-time Fourier transform.

    Returns:
        complex array shape (frames, bins)
    """
    x = np.asarray(y, dtype=np.float64)
    if x.ndim != 1:
        x = to_mono(x)

    n_fft = int(n_fft)
    hop = int(hop)

    if center:
        pad = n_fft // 2
        x = np.pad(x, (pad, pad), mode="constant")

    if len(x) < n_fft:
        x = pad_to_length(x, n_fft, value=0.0)

    if window == "hann":
        w = np.hanning(n_fft)
    elif window is None:
        w = np.ones(n_fft)
    else:
        raise ValueError("window must be 'hann' or None")

    n_frames = 1 + (len(x) - n_fft) // hop
    out = np.empty((n_frames, n_fft // 2 + 1), dtype=np.complex128)

    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + n_fft] * w
        out[i] = np.fft.rfft(frame)

    return out


def stft_mag_db(y, sr, n_fft=1024, hop=256):
    """
    Returns:
      f (Hz) shape (n_freq,)
      t (sec) shape (n_frames,)
      S_db shape (n_freq, n_frames)
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # pad to at least one frame
    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))

    w = np.hanning(n_fft)

    # build frames
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    if frames.ndim != 2 or frames.shape[0] == 0:
        frames = y[:n_fft][None, :]

    X = np.fft.rfft(frames * w[None, :], axis=1)  # (n_frames, n_freq)
    mag = np.abs(X)
    S_db = (20.0 * np.log10(mag + 1e-12)).T       # (n_freq, n_frames)

    f = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    t = (np.arange(S_db.shape[1]) * hop) / float(sr)
    return f, t, S_db
