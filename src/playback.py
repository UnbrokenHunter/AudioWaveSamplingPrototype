import numpy as np
import sounddevice as sd


def _to_channel_last(samples):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return samples[:, None]
    if samples.shape[0] in (1, 2) and samples.shape[0] < samples.shape[1]:
        return samples.T
    return samples


def play_audio(samples, sr=44100, blocking=True):
    y = _to_channel_last(samples)

    if y.dtype != np.float32:
        y = y.astype(np.float32)

    y = np.clip(y, -1.0, 1.0)
    sd.play(y, samplerate=sr, blocking=blocking)


def stop_audio():
    sd.stop()
    