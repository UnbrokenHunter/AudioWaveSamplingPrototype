import numpy as np
import sounddevice as sd

def play_audio(samples, sr=44100, blocking=True):
    y = np.asarray(samples)

    if y.dtype != np.float32:
        y = y.astype(np.float32)

    y = np.clip(y, -1.0, 1.0)
    sd.play(y, samplerate=sr, blocking=blocking)


def stop_audio():
    sd.stop()
    