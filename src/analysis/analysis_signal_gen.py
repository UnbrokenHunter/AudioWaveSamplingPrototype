import numpy as np

def sine(freq, sr, duration, amp=1.0, phase=0.0):
    t = np.linspace(0.0, float(duration), int(sr * duration), endpoint=False)
    return amp * np.sin(2.0 * np.pi * float(freq) * t + float(phase))

def square(freq, sr, duration, amp=1.0):
    t = np.linspace(0.0, float(duration), int(sr * duration), endpoint=False)
    return amp * np.sign(np.sin(2.0 * np.pi * float(freq) * t))

def noise(sr, duration, amp=1.0):
    n = int(sr * duration)
    return amp * np.random.uniform(-1.0, 1.0, size=n)

def impulse(sr, duration, at_time=0.0, amp=1.0):
    n = int(sr * duration)
    y = np.zeros(n, dtype=np.float64)
    i = int(round(at_time * sr))
    if 0 <= i < n:
        y[i] = amp
    return y
