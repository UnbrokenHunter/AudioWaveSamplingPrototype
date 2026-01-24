from tkinter import Tk, filedialog
import librosa

from analysis import rms

def select_file(mono=False):
    Tk().withdraw()  # hide the tiny tk window
    path = filedialog.askopenfilename(
        title="Choose an audio file",
        filetypes=[
            ("Audio files", "*.wav *.flac *.mp3 *.ogg *.m4a *.aiff"),
            ("All files", "*.*"),
        ],
    )

    y, sr = librosa.load(path, sr=None, mono=mono)  # sr=None keeps original sample rate

    print("Selected:", path)
    print("Sample Rate:", sr)
    print("Shape:", y.shape)  # (n,) mono OR (channels, n) when mono=False
    print("RMS: ", rms(y))

    return y, sr, path
