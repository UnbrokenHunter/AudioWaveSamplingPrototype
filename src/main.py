from loader import *
from analysis import *
from visualize import *
from playback import *
from reconstructions import *

import tkinter as tk
import numpy as np

class WaveformApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Waveform Viewer")
        self.geometry("900x600")

        y, sr, path = select_file()

        self.y = y
        self.sr = sr

        tkinter_figure(
            self,
            [y, np.zeros_like(y), np.zeros_like(y)],
            sr,
            labels=["original", "direct reconstruction", "linear reconstruction"],
            title=str(path),
            zoom_seconds=0.01
        )

        self.sample_frequency = self.ui.bind_slider(
            "sample_frequency",
            0.5, 48000.0, 7.0,
            step=0.1,
            label="Sample Frequency (Hz)",
        )

        self._recompute_job = None
        self.sample_frequency.trace_add("write", lambda *_: self.request_recompute())
        self.recompute()

    def request_recompute(self):
        if self._recompute_job is not None:
            self.after_cancel(self._recompute_job)
        self._recompute_job = self.after(40, self.recompute)

    def recompute(self):
        self._recompute_job = None

        y = self.y
        sr = self.sr

        fs = float(self.sample_frequency.get())
        idx = sample_indices(len(y), sr, fs)

        reconstructions = [
            direct_reconstruction(y, idx),
            linear_reconstruction(y, idx),
        ]

        for line, data in zip(self._plot_lines[1:], reconstructions):
            line.set_ydata(data)

        self.ui.ctx["canvas"].draw_idle()


def sample_indices(n, sr, fs):
    k = np.arange(int(np.ceil((n / sr) * fs)))
    idx = np.round(k * sr / fs).astype(int)
    idx = np.clip(idx, 0, n - 1)
    idx = np.unique(idx)
    return idx

if __name__ == "__main__":
    WaveformApp().mainloop()
