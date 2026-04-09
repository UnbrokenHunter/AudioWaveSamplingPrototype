from loader import *
from analysis import *
from visualize import *
from playback import *
from reconstructions import *

import tkinter as tk
import numpy as np

from cache import ReconstructionCacheManager


class WaveformApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Waveform Viewer")
        self.geometry("1200x700")

        y, sr, path = select_file()

        # If stereo (or multi-channel), convert to mono
        if y.ndim == 2:
            # common layouts are (channels, samples) or (samples, channels)
            if y.shape[0] <= 8 and y.shape[0] < y.shape[1]:
                y = y.mean(axis=0)      # (channels, samples) -> (samples,)
            else:
                y = y.mean(axis=1)      # (samples, channels) -> (samples,)

        self.y = np.asarray(y, dtype=np.float64)
        self.sr = sr

        self.signal_labels = [
            "original",
            "direct reconstruction",
            "linear reconstruction",
            "nearest reconstruction",
            "dac reconstruction",
            "sinc reconstruction",
            "sinc reconstruction (lowpassed)",
            "direct subtract",
        ]
        self.signal_data = {
            "original": self.y,
            "direct reconstruction": np.zeros_like(self.y),
            "linear reconstruction": np.zeros_like(self.y),
            "nearest reconstruction": np.zeros_like(self.y),
            "dac reconstruction": np.zeros_like(self.y),
            "sinc reconstruction": np.zeros_like(self.y),
            "sinc reconstruction (lowpassed)": np.zeros_like(self.y),
            "direct subtract": np.zeros_like(self.y),
        }

        self.cache_manager = ReconstructionCacheManager(self.y, self.sr, sample_indices)

        tkinter_figure(
            self,
            self.signal_data,
            sr,
            labels=self.signal_labels,
            title=str(path),
            zoom_seconds=0.01
        )

        self.sample_frequency_bounds = (0.5, 48000.0)
        self.sample_frequency = self.ui.bind_slider(
            "sample_frequency",
            self.sample_frequency_bounds[0], self.sample_frequency_bounds[1], 7.0,
            step=0.1,
            label="Sample Frequency (Hz)",
            length=600,
            log=True,
        )
        self._install_sample_frequency_cache_buttons()

        self._recompute_job = None
        self.sample_frequency.trace_add("write", lambda *_: self.request_recompute())
        self.recompute()

    def _install_sample_frequency_cache_buttons(self):
        slider = getattr(self, "sample_frequency_slider", None)
        if slider is None:
            return

        parent = slider.master
        self.sample_frequency_prev_button = tk.Button(
            parent,
            text="<",
            width=3,
            command=self.jump_to_prev_sample_frequency_bucket,
        )
        self.sample_frequency_prev_button.pack(side=tk.LEFT, padx=(6, 2))

        self.sample_frequency_next_button = tk.Button(
            parent,
            text=">",
            width=3,
            command=self.jump_to_next_sample_frequency_bucket,
        )
        self.sample_frequency_next_button.pack(side=tk.LEFT, padx=(2, 6))

        self.sample_frequency_bucket_label_var = tk.StringVar()
        self.sample_frequency_bucket_label = tk.Label(
            parent,
            textvariable=self.sample_frequency_bucket_label_var,
            width=18,
            anchor="w",
        )
        self.sample_frequency_bucket_label.pack(side=tk.LEFT, padx=(0, 6))

        self.sample_frequency.trace_add(
            "write",
            lambda *_: self.cache_manager.update_navigation_ui(self, float(self.sample_frequency.get())),
        )
        self.cache_manager.update_navigation_ui(self, float(self.sample_frequency.get()))

    def jump_to_prev_sample_frequency_bucket(self):
        fs = float(self.sample_frequency.get())
        prev_fs = self.cache_manager.adjacent_cached_sample_frequency(fs, -1)
        if prev_fs is not None:
            self.sample_frequency.set(prev_fs)

    def jump_to_next_sample_frequency_bucket(self):
        fs = float(self.sample_frequency.get())
        next_fs = self.cache_manager.adjacent_cached_sample_frequency(fs, +1)
        if next_fs is not None:
            self.sample_frequency.set(next_fs)

    def request_recompute(self):
        if self._recompute_job is not None:
            self.after_cancel(self._recompute_job)
        self._recompute_job = self.after(40, self.recompute)

    @time_method(0.05)
    def recompute(self):
        self._recompute_job = None

        y = self.y
        sr = self.sr

        fs = float(self.sample_frequency.get())
        idx = self.cache_manager.get_sample_indices(fs)

        recon_fns = {
            "direct reconstruction": lambda: direct_reconstruction(y, idx),
            "linear reconstruction": lambda: linear_reconstruction(y, idx),
            "nearest reconstruction": lambda: nearest_reconstruction(y, idx),
            "dac reconstruction": lambda: dac_reconstruction(y, idx, sr, fs),
            "sinc reconstruction": lambda: sinc_reconstruction(y, sr, fs),
            "sinc reconstruction (lowpassed)": lambda: sinc_reconstruction_lowpassed(y, sr, fs),
            "direct subtract": lambda: subtract_direct_reconstruction(y, idx),
        }

        for label, fn in recon_fns.items():
            line = self.lines_by_label[label]
            if not line.get_visible():
                continue
            values = self.cache_manager.get_reconstruction(label, fs, fn)
            self.signal_data[label] = values
            line.set_ydata(values)

        self._visible_window_cache = None
        self.cache_manager.update_navigation_ui(self, fs)

        request_bottom_update(self)

        self.ui.ctx["canvas"].draw_idle()


@time_method(0.05)
def sample_indices(n, sr, fs):
    k = np.arange(int(np.ceil((n / sr) * fs)))
    idx = np.round(k * sr / fs).astype(int)
    idx = np.clip(idx, 0, n - 1)
    idx = np.unique(idx)
    return idx



if __name__ == "__main__":
    WaveformApp().mainloop()
