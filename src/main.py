from bisect import bisect_left, bisect_right
from collections import OrderedDict

from loader import *
from analysis import *
from visualize import *
from playback import *
from reconstructions import *

import tkinter as tk
import numpy as np


class LRUCache:
    """Tiny LRU cache for numpy-heavy results keyed by normalized params."""
    def __init__(self, maxsize=8):
        self.maxsize = int(max(1, maxsize))
        self._data = OrderedDict()

    def get(self, key):
        if key not in self._data:
            return None
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key, value):
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()


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
            "dac reconstruction",
            "sinc reconstruction",
            "sinc reconstruction (lowpassed)",
            "direct subtract",
        ]
        self.signal_data = {
            "original": self.y,
            "direct reconstruction": np.zeros_like(self.y),
            "linear reconstruction": np.zeros_like(self.y),
            "dac reconstruction": np.zeros_like(self.y),
            "sinc reconstruction": np.zeros_like(self.y),
            "sinc reconstruction (lowpassed)": np.zeros_like(self.y),
            "direct subtract": np.zeros_like(self.y),
        }

        # Caches for interactive recompute.
        # fs values come from a log slider, so normalize keys to a stable precision.
        self._idx_cache = LRUCache(maxsize=30)
        self._recon_cache = {
            "direct reconstruction": LRUCache(maxsize=30),
            "linear reconstruction": LRUCache(maxsize=30),
            "dac reconstruction": LRUCache(maxsize=30),
            "sinc reconstruction": LRUCache(maxsize=30),
            "sinc reconstruction (lowpassed)": LRUCache(maxsize=30),
            "direct subtract": LRUCache(maxsize=30),
        }

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

    @staticmethod
    def _fs_cache_key(fs):
        """Normalize slider-derived float values so nearby identical states reuse cache."""
        return float(f"{float(fs):.12g}")

    def _get_sample_indices(self, fs):
        key = self._fs_cache_key(fs)
        idx = self._idx_cache.get(key)
        if idx is None:
            idx = sample_indices(len(self.y), self.sr, fs)
            self._idx_cache.put(key, idx)
        return idx

    def _get_reconstruction(self, label, fs, compute_fn):
        key = self._fs_cache_key(fs)
        cache = self._recon_cache[label]
        values = cache.get(key)
        if values is None:
            values = np.asarray(compute_fn(), dtype=np.float64)
            cache.put(key, values)
        return values
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

        self.sample_frequency.trace_add("write", lambda *_: self._update_sample_frequency_bucket_ui())
        self._update_sample_frequency_bucket_ui()

    def _cached_sample_frequency_values(self):
        values = set()

        values.update(float(k) for k in self._idx_cache._data.keys())
        for cache in self._recon_cache.values():
            values.update(float(k) for k in cache._data.keys())

        return sorted(values)

    def _nearest_cached_sample_frequency(self, fs):
        cached = self._cached_sample_frequency_values()
        if not cached:
            return None

        i = bisect_left(cached, fs)
        if i <= 0:
            return float(cached[0])
        if i >= len(cached):
            return float(cached[-1])

        left = float(cached[i - 1])
        right = float(cached[i])
        if abs(fs - left) <= abs(right - fs):
            return left
        return right

    def _adjacent_cached_sample_frequency(self, fs, direction):
        cached = self._cached_sample_frequency_values()
        if not cached:
            return None

        if direction < 0:
            i = bisect_left(cached, fs) - 1
            if i < 0:
                return None
            return float(cached[i])

        i = bisect_right(cached, fs)
        if i >= len(cached):
            return None
        return float(cached[i])

    def jump_to_prev_sample_frequency_bucket(self):
        fs = float(self.sample_frequency.get())
        prev_fs = self._adjacent_cached_sample_frequency(fs, -1)
        if prev_fs is not None:
            self.sample_frequency.set(prev_fs)

    def jump_to_next_sample_frequency_bucket(self):
        fs = float(self.sample_frequency.get())
        next_fs = self._adjacent_cached_sample_frequency(fs, +1)
        if next_fs is not None:
            self.sample_frequency.set(next_fs)

    def _update_sample_frequency_bucket_ui(self):
        fs = float(self.sample_frequency.get())
        nearest = self._nearest_cached_sample_frequency(fs)
        if nearest is None:
            self.sample_frequency_bucket_label_var.set("Nearest cache: none")
        else:
            self.sample_frequency_bucket_label_var.set(f"Nearest cache: {nearest:g} Hz")

        prev_value = self._adjacent_cached_sample_frequency(fs, -1)
        next_value = self._adjacent_cached_sample_frequency(fs, +1)

        if hasattr(self, "sample_frequency_prev_button"):
            state = tk.NORMAL if prev_value is not None else tk.DISABLED
            self.sample_frequency_prev_button.configure(state=state)
        if hasattr(self, "sample_frequency_next_button"):
            state = tk.NORMAL if next_value is not None else tk.DISABLED
            self.sample_frequency_next_button.configure(state=state)

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
        idx = self._get_sample_indices(fs)

        recon_fns = {
            "direct reconstruction": lambda: direct_reconstruction(y, idx),
            "linear reconstruction": lambda: linear_reconstruction(y, idx),
            "dac reconstruction": lambda: dac_reconstruction(y, idx, sr, fs),
            "sinc reconstruction": lambda: sinc_reconstruction(y, sr, fs),
            "sinc reconstruction (lowpassed)": lambda: sinc_reconstruction_lowpassed(y, sr, fs),
            "direct subtract": lambda: subtract_direct_reconstruction(y, idx),
        }

        for label, fn in recon_fns.items():
            line = self.lines_by_label[label]
            if not line.get_visible():
                continue
            values = self._get_reconstruction(label, fs, fn)
            self.signal_data[label] = values
            line.set_ydata(values)

        self._visible_window_cache = None
        self._update_sample_frequency_bucket_ui()

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
