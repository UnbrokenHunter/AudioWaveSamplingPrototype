from bisect import bisect_left, bisect_right
from collections import OrderedDict
import numpy as np


class LRUCache:
    """Tiny LRU cache for numpy-heavy results keyed by normalized params."""
    def __init__(self, maxsize=30):
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

    def keys(self):
        return list(self._data.keys())

    def clear(self):
        self._data.clear()


class ReconstructionCacheManager:
    """Owns per-frequency caches and UI helpers for cached sample-frequency navigation."""

    def __init__(self, y, sr, sample_indices_fn, recon_cache_sizes=None, idx_cache_size=30):
        self.y = np.asarray(y, dtype=np.float64)
        self.sr = sr
        self.sample_indices_fn = sample_indices_fn
        self.idx_cache = LRUCache(maxsize=idx_cache_size)

        default_sizes = {
            "direct reconstruction": 30,
            "linear reconstruction": 30,
            "nearest reconstruction": 30,
            "dac reconstruction": 30,
            "sinc reconstruction": 30,
            "sinc reconstruction (lowpassed)": 30,
            "direct subtract": 30,
        }
        if recon_cache_sizes:
            default_sizes.update(recon_cache_sizes)
        self.recon_cache = {
            label: LRUCache(maxsize=size)
            for label, size in default_sizes.items()
        }

    @staticmethod
    def fs_cache_key(fs):
        return round(float(fs), 3)
    
    def get_sample_indices(self, fs):
        key = self.fs_cache_key(fs)
        idx = self.idx_cache.get(key)
        if idx is None:
            idx = self.sample_indices_fn(len(self.y), self.sr, fs)
            self.idx_cache.put(key, idx)
        return idx

    def get_reconstruction(self, label, fs, compute_fn):
        key = self.fs_cache_key(fs)
        cache = self.recon_cache[label]
        values = cache.get(key)
        if values is None:
            values = np.asarray(compute_fn(), dtype=np.float64)
            cache.put(key, values)
        return values

    def cached_sample_frequency_values(self):
        values = set(float(k) for k in self.idx_cache.keys())
        for cache in self.recon_cache.values():
            values.update(float(k) for k in cache.keys())
        return sorted(values)

    def nearest_cached_sample_frequency(self, fs):
        cached = self.cached_sample_frequency_values()
        if not cached:
            return None

        i = bisect_left(cached, fs)
        if i <= 0:
            return float(cached[0])
        if i >= len(cached):
            return float(cached[-1])

        left = float(cached[i - 1])
        right = float(cached[i])
        return left if abs(fs - left) <= abs(right - fs) else right

    def adjacent_cached_sample_frequency(self, fs, direction):
        cached = self.cached_sample_frequency_values()
        if not cached:
            return None

        if direction < 0:
            i = bisect_left(cached, fs) - 1
            return None if i < 0 else float(cached[i])

        i = bisect_right(cached, fs)
        return None if i >= len(cached) else float(cached[i])

    def update_navigation_ui(self, app, fs):
        nearest = self.nearest_cached_sample_frequency(fs)
        if nearest is None:
            app.sample_frequency_bucket_label_var.set("Nearest cache: none")
        else:
            app.sample_frequency_bucket_label_var.set(f"Nearest cache: {nearest:g} Hz")

        prev_value = self.adjacent_cached_sample_frequency(fs, -1)
        next_value = self.adjacent_cached_sample_frequency(fs, +1)

        if hasattr(app, "sample_frequency_prev_button"):
            app.sample_frequency_prev_button.configure(
                state=("normal" if prev_value is not None else "disabled")
            )
        if hasattr(app, "sample_frequency_next_button"):
            app.sample_frequency_next_button.configure(
                state=("normal" if next_value is not None else "disabled")
            )
