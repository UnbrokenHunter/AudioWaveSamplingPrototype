# visualize.py
import tkinter as tk
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.ticker as mticker

# 3D axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.cm as cm

from debug.timer import time_method
from playback import play_audio, stop_audio


# ============================================================
# UI helper
# ============================================================

class UI:
    def __init__(self, app, parent, ctx):
        self.app = app
        self.parent = parent
        self.ctx = ctx

    def row(self):
        r = tk.Frame(self.parent)
        r.pack(fill=tk.X, padx=6, pady=4)
        return r

    @time_method(0.05)
    def bind_slider(
        self,
        name,
        from_,
        to,
        value,
        *,
        step=0.1,
        label=None,
        length=300,
        log=False,
        entry_width=10,
    ):
        """
        Slider + entry bound to the same tk.DoubleVar.

        - log=False: slider is linear in value
        - log=True:  slider is linear in log10(value), but returned var is still linear value.
        """
        r = self.row()
        if label is None:
            label = name

        lo = float(from_)
        hi = float(to)

        # log sliders require positive range
        if log:
            lo = max(lo, 1e-12)
            hi = max(hi, lo * 1.000001)
            value = max(float(value), lo)

        tk.Label(r, text=f"{label}:").pack(side=tk.LEFT)

        # real value var (linear space)
        var = tk.DoubleVar(value=float(value))
        setattr(self.app, name, var)

        inner = tk.Frame(r)
        inner.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # slider var (maybe log)
        if log:
            slider_var = tk.DoubleVar(value=float(np.log10(var.get())))
            slider_from = float(np.log10(lo))
            slider_to = float(np.log10(hi))
            slider_res = 0.001
        else:
            slider_var = var
            slider_from = lo
            slider_to = hi
            slider_res = float(step)

        slider = tk.Scale(
            inner,
            from_=slider_from,
            to=slider_to,
            resolution=slider_res,
            orient="horizontal",
            variable=slider_var,
            length=length,
            showvalue=False,
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        setattr(self.app, f"{name}_slider", slider)

        entry_frame = tk.Frame(inner, padx=6)
        entry_frame.pack(side=tk.LEFT)

        entry_var = tk.StringVar(value=f"{var.get():g}")
        entry = tk.Entry(
            entry_frame,
            textvariable=entry_var,
            width=entry_width,
            takefocus=True,
            bd=2,
            relief="solid",
            justify="right",
        )
        entry.pack(side=tk.LEFT)
        setattr(self.app, f"{name}_entry", entry)

        def clamp(v):
            return max(lo, min(hi, float(v)))

        editing = {"active": False}

        def sync_entry():
            if not editing["active"]:
                entry_var.set(f"{var.get():g}")

        def begin_edit(_event=None):
            editing["active"] = True
            entry.after_idle(entry.focus_force)
            return None

        def set_from_slider():
            if log:
                v = 10 ** float(slider_var.get())
                v = clamp(v)
                var.set(v)
            sync_entry()

        def commit_entry(_event=None):
            editing["active"] = False
            try:
                v = float(entry_var.get())
            except ValueError:
                sync_entry()
                return None

            v = clamp(v)
            var.set(v)
            if log:
                slider_var.set(float(np.log10(v)))
            entry_var.set(f"{var.get():g}")
            return None

        def cancel_entry(_event=None):
            editing["active"] = False
            sync_entry()
            return "break"

        slider.configure(command=lambda _v: set_from_slider())
        entry.bind("<Button-1>", begin_edit, add="+")
        entry.bind("<FocusIn>", begin_edit, add="+")
        entry.bind("<KeyPress>", begin_edit, add="+")
        entry.bind("<Return>", commit_entry)
        entry.bind("<KP_Enter>", commit_entry)
        entry.bind("<FocusOut>", commit_entry)
        entry.bind("<Escape>", cancel_entry)

        var.trace_add("write", lambda *_: sync_entry())

        sync_entry()
        if log:
            slider_var.set(float(np.log10(var.get())))

        return var


# ============================================================
# Array helpers
# ============================================================

def _to_channel_last(samples):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return samples[:, None]
    if samples.shape[0] in (1, 2) and samples.shape[0] < samples.shape[1]:
        return samples.T
    return samples


def _compute_total_duration(samples_list, sr):
    durations = []
    for s in samples_list:
        s_arr = np.asarray(s)
        n = s_arr.shape[0] if s_arr.ndim >= 1 else 0
        durations.append(n / sr if sr else 0.0)
    return max(durations) if durations else 0.0


def _db(x):
    x = np.asarray(x)
    return 20.0 * np.log10(np.maximum(x, 1e-12))


# ============================================================
# FFT / STFT helpers
# ============================================================

@time_method(0.05)
def stft_mag_db(y, sr, n_fft=1024, hop=256):
    """
    Returns:
      f (Hz) shape (n_freq,)
      t (sec) shape (n_frames,)
      S_db shape (n_freq, n_frames)
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    if y.size < n_fft:
        y = np.pad(y, (0, n_fft - y.size))

    w = np.hanning(n_fft)

    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    if frames.ndim != 2 or frames.shape[0] == 0:
        frames = y[:n_fft][None, :]

    X = np.fft.rfft(frames * w[None, :], axis=1)  # (n_frames, n_freq)
    mag = np.abs(X)
    S_db = _db(mag).T  # (n_freq, n_frames)

    f = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    t = (np.arange(S_db.shape[1]) * hop) / float(sr)
    return f, t, S_db


def _visible_sample_bounds(app):
    if not bool(app.spec_follow_view.get()):
        return None

    sr = float(app.sr)
    x0, x1 = app.ax.get_xlim()
    i0 = max(0, int(round(x0 * sr)))
    i1 = max(i0 + 2, int(round(x1 * sr)))
    return i0, i1


def _slice_visible_window(app, y):
    bounds = _visible_sample_bounds(app)
    if bounds is None:
        return y
    i0, i1 = bounds
    return y[i0:i1]


def _visible_window_cache_key(app):
    bounds = _visible_sample_bounds(app)
    if bounds is None:
        bounds_key = None
    else:
        bounds_key = tuple(bounds)

    return (
        bounds_key,
        tuple(
            (label, bool(app.lines_by_label[label].get_visible()))
            for label in app.labels
        ),
        tuple(
            id(app.signal_data[label])
            for label in app.labels
        ),
    )


def _get_visible_signal_windows(app):
    sr = float(app.sr)
    x0, x1 = app.ax.get_xlim()
    i0 = max(0, int(round(x0 * sr)))
    i1 = max(i0 + 2, int(round(x1 * sr)))

    out = []
    for line in app._plot_lines:
        if not line.get_visible():
            continue

        label = line.get_label()
        y = np.asarray(app.signal_data[label], dtype=np.float64).reshape(-1)
        out.append((line, y[i0:i1]))

    return out

@time_method(0.05)
def request_bottom_update(app, delay_ms=80):
    if app.bottom_mode.get() == "OFF":
        return
    job = getattr(app, "_bottom_job", None)
    if job is not None:
        app.after_cancel(job)
    app._bottom_job = app.after(delay_ms, lambda: _update_bottom_plot(app))


# ============================================================
# Tkinter figure builder
# ============================================================
@time_method(0.05)
def tkinter_figure(self, samples_list, sr, labels=None, title="Waveforms", zoom_seconds=1.0):
    if isinstance(samples_list, dict):
        if labels is None:
            labels = list(samples_list.keys())
        signal_data = {label: np.asarray(samples_list[label]) for label in labels}
    else:
        if not isinstance(samples_list, (list, tuple)) or len(samples_list) == 0:
            raise TypeError("samples_list must be a non-empty list/tuple of arrays")
        if labels is None:
            labels = [f"signal {i}" for i in range(len(samples_list))]
        if len(labels) != len(samples_list):
            raise ValueError("labels must match samples_list length")
        signal_data = {label: np.asarray(samples) for label, samples in zip(labels, samples_list)}

    self.signal_data = signal_data
    self.labels = list(labels)
    self.samples_list = [self.signal_data[label] for label in self.labels]
    self.sr = sr
    self.total_duration = _compute_total_duration(self.samples_list, sr)
    self._visible_window_cache = None

    # --- layout frames (plots stacked: waveform on top, bottom viz below) ---
    plots = tk.Frame(self)
    plots.pack(fill=tk.BOTH, expand=True)

    plot_top = tk.Frame(plots)
    plot_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    plot_bottom = tk.Frame(plots)
    plot_bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    # plot_bottom.configure(height=170)     # make it a wide strip
    # plot_bottom.pack_propagate(False)

    # --- top: waveform plot ---
    self.fig = Figure(figsize=(6, 2), dpi=100, constrained_layout=True)
    self.ax = self.fig.add_subplot(111)
    self.ax.set_title(title)
    self.ax.set_xlabel("Time (seconds)")
    self.ax.set_ylabel("Amplitude")

    self._plot_lines = []
    for label in self.labels:
        y = _to_channel_last(self.signal_data[label])
        n_samples, n_channels = y.shape
        t = np.arange(n_samples) / float(sr)

        for ch in range(n_channels):
            ch_label = f"{label} (ch {ch})" if n_channels > 1 else label
            line, = self.ax.plot(t, y[:, ch], label=ch_label)
            self._plot_lines.append(line)

    self.lines_by_label = {line.get_label(): line for line in self._plot_lines}

    # Legend now; clickable in _wire_legend_toggle()
    self.ax.legend(loc="upper right")

    self.canvas = FigureCanvasTkAgg(self.fig, master=plot_top)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- bottom container: we'll toggle between FFT (2D) and STFT (3D) ---
    self.bottom_container = tk.Frame(plot_bottom)
    self.bottom_container.pack(fill=tk.BOTH, expand=True)
    self._plot_bottom_frame = plot_bottom

    # 2D FFT
    self.fft_fig = Figure(figsize=(16, 1.6), dpi=100, constrained_layout=True)
    self.fft_ax = self.fft_fig.add_subplot(111)
    self.fft_ax.set_xlabel("Frequency (Hz)")
    self.fft_ax.set_ylabel("Magnitude (dB)")
    self.fft_canvas = FigureCanvasTkAgg(self.fft_fig, master=self.bottom_container)
    self.fft_canvas.draw()
    self.fft_widget = self.fft_canvas.get_tk_widget()
    self.bottom_toolbar = NavigationToolbar2Tk(self.fft_canvas, plot_bottom)
    self.bottom_toolbar.update()
    self.bottom_toolbar.pack_forget()                  # take control
    self.bottom_toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    # 3D STFT
    self.spec_fig = Figure(figsize=(16, 1.6), dpi=100, constrained_layout=False)
    self.spec_ax = self.spec_fig.add_subplot(111, projection="3d")
    self.spec_ax.set_xlabel("Time (s)")
    self.spec_ax.set_ylabel("Signal")
    self.spec_ax.set_zlabel("Freq (Hz)")
    self.spec_canvas = FigureCanvasTkAgg(self.spec_fig, master=self.bottom_container)
    self.spec_canvas.draw()
    self.spec_widget = self.spec_canvas.get_tk_widget()

    # Only show one at a time
    self.bottom_mode = tk.StringVar(value="FFT")  # "FFT" or "STFT"
    self._show_bottom_mode("FFT")

    # --- controls row ---
    top_controls = tk.Frame(self)
    top_controls.pack(fill=tk.X, padx=6, pady=6)

    _build_playback_controls(self, top_controls, labels)
    _build_window_controls(self, top_controls, zoom_seconds)
    _build_bottom_controls(self, top_controls)

    # --- hooks area ---
    self.extras = tk.LabelFrame(self, text="Hooks")
    self.extras.pack(fill=tk.X, padx=6, pady=6)

    # Bottom plot knobs
    self.spec_follow_view = tk.BooleanVar(value=True)

    self.stft_nfft = tk.DoubleVar(value=1024.0)
    self.stft_hop = tk.DoubleVar(value=256.0)
    self.stft_log_freq = tk.BooleanVar(value=False)

    self.fft_log_x = tk.BooleanVar(value=True)   # log frequency axis for FFT
    self.fft_db = tk.BooleanVar(value=True)      # magnitude in dB

    # Simple controls in Hooks
    row = tk.Frame(self.extras)
    row.pack(fill=tk.X, padx=6, pady=4)
    tk.Checkbutton(row, text="Bottom follows visible window", variable=self.spec_follow_view,
                   command=lambda: request_bottom_update(self)).pack(side=tk.LEFT)
    tk.Checkbutton(row, text="STFT log-frequency axis", variable=self.stft_log_freq,
                   command=lambda: request_bottom_update(self)).pack(side=tk.LEFT, padx=(10, 0))
    tk.Checkbutton(row, text="FFT log-frequency axis", variable=self.fft_log_x,
                   command=lambda: request_bottom_update(self)).pack(side=tk.LEFT, padx=(10, 0))

    # Expose context
    ctx = {
        "sr": self.sr,
        "ax": self.ax,
        "fig": self.fig,
        "canvas": self.canvas,
        "fft_ax": self.fft_ax,
        "fft_fig": self.fft_fig,
        "fft_canvas": self.fft_canvas,
        "spec_ax": self.spec_ax,
        "spec_fig": self.spec_fig,
        "spec_canvas": self.spec_canvas,
        "apply_view": lambda: _apply_view(self),
        "update_bottom": lambda: request_bottom_update(self),
        "zoom_var": self.zoom_var,
        "pos_var": self.pos_var,
    }
    self.ui = UI(self, self.extras, ctx)

    _wire_legend_toggle(self)

    # initial view + bottom plot
    _apply_view(self)
    request_bottom_update(self)

# ============================================================
# Controls builders
# ============================================================

def _build_playback_controls(app, parent, labels):
    controls = tk.LabelFrame(parent, text="Playback")
    controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

    app.selected_label = tk.StringVar(value=labels[0])
    tk.Label(controls, text="Play:").pack(side=tk.LEFT, padx=(8, 4), pady=6)
    tk.OptionMenu(controls, app.selected_label, *labels).pack(side=tk.LEFT, padx=4, pady=6)

    app.selected_channel = tk.IntVar(value=0)
    tk.Label(controls, text="Channel:").pack(side=tk.LEFT, padx=(12, 4), pady=6)
    tk.Spinbox(controls, from_=0, to=16, width=3, textvariable=app.selected_channel).pack(
        side=tk.LEFT, padx=4, pady=6
    )

    tk.Button(controls, text="Play", command=lambda: _on_play(app)).pack(side=tk.LEFT, padx=8, pady=6)
    tk.Button(controls, text="Stop", command=stop_audio).pack(side=tk.LEFT, padx=4, pady=6)
   
def _hide_bottom_area(app):
    # hide plot area + toolbar
    if hasattr(app, "bottom_toolbar") and app.bottom_toolbar is not None:
        app.bottom_toolbar.pack_forget()
    app._plot_bottom_frame.pack_forget()

def _show_bottom_area(app):
    # show plot area + toolbar
    app._plot_bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    if hasattr(app, "bottom_toolbar") and app.bottom_toolbar is not None:
        # toolbar lives in plot_bottom, so repack it
        app.bottom_toolbar.pack(side=tk.BOTTOM, fill=tk.X)


@time_method(0.05)
def _build_window_controls(app, parent, zoom_seconds):
    sliders = tk.LabelFrame(parent, text="Window")
    sliders.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

    ZOOM_MIN = 0.001
    ZOOM_MAX = max(app.total_duration, ZOOM_MIN)

    app.zoom_var = tk.DoubleVar(value=float(zoom_seconds if zoom_seconds is not None else ZOOM_MAX))
    app.pos_var = tk.DoubleVar(value=0.0)

    app.zoom_var.set(max(ZOOM_MIN, min(float(app.zoom_var.get()), ZOOM_MAX)))
    app.zoom_log_var = tk.DoubleVar(value=float(np.log10(app.zoom_var.get())))

    tk.Label(sliders, text="Zoom (s):").pack(side=tk.LEFT, padx=(8, 4))

    app.zoom_slider = tk.Scale(
        sliders,
        from_=float(np.log10(ZOOM_MIN)),
        to=float(np.log10(ZOOM_MAX)),
        resolution=0.001,
        orient="horizontal",
        variable=app.zoom_log_var,
        command=lambda _v: _on_zoom_slider(app, ZOOM_MIN, ZOOM_MAX),
        length=150,
        showvalue=False,
    )
    app.zoom_slider.pack(side=tk.LEFT, padx=4, pady=6)

    app.zoom_entry_var = tk.StringVar(value=f"{app.zoom_var.get():g}")
    zoom_entry = tk.Entry(sliders, textvariable=app.zoom_entry_var, width=8, justify="right", bd=2, relief="solid")
    zoom_entry.pack(side=tk.LEFT, padx=(6, 0))
    zoom_entry.bind("<Return>", lambda _e: _commit_zoom_entry(app, ZOOM_MIN, ZOOM_MAX))
    zoom_entry.bind("<FocusOut>", lambda _e: _commit_zoom_entry(app, ZOOM_MIN, ZOOM_MAX))

    tk.Label(sliders, text="Pos (s):").pack(side=tk.LEFT, padx=(12, 4))
    app.pos_slider = tk.Scale(
        sliders,
        from_=0.0,
        to=max(app.total_duration, 0.0),
        resolution=0.001,
        orient="horizontal",
        variable=app.pos_var,
        command=lambda _v: _apply_view(app),
        length=150,
        showvalue=False,
    )
    app.pos_slider.pack(side=tk.LEFT, padx=4, pady=6)


def _build_bottom_controls(app, parent):
    bottom = tk.LabelFrame(parent, text="Bottom Plot")
    bottom.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(6, 0))

    tk.Radiobutton(bottom, text="Off", value="OFF", variable=app.bottom_mode,
                   command=lambda: _on_bottom_mode(app)).pack(side=tk.LEFT, padx=8, pady=6)

    tk.Radiobutton(bottom, text="FFT (2D)", value="FFT", variable=app.bottom_mode,
                   command=lambda: _on_bottom_mode(app)).pack(side=tk.LEFT, padx=8, pady=6)

    tk.Radiobutton(bottom, text="STFT (3D)", value="STFT", variable=app.bottom_mode,
                   command=lambda: _on_bottom_mode(app)).pack(side=tk.LEFT, padx=8, pady=6)


# ============================================================
# View + legend + bottom plot update
# ============================================================

def _on_zoom_slider(app, ZOOM_MIN, ZOOM_MAX):
    z = 10 ** float(app.zoom_log_var.get())
    z = max(ZOOM_MIN, min(z, ZOOM_MAX))
    app.zoom_var.set(z)
    app.zoom_entry_var.set(f"{z:g}")
    _apply_view(app)


def _commit_zoom_entry(app, ZOOM_MIN, ZOOM_MAX):
    try:
        z = float(app.zoom_entry_var.get())
    except ValueError:
        app.zoom_entry_var.set(f"{app.zoom_var.get():g}")
        return
    z = max(ZOOM_MIN, min(z, ZOOM_MAX))
    app.zoom_var.set(z)
    app.zoom_log_var.set(float(np.log10(z)))
    app.zoom_entry_var.set(f"{z:g}")
    _apply_view(app)


def _apply_view(app):
    ZOOM_MIN = 0.001
    ZOOM_MAX = max(app.total_duration, ZOOM_MIN)

    z = float(app.zoom_var.get())
    z = max(ZOOM_MIN, min(z, ZOOM_MAX))
    app.zoom_var.set(z)

    max_pos = max(0.0, app.total_duration - z)
    p = float(app.pos_var.get())
    p = max(0.0, min(p, max_pos))
    app.pos_var.set(p)

    app.ax.set_xlim(p, p + z)
    app._visible_window_cache = None
    app.canvas.draw_idle()

    # bottom plot follows view (debounced)
    request_bottom_update(app)


def _wire_legend_toggle(app):
    leg = app.ax.legend(loc="upper right")
    app._legend_map = {}

    for leg_line, orig_line in zip(leg.get_lines(), app._plot_lines):
        leg_line.set_picker(True)
        leg_line.set_pickradius(6)
        app._legend_map[leg_line] = orig_line

    for leg_text, orig_line in zip(leg.get_texts(), app._plot_lines):
        leg_text.set_picker(True)
        app._legend_map[leg_text] = orig_line

    def on_pick(event):
        artist = event.artist
        orig = app._legend_map.get(artist)
        if orig is None:
            return

        vis = not orig.get_visible()
        orig.set_visible(vis)

        try:
            idx = app._plot_lines.index(orig)
        except ValueError:
            idx = None

        if idx is not None:
            if idx < len(leg.get_lines()):
                leg.get_lines()[idx].set_alpha(1.0 if vis else 0.2)
            if idx < len(leg.get_texts()):
                leg.get_texts()[idx].set_alpha(1.0 if vis else 0.2)

        app._visible_window_cache = None
        app.canvas.draw_idle()

        # update bottom
        request_bottom_update(app)

        # If enabled, ask the app to recompute (your main can hook this)
        if vis and hasattr(app, "request_recompute"):
            app.request_recompute()

    app._pick_cid = app.fig.canvas.mpl_connect("pick_event", on_pick)


def _on_bottom_mode(app):
    mode = app.bottom_mode.get()
    app._show_bottom_mode(mode)

    # Cancel any scheduled bottom update if turning OFF
    if mode == "OFF":
        job = getattr(app, "_bottom_job", None)
        if job is not None:
            app.after_cancel(job)
            app._bottom_job = None
        return

    request_bottom_update(app)


def _show_one(widget_to_show, widget_to_hide):
    widget_to_hide.pack_forget()
    widget_to_show.pack(fill=tk.BOTH, expand=True)


def _update_bottom_plot(app):
    app._bottom_job = None

    mode = app.bottom_mode.get()
    if mode == "OFF":
        return
    if mode == "FFT":
        _update_fft_2d(app)
    else:
        _update_stft_3d(app)


@time_method(0.05)
def _update_fft_2d(app):
    sr = float(app.sr)
    if sr <= 0:
        return

    app.fft_ax.cla()
    app.fft_ax.set_xlabel("Frequency (Hz)")
    app.fft_ax.set_ylabel("Magnitude (dB)" if app.fft_db.get() else "Magnitude")

    any_plotted = False

    for line, y in _get_visible_signal_windows(app):

        if y.size < 16:
            continue

        # window
        if not hasattr(app, "_hann_cache"):
            app._hann_cache = {}

        def get_hann(app, n):
            w = app._hann_cache.get(n)
            if w is None:
                w = np.hanning(n)
                app._hann_cache[n] = w
            return w
        
        w = get_hann(app, y.size)
        yw = y * w

        Y = np.fft.rfft(yw)
        f = np.fft.rfftfreq(yw.size, d=1.0 / sr)
        mag = np.abs(Y)

        if app.fft_db.get():
            mag = _db(mag)

        app.fft_ax.plot(
            f,
            mag,
            label=line.get_label(),
            color=line.get_color(),
        )
        any_plotted = True

    if app.fft_log_x.get():
        app.fft_ax.set_xscale("log")

        # limits
        fmin = 20
        fmax = sr / 2.0
        app.fft_ax.set_xlim(fmin, fmax)

        # ---- nice log ticks ----
        # Major ticks at decades (10, 100, 1000, 10000, ...)
        app.fft_ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=12))

        # Minor ticks between decades (2..9)
        app.fft_ax.xaxis.set_minor_locator(
            mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
        )

        # Label majors as plain numbers (no 10^x)
        app.fft_ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        app.fft_ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        # Make sure ScalarFormatter actually uses plain formatting
        app.fft_ax.ticklabel_format(axis="x", style="plain")

        # Optional: hide crazy labels outside range
        app.fft_ax.xaxis.get_major_formatter().set_useOffset(False)

        # ---- grid styling ----
        app.fft_ax.grid(True, which="major", axis="x", linewidth=0.8)
        app.fft_ax.grid(True, which="minor", axis="x", linewidth=0.3, alpha=0.5)
        app.fft_ax.grid(True, which="major", axis="y", linewidth=0.6, alpha=0.4)

    else:
        app.fft_ax.set_xlim(0, sr / 2.0)
        app.fft_ax.grid(True, which="both", axis="both", alpha=0.3)

    if any_plotted:
        app.fft_ax.legend(loc="upper right", fontsize=8)

    # no tight_layout spam; one-time margin is fine
    app.fft_fig.subplots_adjust(left=0.05, right=0.99, bottom=0.22, top=0.95)
    app.fft_canvas.draw_idle()


def _log_freq(f, fmin):
    return np.log10(np.maximum(f, float(fmin)))

@time_method(0.05)
def _update_stft_3d(app):
    sr = float(app.sr)
    if sr <= 0:
        return

    app.spec_ax.cla()
    app.spec_ax.set_xlabel("Time (s)")
    app.spec_ax.set_ylabel("Signal")
    app.spec_ax.set_zlabel("Frequency (Hz)" + (" (log)" if app.stft_log_freq.get() else ""))

    visible_windows = _get_visible_signal_windows(app)
    if not visible_windows:
        app.spec_canvas.draw_idle()
        return

    # requested settings
    n_fft_req = int(max(128, round(float(app.stft_nfft.get()))))
    hop_req = int(max(16, round(float(app.stft_hop.get()))))

    # speed knobs: decimate surface
    max_f_bins = 80
    max_t_bins = 140

    any_drawn = False

    plotted_labels = []

    for si, (label, y) in enumerate(visible_windows):

        if y.size < 32:
            continue

        # make n_fft fit the slice (prevents "nothing drawn")
        n_fft = min(n_fft_req, int(2 ** np.floor(np.log2(max(64, y.size)))))
        n_fft = max(64, n_fft)
        hop = min(hop_req, max(16, n_fft // 4))

        f, tt, S_db = stft_mag_db(y, sr, n_fft=n_fft, hop=hop)
        if f.size < 2 or tt.size < 1:
            continue

        # optional log-frequency axis for 3D (transform data)
        if app.stft_log_freq.get():
            fmin = 20.0
            f_plot = _log_freq(f, fmin)
        else:
            f_plot = f

        # decimate for 3D speed
        f_step = max(1, S_db.shape[0] // max_f_bins)
        t_step = max(1, S_db.shape[1] // max_t_bins)

        f2 = f_plot[::f_step]
        tt2 = tt[::t_step]
        S2 = S_db[::f_step, ::t_step]

        if f2.size < 2 or tt2.size < 2:
            continue

        T, Fm = np.meshgrid(tt2, f2)
        Y = np.full_like(T, float(si))

        vmin = np.percentile(S2, 10)
        vmax = np.percentile(S2, 99)
        norm = (S2 - vmin) / (vmax - vmin + 1e-12)
        norm = np.clip(norm, 0, 1)
        facecolors = cm.viridis(norm)

        app.spec_ax.plot_surface(
            T, Y, Fm,
            rstride=1,
            cstride=1,
            facecolors=facecolors,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        plotted_labels.append(label)
        any_drawn = True

    # stacked labels
    app.spec_ax.set_yticks(range(len(plotted_labels)))
    app.spec_ax.set_yticklabels(plotted_labels, fontsize=8)

    # If log axis, put nice Hz tick labels
    if app.stft_log_freq.get():
        hz_ticks = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], dtype=float)
        hz_ticks = hz_ticks[hz_ticks <= sr / 2.0]
        app.spec_ax.set_zticks(np.log10(hz_ticks))
        app.spec_ax.set_zticklabels([f"{int(hz)}" for hz in hz_ticks])

    # view angle
    app.spec_ax.view_init(elev=25, azim=-20)

    if not any_drawn:
        app.spec_ax.text2D(0.02, 0.85, "No data drawn (window too small?)", transform=app.spec_ax.transAxes)

    # one-time-ish margins (cheap)
    app.spec_fig.subplots_adjust(left=0.02, right=0.99, bottom=0.02, top=0.98)
    app.spec_canvas.draw_idle()


def _show_bottom_mode(self, mode):
    if mode == "OFF":
        _hide_bottom_area(self)
        return

    _show_bottom_area(self)
    if mode == "FFT":
        _show_one(self.fft_widget, self.spec_widget)
    else:
        _show_one(self.spec_widget, self.fft_widget)


# monkey attach helper to instances
def _attach_show_mode():
    def _show(self, mode):
        _show_bottom_mode(self, mode)
    return _show


# attach method name used above
# (keeps your code style: self._show_bottom_mode(...))
setattr(tk.Misc, "_show_bottom_mode", _attach_show_mode())


# ============================================================
# Playback
# ============================================================

def _on_play(app):
    label = app.selected_label.get()
    try:
        idx = app.labels.index(label)
    except ValueError:
        idx = 0

    y = np.asarray(app.signal_data[app.labels[idx]], dtype=np.float32).reshape(-1)
    play_audio(y, app.sr, blocking=False)
