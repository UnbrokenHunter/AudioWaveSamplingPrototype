import tkinter as tk
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from playback import play_audio, stop_audio

class UI:
    def __init__(self, app, parent, ctx):
        self.app = app
        self.parent = parent
        self.ctx = ctx

    def row(self):
        r = tk.Frame(self.parent)
        r.pack(fill=tk.X, padx=6, pady=4)
        return r

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
        log=False,              # NEW: log slider option
        entry_width=10,         # NEW: nicer entry
    ):
        r = self.row()
        if label is None:
            label = name

        lo = float(from_)
        hi = float(to)

        # For log sliders, the range must be > 0
        if log:
            lo = max(lo, 1e-12)
            hi = max(hi, lo * 1.000001)
            value = max(float(value), lo)

        tk.Label(r, text=f"{label}:").pack(side=tk.LEFT)

        # "real" value variable (always linear space)
        var = tk.DoubleVar(value=float(value))
        setattr(self.app, name, var)

        # A tiny sub-frame for slider + entry so the entry never feels cramped
        inner = tk.Frame(r)
        inner.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # --- Slider variable ---
        if log:
            # slider controls log10(value)
            slider_var = tk.DoubleVar(value=float(np.log10(var.get())))
            slider_from = float(np.log10(lo))
            slider_to = float(np.log10(hi))
            slider_res = 0.001  # log slider smoothness
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
            showvalue=False,   # nicer since entry shows the value
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        setattr(self.app, f"{name}_slider", slider)

        # --- Better Entry box (bigger hitbox, always clickable) ---
        entry_frame = tk.Frame(inner, padx=4)
        entry_frame.pack(side=tk.LEFT)

        entry_var = tk.StringVar(value=f"{var.get():g}")
        entry = tk.Entry(
            entry_frame,
            textvariable=entry_var,
            width=entry_width,
            takefocus=True,
            bd=2,              # more clickable border
            relief="groove",
            justify="right",
        )
        entry.pack(side=tk.LEFT)
        setattr(self.app, f"{name}_entry", entry)

        def clamp(v):
            return max(lo, min(hi, v))

        def sync_entry():
            entry_var.set(f"{var.get():g}")

        def set_var_from_entry():
            try:
                v = float(entry_var.get())
            except ValueError:
                sync_entry()
                return
            v = clamp(v)
            var.set(v)

            if log:
                slider_var.set(float(np.log10(v)))

            sync_entry()

        def set_var_from_slider():
            if log:
                v = 10 ** float(slider_var.get())
                v = clamp(v)
                var.set(v)
            # if not log, slider_var is var already
            sync_entry()

        # Slider movement updates var + entry
        slider.configure(command=lambda _v: set_var_from_slider())

        # Entry commit
        entry.bind("<Return>", lambda _e: set_var_from_entry())
        entry.bind("<FocusOut>", lambda _e: set_var_from_entry())

        # Make clicking anywhere near the entry focus it (helps the “hard to click” feeling)
        entry_frame.bind("<Button-1>", lambda _e: entry.focus_set())
        entry.bind("<Button-1>", lambda _e: entry.focus_set())

        # If var changes externally (rare), keep entry in sync
        var.trace_add("write", lambda *_: sync_entry())

        # Initialize consistency
        sync_entry()
        if log:
            slider_var.set(float(np.log10(var.get())))

        return var

def _to_channel_last(samples):
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return samples[:, None]
    if samples.shape[0] in (1, 2) and samples.shape[0] < samples.shape[1]:
        return samples.T
    return samples


def tkinter_figure(self, samples_list, sr, labels=None, title="Waveforms", zoom_seconds=1.0):
    _validate_inputs(samples_list, labels)

    self.samples_list = samples_list
    self.labels = labels if labels is not None else [f"signal {i}" for i in range(len(samples_list))]
    self.sr = sr
    self.total_duration = _compute_total_duration(samples_list, sr)

    plot_top, plot_bottom = _build_plot_frames(self)

    # Waveform plot
    self.fig, self.ax, self._plot_lines = _build_waveform_plot(
        self, plot_top, samples_list, sr, self.labels, title
    )

    # FFT plot
    self.fft_fig, self.fft_ax, self.fft_canvas = _build_fft_plot(self, plot_bottom)

    # Controls
    top_controls = tk.Frame(self)
    top_controls.pack(fill=tk.X, padx=6, pady=6)

    _build_playback_controls(self, top_controls, self.labels)
    _build_window_controls(self, top_controls, zoom_seconds)

    # Hooks area + UI helper
    self.extras = tk.LabelFrame(self, text="Hooks")
    self.extras.pack(fill=tk.X, padx=6, pady=6)

    # Expose helpers to the outside world
    ctx = {
        "sr": self.sr,
        "ax": self.ax,
        "fig": self.fig,
        "canvas": self.canvas,
        "fft_ax": self.fft_ax,
        "fft_fig": self.fft_fig,
        "fft_canvas": self.fft_canvas,
        "update_fft": lambda: _update_fft(self, use_visible_window=True),
        "apply_view": lambda: _apply_view(self),
        "zoom_var": self.zoom_var,
        "pos_var": self.pos_var,
    }
    self.ui = UI(self, self.extras, ctx)

    # Legend toggle wiring (needs update_fft + recompute hooks)
    _wire_legend_toggle(self)

    # Initial draw
    _apply_view(self)
    _update_fft(self, use_visible_window=True)


# -------------------------
# Helpers
# -------------------------

def _validate_inputs(samples_list, labels):
    if not isinstance(samples_list, (list, tuple)) or len(samples_list) == 0:
        raise TypeError("samples_list must be a non-empty list/tuple of arrays")
    if labels is not None and len(labels) != len(samples_list):
        raise ValueError("labels must match samples_list length")


def _compute_total_duration(samples_list, sr):
    durations = []
    for s in samples_list:
        s_arr = np.asarray(s)
        n = s_arr.shape[0] if s_arr.ndim >= 1 else 0
        durations.append(n / sr if sr else 0.0)
    return max(durations) if durations else 0.0


def _build_plot_frames(app):
    plots = tk.Frame(app)
    plots.pack(fill=tk.BOTH, expand=True)

    plot_top = tk.Frame(plots)
    plot_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    plot_bottom = tk.Frame(plots)
    plot_bottom.pack(side=tk.BOTTOM, fill=tk.X)

    return plot_top, plot_bottom


def _build_waveform_plot(app, parent, samples_list, sr, labels, title):
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")

    plot_lines = []
    for samples, label in zip(samples_list, labels):
        y = _to_channel_last(samples)
        n_samples, n_channels = y.shape
        t = np.arange(n_samples) / sr

        for ch in range(n_channels):
            ch_label = f"{label} (ch {ch})" if n_channels > 1 else label
            line, = ax.plot(t, y[:, ch], label=ch_label)
            plot_lines.append(line)

    # Canvas
    app.canvas = FigureCanvasTkAgg(fig, master=parent)
    app.canvas.draw()
    app.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Legend (we’ll wire picker later)
    ax.legend(loc="upper right")

    return fig, ax, plot_lines


def _build_fft_plot(app, parent):
    fft_fig = Figure(figsize=(6, 2), dpi=100)
    fft_ax = fft_fig.add_subplot(111)
    fft_ax.set_xlabel("Frequency (Hz)")
    fft_ax.set_ylabel("Magnitude (dB)")

    # ✅ log frequency axis
    fft_ax.set_xscale("log")

    fft_canvas = FigureCanvasTkAgg(fft_fig, master=parent)
    fft_canvas.draw()
    fft_canvas.get_tk_widget().pack(fill=tk.X, expand=True)

    return fft_fig, fft_ax, fft_canvas


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


def _build_window_controls(app, parent, zoom_seconds):
    sliders = tk.LabelFrame(parent, text="Window")
    sliders.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

    ZOOM_MIN = 0.001
    ZOOM_MAX = max(app.total_duration, ZOOM_MIN)

    app.zoom_var = tk.DoubleVar(value=float(zoom_seconds if zoom_seconds is not None else ZOOM_MAX))
    app.pos_var = tk.DoubleVar(value=0.0)

    # Log zoom slider var
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
    )
    app.zoom_slider.pack(side=tk.LEFT, padx=4, pady=6)

    app.zoom_entry_var = tk.StringVar(value=f"{app.zoom_var.get():g}")
    zoom_entry = tk.Entry(sliders, textvariable=app.zoom_entry_var, width=8)
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
    )
    app.pos_slider.pack(side=tk.LEFT, padx=4, pady=6)


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
    app.canvas.draw_idle()

    _update_fft(app, use_visible_window=True)


def _wire_legend_toggle(app):
    # Make a fresh legend so we can attach pickers
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

        # fade legend entry
        try:
            idx = app._plot_lines.index(orig)
        except ValueError:
            idx = None

        if idx is not None:
            if idx < len(leg.get_lines()):
                leg.get_lines()[idx].set_alpha(1.0 if vis else 0.2)
            if idx < len(leg.get_texts()):
                leg.get_texts()[idx].set_alpha(1.0 if vis else 0.2)

        app.canvas.draw_idle()
        _update_fft(app, use_visible_window=True)

        if vis and hasattr(app, "request_recompute"):
            app.request_recompute()

    app._pick_cid = app.fig.canvas.mpl_connect("pick_event", on_pick)


def _update_fft(app, use_visible_window=True):
    sr = float(app.sr)
    if sr <= 0:
        return

    app.fft_ax.cla()
    app.fft_ax.set_xlabel("Frequency (Hz)")
    app.fft_ax.set_ylabel("Magnitude (dB)")
    app.fft_ax.set_xscale("log")  # ✅ keep log scale after cla()

    # choose time range
    if use_visible_window:
        x0, x1 = app.ax.get_xlim()
        i0 = max(0, int(round(x0 * sr)))
        i1 = max(i0 + 2, int(round(x1 * sr)))
    else:
        i0, i1 = 0, None

    any_plotted = False

    for line in app._plot_lines:
        if not line.get_visible():
            continue

        y = np.asarray(line.get_ydata(), dtype=np.float64)
        if i1 is not None:
            y = y[i0:i1]

        if y.size < 16:
            continue

        # window
        w = np.hanning(y.size)
        yw = y * w

        Y = np.fft.rfft(yw)
        f = np.fft.rfftfreq(yw.size, d=1.0 / sr)

        mag_db = 20.0 * np.log10(np.abs(Y) + 1e-12)

        # skip DC for log-axis
        f = f[1:]
        mag_db = mag_db[1:]
        if f.size == 0:
            continue

        app.fft_ax.plot(f, mag_db, label=line.get_label())
        any_plotted = True

    # log axis can't include 0
    fmin = 1.0
    app.fft_ax.set_xlim(fmin, sr / 2.0)

    if any_plotted:
        app.fft_ax.legend(loc="upper right", fontsize=8)

    app.fft_fig.tight_layout()
    app.fft_canvas.draw_idle()

def _on_play(app):
    label = app.selected_label.get()
    try:
        idx = app.labels.index(label)
    except ValueError:
        idx = 0

    y = app._plot_lines[idx].get_ydata()
    play_audio(y, app.sr, blocking=False)

