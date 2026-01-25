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


def tkinter_figure(
    self,
    samples_list,
    sr,
    labels=None,
    title="Waveforms",
    zoom_seconds=1.0,
):
    if not isinstance(samples_list, (list, tuple)) or len(samples_list) == 0:
        raise TypeError("samples_list must be a non-empty list/tuple of arrays")

    if labels is None:
        labels = [f"signal {i}" for i in range(len(samples_list))]
    if len(labels) != len(samples_list):
        raise ValueError("labels must match samples_list length")

    # Store these on the app for callbacks
    self.samples_list = samples_list
    self.labels = labels
    self.sr = sr

    # Compute overall max duration (seconds) across all signals
    durations = []
    for s in samples_list:
        s_arr = np.asarray(s)
        n = s_arr.shape[0] if s_arr.ndim >= 1 else 0
        durations.append(n / sr if sr else 0.0)
    self.total_duration = max(durations) if durations else 0.0

    # --- Figure ---
    self.fig = Figure(figsize=(6, 4), dpi=100)
    self.ax = self.fig.add_subplot(111)
    self.ax.set_title(title)
    self.ax.set_xlabel("Time (seconds)")
    self.ax.set_ylabel("Amplitude")

    # Plot ALL waveforms overlaid + keep refs for legend toggling
    self._plot_lines = []  # list of Line2D objects in plot order
    for samples, label in zip(samples_list, labels):
        y = _to_channel_last(samples)
        n_samples, n_channels = y.shape
        t = np.arange(n_samples) / sr

        for ch in range(n_channels):
            ch_label = f"{label} (ch {ch})" if n_channels > 1 else label
            line, = self.ax.plot(t, y[:, ch], label=ch_label)
            self._plot_lines.append(line)

    # Legend + click-to-toggle
    leg = self.ax.legend(loc="upper right")

    # Map legend artists (both line + text) -> original plotted line
    self._legend_map = {}

    # Legend lines
    for leg_line, orig_line in zip(leg.get_lines(), self._plot_lines):
        leg_line.set_picker(True)
        leg_line.set_pickradius(6)
        self._legend_map[leg_line] = orig_line

    # Legend text (also clickable)
    for leg_text, orig_line in zip(leg.get_texts(), self._plot_lines):
        leg_text.set_picker(True)
        self._legend_map[leg_text] = orig_line

    def on_pick(event):
        artist = event.artist
        orig = self._legend_map.get(artist)
        if orig is None:
            return

        vis = not orig.get_visible()
        orig.set_visible(vis)

        # Fade matching legend entries (both line + text)
        # Find which plotted line this corresponds to:
        try:
            idx = self._plot_lines.index(orig)
        except ValueError:
            idx = None

        if idx is not None:
            leg_lines = leg.get_lines()
            leg_texts = leg.get_texts()

            if idx < len(leg_lines):
                leg_lines[idx].set_alpha(1.0 if vis else 0.2)
            if idx < len(leg_texts):
                leg_texts[idx].set_alpha(1.0 if vis else 0.2)

        self.canvas.draw_idle()

        # If a line was just enabled, ask the app to recompute
        if vis and hasattr(self, "request_recompute"):
            self.request_recompute()


    # Connect pick handler
    self._pick_cid = self.fig.canvas.mpl_connect("pick_event", on_pick)

    self.canvas = FigureCanvasTkAgg(self.fig, master=self)
    self.canvas.draw()
    self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    top_controls = tk.Frame(self)
    top_controls.pack(fill=tk.X, padx=6, pady=6)

    # --- Controls row ---
    controls = tk.LabelFrame(top_controls, text="Playback")
    controls.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

    # Dropdown to pick which waveform to play
    self.selected_label = tk.StringVar(value=labels[0])
    tk.Label(controls, text="Play:").pack(side=tk.LEFT, padx=(8, 4), pady=6)
    tk.OptionMenu(controls, self.selected_label, *labels).pack(side=tk.LEFT, padx=4, pady=6)

    # Channel selector
    self.selected_channel = tk.IntVar(value=0)
    tk.Label(controls, text="Channel:").pack(side=tk.LEFT, padx=(12, 4), pady=6)
    tk.Spinbox(
        controls, from_=0, to=16, width=3, textvariable=self.selected_channel
    ).pack(side=tk.LEFT, padx=4, pady=6)

    tk.Button(controls, text="Play", command=lambda: _on_play(self)).pack(side=tk.LEFT, padx=8, pady=6)
    tk.Button(controls, text="Stop", command=stop_audio).pack(side=tk.LEFT, padx=4, pady=6)

    # --- Sliders row (Position + Zoom) ---
    sliders = tk.LabelFrame(top_controls, text="Window")
    sliders.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

    # State vars
    ZOOM_MIN = 0.001  # 1 ms
    ZOOM_MAX = max(self.total_duration, ZOOM_MIN)
    
    self.zoom_var = tk.DoubleVar(
        value=float(zoom_seconds if zoom_seconds is not None else max(self.total_duration, 1e-3))
    )
    self.pos_var = tk.DoubleVar(value=0.0)

    def apply_view(*_):
        # z comes from zoom_var (seconds)
        z = float(self.zoom_var.get())
        z = max(ZOOM_MIN, min(z, ZOOM_MAX))
        self.zoom_var.set(z)

        max_pos = max(0.0, self.total_duration - z)
        p = float(self.pos_var.get())
        p = max(0.0, min(p, max_pos))
        self.pos_var.set(p)

        self.ax.set_xlim(p, p + z)
        self.canvas.draw_idle()

    # --- Log Zoom (slider controls log10(zoom_seconds), entry edits zoom_seconds) ---
    tk.Label(sliders, text="Zoom (s):").pack(side=tk.LEFT, padx=(8, 4))

    # real zoom in seconds (already exists)
    self.zoom_var.set(max(ZOOM_MIN, min(float(self.zoom_var.get()), ZOOM_MAX)))

    # slider variable in log space
    self.zoom_log_var = tk.DoubleVar(value=float(np.log10(self.zoom_var.get())))

    # log slider
    self.zoom_slider = tk.Scale(
        sliders,
        from_=float(np.log10(ZOOM_MIN)),
        to=float(np.log10(ZOOM_MAX)),
        resolution=0.001,  # smaller = smoother
        orient="horizontal",
        variable=self.zoom_log_var,
        command=lambda _v: _on_zoom_slider(),
        length=150,
    )
    self.zoom_slider.pack(side=tk.LEFT, padx=4, pady=6)

    # entry tied to zoom_var (seconds)
    self.zoom_entry_var = tk.StringVar(value=f"{self.zoom_var.get():g}")
    zoom_entry = tk.Entry(sliders, textvariable=self.zoom_entry_var, width=8)
    zoom_entry.pack(side=tk.LEFT, padx=(6, 0))

    def _clamp_zoom(z):
        return max(ZOOM_MIN, min(float(z), ZOOM_MAX))

    def _sync_zoom_entry():
        self.zoom_entry_var.set(f"{self.zoom_var.get():g}")

    def _sync_zoom_slider_from_zoom():
        # set slider based on zoom_var
        z = _clamp_zoom(self.zoom_var.get())
        self.zoom_var.set(z)
        self.zoom_log_var.set(float(np.log10(z)))

    def _on_zoom_slider():
        # slider changed -> update zoom_var (seconds)
        z = 10 ** float(self.zoom_log_var.get())
        self.zoom_var.set(_clamp_zoom(z))
        _sync_zoom_entry()
        apply_view()

    def _commit_zoom_entry(_event=None):
        # entry changed -> update zoom_var + slider
        try:
            z = float(self.zoom_entry_var.get())
        except ValueError:
            _sync_zoom_entry()
            return
        self.zoom_var.set(_clamp_zoom(z))
        _sync_zoom_slider_from_zoom()
        apply_view()

    zoom_entry.bind("<Return>", _commit_zoom_entry)
    zoom_entry.bind("<FocusOut>", _commit_zoom_entry)

    # initialize consistency
    _sync_zoom_slider_from_zoom()
    _sync_zoom_entry()

    # Position slider
    tk.Label(sliders, text="Pos (s):").pack(side=tk.LEFT, padx=(12, 4))
    self.pos_slider = tk.Scale(
        sliders,
        from_=0.0,
        to=max(self.total_duration, 0.0),
        resolution=0.001,
        orient="horizontal",
        variable=self.pos_var,
        command=lambda _v: apply_view(),
        length=150,
    )
    self.pos_slider.pack(side=tk.LEFT, padx=4, pady=6)

    # Initialize view
    apply_view()

    # --- Extras area for inline UI variables ---
    self.extras = tk.LabelFrame(self, text="Hooks")
    self.extras.pack(fill=tk.X, padx=6, pady=6)

    ctx = {
        "sr": self.sr,
        "samples_list": self.samples_list,
        "labels": self.labels,
        "ax": self.ax,
        "fig": self.fig,
        "canvas": self.canvas,
        "apply_view": apply_view,
        "zoom_var": self.zoom_var,
        "pos_var": self.pos_var,
    }

    self.ui = UI(self, self.extras, ctx)


def _on_play(app):
    label = app.selected_label.get()
    try:
        idx = app.labels.index(label)
    except ValueError:
        idx = 0

    y = app._plot_lines[idx].get_ydata()
    play_audio(y, app.sr, blocking=False)
