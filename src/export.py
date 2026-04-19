from pathlib import Path
import re
import numpy as np
import soundfile as sf
from tkinter import filedialog


def _slug(text):
    text = str(text).strip().lower()
    text = text.replace(" reconstruction", "")
    text = text.replace(" ", "-")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    return text or "audio"


def _format_frequency(fs):
    fs = float(fs)
    if fs.is_integer():
        return str(int(fs))
    return f"{fs:g}".replace('.', 'p')


def build_default_save_name(app):
    source_path = Path(getattr(app, 'source_path', 'audio'))
    base_name = source_path.stem or 'audio'

    selected_label = getattr(app, 'selected_label').get()
    interp_name = _slug(selected_label)

    sample_frequency_var = getattr(app, 'sample_frequency', None)
    if sample_frequency_var is None:
        freq_text = str(int(round(float(getattr(app, 'sr', 44100)))))
    else:
        freq_text = _format_frequency(sample_frequency_var.get())

    return f"{base_name}-{interp_name}-{freq_text}.wav"


def save_selected_audio(app):
    label = app.selected_label.get()
    samples = np.asarray(app.signal_data[label], dtype=np.float32).reshape(-1)
    samples = np.clip(samples, -1.0, 1.0)

    default_name = build_default_save_name(app)
    target_path = filedialog.asksaveasfilename(
        title='Save audio file',
        initialfile=default_name,
        defaultextension='.wav',
        filetypes=[
            ('WAV file', '*.wav'),
            ('FLAC file', '*.flac'),
            ('OGG file', '*.ogg'),
            ('All files', '*.*'),
        ],
    )
    if not target_path:
        return

    sf.write(target_path, samples, int(round(float(app.sr))))
