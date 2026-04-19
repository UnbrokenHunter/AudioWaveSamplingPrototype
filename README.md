# Interpolating Undersampled Audio Signals

## About

This program lets you open an audio file, view the waveform, play different reconstruction/interpolation versions of the signal, and save the currently selected version to a new audio file.

## Installation

1. Download the project, or clone it from Git:

```bash
git clone UnbrokenHunter/AudioWaveSamplingPrototype
cd AudioWaveSamplingPrototype
python -m venv .venv
```

2. Activate the virtual environment:

**Windows (PowerShell):**

```bash
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```bash
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

3. Install the requirements:

```bash
pip install -r requirements.txt
```

4. Run the main file inside `src`:

```bash
python src/main.py
```

5. When the program opens, select an audio file.
