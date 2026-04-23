# Interpolating Undersampled Audio Signals

## About

This program lets you open an audio file, view the waveform, play different reconstruction/interpolation versions of the signal, and save the currently selected version to a new audio file.

## Installation

0. Ensure prerequisites are installed:

Ensure Python 3.14 is installed, along with Python3.14-tk for the Tkinter dependancy. Also ensure PortAudio is installed on the system.

1. Download the project, or clone it from Git:

```bash
git clone UnbrokenHunter/AudioWaveSamplingPrototype
cd AudioWaveSamplingPrototype
python -m venv c
```

2. Activate the virtual environment:

**Windows (PowerShell):**

```bash
myenv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```bash
myenv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source myenv/bin/activate
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
