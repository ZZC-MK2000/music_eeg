"""Project-wide PsychoPy configuration helpers."""

from psychopy import prefs


def configure_psychopy():
    """Configure PsychoPy before importing audio/visual modules."""
    # Prefer pygame for stable MP3/WAV playback
    prefs.hardware['audioLib'] = ['pygame', 'sounddevice', 'ptb']
    prefs.hardware['audioLatencyMode'] = 3
