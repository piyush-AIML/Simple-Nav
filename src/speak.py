"""Optional text-to-speech for turn-by-turn directions.

Uses pyttsx3 (offline, no internet/API key needed). Wrapped defensively:
some environments (headless servers, containers without an audio backend)
don't have a working TTS engine, and that should never crash the app —
it should just log a note and move on.
"""

from __future__ import annotations

from src.utils import setup_logger

logger = setup_logger("speak")


def speak(text: str) -> bool:
    """Speak `text` aloud. Returns True if it worked, False if TTS wasn't
    available (missing engine, no audio device, etc.) — callers should treat
    False as "not fatal, just skip it."
    """
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        return True
    except Exception as e:
        logger.warning(
            f"Text-to-speech unavailable ({e}). This needs pyttsx3 plus a working "
            "system audio/TTS backend (e.g. espeak on Linux, or the built-in "
            "engines on Windows/macOS) — directions are still shown as text."
        )
        return False
