"""
LyricFuse - Forced alignment module.

Uses WhisperX's wav2vec2 alignment to match user-provided lyrics to the
exact word-level timestamps in the audio track. No Whisper transcription
step is needed because the text is already known.
"""
from __future__ import annotations

import torch
import whisperx


# Human-readable language -> WhisperX language code
# Sanskrit falls back to Hindi's alignment model (shared Devanagari + many
# overlapping phonemes). Accuracy may be lower for pure chants.
LANGUAGE_MAP = {
    "english": "en",
    "hindi": "hi",
    "punjabi": "pa",
    "sanskrit": "hi",   # Fallback
}


def get_device() -> str:
    """Return 'cuda' if a GPU is available, otherwise 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def align_lyrics(
    audio_path: str,
    lyrics_text: str,
    language: str,
    progress_callback=None,
):
    """
    Forced-align lyrics to audio using WhisperX.

    Args:
        audio_path:   Path to 16 kHz mono WAV file.
        lyrics_text:  Plain-text lyrics, one line per row.
        language:     Key from LANGUAGE_MAP (case-insensitive) or 2-letter code.
        progress_callback: Optional callable(progress_0_to_1, message).

    Returns:
        (word_timings, lyric_lines) tuple where
            word_timings = [{"word": str, "start": float, "end": float}, ...]
            lyric_lines  = non-empty original lines from user input
    """
    if not lyrics_text or not lyrics_text.strip():
        raise ValueError("Lyrics text is empty.")

    device = get_device()
    lang_key = language.lower().strip()
    lang_code = LANGUAGE_MAP.get(lang_key, lang_key[:2])

    if progress_callback:
        progress_callback(0.05, f"Loading alignment model for {language}...")

    # Load wav2vec2 alignment model for this language. This is the only
    # model we need — no Whisper transcription is required when the text
    # is already known (the forced-alignment use case).
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=lang_code, device=device,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load alignment model for '{language}' (code='{lang_code}'). "
            f"Underlying error: {e}"
        ) from e

    if progress_callback:
        progress_callback(0.25, "Loading audio...")

    audio = whisperx.load_audio(audio_path)
    audio_duration = len(audio) / 16000.0  # WhisperX resamples to 16 kHz

    # Keep the user's line structure; drop blank lines.
    lyric_lines = [ln.strip() for ln in lyrics_text.splitlines() if ln.strip()]
    if not lyric_lines:
        raise ValueError("Lyrics contained only blank lines.")

    full_text = " ".join(lyric_lines)

    # One big segment covering the whole audio — WhisperX's align() will
    # distribute word timings inside this range.
    segments = [{
        "start": 0.0,
        "end": audio_duration,
        "text": full_text,
    }]

    if progress_callback:
        progress_callback(0.4, "Aligning lyrics to audio (this may take a minute)...")

    result = whisperx.align(
        segments,
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Flatten word-level timings from all aligned segments.
    word_timings = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            # Some words may lack timings (e.g. pure punctuation) — skip them.
            if "start" in w and "end" in w:
                word_timings.append({
                    "word": w.get("word", "").strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })

    if not word_timings:
        raise RuntimeError(
            "Alignment produced no word timings. The audio may be silent, "
            "the language may be mis-set, or the model may not support this language."
        )

    if progress_callback:
        progress_callback(0.7, f"Alignment complete ({len(word_timings)} words).")

    return word_timings, lyric_lines
