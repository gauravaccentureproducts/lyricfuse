"""
LyricFuse - Video processing module.

Thin wrapper around FFmpeg for two jobs:
  1. Extract 16 kHz mono WAV audio (the format WhisperX expects).
  2. Burn a generated SRT file onto the input video while leaving the
     original audio stream untouched (-c:a copy).
"""
from __future__ import annotations

import os
import subprocess


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is runnable from PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _run(cmd: list) -> None:
    """Run an FFmpeg command and raise with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed (exit {proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {proc.stderr[-2000:]}"  # Tail of stderr
        )


def extract_audio(video_path: str, audio_path: str) -> None:
    """
    Extract the audio track as 16 kHz mono WAV (WhisperX requirement).

    The original video and its audio stream remain untouched — this is a
    read-only operation on the input.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16 kHz (WhisperX)
        "-ac", "1",               # Mono
        audio_path,
    ]
    _run(cmd)


def _escape_subtitles_path(path: str) -> str:
    """
    FFmpeg's `subtitles` filter on Windows needs forward slashes and the
    drive-letter colon escaped, e.g. `C\\:/tmp/subs.srt`.
    """
    escaped = path.replace("\\", "/")
    # Escape the first ':' (drive letter on Windows).
    if len(escaped) >= 2 and escaped[1] == ":":
        escaped = escaped[0] + "\\:" + escaped[2:]
    return escaped


def burn_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    font_size: int = 24,
) -> None:
    """
    Burn SRT subtitles into the video as a permanent visual layer.

    Audio stream is copied verbatim (-c:a copy) so the original audio
    quality is preserved exactly. Video is re-encoded with x264 at
    CRF 18 (visually near-lossless).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"Subtitle file not found: {srt_path}")

    srt_for_filter = _escape_subtitles_path(srt_path)

    # ASS styling for the burned subtitles:
    #   FontName   - widely available
    #   FontSize   - caller-tunable
    #   PrimaryColour - white text (&HBBGGRR& in ASS hex)
    #   OutlineColour - black outline for readability
    #   BorderStyle=3 - opaque box background
    #   Outline=2     - outline thickness
    #   Alignment=2   - bottom-center
    style = (
        f"FontName=Arial,"
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=1,"
        f"Outline=2,"
        f"Shadow=1,"
        f"Alignment=2,"
        f"MarginV=40"
    )

    vf = f"subtitles='{srt_for_filter}':force_style='{style}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", vf,
        "-c:a", "copy",            # Preserve original audio exactly
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",              # Near-lossless visual quality
        output_path,
    ]
    _run(cmd)
