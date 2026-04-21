"""
LyricFuse - Subtitle generation module.

Takes word-level timestamps from alignment plus the user's original
line breaks, groups the words back into those lines, and writes a
standard SRT file. Line-by-line display (MVP): each line appears at
its first word's start time and disappears at its last word's end time.
"""
from __future__ import annotations

from typing import List, Tuple


def group_words_to_lines(
    word_timings: List[dict],
    lyric_lines: List[str],
) -> List[Tuple[float, float, str]]:
    """
    Map the flat word-timing list back onto the user's original lines.

    Args:
        word_timings: [{"word", "start", "end"}, ...] from aligner.
        lyric_lines:  Original non-blank lines from the user.

    Returns:
        [(start_sec, end_sec, line_text), ...] - one tuple per line that
        was successfully timed.
    """
    result: List[Tuple[float, float, str]] = []
    word_idx = 0
    total_words = len(word_timings)

    for line in lyric_lines:
        words_in_line = line.split()
        n = len(words_in_line)
        if n == 0:
            continue

        if word_idx >= total_words:
            # Ran out of aligned words before finishing the lyrics.
            # This can happen if alignment dropped some words.
            break

        # Slice the next n timings for this line.
        chunk = word_timings[word_idx:word_idx + n]
        if not chunk:
            break

        start = chunk[0]["start"]
        # Use last word's end, or last available timing if chunk is short.
        end = chunk[-1]["end"]

        # Guard against zero-duration lines.
        if end <= start:
            end = start + 0.5

        result.append((start, end, line))
        word_idx += n

    return result


def format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(
    lines: List[Tuple[float, float, str]],
    output_path: str,
) -> None:
    """
    Write a UTF-8 SRT file.

    Args:
        lines:       Output of group_words_to_lines().
        output_path: Destination .srt path.
    """
    if not lines:
        raise ValueError("No subtitle lines to write.")

    with open(output_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(lines, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
            f.write(f"{text}\n\n")
