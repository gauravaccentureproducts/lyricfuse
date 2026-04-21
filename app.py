"""
LyricFuse - Streamlit UI.

Workflow:
  1. Upload MP4 video + provide lyrics (paste or .txt upload).
  2. Pick language (English, Hindi, Sanskrit, Punjabi).
  3. Click "Analyze & Preview" -> forced alignment runs, timings shown.
  4. Review preview -> click "Generate Final Video" to burn subtitles.
  5. Download the output MP4 (same resolution, original audio preserved).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from aligner import align_lyrics
from subtitle_generator import group_words_to_lines, write_srt
from video_processor import burn_subtitles, check_ffmpeg, extract_audio


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="LyricFuse", page_icon="🎵", layout="centered")

st.title("🎵 LyricFuse")
st.caption("Fuse lyrics onto a video — exactly on time, line by line.")

# FFmpeg must be on PATH.
if not check_ffmpeg():
    st.error(
        "❌ **FFmpeg not found.** LyricFuse needs FFmpeg for video work.\n\n"
        "Install from https://ffmpeg.org/download.html, then make sure "
        "`ffmpeg` is on your system PATH and restart this app."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Session-state init — this app has 3 stages: input -> preview -> done.
# ---------------------------------------------------------------------------
def _init_state():
    defaults = {
        "stage": "input",         # input | preview | done
        "work_dir": None,         # persistent temp dir across reruns
        "video_path": None,
        "video_name": None,
        "srt_path": None,
        "output_path": None,
        "lines": None,            # [(start, end, text), ...]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _reset():
    """Blow away state + temp dir and start over."""
    wd = st.session_state.get("work_dir")
    if wd and os.path.isdir(wd):
        import shutil
        shutil.rmtree(wd, ignore_errors=True)
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    _init_state()


# ---------------------------------------------------------------------------
# STAGE 1: Input
# ---------------------------------------------------------------------------
if st.session_state.stage == "input":
    st.subheader("1. Upload video")
    video_file = st.file_uploader(
        "Drag & drop your MP4 here, or click Browse",
        type=["mp4"],
    )

    st.subheader("2. Provide lyrics")
    mode = st.radio(
        "Lyric source",
        ["Paste text", "Upload .txt file"],
        horizontal=True,
    )
    lyrics_text = ""
    if mode == "Paste text":
        lyrics_text = st.text_area(
            "Lyrics (one line per row)",
            height=220,
            placeholder="First line of the song\nSecond line of the song\n...",
        )
    else:
        txt_file = st.file_uploader("Upload .txt", type=["txt"])
        if txt_file is not None:
            lyrics_text = txt_file.read().decode("utf-8", errors="replace")
            st.text_area("Preview of uploaded lyrics", value=lyrics_text,
                         height=180, disabled=True)

    st.subheader("3. Language")
    language = st.selectbox(
        "Language of the lyrics",
        ["English", "Hindi", "Punjabi", "Sanskrit"],
        help="Sanskrit uses the Hindi alignment model as a fallback — "
             "accuracy may be lower for pure chants.",
    )

    can_go = bool(video_file) and bool(lyrics_text.strip())
    if st.button("🎯 Analyze & Preview", type="primary", disabled=not can_go):
        # Persist uploads to a stable temp dir for the rest of the session.
        work_dir = tempfile.mkdtemp(prefix="lyricfuse_")
        video_path = os.path.join(work_dir, "input.mp4")
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        audio_path = os.path.join(work_dir, "audio.wav")
        srt_path = os.path.join(work_dir, "subs.srt")

        progress = st.progress(0, "Starting...")
        status = st.empty()

        try:
            status.info("📤 Extracting audio from video...")
            progress.progress(5)
            extract_audio(video_path, audio_path)

            def on_prog(p: float, msg: str):
                # Align stage occupies 10% → 85% of the overall bar.
                pct = 10 + int(p * 75)
                progress.progress(min(pct, 90))
                status.info(f"🎯 {msg}")

            word_timings, lyric_lines = align_lyrics(
                audio_path, lyrics_text, language, on_prog,
            )

            status.info("📝 Building subtitle file...")
            progress.progress(95)
            lines = group_words_to_lines(word_timings, lyric_lines)
            if not lines:
                st.error("❌ Could not group words into lines. Check that your "
                         "lyrics match the audio.")
                st.stop()
            write_srt(lines, srt_path)

            progress.progress(100)
            status.success("✅ Alignment done — review the preview below.")

            # Stash for stage 2.
            st.session_state.update({
                "stage": "preview",
                "work_dir": work_dir,
                "video_path": video_path,
                "video_name": video_file.name,
                "srt_path": srt_path,
                "lines": lines,
            })
            st.rerun()

        except Exception as e:
            st.error(f"❌ Failed: {e}")
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# STAGE 2: Preview
# ---------------------------------------------------------------------------
elif st.session_state.stage == "preview":
    st.subheader("🔍 Preview — subtitle timings")
    lines = st.session_state.lines
    st.caption(f"{len(lines)} lines aligned. Review timings, then generate.")

    preview_rows = [
        {"#": i + 1,
         "Start": f"{s:7.2f}s",
         "End":   f"{e:7.2f}s",
         "Line":  t}
        for i, (s, e, t) in enumerate(lines)
    ]
    st.dataframe(preview_rows, hide_index=True, use_container_width=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🎬 Generate Final Video", type="primary"):
            out_path = os.path.join(
                st.session_state.work_dir, "output.mp4",
            )
            progress = st.progress(0, "Burning subtitles...")
            status = st.empty()
            try:
                status.info("🎬 Burning subtitles onto video (preserving audio)...")
                progress.progress(20)
                burn_subtitles(
                    st.session_state.video_path,
                    st.session_state.srt_path,
                    out_path,
                )
                progress.progress(100)
                status.success("✅ Done!")
                st.session_state.output_path = out_path
                st.session_state.stage = "done"
                st.rerun()
            except Exception as e:
                st.error(f"❌ Burn failed: {e}")

    with col2:
        if st.button("🔄 Start Over"):
            _reset()
            st.rerun()


# ---------------------------------------------------------------------------
# STAGE 3: Done
# ---------------------------------------------------------------------------
elif st.session_state.stage == "done":
    st.success("🎉 Your lyric video is ready.")

    out_path = st.session_state.output_path
    with open(out_path, "rb") as f:
        video_bytes = f.read()

    st.video(video_bytes)

    orig_name = st.session_state.video_name or "output.mp4"
    stem = Path(orig_name).stem

    # Read the SRT we generated earlier so the user can verify timings
    # independently of the burned video.
    srt_path = st.session_state.srt_path
    with open(srt_path, "rb") as f:
        srt_bytes = f.read()

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download lyric video",
            data=video_bytes,
            file_name=f"{stem}_lyrics.mp4",
            mime="video/mp4",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            "📄 Download SRT (for verification)",
            data=srt_bytes,
            file_name=f"{stem}_lyrics.srt",
            mime="application/x-subrip",
            use_container_width=True,
            help="Open in any text editor or subtitle tool to inspect/edit timings.",
        )

    if st.button("🔄 Make another"):
        _reset()
        st.rerun()
