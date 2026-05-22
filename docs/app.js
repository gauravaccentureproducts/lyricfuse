/*
 * LyricFuse — browser-only forced alignment + subtitle burning.
 *
 * Pipeline:
 *   1. User drops MP4 + pastes lyrics + picks language.
 *   2. FFmpeg.wasm extracts 16 kHz mono WAV from the video.
 *   3. Whisper (transformers.js) transcribes the audio with word-level timestamps.
 *   4. Whisper's word timestamps are mapped onto the user's line structure
 *      (positional grouping — user's lyrics text is the authoritative content,
 *      Whisper's timestamps are the authoritative timing).
 *   5. SRT generated and previewed.
 *   6. FFmpeg.wasm burns subtitles onto a copy of the video (audio stream copied
 *      bit-for-bit). Result is downloaded entirely client-side.
 *
 * No data leaves the user's browser at any point.
 */

import { FFmpeg } from "https://unpkg.com/@ffmpeg/ffmpeg@0.12.10/dist/esm/index.js";
import { fetchFile, toBlobURL } from "https://unpkg.com/@ffmpeg/util@0.12.1/dist/esm/index.js";
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

// Force transformers.js to fetch models from the hub (we don't host them locally).
env.allowLocalModels = false;
env.useBrowserCache = true;

// ----- single-threaded FFmpeg core: no SharedArrayBuffer needed -----
// (Slower than the multi-threaded core, but works on GitHub Pages without
// custom COOP/COEP headers.)
const FFMPEG_CORE_BASE = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm";

// The @ffmpeg/ffmpeg main thread instantiates a module Worker (worker.js).
// That Worker can't be loaded cross-origin AND it imports ./const.js and
// ./errors.js by relative path — which a Blob URL can't resolve either.
// Solution: self-host worker.js + its two siblings under docs/ffmpeg/ so
// they all live at the same same-origin directory. Version MUST stay in
// lockstep with the @ffmpeg/ffmpeg ESM import URL (0.12.10).
const LOCAL_FFMPEG_WORKER = "ffmpeg/worker.js";  // resolved against window.location at runtime

// ----- Whisper model choice -----
// 'Xenova/whisper-base' is the multilingual base model (~75 MB).
// 'tiny' is faster but markedly worse for Hindi/Punjabi. 'small' is better
// but ~250 MB and noticeably slower in browser.
const WHISPER_MODEL = "Xenova/whisper-base";

// ===================================================================
// Application state
// ===================================================================
const state = {
  videoFile: null,        // File object selected by user
  videoBytes: null,       // Uint8Array of the video
  lyricsText: "",         // Raw lyrics string
  language: "en",         // BCP-style code chosen in <select>

  ffmpeg: null,           // FFmpeg instance (lazy)
  transcriber: null,      // Whisper pipeline (lazy, per-language)
  transcriberLang: null,  // Track which language the current transcriber was loaded with

  timedLines: null,       // [{start, end, text}, ...]
  srtText: null,          // Generated SRT string
  outputBlob: null,       // Blob of final mp4
};

// ===================================================================
// DOM references
// ===================================================================
const $ = (id) => document.getElementById(id);

const els = {
  // Stage 1
  stageInput: $("stage-input"),
  dropZone: $("drop-zone"),
  videoInput: $("video-input"),
  videoFilename: $("video-filename"),
  lyricsText: $("lyrics-text"),
  lyricsFile: $("lyrics-file"),
  lyricsFilePreview: $("lyrics-file-preview"),
  languageSelect: $("language-select"),
  btnAnalyze: $("btn-analyze"),
  progressArea: $("progress-area"),
  progressFill: $("progress-fill"),
  progressPct: $("progress-pct"),
  progressStatus: $("progress-status"),
  errorBox: $("error-box"),

  // Stage 2
  stagePreview: $("stage-preview"),
  previewCount: $("preview-count"),
  previewTbody: $("preview-tbody"),
  btnGenerate: $("btn-generate"),
  btnRestartFromPreview: $("btn-restart-from-preview"),
  burnProgressArea: $("burn-progress-area"),
  burnProgressFill: $("burn-progress-fill"),
  burnProgressPct: $("burn-progress-pct"),
  burnProgressStatus: $("burn-progress-status"),

  // Stage 3
  stageDone: $("stage-done"),
  resultVideo: $("result-video"),
  dlVideo: $("dl-video"),
  dlSrt: $("dl-srt"),
  btnRestart: $("btn-restart"),
};

// ===================================================================
// UI helpers
// ===================================================================
function showStage(stage) {
  els.stageInput.classList.toggle("active", stage === "input");
  els.stagePreview.classList.toggle("active", stage === "preview");
  els.stageDone.classList.toggle("active", stage === "done");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

/*
 * Build a single closure that bundles all four DOM nodes for one progress UI
 * (bar area, fill, percent label, status text). Returns a function the
 * pipeline calls as `progress(pct, message)`. Passing `null`/undefined for
 * `message` leaves the existing status text alone — useful for the
 * sub-second updates coming from ffmpeg's progress event or whisper's
 * chunk callback where the verb hasn't changed.
 */
function makeProgressUpdater(area, fill, pctEl, statusEl) {
  let lastMessage = "";
  return function progress(pct, message) {
    area.classList.remove("hidden");
    const clamped = Math.max(0, Math.min(100, pct));
    fill.style.width = `${clamped}%`;
    pctEl.textContent = `${Math.floor(clamped)}%`;
    if (message != null && message !== lastMessage) {
      statusEl.textContent = message;
      lastMessage = message;
    }
  };
}

/*
 * Maps a sub-step's [0..1] internal progress onto a slice of the overall
 * progress bar [from..to]. Lets each pipeline step report 0..1 of itself
 * without knowing where it lives in the global timeline.
 */
function makeSubProgress(progress, from, to, label) {
  const span = to - from;
  return function (frac, sublabel) {
    const pct = from + Math.max(0, Math.min(1, frac)) * span;
    const msg = sublabel ? `${label} — ${sublabel}` : label;
    progress(pct, msg);
  };
}

function clearError() {
  els.errorBox.classList.add("hidden");
  els.errorBox.textContent = "";
}

function showError(msgOrErr) {
  els.errorBox.classList.remove("hidden");
  let text;
  if (msgOrErr && typeof msgOrErr === "object") {
    // Surface the FULL technical detail so we can debug from the UI alone
    // without forcing the user to open DevTools.
    const parts = [];
    if (msgOrErr.name) parts.push(msgOrErr.name);
    if (msgOrErr.message) parts.push(msgOrErr.message);
    if (msgOrErr.cause)   parts.push("cause: " + (msgOrErr.cause.message || msgOrErr.cause));
    if (msgOrErr.stack)   parts.push("\n--- stack ---\n" + msgOrErr.stack);
    text = parts.join(" — ");
  } else {
    text = String(msgOrErr);
  }
  els.errorBox.textContent = text;
  console.error("[LyricFuse]", msgOrErr);
}

function updateAnalyzeButton() {
  const ready = !!state.videoFile && state.lyricsText.trim().length > 0;
  els.btnAnalyze.disabled = !ready;
}

// ===================================================================
// Event wiring
// ===================================================================
function setupEventListeners() {
  // --- Drag/drop video ---
  const dz = els.dropZone;
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
    })
  );
  dz.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files?.[0];
    if (f) handleVideoSelected(f);
  });
  els.videoInput.addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) handleVideoSelected(f);
  });

  // --- Tabs for lyrics input ---
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    });
  });

  // --- Lyrics text/file ---
  els.lyricsText.addEventListener("input", (e) => {
    state.lyricsText = e.target.value;
    updateAnalyzeButton();
  });
  els.lyricsFile.addEventListener("change", async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const text = await f.text();
    state.lyricsText = text;
    els.lyricsText.value = text;
    els.lyricsFilePreview.textContent = text;
    els.lyricsFilePreview.classList.remove("hidden");
    updateAnalyzeButton();
  });

  // --- Language ---
  els.languageSelect.addEventListener("change", (e) => {
    state.language = e.target.value;
  });

  // --- Main actions ---
  els.btnAnalyze.addEventListener("click", handleAnalyze);
  els.btnGenerate.addEventListener("click", handleGenerate);
  els.btnRestart.addEventListener("click", handleRestart);
  els.btnRestartFromPreview.addEventListener("click", handleRestart);
}

function handleVideoSelected(file) {
  if (file.type && !file.type.startsWith("video/")) {
    showError("That doesn't look like a video file. Please drop an MP4.");
    return;
  }
  clearError();
  state.videoFile = file;
  els.videoFilename.textContent = `✓ ${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  els.videoFilename.classList.remove("hidden");
  updateAnalyzeButton();
}

// ===================================================================
// FFmpeg lazy loader
//
// Loading is genuinely three serial fetches (core JS, core WASM, wrapper
// worker JS) plus an initialization handshake. We report each as a slice
// of [0..1] so the caller can map them onto the overall bar.
// ===================================================================
async function getFFmpeg(subProgress) {
  if (state.ffmpeg && state.ffmpeg.loaded) {
    if (subProgress) subProgress(1.0, "already loaded");
    return state.ffmpeg;
  }

  const ffmpeg = new FFmpeg();
  ffmpeg.on("log", ({ message }) => console.debug("[ffmpeg]", message));

  if (subProgress) subProgress(0.0, "fetching core script");
  const coreURL = await toBlobURL(`${FFMPEG_CORE_BASE}/ffmpeg-core.js`, "text/javascript");
  if (subProgress) subProgress(0.4, "fetching WASM (~20 MB)");
  const wasmURL = await toBlobURL(`${FFMPEG_CORE_BASE}/ffmpeg-core.wasm`, "application/wasm");
  if (subProgress) subProgress(0.85, "fetching worker");
  const classWorkerURL = new URL(LOCAL_FFMPEG_WORKER, window.location.href).toString();

  if (subProgress) subProgress(0.92, "initializing engine");
  await ffmpeg.load({ coreURL, wasmURL, classWorkerURL });
  if (subProgress) subProgress(1.0, "ready");

  state.ffmpeg = ffmpeg;
  return ffmpeg;
}

// ===================================================================
// Whisper lazy loader (per language)
// ===================================================================
async function getTranscriber(language, subProgress) {
  // Sanskrit falls back to Hindi (Whisper has very limited 'sa' data).
  const effectiveLang = language === "sa" ? "hi" : language;

  if (state.transcriber && state.transcriberLang === effectiveLang) {
    if (subProgress) subProgress(1.0, "already in memory");
    return state.transcriber;
  }
  if (subProgress) subProgress(0.0, "starting download");

  // transformers.js fires progress_callback with status 'progress' and
  // numeric progress 0..100 per file. Multiple files load sequentially —
  // we track which file is current and surface that to the user.
  state.transcriber = await pipeline(
    "automatic-speech-recognition",
    WHISPER_MODEL,
    {
      progress_callback: (data) => {
        if (!subProgress) return;
        if (data.status === "progress") {
          const pct = (data.progress || 0) / 100;
          const mb = data.loaded ? ` (${(data.loaded / 1024 / 1024).toFixed(1)} MB)` : "";
          subProgress(pct, `${data.file}${mb}`);
        } else if (data.status === "done") {
          subProgress(1.0, `${data.file} cached`);
        } else if (data.status === "ready") {
          subProgress(1.0, "model ready");
        }
      },
    }
  );
  state.transcriberLang = effectiveLang;
  if (subProgress) subProgress(1.0, "model ready");
  return state.transcriber;
}

// ===================================================================
// Step 1: extract 16 kHz mono WAV from video
//
// FFmpeg's `progress` event emits { progress: 0..1, time: us } during exec.
// We attach a temporary listener for the duration of this call so the bar
// moves smoothly while ffmpeg is decoding/resampling/encoding the audio.
// ===================================================================
async function extractAudio(ffmpeg, videoBytes, subProgress) {
  if (subProgress) subProgress(0, "writing video to virtual FS");
  await ffmpeg.writeFile("input.mp4", videoBytes);

  // Live progress wiring. We CANNOT use the global onProgress option of
  // ffmpeg.load() because it would also fire during the later burn step;
  // instead we attach + detach a listener around this exec.
  const onFFmpegProgress = ({ progress }) => {
    if (subProgress && progress >= 0 && progress <= 1) {
      // Reserve the first 5% for the writeFile, last 5% for readFile.
      subProgress(0.05 + progress * 0.90, `decoding (${Math.floor(progress * 100)}%)`);
    }
  };
  ffmpeg.on("progress", onFFmpegProgress);

  try {
    if (subProgress) subProgress(0.05, "starting ffmpeg decode");
    // 16 kHz mono PCM is what Whisper wants.
    await ffmpeg.exec([
      "-i", "input.mp4",
      "-vn",
      "-acodec", "pcm_s16le",
      "-ar", "16000",
      "-ac", "1",
      "audio.wav",
    ]);
  } finally {
    ffmpeg.off("progress", onFFmpegProgress);
  }

  if (subProgress) subProgress(0.95, "reading WAV bytes back");
  const data = await ffmpeg.readFile("audio.wav");
  if (subProgress) subProgress(1.0, `${(data.byteLength / 1024 / 1024).toFixed(1)} MB extracted`);
  return data; // Uint8Array
}

// ===================================================================
// Step 2: decode WAV bytes -> Float32Array (mono, 16 kHz) for Whisper
//
// We control the producer: FFmpeg was invoked with
//   `-acodec pcm_s16le -ar 16000 -ac 1`
// so the WAV is GUARANTEED to be 16 kHz, mono, 16-bit little-endian PCM.
// Parsing it directly is faster, deterministic, and — critically — has no
// dependency on AudioContext, which on mobile Chrome has been observed to
// hang indefinitely when the context is created in a suspended state.
//
// We still fall back to AudioContext if the WAV header somehow comes back
// with unexpected parameters (FFmpeg version drift, etc.) — but that path
// should never trigger in normal use.
// ===================================================================
async function decodeWavToFloat32(wavBytes, subProgress) {
  if (subProgress) subProgress(0.0, "parsing WAV header");

  // RIFF/WAVE header walk. Layout reference: http://soundfile.sapp.org/doc/WaveFormat/
  const dv = new DataView(wavBytes.buffer, wavBytes.byteOffset, wavBytes.byteLength);
  if (dv.byteLength < 44) {
    throw new Error(`WAV file too short to contain a header (${dv.byteLength} bytes).`);
  }
  // "RIFF" + size + "WAVE"
  if (dv.getUint32(0,  false) !== 0x52494646) throw new Error("Not a RIFF file (missing 'RIFF' magic).");
  if (dv.getUint32(8,  false) !== 0x57415645) throw new Error("Not a WAVE file (missing 'WAVE' magic).");

  let offset = 12;
  let audioFormat = 0, numChannels = 0, sampleRate = 0, bitsPerSample = 0;
  let dataOffset = 0, dataSize = 0;

  // Walk sub-chunks. Most WAVs have 'fmt ' then 'data', but some encoders
  // insert 'LIST' / 'JUNK' / 'fact' in between — we skip whatever we don't know.
  while (offset <= dv.byteLength - 8) {
    const id   = dv.getUint32(offset,     false);
    const size = dv.getUint32(offset + 4, true);
    if (id === 0x666d7420 /* 'fmt ' */) {
      audioFormat   = dv.getUint16(offset + 8,  true);
      numChannels   = dv.getUint16(offset + 10, true);
      sampleRate    = dv.getUint32(offset + 12, true);
      bitsPerSample = dv.getUint16(offset + 22, true);
    } else if (id === 0x64617461 /* 'data' */) {
      dataOffset = offset + 8;
      dataSize   = size;
      break;
    }
    // Chunks are word-aligned: round up to even length.
    offset += 8 + size + (size & 1);
  }
  if (!dataOffset) throw new Error("WAV file has no 'data' chunk.");

  if (subProgress) {
    subProgress(0.2, `${sampleRate} Hz, ${numChannels}ch, ${bitsPerSample}-bit, ${(dataSize / 1024 / 1024).toFixed(1)} MB`);
  }

  // Fast path: 16-bit PCM, mono, 16 kHz — exactly what FFmpeg gave us and
  // exactly what Whisper wants. Convert int16 -> float32 in [-1, 1].
  const FORMAT_PCM = 1;
  if (audioFormat === FORMAT_PCM && bitsPerSample === 16 && numChannels === 1 && sampleRate === 16000) {
    const numSamples = (dataSize / 2) | 0;
    const out = new Float32Array(numSamples);
    // Yield control to the UI periodically so the progress bar can repaint.
    const CHUNK = 1 << 18; // 262 144 samples per yield ≈ 16 s of 16 kHz
    let written = 0;
    while (written < numSamples) {
      const end = Math.min(written + CHUNK, numSamples);
      for (let i = written; i < end; i++) {
        out[i] = dv.getInt16(dataOffset + i * 2, true) / 32768;
      }
      written = end;
      if (subProgress) {
        subProgress(0.2 + 0.8 * (written / numSamples),
          `${(written / 16000).toFixed(0)}s / ${(numSamples / 16000).toFixed(0)}s converted`);
      }
      // Yield to event loop so the UI can paint the new %.
      await new Promise((r) => setTimeout(r, 0));
    }
    if (subProgress) subProgress(1.0, `${(numSamples / 16000).toFixed(1)}s ready`);
    return out;
  }

  // Slow path: unexpected format (shouldn't happen with our FFmpeg call,
  // but degrade gracefully). Use AudioContext + OfflineAudioContext.
  console.warn(
    "[decodeWav] Unexpected WAV params, falling back to AudioContext.",
    { audioFormat, numChannels, sampleRate, bitsPerSample }
  );
  if (subProgress) subProgress(0.25, "unexpected format — using AudioContext fallback");
  return await decodeWavFallback(wavBytes, subProgress);
}

async function decodeWavFallback(wavBytes, subProgress) {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) throw new Error("Browser lacks Web Audio API and WAV format was non-standard.");
  const ctx = new AC();
  if (ctx.state === "suspended" && ctx.resume) {
    try { await ctx.resume(); } catch { /* best effort */ }
  }
  const ab = wavBytes.buffer.slice(wavBytes.byteOffset, wavBytes.byteOffset + wavBytes.byteLength);
  // Race decodeAudioData against a timeout so we surface the hang instead of
  // sitting there for 30 minutes.
  const decoded = await Promise.race([
    ctx.decodeAudioData(ab),
    new Promise((_, rej) =>
      setTimeout(() => rej(new Error("AudioContext.decodeAudioData hung for 60s — aborting.")), 60000)
    ),
  ]);
  if (ctx.close) ctx.close().catch(() => {});

  if (decoded.sampleRate === 16000 && decoded.numberOfChannels === 1) {
    return decoded.getChannelData(0);
  }
  if (subProgress) subProgress(0.6, `resampling ${decoded.sampleRate} Hz → 16000 Hz`);
  const targetLen = Math.max(1, Math.ceil(decoded.duration * 16000));
  const offline = new OfflineAudioContext(1, targetLen, 16000);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start(0);
  const resampled = await offline.startRendering();
  return resampled.getChannelData(0);
}

// ===================================================================
// Step 3: transcribe with word timestamps
//
// Transcription is the slowest pipeline step (often 30s-5min on mobile).
// Two complementary progress sources keep the bar moving:
//
//   1. transformers.js fires `chunk_callback(chunk)` after each 30-second
//      window completes. Total chunks ≈ ceil(audio_seconds / stride_window).
//   2. As a safety net, a 1 Hz wall-clock ticker advances the bar by a
//      small fraction per second so users see life even if chunk_callback
//      stalls (which can happen when the model is loading internal weights
//      lazily on the first chunk).
// ===================================================================
const CHUNK_LENGTH_S = 30;
const STRIDE_LENGTH_S = 5;

async function transcribeWithWordTimestamps(transcriber, float32Audio, language, subProgress) {
  const audioSeconds = float32Audio.length / 16000;
  // Stride between successive chunks; matches transformers.js's internal math.
  const effectiveStride = CHUNK_LENGTH_S - STRIDE_LENGTH_S;
  const totalChunks = Math.max(1, Math.ceil(audioSeconds / effectiveStride));

  let chunksDone = 0;
  const started = performance.now();

  if (subProgress) {
    subProgress(0.0, `0 / ${totalChunks} chunks (audio is ${audioSeconds.toFixed(1)}s)`);
  }

  // Safety-net ticker: even if chunk_callback never fires, advance the bar
  // toward — but never past — the next chunk boundary, based on a rough
  // estimate (mobile WASM Whisper ≈ 20s real time per 30s chunk).
  const ESTIMATED_SECONDS_PER_CHUNK = 20;
  const ticker = setInterval(() => {
    if (!subProgress) return;
    const elapsedS = (performance.now() - started) / 1000;
    const estChunks = Math.min(elapsedS / ESTIMATED_SECONDS_PER_CHUNK, totalChunks);
    // Only step forward if the wall-clock estimate is ahead of chunksDone.
    const frac = Math.max(chunksDone, estChunks) / totalChunks;
    subProgress(
      Math.min(0.97, frac),
      `${chunksDone} / ${totalChunks} chunks · ${Math.floor(elapsedS)}s elapsed`
    );
  }, 1000);

  let result;
  try {
    result = await transcriber(float32Audio, {
      return_timestamps: "word",
      language,
      task: "transcribe",
      chunk_length_s: CHUNK_LENGTH_S,
      stride_length_s: STRIDE_LENGTH_S,
      chunk_callback: (chunk) => {
        chunksDone += 1;
        if (subProgress) {
          const elapsedS = (performance.now() - started) / 1000;
          subProgress(
            chunksDone / totalChunks,
            `${chunksDone} / ${totalChunks} chunks · ${Math.floor(elapsedS)}s elapsed`
          );
        }
      },
    });
  } finally {
    clearInterval(ticker);
  }

  if (subProgress) subProgress(1.0, `transcribed (${(result.chunks || []).length} words)`);

  // result.chunks = [{ text, timestamp: [start, end] }, ...]
  return (result.chunks || [])
    .filter((c) => c.timestamp && c.timestamp[0] != null && c.timestamp[1] != null)
    .map((c) => ({
      word: c.text.trim(),
      start: c.timestamp[0],
      end: c.timestamp[1],
    }));
}

// ===================================================================
// Step 4: map Whisper word-timings onto user's line structure
// ===================================================================
function groupWordsToLines(words, lyricsText) {
  const userLines = lyricsText
    .split("\n")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  const result = [];
  let wi = 0;

  for (const line of userLines) {
    const wordCount = line.split(/\s+/).length;
    if (wi >= words.length) break;

    const chunk = words.slice(wi, wi + wordCount);
    if (chunk.length === 0) break;

    let start = chunk[0].start;
    let end = chunk[chunk.length - 1].end;
    if (end <= start) end = start + 0.5;

    result.push({ start, end, text: line });
    wi += wordCount;
  }
  return result;
}

// ===================================================================
// Step 5: write SRT
// ===================================================================
function formatSrtTime(seconds) {
  if (seconds < 0) seconds = 0;
  const totalMs = Math.round(seconds * 1000);
  const h = Math.floor(totalMs / 3_600_000);
  const m = Math.floor((totalMs % 3_600_000) / 60_000);
  const s = Math.floor((totalMs % 60_000) / 1000);
  const ms = totalMs % 1000;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")},${String(ms).padStart(3, "0")}`;
}

function generateSRT(timedLines) {
  return timedLines
    .map(
      (line, i) =>
        `${i + 1}\n${formatSrtTime(line.start)} --> ${formatSrtTime(line.end)}\n${line.text}\n`
    )
    .join("\n");
}

// ===================================================================
// Step 6: burn subtitles into video (FFmpeg)
//
// Same progress pattern as extractAudio: attach a temporary listener for
// the encode's duration so the bar reflects ffmpeg's own progress events.
// ===================================================================
async function burnSubtitles(ffmpeg, videoBytes, srtText, subProgress) {
  if (subProgress) subProgress(0.0, "writing video to virtual FS");
  await ffmpeg.writeFile("input.mp4", videoBytes);
  if (subProgress) subProgress(0.05, "writing subtitle file");
  await ffmpeg.writeFile("subs.srt", new TextEncoder().encode(srtText));

  const forceStyle =
    "FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=40";

  const onFFmpegProgress = ({ progress }) => {
    if (subProgress && progress >= 0 && progress <= 1) {
      // Reserve 0-7% for writes, 90-100% for read-back.
      subProgress(0.07 + progress * 0.83, `encoding (${Math.floor(progress * 100)}%)`);
    }
  };
  ffmpeg.on("progress", onFFmpegProgress);

  try {
    if (subProgress) subProgress(0.07, "starting encode");
    await ffmpeg.exec([
      "-i", "input.mp4",
      "-vf", `subtitles=subs.srt:force_style='${forceStyle}'`,
      "-c:a", "copy",
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-crf", "22",
      "output.mp4",
    ]);
  } finally {
    ffmpeg.off("progress", onFFmpegProgress);
  }

  if (subProgress) subProgress(0.92, "reading output MP4");
  const out = await ffmpeg.readFile("output.mp4");
  if (subProgress) subProgress(1.0, `${(out.byteLength / 1024 / 1024).toFixed(1)} MB ready`);
  return out;
}

// ===================================================================
// Handlers
// ===================================================================
async function handleAnalyze() {
  clearError();
  els.btnAnalyze.disabled = true;

  // Build a single progress updater bound to the analyze UI nodes, then
  // carve it into named sub-ranges per pipeline phase. The width of each
  // range is roughly proportional to how long that phase takes in practice,
  // so the bar moves at a roughly steady pace from end to end.
  const progress = makeProgressUpdater(
    els.progressArea, els.progressFill, els.progressPct, els.progressStatus
  );

  // Phase ranges (must sum to 100):
  const R = {
    read:     [0,   3 ],   //  3 — instant read of the file bytes
    ffmpeg:   [3,   10],   //  7 — fetching 3 scripts + init
    extract:  [10,  25],   // 15 — ffmpeg exec, progress events live
    model:    [25,  48],   // 23 — Whisper model download (first run dominates)
    decode:   [48,  55],   //  7 — WAV decode + resample
    asr:      [55,  92],   // 37 — transcription (slowest, chunk_callback live)
    align:    [92,  96],   //  4 — word-to-line mapping
    srt:      [96,  100],  //  4 — text gen
  };
  const phase = (key, label) =>
    makeSubProgress(progress, R[key][0], R[key][1], label);

  try {
    const readSub = phase("read", "Reading video file");
    readSub(0.0, "loading bytes from disk");
    state.videoBytes = new Uint8Array(await state.videoFile.arrayBuffer());
    readSub(1.0, `${(state.videoBytes.byteLength / 1024 / 1024).toFixed(1)} MB loaded`);

    const ffmpeg = await getFFmpeg(phase("ffmpeg", "Loading FFmpeg engine"));

    const wavBytes = await extractAudio(
      ffmpeg, state.videoBytes, phase("extract", "Extracting audio")
    );

    const transcriber = await getTranscriber(
      state.language, phase("model", "Loading Whisper model")
    );

    const float32 = await decodeWavToFloat32(wavBytes, phase("decode", "Decoding audio"));

    const lang = state.language === "sa" ? "hi" : state.language;
    const words = await transcribeWithWordTimestamps(
      transcriber, float32, lang, phase("asr", "Transcribing audio")
    );

    if (words.length === 0) {
      throw new Error("Whisper produced no word timestamps. Audio may be silent or in an unsupported language.");
    }

    const alignSub = phase("align", "Aligning lyric lines");
    alignSub(0.5, `${words.length} words → your lyric structure`);
    const timedLines = groupWordsToLines(words, state.lyricsText);
    if (timedLines.length === 0) {
      throw new Error("Could not align any lyric lines.");
    }
    alignSub(1.0, `${timedLines.length} lines aligned`);
    state.timedLines = timedLines;

    const srtSub = phase("srt", "Generating SRT");
    srtSub(0.5, "encoding timestamps");
    state.srtText = generateSRT(timedLines);
    srtSub(1.0, `${timedLines.length} cues ready`);

    progress(100, `Done — ${timedLines.length} lines aligned`);

    populatePreviewTable(timedLines);
    showStage("preview");
  } catch (err) {
    // Wrap with our prefix but pass the original error so showError
    // can render the full stack/cause chain.
    const wrapped = new Error("Analysis failed — " + (err.message || err));
    wrapped.cause = err;
    wrapped.stack = err.stack;
    showError(wrapped);
    els.btnAnalyze.disabled = false;
  }
}

function populatePreviewTable(lines) {
  els.previewCount.textContent =
    `${lines.length} lines aligned. Review timings, then generate.`;
  els.previewTbody.innerHTML = lines
    .map((l, i) =>
      `<tr>
        <td>${i + 1}</td>
        <td>${l.start.toFixed(2)}s</td>
        <td>${l.end.toFixed(2)}s</td>
        <td>${escapeHtml(l.text)}</td>
      </tr>`
    )
    .join("");
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function handleGenerate() {
  els.btnGenerate.disabled = true;
  const progress = makeProgressUpdater(
    els.burnProgressArea, els.burnProgressFill, els.burnProgressPct, els.burnProgressStatus
  );

  // FFmpeg should already be loaded from the analyze step (cached on state),
  // so its phase is normally a quick "already loaded" — but we leave a small
  // budget in case the user refreshed between analyze and generate.
  const R = {
    ffmpeg: [0,   5  ],
    burn:   [5,   100],
  };
  const phase = (k, label) => makeSubProgress(progress, R[k][0], R[k][1], label);

  try {
    const ffmpeg = await getFFmpeg(phase("ffmpeg", "Loading FFmpeg engine"));
    const outBytes = await burnSubtitles(
      ffmpeg, state.videoBytes, state.srtText, phase("burn", "Burning subtitles into video")
    );

    progress(100, "Done — your subtitled video is ready");

    const blob = new Blob([outBytes.buffer], { type: "video/mp4" });
    state.outputBlob = blob;
    const url = URL.createObjectURL(blob);
    els.resultVideo.src = url;
    els.dlVideo.href = url;

    const baseName = (state.videoFile.name || "output").replace(/\.[^/.]+$/, "");
    els.dlVideo.download = `${baseName}_lyrics.mp4`;

    const srtBlob = new Blob([state.srtText], { type: "application/x-subrip" });
    els.dlSrt.href = URL.createObjectURL(srtBlob);
    els.dlSrt.download = `${baseName}_lyrics.srt`;

    showStage("done");
  } catch (err) {
    const wrapped = new Error("Subtitle burn failed — " + (err.message || err));
    wrapped.cause = err;
    wrapped.stack = err.stack;
    showError(wrapped);
    els.btnGenerate.disabled = false;
  }
}

function handleRestart() {
  // Revoke object URLs so the browser can free memory.
  if (els.dlVideo.href.startsWith("blob:")) URL.revokeObjectURL(els.dlVideo.href);
  if (els.dlSrt.href.startsWith("blob:")) URL.revokeObjectURL(els.dlSrt.href);
  if (els.resultVideo.src.startsWith("blob:")) URL.revokeObjectURL(els.resultVideo.src);

  // Reset state
  state.videoFile = null;
  state.videoBytes = null;
  state.lyricsText = "";
  state.timedLines = null;
  state.srtText = null;
  state.outputBlob = null;

  els.videoInput.value = "";
  els.lyricsFile.value = "";
  els.lyricsText.value = "";
  els.lyricsFilePreview.textContent = "";
  els.lyricsFilePreview.classList.add("hidden");
  els.videoFilename.textContent = "";
  els.videoFilename.classList.add("hidden");
  els.progressArea.classList.add("hidden");
  els.burnProgressArea.classList.add("hidden");
  els.btnAnalyze.disabled = true;
  els.btnGenerate.disabled = false;
  clearError();

  showStage("input");
}

// ===================================================================
// Boot
// ===================================================================
setupEventListeners();
console.log("[LyricFuse] Ready.");
