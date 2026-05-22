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
// custom COOP/COEP headers. The coi-serviceworker is preloaded so we *could*
// upgrade to @ffmpeg/core-mt later for a speed boost.)
const FFMPEG_CORE_BASE = "https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm";

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

function setProgress(area, fill, status, pct, message) {
  area.classList.remove("hidden");
  fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  status.textContent = message;
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
// ===================================================================
async function getFFmpeg(onProgress) {
  if (state.ffmpeg && state.ffmpeg.loaded) return state.ffmpeg;

  const ffmpeg = new FFmpeg();
  ffmpeg.on("log", ({ message }) => console.debug("[ffmpeg]", message));
  ffmpeg.on("progress", ({ progress }) => {
    if (onProgress) onProgress(progress);
  });

  await ffmpeg.load({
    coreURL: await toBlobURL(`${FFMPEG_CORE_BASE}/ffmpeg-core.js`, "text/javascript"),
    wasmURL: await toBlobURL(`${FFMPEG_CORE_BASE}/ffmpeg-core.wasm`, "application/wasm"),
  });
  state.ffmpeg = ffmpeg;
  return ffmpeg;
}

// ===================================================================
// Whisper lazy loader (per language)
// ===================================================================
async function getTranscriber(language, onProgress) {
  // Sanskrit falls back to Hindi (Whisper has very limited 'sa' data).
  const effectiveLang = language === "sa" ? "hi" : language;

  if (state.transcriber && state.transcriberLang === effectiveLang) {
    return state.transcriber;
  }
  state.transcriber = await pipeline(
    "automatic-speech-recognition",
    WHISPER_MODEL,
    {
      progress_callback: (data) => {
        if (data.status === "progress" && onProgress) {
          onProgress(data.progress / 100, data.file);
        }
      },
    }
  );
  state.transcriberLang = effectiveLang;
  return state.transcriber;
}

// ===================================================================
// Step 1: extract 16 kHz mono WAV from video
// ===================================================================
async function extractAudio(ffmpeg, videoBytes) {
  await ffmpeg.writeFile("input.mp4", videoBytes);
  // 16 kHz mono PCM is what Whisper wants.
  await ffmpeg.exec([
    "-i", "input.mp4",
    "-vn",
    "-acodec", "pcm_s16le",
    "-ar", "16000",
    "-ac", "1",
    "audio.wav",
  ]);
  const data = await ffmpeg.readFile("audio.wav");
  return data; // Uint8Array
}

// ===================================================================
// Step 2: decode WAV bytes -> Float32Array (mono, 16 kHz) for Whisper
//
// Why this is non-trivial:
//   - `new AudioContext({ sampleRate: 16000 })` throws/ignores on Safari and
//     some mobile Chrome versions, silently falling back to 44.1/48 kHz.
//   - If we hand non-16 kHz audio to Whisper, it misinterprets duration and
//     produces garbage word timestamps (or hangs).
//   - Solution: decode at the context's native rate, then explicitly resample
//     to 16 kHz mono using an OfflineAudioContext (which always honors its
//     target sample rate).
// ===================================================================
async function decodeWavToFloat32(wavBytes) {
  const AC = window.AudioContext || window.webkitAudioContext;
  if (!AC) throw new Error("This browser does not support the Web Audio API.");
  const ctx = new AC();

  // decodeAudioData wants a real ArrayBuffer slice (not a SharedArrayBuffer view).
  const ab = wavBytes.buffer.slice(
    wavBytes.byteOffset,
    wavBytes.byteOffset + wavBytes.byteLength
  );

  let decoded;
  try {
    decoded = await ctx.decodeAudioData(ab);
  } catch (e) {
    throw new Error("Could not decode the extracted WAV audio. " + (e.message || e));
  } finally {
    if (ctx.close) ctx.close().catch(() => {});
  }

  // Fast path: already exactly what Whisper wants.
  if (decoded.sampleRate === 16000 && decoded.numberOfChannels === 1) {
    return decoded.getChannelData(0);
  }

  // Resample to 16 kHz mono via OfflineAudioContext (always honors its
  // declared sampleRate; auto-downmixes when the destination has 1 channel).
  const targetRate = 16000;
  const targetLen = Math.max(1, Math.ceil(decoded.duration * targetRate));
  const offline = new OfflineAudioContext(1, targetLen, targetRate);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start(0);
  const resampled = await offline.startRendering();
  return resampled.getChannelData(0);
}

// ===================================================================
// Step 3: transcribe with word timestamps
// ===================================================================
async function transcribeWithWordTimestamps(transcriber, float32Audio, language) {
  const result = await transcriber(float32Audio, {
    return_timestamps: "word",
    language,
    task: "transcribe",
    chunk_length_s: 30,
    stride_length_s: 5,
  });
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
// ===================================================================
async function burnSubtitles(ffmpeg, videoBytes, srtText) {
  await ffmpeg.writeFile("input.mp4", videoBytes);
  await ffmpeg.writeFile("subs.srt", new TextEncoder().encode(srtText));

  const forceStyle =
    "FontName=Arial,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=40";

  await ffmpeg.exec([
    "-i", "input.mp4",
    "-vf", `subtitles=subs.srt:force_style='${forceStyle}'`,
    "-c:a", "copy",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "22",
    "output.mp4",
  ]);
  return await ffmpeg.readFile("output.mp4");
}

// ===================================================================
// Handlers
// ===================================================================
async function handleAnalyze() {
  clearError();
  els.btnAnalyze.disabled = true;

  try {
    setProgress(els.progressArea, els.progressFill, els.progressStatus, 2, "Reading video...");
    state.videoBytes = new Uint8Array(await state.videoFile.arrayBuffer());

    setProgress(els.progressArea, els.progressFill, els.progressStatus, 5,
      "Loading FFmpeg (first time only, ~25 MB)...");
    const ffmpeg = await getFFmpeg();

    setProgress(els.progressArea, els.progressFill, els.progressStatus, 15,
      "Extracting audio from video...");
    const wavBytes = await extractAudio(ffmpeg, state.videoBytes);

    setProgress(els.progressArea, els.progressFill, els.progressStatus, 25,
      "Loading Whisper model (first time only, ~75 MB)...");
    const transcriber = await getTranscriber(state.language, (p, file) => {
      const pct = 25 + Math.floor(p * 25);
      setProgress(els.progressArea, els.progressFill, els.progressStatus, pct,
        `Downloading model: ${file} (${Math.floor(p * 100)}%)`);
    });

    setProgress(els.progressArea, els.progressFill, els.progressStatus, 55,
      "Decoding audio...");
    const float32 = await decodeWavToFloat32(wavBytes);

    setProgress(els.progressArea, els.progressFill, els.progressStatus, 60,
      "Transcribing with Whisper (this is the slow part — be patient)...");
    const lang = state.language === "sa" ? "hi" : state.language;
    const words = await transcribeWithWordTimestamps(transcriber, float32, lang);

    if (words.length === 0) {
      throw new Error("Whisper produced no word timestamps. Audio may be silent or in an unsupported language.");
    }

    setProgress(els.progressArea, els.progressFill, els.progressStatus, 92,
      "Mapping words to your lyric lines...");
    const timedLines = groupWordsToLines(words, state.lyricsText);
    if (timedLines.length === 0) {
      throw new Error("Could not align any lyric lines.");
    }
    state.timedLines = timedLines;
    state.srtText = generateSRT(timedLines);

    setProgress(els.progressArea, els.progressFill, els.progressStatus, 100,
      `Done — ${timedLines.length} lines aligned.`);

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
  try {
    setProgress(els.burnProgressArea, els.burnProgressFill, els.burnProgressStatus, 10,
      "Burning subtitles onto video...");
    const ffmpeg = await getFFmpeg();
    const outBytes = await burnSubtitles(ffmpeg, state.videoBytes, state.srtText);

    setProgress(els.burnProgressArea, els.burnProgressFill, els.burnProgressStatus, 100, "Done!");

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
