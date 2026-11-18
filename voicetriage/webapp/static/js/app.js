/* ═══════════════════════════════════════════════════════════
   VoiceTriage  —  Frontend Application  v4.0
   FIXED: Voice → Symptom Sync, State Reset, Transcript Priority
   ═══════════════════════════════════════════════════════════ */
"use strict";

// ── State ─────────────────────────────────────────────────────
const State = {
  mode:            "voice",
  currentResult:   null,
  allCases:        [],
  filteredCases:   [],
  urgencyFilter:   "ALL",
  historyData:     [],

  // Voice recording state
  isRecording:     false,
  voiceSeconds:    0,
  voiceTimerId:    null,
  audioContext:    null,
  mediaStream:     null,
  audioInput:      null,
  scriptProcessor: null,
  analyser:        null,
  recordedPcm:     [],
  recordedWavBlob: null,
  speechRecognizer:null,
  accumulatedSpeech:"",   // RESET on every new recording
  interimSpeech:   "",
  animFrameId:     null,
  speechAPIWorking:false, // track if Web Speech API is working
};

// ── Speaking prompts ───────────────────────────────────────────
const VOICE_PROMPTS = {
  chest: {
    lang: "hi",
    text: "मुझे पिछले दो घंटे से सीने में बहुत तेज़ दर्द हो रहा है। दर्द बाईं बाँह तक जा रहा है और बहुत पसीना आ रहा है। साँस लेने में भी बहुत तकलीफ़ है।",
  },
  stroke: {
    lang: "en",
    text: "My husband's face has suddenly drooped on one side. He cannot lift his right arm and his speech is completely slurred. This started ten minutes ago.",
  },
  fever: {
    lang: "hi",
    text: "मुझे चार दिनों से बहुत तेज़ बुखार है और पूरे शरीर पर लाल दाने निकल आए हैं। बहुत कमज़ोरी है और जोड़ों में असहनीय दर्द है।",
  },
  cold: {
    lang: "hi",
    text: "मुझे दो दिन से हल्की सर्दी और खाँसी है। हल्का बुखार निन्यानवे डिग्री है और गले में थोड़ी खराश है। कोई गंभीर तकलीफ़ नहीं है।",
  },
};

const TEXT_SAMPLES = {
  emergency: "I have been having severe crushing chest pain for the past two hours. The pain radiates to my left arm and jaw. I am sweating profusely and breathless.",
  stroke:    "My father's face suddenly drooped to the right side. He is unable to raise his right arm and cannot speak coherent words. Started 15 minutes ago.",
  tb:        "I have had a chronic cough for over three weeks with blood-tinged sputum, night sweats, and unexplained weight loss.",
  cold:      "Mild runny nose, sneezing, and slight sore throat for two days without high fever or difficulty breathing.",
  suicide:   "I have severe feelings of hopelessness and took multiple sleeping pills with suicidal intent.",
};

// ── Language map for Web Speech API ───────────────────────────
const SPEECH_LANG_MAP = {
  hi: "hi-IN",
  en: "en-IN",
  mr: "mr-IN",
  ta: "ta-IN",
  te: "te-IN",
  bn: "bn-IN",
  kn: "kn-IN",
  gu: "gu-IN",
  pa: "pa-IN",
};


// ═══════════════════════════════════════════════════════════
//  INITIALIZATION
// ═══════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadDemoCases();
  setupTextInput();
  initWaveformCanvas();
  checkSpeechAPISupport();
});

function setupTextInput() {
  const ta = document.getElementById("txtInput");
  if (!ta) return;
  ta.addEventListener("input", () => {
    document.getElementById("charCount").textContent = ta.value.length;
  });
}

function checkSpeechAPISupport() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    showVoiceNote("⚠️ Your browser does not support live speech recognition. Please use Google Chrome for voice input.", "warn");
  } else if (location.protocol === "http:" && location.hostname !== "127.0.0.1" && location.hostname !== "localhost") {
    showVoiceNote("ℹ️ For best voice recognition on phones, use the HTTPS link (run: python run.py --share).", "info");
  }
}

function showVoiceNote(msg, type) {
  const el = document.getElementById("voiceApiNote");
  if (!el) return;
  el.textContent = msg;
  el.className = `voice-note voice-note-${type}`;
  el.style.display = "block";
}


// ═══════════════════════════════════════════════════════════
//  MODE SWITCHING
// ═══════════════════════════════════════════════════════════
function setMode(mode) {
  State.mode = mode;
  ["voice", "text", "demo"].forEach(m => {
    const btn = document.getElementById(`mode${cap(m)}`);
    const content = document.getElementById(`mode${cap(m)}Content`);
    if (btn) btn.classList.toggle("active", m === mode);
    if (content) content.classList.toggle("active", m === mode);
  });
}
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);


// ═══════════════════════════════════════════════════════════
//  HEALTH CHECK
// ═══════════════════════════════════════════════════════════
async function checkHealth() {
  const pill = document.getElementById("statusPill");
  const dot  = pill ? pill.querySelector(".dot") : null;
  const txt  = document.getElementById("statusText");
  try {
    const r = await fetch("/api/health");
    const d = await r.json();
    if (d.status === "ok") {
      if (dot) dot.className = "dot dot-green";
      if (txt) txt.textContent = `Live · ${d.whisper_model}`;
    } else { throw new Error(); }
  } catch {
    if (dot) dot.className = "dot dot-red";
    if (txt) txt.textContent = "Offline";
  }
}


// ═══════════════════════════════════════════════════════════
//  DIRECT VOICE RECORDING & LIVE SPEAKING
// ═══════════════════════════════════════════════════════════

function toggleVoiceRecording() {
  if (State.isRecording) {
    stopVoiceRecording();
  } else {
    startVoiceRecording();
  }
}

async function startVoiceRecording() {
  // ── FIX 1: Reset ALL state on every new recording ──
  State.accumulatedSpeech = "";
  State.interimSpeech     = "";
  State.recordedWavBlob   = null;
  State.recordedPcm       = [];
  State.speechAPIWorking  = false;
  renderLiveSpeechText("", "");

  try {
    State.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: 16000,
        echoCancellation: true,
        noiseSuppression: true,
      }
    });

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    State.audioContext = new AudioContextClass({ sampleRate: 16000 });
    State.audioInput = State.audioContext.createMediaStreamSource(State.mediaStream);

    // Setup Analyser for real-time waveform
    State.analyser = State.audioContext.createAnalyser();
    State.analyser.fftSize = 256;
    State.audioInput.connect(State.analyser);

    // Setup ScriptProcessor for PCM capture (WAV fallback)
    State.scriptProcessor = State.audioContext.createScriptProcessor(4096, 1, 1);

    State.scriptProcessor.onaudioprocess = (e) => {
      if (!State.isRecording) return;
      State.recordedPcm.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };

    State.audioInput.connect(State.scriptProcessor);
    State.scriptProcessor.connect(State.audioContext.destination);

    // Start Speech Recognition (primary transcript source)
    setupLiveSpeechRecognition();

    State.isRecording   = true;
    State.voiceSeconds  = 0;

    // Update UI
    const micBtn     = document.getElementById("micBtn");
    const voiceStatus = document.getElementById("voiceStatus");
    const triageBtn  = document.getElementById("voiceTriageBtn");

    if (micBtn) {
      micBtn.classList.add("recording");
      document.getElementById("micIcon").textContent = "⏹️";
    }
    if (voiceStatus) {
      voiceStatus.classList.add("recording");
      voiceStatus.textContent = "🔴 Listening... Speak patient symptoms clearly";
    }
    if (triageBtn) {
      triageBtn.disabled = true;
      triageBtn.textContent = "⏹️ Stop & Analyze";
    }

    updateVoiceTimerDisplay();
    State.voiceTimerId = setInterval(() => {
      State.voiceSeconds++;
      updateVoiceTimerDisplay();
      // Auto-stop at 90 seconds
      if (State.voiceSeconds >= 90) stopVoiceRecording();
    }, 1000);

    drawLiveWaveform();
    toast("Microphone active — speak the patient's symptoms!", "info");

  } catch (err) {
    console.error("Microphone error:", err);
    if (err.name === "NotAllowedError") {
      toast("Microphone permission denied. Please allow microphone access in your browser.", "warn");
    } else {
      toast("Could not access microphone: " + err.message, "warn");
    }
    _cleanupAudio();
  }
}

function stopVoiceRecording() {
  if (!State.isRecording && !State.mediaStream) return;
  State.isRecording = false;

  if (State.voiceTimerId) {
    clearInterval(State.voiceTimerId);
    State.voiceTimerId = null;
  }

  // Stop Speech Recognition
  if (State.speechRecognizer) {
    try { State.speechRecognizer.stop(); } catch (e) {}
    State.speechRecognizer = null;
  }

  // ── FIX 2: Save any pending interim speech as final ──
  if (State.interimSpeech && State.interimSpeech.trim()) {
    State.accumulatedSpeech += (State.accumulatedSpeech ? " " : "") + State.interimSpeech.trim();
    State.interimSpeech = "";
  }

  // Encode WAV (backup only — transcript is primary)
  if (State.recordedPcm.length > 0) {
    State.recordedWavBlob = encodePcmToWav(State.recordedPcm, 16000);
  }

  _cleanupAudio();

  if (State.animFrameId) {
    cancelAnimationFrame(State.animFrameId);
    State.animFrameId = null;
    drawIdleWaveform();
  }

  // Update UI
  const micBtn     = document.getElementById("micBtn");
  const voiceStatus = document.getElementById("voiceStatus");
  const triageBtn  = document.getElementById("voiceTriageBtn");

  if (micBtn) {
    micBtn.classList.remove("recording");
    document.getElementById("micIcon").textContent = "🎙️";
  }

  const spokenWords = State.accumulatedSpeech.trim();

  if (voiceStatus) {
    voiceStatus.classList.remove("recording");
    if (spokenWords) {
      voiceStatus.textContent = `✅ Got ${spokenWords.split(" ").length} words. Click Analyze below.`;
    } else {
      voiceStatus.textContent = "⚠️ No speech detected. Try typing in the Text tab instead.";
    }
  }

  if (triageBtn) {
    if (spokenWords || State.recordedWavBlob) {
      triageBtn.disabled = false;
      triageBtn.textContent = "⚡ Analyze Spoken Symptoms ▶";
    } else {
      triageBtn.disabled = true;
      triageBtn.textContent = "⚡ Analyze Spoken Symptoms ▶";
    }
  }

  renderLiveSpeechText(State.accumulatedSpeech, "");
}

function _cleanupAudio() {
  if (State.scriptProcessor) {
    try { State.scriptProcessor.disconnect(); } catch(e){}
    State.scriptProcessor = null;
  }
  if (State.audioInput) {
    try { State.audioInput.disconnect(); } catch(e){}
    State.audioInput = null;
  }
  if (State.mediaStream) {
    State.mediaStream.getTracks().forEach(t => t.stop());
    State.mediaStream = null;
  }
  if (State.audioContext) {
    try { State.audioContext.close(); } catch(e){}
    State.audioContext = null;
  }
  State.analyser = null;
}

function updateVoiceTimerDisplay() {
  const el = document.getElementById("voiceTimer");
  if (!el) return;
  const m = String(Math.floor(State.voiceSeconds / 60)).padStart(2, "0");
  const s = String(State.voiceSeconds % 60).padStart(2, "0");
  el.textContent = `${m}:${s}`;
  el.classList.toggle("recording", State.isRecording);
}


// ═══════════════════════════════════════════════════════════
//  SPEECH RECOGNITION (WEB SPEECH API)
// ═══════════════════════════════════════════════════════════

function setupLiveSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    showVoiceNote("⚠️ Speech recognition not supported. Use Chrome browser for voice input.", "warn");
    return;
  }

  const selectedLang = document.getElementById("voiceLang")?.value || "hi";
  const bcp47 = SPEECH_LANG_MAP[selectedLang] || "hi-IN";

  State.speechRecognizer = new SpeechRecognition();
  State.speechRecognizer.continuous      = true;
  State.speechRecognizer.interimResults  = true;
  State.speechRecognizer.maxAlternatives = 1;
  State.speechRecognizer.lang            = bcp47;

  State.speechRecognizer.onstart = () => {
    State.speechAPIWorking = true;
    const note = document.getElementById("voiceApiNote");
    if (note) note.style.display = "none"; // hide warning if API works
  };

  State.speechRecognizer.onresult = (event) => {
    State.speechAPIWorking = true;
    let interimText = "";

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        // ── FIX 3: Accumulate FINAL results cleanly, avoid duplicates ──
        State.accumulatedSpeech += (State.accumulatedSpeech ? " " : "") + transcript.trim();
        State.interimSpeech = "";
      } else {
        interimText += transcript;
        State.interimSpeech = interimText;
      }
    }

    renderLiveSpeechText(State.accumulatedSpeech, interimText);

    // Enable triage button once some words captured
    const triageBtn = document.getElementById("voiceTriageBtn");
    const totalText = State.accumulatedSpeech + " " + interimText;
    if (triageBtn && totalText.trim().length > 5) triageBtn.disabled = false;
  };

  State.speechRecognizer.onerror = (e) => {
    console.warn("Speech recognition error:", e.error);
    if (e.error === "no-speech") {
      // Normal — just waiting for user to speak
    } else if (e.error === "not-allowed") {
      showVoiceNote("❌ Microphone blocked. Click the 🔒 icon in browser address bar → Allow microphone.", "warn");
    } else if (e.error === "network") {
      showVoiceNote("⚠️ Speech API needs internet. Words will still be captured via audio.", "info");
    } else if (e.error === "aborted") {
      // Intentional stop — ignore
    }
  };

  State.speechRecognizer.onend = () => {
    // ── FIX 4: Auto-restart recognition while still recording ──
    if (State.isRecording && State.speechRecognizer) {
      try {
        State.speechRecognizer.start();
      } catch (e) {
        // If it can't restart, that's okay — WAV fallback covers it
      }
    }
  };

  try {
    State.speechRecognizer.start();
  } catch (e) {
    console.warn("Could not start speech recognition:", e);
    showVoiceNote("⚠️ Speech recognition failed to start. Use Text tab as fallback.", "warn");
  }
}

function onVoiceLangChange() {
  if (State.isRecording && State.speechRecognizer) {
    try { State.speechRecognizer.stop(); } catch (e) {}
    setTimeout(() => setupLiveSpeechRecognition(), 300);
  }
}

function renderLiveSpeechText(finalText, interimText) {
  const el = document.getElementById("liveTranscript");
  if (!el) return;

  const full = (finalText + (interimText ? " " + interimText : "")).trim();
  if (!full) {
    el.innerHTML = `<span class="lt-placeholder">Click microphone above and speak your symptoms. Your words will appear here live...</span>`;
    return;
  }

  el.innerHTML = `<strong>${esc(finalText)}</strong> <span style="color:var(--accent);opacity:0.8">${esc(interimText || "")}</span>`;
  el.scrollTop = el.scrollHeight;

  // Show word count
  const wordCount = finalText.trim().split(/\s+/).filter(w => w).length;
  const wc = document.getElementById("wordCountBadge");
  if (wc) wc.textContent = wordCount > 0 ? `${wordCount} words` : "";
}

function clearVoiceTranscript() {
  State.accumulatedSpeech = "";
  State.interimSpeech     = "";
  State.recordedWavBlob   = null;
  renderLiveSpeechText("", "");
  const triageBtn = document.getElementById("voiceTriageBtn");
  if (triageBtn && !State.isRecording) triageBtn.disabled = true;
  toast("Voice transcript cleared.", "info");
}

function speakPrompt(key) {
  const p = VOICE_PROMPTS[key];
  if (!p) return;
  const langSel = document.getElementById("voiceLang");
  if (langSel) langSel.value = p.lang;
  State.accumulatedSpeech = p.text;
  State.interimSpeech     = "";
  renderLiveSpeechText(p.text, "");
  const triageBtn = document.getElementById("voiceTriageBtn");
  if (triageBtn) triageBtn.disabled = false;
  toast("Practice symptoms loaded! Click 'Analyze' to run triage.", "ok");
}


// ═══════════════════════════════════════════════════════════
//  WAVEFORM VISUALIZER (CANVAS)
// ═══════════════════════════════════════════════════════════

function initWaveformCanvas() {
  const canvas = document.getElementById("waveformCanvas");
  if (!canvas) return;
  drawIdleWaveform();
}

function drawIdleWaveform() {
  const canvas = document.getElementById("waveformCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#2e344d";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
}

function drawLiveWaveform() {
  const canvas = document.getElementById("waveformCanvas");
  if (!canvas || !State.analyser) return;

  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  const bufferLength = State.analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  const draw = () => {
    if (!State.isRecording) return;
    State.animFrameId = requestAnimationFrame(draw);
    State.analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = "#121520";
    ctx.fillRect(0, 0, w, h);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "#38bdf8";
    ctx.beginPath();

    const sliceWidth = w / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = (v * h) / 2;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
      x += sliceWidth;
    }

    ctx.lineTo(w, h / 2);
    ctx.stroke();
  };

  draw();
}


// ═══════════════════════════════════════════════════════════
//  SUBMIT VOICE TRIAGE  — FIXED PRIORITY LOGIC
// ═══════════════════════════════════════════════════════════

async function submitVoiceTriage() {
  if (State.isRecording) stopVoiceRecording();

  // Small delay to let any final speech results arrive
  await new Promise(r => setTimeout(r, 300));

  // ── FIX 5: ALWAYS prefer live speech transcript over WAV ──
  // Web Speech API gives us real text; WAV needs Whisper (not always installed)
  const spokenText = State.accumulatedSpeech.trim();
  const language   = document.getElementById("voiceLang")?.value || "hi";

  if (!spokenText && !State.recordedWavBlob) {
    toast("No voice recorded yet. Click the microphone and speak.", "warn");
    return;
  }

  // If no speech text but have WAV audio, warn user
  if (!spokenText && State.recordedWavBlob) {
    toast("⚠️ No live transcript captured. Sending audio for processing (needs Whisper model)…", "warn");
  }

  showLoading("Analysing voice symptoms across clinical layers…");

  try {
    let res, data;

    if (spokenText) {
      // ── PRIMARY PATH: Send transcript text directly (always works) ──
      res = await fetch("/api/triage/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          transcript: spokenText,
          language:   language,
          case_id:    `VOICE_${Date.now().toString(16).toUpperCase()}`,
        }),
      });
    } else {
      // ── FALLBACK PATH: Send WAV audio (needs Whisper installed) ──
      const formData = new FormData();
      formData.append("file", State.recordedWavBlob, "voice_recording.wav");
      formData.append("language", language);
      res = await fetch("/api/triage/voice", {
        method: "POST",
        body: formData,
      });
    }

    data = await res.json();
    if (!data.success) throw new Error(data.error || "Triage processing failed");

    // Tag the result as voice-sourced for display
    if (spokenText) {
      data.stt_model = data.stt_model || "Live Voice (Web Speech API)";
    }

    renderResult(data);
    toast("Voice triage complete!", "ok");

  } catch (err) {
    showErrorState(err.message);
  } finally {
    hideLoading();
  }
}


// ═══════════════════════════════════════════════════════════
//  PCM TO 16kHz WAV ENCODER (PURE JS — backup only)
// ═══════════════════════════════════════════════════════════

function encodePcmToWav(chunks, sampleRate) {
  let totalLength = 0;
  for (let c of chunks) totalLength += c.length;

  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (let c of chunks) {
    merged.set(c, offset);
    offset += c.length;
  }

  const buffer = new ArrayBuffer(44 + merged.length * 2);
  const view   = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + merged.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, merged.length * 2, true);

  let idx = 44;
  for (let i = 0; i < merged.length; i++) {
    let s = Math.max(-1, Math.min(1, merged[i]));
    view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    idx += 2;
  }

  return new Blob([view], { type: "audio/wav" });
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}


// ═══════════════════════════════════════════════════════════
//  TEXT TRIAGE
// ═══════════════════════════════════════════════════════════

async function runTextTriage() {
  const transcript = document.getElementById("txtInput").value.trim();
  if (!transcript) { toast("Please type a patient description first.", "warn"); return; }
  const language = document.getElementById("txtLang").value;

  showLoading("Analysing patient description…");
  setBtn("txtRunBtn", true, "Running…");

  try {
    const r = await fetch("/api/triage/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript, language }),
    });
    const d = await r.json();
    if (!d.success) throw new Error(d.error || "Server error");
    renderResult(d);
    toast("Triage complete!", "ok");
  } catch (e) {
    showErrorState(e.message);
  } finally {
    hideLoading();
    setBtn("txtRunBtn", false, "Run Triage ▶");
  }
}

function useSample(type) {
  document.getElementById("txtInput").value = TEXT_SAMPLES[type] || "";
  document.getElementById("charCount").textContent = (TEXT_SAMPLES[type] || "").length;
  setMode("text");
  toast("Sample loaded — click Run Triage ▶", "info");
}


// ═══════════════════════════════════════════════════════════
//  DEMO CASES
// ═══════════════════════════════════════════════════════════

async function loadDemoCases() {
  try {
    const r = await fetch("/api/cases");
    const d = await r.json();
    State.allCases = d.cases || [];
    State.filteredCases = [...State.allCases];
    renderCasesList();
  } catch {
    document.getElementById("casesList").innerHTML =
      `<div class="loading-row">Failed to load cases. Is the server running?</div>`;
  }
}

function renderCasesList() {
  const list = document.getElementById("casesList");
  const q    = (document.getElementById("demoSearch")?.value || "").toLowerCase();
  let cases  = State.filteredCases.filter(c => {
    const matchText = !q || c.case_id.toLowerCase().includes(q) ||
                      c.transcript_preview.toLowerCase().includes(q) ||
                      (c.symptoms||[]).join(",").toLowerCase().includes(q);
    const matchUrg  = State.urgencyFilter === "ALL" || c.urgency === State.urgencyFilter;
    return matchText && matchUrg;
  });

  if (!cases.length) {
    list.innerHTML = `<div class="loading-row">No cases match your filter.</div>`;
    return;
  }

  list.innerHTML = cases.map(c => `
    <div class="case-row" onclick="runDemo('${c.case_id}')">
      <div class="case-dot" style="background:${urgColor(c.urgency)}"></div>
      <div class="case-info">
        <div class="case-id">${esc(c.case_id)} · ${esc(c.language_name)}</div>
        <div class="case-preview">${esc(c.transcript_preview)}</div>
        <div class="case-meta">${esc(c.urgency)} → ${esc(c.specialty)}</div>
      </div>
    </div>
  `).join("");
}

function filterCases(query) { renderCasesList(); }

function filterByUrgency(urg, btn) {
  State.urgencyFilter = urg;
  document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  renderCasesList();
}

async function runDemo(caseId) {
  showLoading(`Running demo: ${caseId}…`);
  try {
    const r = await fetch(`/api/demo/${caseId || ""}`);
    const d = await r.json();
    if (!d.success) throw new Error(d.error || "Server error");
    renderResult(d);
    toast(`Demo case ${caseId} complete`, "ok");
  } catch (e) {
    showErrorState(e.message);
  } finally {
    hideLoading();
  }
}


// ═══════════════════════════════════════════════════════════
//  RENDER RESULT
// ═══════════════════════════════════════════════════════════

function renderResult(data) {
  State.currentResult = data;

  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("errorState").classList.add("hidden");
  document.getElementById("resultWrap").classList.remove("hidden");

  const urg  = data.urgency_level || "ROUTINE";
  const icon = { EMERGENCY:"🔴", URGENT:"🟡", ROUTINE:"🟢" }[urg] || "🟢";

  // Urgency banner
  const banner = document.getElementById("urgencyBanner");
  banner.className = `urgency-banner ${urg}`;
  set("ubIcon",     icon);
  set("ubUrgency",  urg);
  set("ubSpecialty",data.suggested_specialty || "–");
  set("ubConf",     `${Math.round((data.confidence_score||0)*100)}%`);

  // Pipeline steps
  ["ps1","ps2","ps3","ps4"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.className = "pstep done";
  });

  // Transcript
  set("transcriptBox", data.transcript || "–");
  set("langBadge",     data.detected_language || "–");
  set("modelBadge",    data.stt_model || "–");

  // Symptoms
  const symptomsEl = document.getElementById("symptomsBox");
  if (symptomsEl) {
    if (data.symptoms && data.symptoms.length > 0) {
      symptomsEl.innerHTML = tagsHtml(data.symptoms, "tag-blue");
    } else {
      symptomsEl.innerHTML = `<span style="color:var(--muted)">None extracted — try speaking more clearly or use the Text tab.</span>`;
    }
  }

  // Patient info
  set("durVal", data.duration  || "–");
  set("sevVal", data.severity  || "–");
  set("ageVal", data.age != null ? String(data.age) : "–");
  set("genVal", data.gender    || "–");
  document.getElementById("disVal").innerHTML =
    tagsHtml(data.existing_diseases||[], "tag-gray") ||
    `<span style="color:var(--muted)">None</span>`;

  // Red flags
  const rfs    = data.red_flags_triggered || [];
  const rfCard = document.getElementById("rfCard");
  if (rfCard) rfCard.style.borderColor = rfs.length ? "var(--emergency)" : "";
  document.getElementById("rfBox").innerHTML = rfs.length
    ? tagsHtml(rfs.map(r => r.split(" [")[0]), "tag-red")
    : `<span style="color:var(--muted)">None triggered ✓</span>`;

  // Rules
  document.getElementById("rulesBox").innerHTML =
    tagsHtml(data.rule_ids_matched||[], "tag-yellow") ||
    `<span style="color:var(--muted)">–</span>`;

  // Explanation
  set("reasonBox", data.explanation || "–");

  // Performance
  set("perfTotal", fmt(data.total_time_seconds));
  set("perfSTT",   fmt(data.stt_time));
  set("perfLLM",   fmt(data.llm_time));
  set("perfDec",   fmt(data.decision_time));
  set("perfRAG",   String(data.rag_rules_count ?? "–"));
  set("perfCase",  data.case_id || "–");

  document.getElementById("outputCard").scrollIntoView({behavior:"smooth",block:"start"});
}

const fmt = v => v != null ? `${Number(v).toFixed(2)}s` : "–";


// ═══════════════════════════════════════════════════════════
//  HISTORY
// ═══════════════════════════════════════════════════════════

async function openHistory() {
  document.getElementById("historyOverlay").classList.remove("hidden");
  document.getElementById("historyDrawer").classList.add("open");
  await loadHistory();
}

function closeDrawer(which) {
  document.getElementById(`${which}Overlay`).classList.add("hidden");
  document.getElementById(`${which}Drawer`).classList.remove("open");
}

async function loadHistory(query) {
  const list = document.getElementById("historyList");
  try {
    const r = await fetch("/api/history?limit=40");
    const d = await r.json();
    State.historyData = d.results || [];
    renderHistory(query || "");
  } catch {
    list.innerHTML = `<div class="loading-row">Failed to load history.</div>`;
  }
}

function filterHistory(q) { renderHistory(q); }

function renderHistory(q) {
  const list = document.getElementById("historyList");
  q = (q||"").toLowerCase();
  let items = State.historyData;
  if (q) items = items.filter(r =>
    (r.urgency_level||"").toLowerCase().includes(q) ||
    (r.suggested_specialty||"").toLowerCase().includes(q) ||
    (r.symptoms||[]).join(",").toLowerCase().includes(q)
  );

  if (!items.length) {
    list.innerHTML = `<div class="loading-row">No history yet. Record some voice triages first!</div>`;
    return;
  }
  list.innerHTML = items.map((r,i) => `
    <div class="hist-item" onclick="restoreResult(${i})">
      <div class="hist-row1">
        <span class="hist-urgency" style="color:${urgColor(r.urgency_level)}">${esc(r.urgency_level||"–")}</span>
        <span class="hist-time">${fmtTime(r.timestamp)}</span>
      </div>
      <div class="hist-spec">${esc(r.suggested_specialty||"–")}</div>
      <div class="hist-syms">${esc((r.symptoms||[]).slice(0,4).join(", ")||"–")}</div>
    </div>
  `).join("");
}

function restoreResult(idx) {
  const r = State.historyData[idx];
  if (r) { renderResult(r); closeDrawer("history"); }
}


// ═══════════════════════════════════════════════════════════
//  STATS
// ═══════════════════════════════════════════════════════════

async function openStats() {
  document.getElementById("statsOverlay").classList.remove("hidden");
  document.getElementById("statsDrawer").classList.add("open");
  await loadStats();
}

async function loadStats() {
  const wrap = document.getElementById("statsContent");
  try {
    const r = await fetch("/api/stats");
    const d = await r.json();

    if (!d.total) {
      wrap.innerHTML = `<div class="loading-row">No data yet. Speak some symptoms first.</div>`;
      return;
    }
    const u     = d.urgency || {};
    const specs = Object.entries(d.top_specialties||{})
      .map(([k,v])=>`<span class="tag tag-blue">${esc(k)}: ${v}</span>`).join(" ");

    wrap.innerHTML = `
      <div class="stat-row">
        <div class="stat-box"><div class="stat-val c-emergency">${u.EMERGENCY||0}</div><div class="stat-key">Emergency</div></div>
        <div class="stat-box"><div class="stat-val c-urgent">${u.URGENT||0}</div><div class="stat-key">Urgent</div></div>
        <div class="stat-box"><div class="stat-val c-routine">${u.ROUTINE||0}</div><div class="stat-key">Routine</div></div>
      </div>
      <div class="stat-row">
        <div class="stat-box"><div class="stat-val c-accent">${d.total}</div><div class="stat-key">Total Triages</div></div>
        <div class="stat-box"><div class="stat-val c-accent">${Math.round((d.avg_confidence||0)*100)}%</div><div class="stat-key">Avg Confidence</div></div>
      </div>
      <div>
        <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:.5rem">Top Specialties</div>
        <div class="tags-wrap">${specs||"–"}</div>
      </div>
    `;
  } catch {
    wrap.innerHTML = `<div class="loading-row">Failed to load stats.</div>`;
  }
}


// ═══════════════════════════════════════════════════════════
//  HELP
// ═══════════════════════════════════════════════════════════
function openHelp()  { document.getElementById("helpOverlay").classList.remove("hidden"); }
function closeHelp() { document.getElementById("helpOverlay").classList.add("hidden"); }


// ═══════════════════════════════════════════════════════════
//  COPY / DOWNLOAD RESULT
// ═══════════════════════════════════════════════════════════
function copyResult() {
  if (!State.currentResult) { toast("No result to copy.", "warn"); return; }
  navigator.clipboard.writeText(JSON.stringify(State.currentResult, null, 2))
    .then(() => toast("Result copied to clipboard!", "ok"))
    .catch(() => toast("Copy failed.", "warn"));
}

function downloadResult() {
  if (!State.currentResult) { toast("No result to download.", "warn"); return; }
  const blob = new Blob([JSON.stringify(State.currentResult, null, 2)], {type:"application/json"});
  const a    = document.createElement("a");
  a.href     = URL.createObjectURL(blob);
  a.download = `triage_${State.currentResult.case_id || "result"}.json`;
  a.click();
  toast("Downloading JSON…", "info");
}


// ═══════════════════════════════════════════════════════════
//  LOADING OVERLAY
// ═══════════════════════════════════════════════════════════
let _stageTimer = null;

function showLoading(title) {
  document.getElementById("loadingOverlay").classList.remove("hidden");
  document.getElementById("loadingTitle").textContent = title || "Running Pipeline…";

  const rows = ["sr1","sr2","sr3","sr4"];
  rows.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.className = "stage-row";
  });

  let step = 0;
  _stageTimer = setInterval(() => {
    if (step > 0) {
      const prev = document.getElementById(rows[step-1]);
      if (prev) prev.className = "stage-row done";
    }
    if (step < rows.length) {
      const curr = document.getElementById(rows[step]);
      if (curr) curr.className = "stage-row active";
      step++;
    } else { clearInterval(_stageTimer); }
  }, 700);
}

function hideLoading() {
  clearInterval(_stageTimer);
  document.getElementById("loadingOverlay").classList.add("hidden");
}


// ═══════════════════════════════════════════════════════════
//  ERROR STATE
// ═══════════════════════════════════════════════════════════
function showErrorState(msg) {
  document.getElementById("emptyState").classList.add("hidden");
  document.getElementById("resultWrap").classList.add("hidden");
  document.getElementById("errorState").classList.remove("hidden");

  let friendlyMsg = msg || "An unexpected error occurred.";
  if (friendlyMsg.toLowerCase().includes("fetch") || friendlyMsg.toLowerCase().includes("network")) {
    friendlyMsg = "Cannot reach the VoiceTriage server. Please run 'python run.py' in your terminal, then try again.";
  }
  document.getElementById("errMsg").textContent = friendlyMsg;
}

function hideError() {
  document.getElementById("errorState").classList.add("hidden");
  document.getElementById("emptyState").classList.remove("hidden");
}


// ═══════════════════════════════════════════════════════════
//  TOAST
// ═══════════════════════════════════════════════════════════
let _toastTimer = null;
function toast(msg, type) {
  const el = document.getElementById("toast");
  if (!el) return;
  const colors = { ok:"var(--routine)", warn:"var(--urgent)", info:"var(--accent)" };
  el.textContent = msg;
  el.style.borderLeftColor = colors[type] || "var(--accent)";
  el.style.borderLeft = `3px solid ${colors[type]||"var(--accent)"}`;
  el.classList.remove("hidden");
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => el.classList.add("hidden"), 4000);
}


// ═══════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════
function set(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function esc(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

function tagsHtml(arr, cls) {
  if (!arr?.length) return "";
  return arr.map(s => `<span class="tag ${cls}">${esc(s)}</span>`).join(" ");
}

function urgColor(u) {
  return { EMERGENCY:"var(--emergency)", URGENT:"var(--urgent)", ROUTINE:"var(--routine)" }[u]
    || "var(--muted)";
}

function fmtTime(ts) {
  try { return new Date(ts).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"}); }
  catch { return ts||"–"; }
}

function setBtn(id, disabled, text) {
  const btn = document.getElementById(id);
  if (!btn) return;
  btn.disabled = disabled;
  btn.textContent = text;
}
