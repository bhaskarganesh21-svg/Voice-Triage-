# ==========================================
# Stage 1: Speech Processing Layer
# VoiceTriage — eSanjeevani Triage Assistant
# ==========================================
# Pipeline:
#   Audio File
#     ↓ Noise Reduction (noisereduce)
#     ↓ Voice Activity Detection (silero-vad)
#     ↓ Speech-to-Text (faster-whisper / IndicWhisper)
#     ↓ Transcribed Text + Metadata
# ==========================================

import os
import sys
import json
import time
import logging
import tempfile
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import soundfile as sf
import noisereduce as nr
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────
# Logger Setup
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Stage1-STT")


# ──────────────────────────────────────────
# Output Data Structure
# ──────────────────────────────────────────
@dataclass
class STTResult:
    """Structured output from Stage 1 Speech Processing"""
    # Core output
    transcript: str                   # Final transcribed text
    detected_language: str            # Language code (e.g. "hi", "ta", "mr")
    detected_language_name: str       # Human-readable language name
    confidence: float                 # Overall transcription confidence (0.0-1.0)

    # Audio metadata
    audio_duration_seconds: float     # Total audio duration
    speech_duration_seconds: float    # Duration of actual speech (post-VAD)
    sample_rate: int                  # Audio sample rate

    # Processing metadata
    model_used: str                   # Whisper model variant used
    noise_reduction_applied: bool     # Whether noise reduction was performed
    vad_applied: bool                 # Whether VAD was performed
    processing_time_seconds: float    # Time taken to process

    # Segments (for detailed analysis)
    segments: list                    # List of timed transcript segments

    # Status
    success: bool                     # Whether processing succeeded
    error_message: Optional[str]      # Error message if failed

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def summary(self) -> str:
        """Human-readable summary for logging"""
        if not self.success:
            return f"❌ STT FAILED: {self.error_message}"
        return (
            f"\n{'='*55}\n"
            f"  ✅  STAGE 1 — SPEECH PROCESSING COMPLETE\n"
            f"{'='*55}\n"
            f"  📝 Transcript   : {self.transcript[:120]}{'...' if len(self.transcript)>120 else ''}\n"
            f"  🌐 Language     : {self.detected_language_name} ({self.detected_language})\n"
            f"  🎯 Confidence   : {self.confidence:.1%}\n"
            f"  🎵 Audio        : {self.audio_duration_seconds:.1f}s total, {self.speech_duration_seconds:.1f}s speech\n"
            f"  ⚡ Processing   : {self.processing_time_seconds:.2f}s\n"
            f"  🔧 Model        : {self.model_used}\n"
            f"{'='*55}"
        )


# ──────────────────────────────────────────
# Language Code to Name Map
# ──────────────────────────────────────────
LANGUAGE_NAMES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "bn": "Bengali",
    "or": "Odia",
    "ur": "Urdu",
    "as": "Assamese",
    "en": "English",
    "unknown": "Unknown"
}


# ──────────────────────────────────────────
# STEP 1: Audio Loading & Validation
# ──────────────────────────────────────────
def load_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio file and resample to 16kHz (required by Whisper).
    Supports: WAV, MP3, M4A, FLAC, OGG, WEBM

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default 16000 Hz for Whisper)

    Returns:
        (audio_array, sample_rate)
    """
    file_path = str(file_path)

    # Try loading with soundfile first (WAV, FLAC, etc.)
    try:
        audio, sr = sf.read(file_path, dtype="float32")
        logger.info(f"Loaded audio: {Path(file_path).name} | {sr} Hz | {len(audio)/sr:.2f}s | {audio.shape}")
    except Exception:
        # Fallback: use librosa for MP3, M4A, etc.
        try:
            import librosa
            audio, sr = librosa.load(file_path, sr=None, mono=False, dtype=np.float32)
            logger.info(f"Loaded (librosa): {Path(file_path).name} | {sr} Hz | {len(audio)/sr:.2f}s")
        except ImportError:
            raise RuntimeError(
                "Cannot load audio file. Install librosa for MP3/M4A support:\n"
                "  pip install librosa"
            )

    # Convert stereo to mono by averaging channels
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
        logger.info("Converted stereo → mono")

    # Resample to target sample rate if needed
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            logger.info(f"Resampled: {sr} Hz → {target_sr} Hz")
        except ImportError:
            logger.warning("librosa not installed — skipping resample. Whisper may handle it.")

    # Normalize audio to [-1.0, 1.0]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
        logger.info(f"Normalized audio (peak was {max_val:.3f})")

    return audio, sr


# ──────────────────────────────────────────
# STEP 2: Noise Reduction
# ──────────────────────────────────────────
def apply_noise_reduction(
    audio: np.ndarray,
    sr: int,
    stationary: bool = True,
    prop_decrease: float = 0.75
) -> np.ndarray:
    """
    Apply spectral noise reduction to clean up background noise from calls.

    Args:
        audio: Input audio array
        sr: Sample rate
        stationary: If True, assumes steady background noise (good for call center hiss)
        prop_decrease: How aggressively to reduce noise (0.0 = none, 1.0 = full)

    Returns:
        Cleaned audio array
    """
    logger.info(f"Applying noise reduction (stationary={stationary}, strength={prop_decrease})")

    try:
        # Use first 0.5s as noise profile (background/silence at call start)
        noise_sample = audio[:int(sr * 0.5)] if len(audio) > sr * 0.5 else audio[:100]

        cleaned = nr.reduce_noise(
            y=audio,
            y_noise=noise_sample,
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease,
            time_constant_s=2.0,
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=50,
            n_std_thresh_stationary=1.5,
        )

        # Measure SNR improvement (approximate)
        original_rms = np.sqrt(np.mean(audio**2))
        cleaned_rms = np.sqrt(np.mean(cleaned**2))
        logger.info(f"Noise reduction complete. RMS: {original_rms:.4f} → {cleaned_rms:.4f}")

        return cleaned

    except Exception as e:
        logger.warning(f"Noise reduction failed ({e}). Using original audio.")
        return audio


# ──────────────────────────────────────────
# STEP 3: Voice Activity Detection (VAD)
# ──────────────────────────────────────────
def apply_vad_silero(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.4,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 500
) -> tuple[np.ndarray, float]:
    """
    Apply Silero VAD to detect and keep only speech segments.
    Removes long silences and non-speech audio.

    Args:
        audio: Input audio array
        sr: Sample rate (must be 16000 or 8000 for Silero)
        threshold: Speech probability threshold (0.0-1.0, higher = stricter)
        min_speech_duration_ms: Minimum speech segment length to keep
        min_silence_duration_ms: Minimum silence to remove

    Returns:
        (speech_only_audio, speech_duration_seconds)
    """
    logger.info("Applying Voice Activity Detection (Silero VAD)...")

    try:
        import torch

        # Load Silero VAD model (cached after first download)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False
        )
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

        # Convert numpy to torch tensor
        audio_tensor = torch.FloatTensor(audio)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor,
            model,
            sampling_rate=sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=False
        )

        if not speech_timestamps:
            logger.warning("VAD found no speech segments! Using full audio.")
            return audio, len(audio) / sr

        # Collect only speech chunks
        speech_audio = collect_chunks(speech_timestamps, audio_tensor)
        speech_audio = speech_audio.numpy()

        speech_duration = len(speech_audio) / sr
        total_duration = len(audio) / sr
        removed = total_duration - speech_duration

        logger.info(
            f"VAD: {len(speech_timestamps)} speech segments found | "
            f"Speech: {speech_duration:.1f}s | Silence removed: {removed:.1f}s"
        )

        return speech_audio, speech_duration

    except ImportError:
        logger.warning("PyTorch not installed — skipping Silero VAD. Install: pip install torch")
        return audio, len(audio) / sr
    except Exception as e:
        logger.warning(f"VAD failed ({e}). Using full audio without VAD.")
        return audio, len(audio) / sr


def apply_vad_simple(audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> tuple[np.ndarray, float]:
    """
    Simple energy-based VAD fallback (no PyTorch required).
    Removes frames below threshold_db energy.

    Args:
        audio: Input audio array
        sr: Sample rate
        threshold_db: Energy threshold below which audio is considered silence

    Returns:
        (speech_audio, speech_duration_seconds)
    """
    logger.info("Applying simple energy-based VAD (fallback)...")

    frame_size = int(sr * 0.025)  # 25ms frames
    hop_size = int(sr * 0.010)    # 10ms hop

    # Build a per-sample boolean mask: mark samples in speech frames as True
    speech_mask = np.zeros(len(audio), dtype=bool)

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        energy = np.sqrt(np.mean(frame ** 2))
        energy_db = 20 * np.log10(energy + 1e-10)
        if energy_db > threshold_db:
            speech_mask[i:i + frame_size] = True

    speech_audio = audio[speech_mask]

    if len(speech_audio) == 0:
        logger.warning("Simple VAD found no speech. Using full audio.")
        return audio, len(audio) / sr

    speech_duration = len(speech_audio) / sr
    total_duration = len(audio) / sr
    logger.info(f"Simple VAD: kept {speech_duration:.1f}s of speech from {total_duration:.1f}s")

    return speech_audio, speech_duration


# ──────────────────────────────────────────
# STEP 4: Speech-to-Text (faster-whisper)
# ──────────────────────────────────────────
def transcribe_with_whisper(
    audio: np.ndarray,
    sr: int,
    model_size: str = "medium",
    language: Optional[str] = None,
    device: str = "auto",
    compute_type: str = "auto"
) -> dict:
    """
    Transcribe audio using faster-whisper (CTranslate2 backend).

    Supports all Whisper model sizes:
        tiny, base, small, medium, large-v2, large-v3
        For Indian languages: medium or large-v3 recommended.

    Args:
        audio: Input audio (16kHz float32 numpy array)
        sr: Sample rate (should be 16000)
        model_size: Whisper model size ("medium" recommended for Indian languages)
        language: Force language (None = auto-detect). Use "hi" for Hindi, "ta" for Tamil etc.
        device: "cpu", "cuda", or "auto"
        compute_type: "int8" (faster/smaller), "float16" (GPU), "float32" (CPU accurate)

    Returns:
        dict with transcript, language, confidence, segments
    """
    start_time = time.time()

    # Auto-detect device and compute type
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "int8" if device == "cpu" else "float16"

    logger.info(f"Loading Whisper model: '{model_size}' | Device: {device} | Compute: {compute_type}")

    try:
        # Load model (cached after first download)
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=str(Path.home() / ".cache" / "voicetriage" / "whisper")
        )

        # Save audio to temp file (faster-whisper accepts file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr, subtype="PCM_16")
            tmp_path = tmp.name

        # Transcription options optimized for Indian languages
        segments_gen, info = model.transcribe(
            tmp_path,
            language=language,           # None = auto-detect
            beam_size=5,                 # Higher = more accurate, slower
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Fallback if confidence low
            condition_on_previous_text=True,
            vad_filter=False,            # We already did VAD
            word_timestamps=False,
            initial_prompt=(             # Hint the model for Indian medical context
                "यह एक मरीज़ की स्वास्थ्य समस्या की बातचीत है। "
                "This is a patient describing their health problem in Hindi or regional Indian language."
            )
        )

        # Collect all segments
        segments = []
        full_transcript = ""
        total_confidence = 0.0

        for seg in segments_gen:
            segments.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4)
            })
            full_transcript += seg.text + " "
            # Convert log-probability to confidence (rough approximation)
            total_confidence += np.exp(seg.avg_logprob)

        # Clean up temp file
        os.unlink(tmp_path)

        # Compute average confidence
        avg_confidence = total_confidence / max(len(segments), 1)
        avg_confidence = min(max(avg_confidence, 0.0), 1.0)

        elapsed = time.time() - start_time
        detected_lang = info.language if info.language else "unknown"
        lang_prob = info.language_probability if info.language_probability else 0.0

        logger.info(
            f"Transcription complete in {elapsed:.2f}s | "
            f"Language: {detected_lang} (confidence: {lang_prob:.1%}) | "
            f"Segments: {len(segments)}"
        )

        return {
            "transcript": full_transcript.strip(),
            "detected_language": detected_lang,
            "language_probability": round(lang_prob, 3),
            "confidence": round(avg_confidence, 3),
            "segments": segments,
            "model_used": f"faster-whisper-{model_size}",
            "processing_time": round(elapsed, 3),
            "success": True,
            "error": None
        }

    except Exception as e:
        logger.error(f"Whisper transcription failed: {e}", exc_info=True)
        return {
            "transcript": "",
            "detected_language": "unknown",
            "language_probability": 0.0,
            "confidence": 0.0,
            "segments": [],
            "model_used": f"faster-whisper-{model_size}",
            "processing_time": time.time() - start_time,
            "success": False,
            "error": str(e)
        }


# ──────────────────────────────────────────
# MAIN PIPELINE ENTRY POINT
# ──────────────────────────────────────────
def process_audio(
    audio_file_path: str,
    whisper_model_size: str = "medium",
    force_language: Optional[str] = None,
    apply_noise_reduce: bool = True,
    apply_vad: bool = True,
    vad_backend: str = "silero",       # "silero" | "simple"
    noise_reduction_strength: float = 0.75,
    device: str = "auto"
) -> STTResult:
    """
    Full Stage 1 Speech Processing Pipeline.

    Args:
        audio_file_path    : Path to patient's voice recording
        whisper_model_size : Whisper model size (tiny|base|small|medium|large-v3)
        force_language     : Override language detection ("hi", "ta", "mr", etc.)
        apply_noise_reduce : Apply spectral noise reduction
        apply_vad          : Apply Voice Activity Detection
        vad_backend        : "silero" (better) or "simple" (no PyTorch needed)
        noise_reduction_strength: 0.0 = none, 1.0 = aggressive (default 0.75)
        device             : "cpu" | "cuda" | "auto"

    Returns:
        STTResult dataclass with transcript and all metadata
    """
    pipeline_start = time.time()

    logger.info(f"\n{'='*55}")
    logger.info(f"  🎙️  STAGE 1: SPEECH PROCESSING STARTING")
    logger.info(f"  📁  File: {Path(audio_file_path).name}")
    logger.info(f"{'='*55}")

    noise_reduction_applied = False
    vad_applied = False

    # ── 1. Load Audio ─────────────────────────────────
    logger.info("[ STEP 1/4 ] Loading audio...")
    try:
        audio, sr = load_audio(audio_file_path, target_sr=16000)
        audio_duration = len(audio) / sr
        logger.info(f"Audio loaded: {audio_duration:.2f}s @ {sr}Hz")
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return STTResult(
            transcript="", detected_language="unknown",
            detected_language_name="Unknown", confidence=0.0,
            audio_duration_seconds=0.0, speech_duration_seconds=0.0,
            sample_rate=0, model_used=f"faster-whisper-{whisper_model_size}",
            noise_reduction_applied=False, vad_applied=False,
            processing_time_seconds=time.time() - pipeline_start,
            segments=[], success=False,
            error_message=f"Audio load failed: {str(e)}"
        )

    # ── 2. Noise Reduction ────────────────────────────
    if apply_noise_reduce and audio_duration > 0.5:
        logger.info("[ STEP 2/4 ] Applying noise reduction...")
        audio = apply_noise_reduction(audio, sr, prop_decrease=noise_reduction_strength)
        noise_reduction_applied = True
    else:
        logger.info("[ STEP 2/4 ] Noise reduction skipped.")

    # ── 3. Voice Activity Detection ───────────────────
    speech_duration = audio_duration
    if apply_vad:
        logger.info(f"[ STEP 3/4 ] Applying VAD (backend: {vad_backend})...")
        if vad_backend == "silero":
            audio, speech_duration = apply_vad_silero(audio, sr)
        else:
            audio, speech_duration = apply_vad_simple(audio, sr)
        vad_applied = True
    else:
        logger.info("[ STEP 3/4 ] VAD skipped.")

    # ── 4. Speech-to-Text ──────────────────────────────
    logger.info(f"[ STEP 4/4 ] Transcribing with Whisper ({whisper_model_size})...")
    stt_result = transcribe_with_whisper(
        audio, sr,
        model_size=whisper_model_size,
        language=force_language,
        device=device
    )

    total_time = time.time() - pipeline_start
    detected_lang = stt_result["detected_language"]
    lang_name = LANGUAGE_NAMES.get(detected_lang, f"Unknown ({detected_lang})")

    # Build final result
    result = STTResult(
        transcript=stt_result["transcript"],
        detected_language=detected_lang,
        detected_language_name=lang_name,
        confidence=stt_result["confidence"],
        audio_duration_seconds=round(audio_duration, 2),
        speech_duration_seconds=round(speech_duration, 2),
        sample_rate=sr,
        model_used=stt_result["model_used"],
        noise_reduction_applied=noise_reduction_applied,
        vad_applied=vad_applied,
        processing_time_seconds=round(total_time, 3),
        segments=stt_result["segments"],
        success=stt_result["success"],
        error_message=stt_result["error"]
    )

    logger.info(result.summary())
    return result


# ──────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="VoiceTriage Stage 1 — Speech Processing Layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stage1_stt.py audio/patient_hindi.wav
  python stage1_stt.py audio/patient.mp3 --model large-v3 --language hi
  python stage1_stt.py audio/patient.wav --no-vad --output result.json
  python stage1_stt.py audio/patient.wav --vad-backend simple --model small
        """
    )

    parser.add_argument("audio_file",
                        help="Path to audio file (WAV, MP3, M4A, FLAC, OGG)")
    parser.add_argument("--model", "-m", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--language", "-l", default=None,
                        help="Force language code (e.g. 'hi', 'ta', 'mr'). Default: auto-detect")
    parser.add_argument("--no-noise-reduction", action="store_true",
                        help="Skip noise reduction step")
    parser.add_argument("--no-vad", action="store_true",
                        help="Skip Voice Activity Detection")
    parser.add_argument("--vad-backend", default="silero",
                        choices=["silero", "simple"],
                        help="VAD backend (silero=better, simple=no PyTorch needed)")
    parser.add_argument("--noise-strength", type=float, default=0.75,
                        help="Noise reduction strength (0.0-1.0, default: 0.75)")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Compute device (default: auto)")
    parser.add_argument("--output", "-o", default=None,
                        help="Save JSON result to this file path")
    parser.add_argument("--transcript-only", action="store_true",
                        help="Print only the transcript text (no metadata)")

    args = parser.parse_args()

    # Check file exists
    if not Path(args.audio_file).exists():
        logger.error(f"Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Run pipeline
    result = process_audio(
        audio_file_path=args.audio_file,
        whisper_model_size=args.model,
        force_language=args.language,
        apply_noise_reduce=not args.no_noise_reduction,
        apply_vad=not args.no_vad,
        vad_backend=args.vad_backend,
        noise_reduction_strength=args.noise_strength,
        device=args.device
    )

    if args.transcript_only:
        print(result.transcript)
    else:
        print(result.to_json())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        logger.info(f"Result saved to: {out_path}")

    sys.exit(0 if result.success else 1)
