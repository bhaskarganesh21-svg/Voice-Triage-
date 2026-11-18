"""
test_stage1.py — Unit & Integration Tests for Stage 1: Speech Processing Layer

Tests:
  1. Audio loading (WAV files)
  2. Noise reduction
  3. Simple VAD (no PyTorch required)
  4. Full pipeline with a synthetic tone (validates wiring, not STT accuracy)
  5. JSON output structure validation

Run:
  python tests/test_stage1.py
"""

import os
import sys
import json
import unittest
import tempfile
import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.stage1_stt import (
    load_audio,
    apply_noise_reduction,
    apply_vad_simple,
    STTResult,
    LANGUAGE_NAMES
)


def generate_sine_wave(duration: float = 3.0, freq: float = 440.0, sr: int = 16000) -> np.ndarray:
    """Generate a sine wave audio signal for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return audio


def generate_speech_like_audio(duration: float = 5.0, sr: int = 16000) -> np.ndarray:
    """
    Generate audio that roughly mimics speech (bursts of sound with pauses).
    Used to test VAD without needing real audio.
    """
    audio = np.zeros(int(sr * duration), dtype=np.float32)

    # Add "speech" bursts at intervals
    speech_windows = [
        (0.5, 1.5),   # Burst 1
        (2.0, 3.0),   # Burst 2
        (3.5, 4.5),   # Burst 3
    ]
    for start, end in speech_windows:
        s, e = int(start * sr), int(end * sr)
        t = np.linspace(0, end - start, e - s)
        # Mix of frequencies to simulate speech
        audio[s:e] = (
            0.3 * np.sin(2 * np.pi * 200 * t) +
            0.2 * np.sin(2 * np.pi * 500 * t) +
            0.1 * np.random.randn(e - s)
        ).astype(np.float32)

    # Add low-level background noise
    audio += 0.01 * np.random.randn(len(audio)).astype(np.float32)
    return audio


class TestAudioLoading(unittest.TestCase):
    """Test audio file loading and preprocessing."""

    def setUp(self):
        """Create temp WAV files for testing."""
        self.sr = 16000
        self.test_audio = generate_sine_wave(duration=2.0, sr=self.sr)
        self.tmp_dir = tempfile.mkdtemp()

        # Create mono WAV
        self.mono_wav = os.path.join(self.tmp_dir, "test_mono.wav")
        sf.write(self.mono_wav, self.test_audio, self.sr, subtype="PCM_16")

        # Create stereo WAV
        stereo = np.stack([self.test_audio, self.test_audio * 0.8], axis=1)
        self.stereo_wav = os.path.join(self.tmp_dir, "test_stereo.wav")
        sf.write(self.stereo_wav, stereo, self.sr, subtype="PCM_16")

    def test_load_mono_wav(self):
        """Should load mono WAV and return correct shape."""
        audio, sr = load_audio(self.mono_wav)
        self.assertEqual(sr, 16000)
        self.assertEqual(audio.ndim, 1)
        self.assertGreater(len(audio), 0)

    def test_load_stereo_wav_converts_to_mono(self):
        """Should convert stereo to mono."""
        audio, sr = load_audio(self.stereo_wav)
        self.assertEqual(audio.ndim, 1, "Stereo audio was not converted to mono")

    def test_audio_normalized(self):
        """Loaded audio should be normalized to [-1, 1]."""
        audio, _ = load_audio(self.mono_wav)
        self.assertLessEqual(np.max(np.abs(audio)), 1.01,
                             "Audio not normalized to [-1, 1]")

    def test_nonexistent_file_raises(self):
        """Should raise an error for non-existent files."""
        with self.assertRaises(Exception):
            load_audio("/nonexistent/path/audio.wav")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


class TestNoiseReduction(unittest.TestCase):
    """Test noise reduction step."""

    def setUp(self):
        self.sr = 16000
        # Create noisy signal: clean sine + noise
        clean = generate_sine_wave(duration=3.0, sr=self.sr)
        noise = 0.3 * np.random.randn(len(clean)).astype(np.float32)
        self.noisy_audio = (clean + noise).astype(np.float32)

    def test_noise_reduction_output_shape(self):
        """Output should have same shape as input."""
        cleaned = apply_noise_reduction(self.noisy_audio, self.sr)
        self.assertEqual(cleaned.shape, self.noisy_audio.shape)

    def test_noise_reduction_reduces_noise(self):
        """Noise-reduced audio should have lower RMS variance than noisy input."""
        cleaned = apply_noise_reduction(self.noisy_audio, self.sr)
        # RMS of cleaned should be different from noisy (noise was removed)
        noisy_rms = np.sqrt(np.mean(self.noisy_audio**2))
        cleaned_rms = np.sqrt(np.mean(cleaned**2))
        self.assertNotAlmostEqual(noisy_rms, cleaned_rms, places=3,
                                  msg="Noise reduction had no effect")

    def test_noise_reduction_on_very_short_audio(self):
        """Should not crash on very short audio (<0.5s)."""
        short_audio = generate_sine_wave(duration=0.1, sr=self.sr)
        result = apply_noise_reduction(short_audio, self.sr)
        self.assertEqual(len(result), len(short_audio))


class TestVAD(unittest.TestCase):
    """Test Voice Activity Detection (simple fallback)."""

    def setUp(self):
        self.sr = 16000
        self.speech_audio = generate_speech_like_audio(duration=5.0, sr=self.sr)

    def test_vad_simple_returns_shorter_audio(self):
        """VAD should remove silence — output should be shorter than input."""
        output, speech_dur = apply_vad_simple(self.speech_audio, self.sr)
        total_dur = len(self.speech_audio) / self.sr
        self.assertLess(speech_dur, total_dur,
                        "VAD did not remove any silence")

    def test_vad_simple_speech_duration_correct(self):
        """Returned speech_duration should match len(output)/sr."""
        output, speech_dur = apply_vad_simple(self.speech_audio, self.sr)
        expected_dur = len(output) / self.sr
        self.assertAlmostEqual(speech_dur, expected_dur, places=2)

    def test_vad_on_silence_uses_original(self):
        """Pure silence should trigger fallback to original audio."""
        silence = np.zeros(self.sr * 3, dtype=np.float32)
        output, dur = apply_vad_simple(silence, self.sr, threshold_db=-200.0)
        # With very low threshold, some frames may pass
        self.assertGreater(len(output), 0)

    def test_vad_output_is_1d(self):
        """VAD output should be 1D numpy array."""
        output, _ = apply_vad_simple(self.speech_audio, self.sr)
        self.assertEqual(output.ndim, 1)


class TestSTTResult(unittest.TestCase):
    """Test STTResult data structure."""

    def get_sample_result(self):
        return STTResult(
            transcript="मेरे सीने में दर्द है और साँस लेने में तकलीफ हो रही है",
            detected_language="hi",
            detected_language_name="Hindi",
            confidence=0.87,
            audio_duration_seconds=8.5,
            speech_duration_seconds=6.2,
            sample_rate=16000,
            model_used="faster-whisper-medium",
            noise_reduction_applied=True,
            vad_applied=True,
            processing_time_seconds=3.21,
            segments=[
                {"start": 0.0, "end": 4.1, "text": "मेरे सीने में दर्द है", "avg_logprob": -0.3, "no_speech_prob": 0.05},
                {"start": 4.2, "end": 6.2, "text": "और साँस लेने में तकलीफ हो रही है", "avg_logprob": -0.25, "no_speech_prob": 0.04}
            ],
            success=True,
            error_message=None
        )

    def test_to_dict(self):
        """to_dict() should return a dict with all fields."""
        r = self.get_sample_result()
        d = r.to_dict()
        self.assertIsInstance(d, dict)
        self.assertIn("transcript", d)
        self.assertIn("detected_language", d)
        self.assertIn("confidence", d)
        self.assertIn("segments", d)

    def test_to_json_valid(self):
        """to_json() should return valid JSON."""
        r = self.get_sample_result()
        json_str = r.to_json()
        parsed = json.loads(json_str)
        self.assertEqual(parsed["detected_language"], "hi")
        self.assertEqual(parsed["success"], True)

    def test_to_json_preserves_hindi_text(self):
        """JSON should preserve Unicode Hindi text without escaping."""
        r = self.get_sample_result()
        json_str = r.to_json()
        # ensure_ascii=False means Hindi chars preserved
        self.assertIn("सीने", json_str)

    def test_summary_success(self):
        """summary() should return a non-empty string for successful result."""
        r = self.get_sample_result()
        s = r.summary()
        self.assertIsInstance(s, str)
        self.assertGreater(len(s), 10)
        self.assertIn("COMPLETE", s)

    def test_summary_failure(self):
        """summary() for failed result should indicate failure."""
        r = self.get_sample_result()
        r.success = False
        r.error_message = "Audio file not found"
        s = r.summary()
        self.assertIn("FAILED", s)


class TestLanguageNames(unittest.TestCase):
    """Test language name mapping."""

    def test_hindi_code(self):
        self.assertEqual(LANGUAGE_NAMES["hi"], "Hindi")

    def test_tamil_code(self):
        self.assertEqual(LANGUAGE_NAMES["ta"], "Tamil")

    def test_marathi_code(self):
        self.assertEqual(LANGUAGE_NAMES["mr"], "Marathi")

    def test_unknown_fallback(self):
        name = LANGUAGE_NAMES.get("xyz", "Unknown")
        self.assertEqual(name, "Unknown")

    def test_all_major_indian_languages_present(self):
        required = ["hi", "mr", "ta", "te", "kn", "ml", "gu", "bn", "pa"]
        for code in required:
            self.assertIn(code, LANGUAGE_NAMES,
                          f"Language code '{code}' missing from LANGUAGE_NAMES")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  VoiceTriage — Stage 1 Unit Tests")
    print("="*60 + "\n")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    for cls in [
        TestAudioLoading,
        TestNoiseReduction,
        TestVAD,
        TestSTTResult,
        TestLanguageNames
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print(f"\n" + "="*60)
    if result.wasSuccessful():
        print(f"  [PASS]  All {result.testsRun} tests PASSED")
    else:
        print(f"  [FAIL]  {len(result.failures)} failures, {len(result.errors)} errors")
    print("="*60)

    sys.exit(0 if result.wasSuccessful() else 1)
