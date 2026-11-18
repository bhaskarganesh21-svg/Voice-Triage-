"""
VoiceTriage Web Application — Flask Backend
============================================
Routes:
  GET  /                    -> Dashboard page
  POST /api/triage/text     -> Triage from text transcript
  POST /api/triage/audio    -> Triage from uploaded audio file
  GET  /api/demo/<case_id>  -> Run a demo synthetic case
  GET  /api/cases           -> List all demo cases
  GET  /api/history         -> Recent triage results
  GET  /api/stats           -> Aggregate stats

Run:
  cd voicetriage
  python webapp/app.py
  # Open http://localhost:5000
"""

import sys
import json
import uuid
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from main import (
    PatientInput, run_pipeline, run_demo,
    list_cases, _load_json, PATHS, CONFIG,
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VoiceTriage.WebApp")

# ── In-memory history (replace with SQLite for production) ───────────────────
_history: list = []
MAX_HISTORY = 100


# ==============================================================================
#  HELPER
# ==============================================================================

def result_to_response(result) -> dict:
    """Convert TriageResult -> JSON-serializable dict for the API."""
    d  = result.stage4_decision
    s1 = result.stage1_stt
    s2 = result.stage2_extraction
    s3 = result.stage3_rag

    return {
        "success":            True,
        "case_id":            result.case_id,
        "timestamp":          result.timestamp,
        "total_time_seconds": result.total_time_seconds,
        # Core output fields
        "urgency_level":      d.urgency_level,
        "urgency_color":      d.urgency_color,
        "suggested_specialty":d.suggested_specialty,
        "confidence_score":   d.confidence_score,
        "explanation":        d.explanation,
        # Patient data
        "transcript":         s1.transcript,
        "detected_language":  s1.detected_language_name,
        "symptoms":           s2.symptoms,
        "duration":           s2.duration,
        "severity":           s2.severity,
        "existing_diseases":  s2.existing_diseases,
        "age":                s2.age,
        "gender":             s2.gender,
        # Evidence
        "red_flags_triggered":d.red_flags_triggered,
        "composite_rules":    d.composite_rules_triggered,
        "rule_ids_matched":   d.rule_ids_matched,
        "rag_rules_count":    len(s3.retrieved_rules),
        # Pipeline info
        "stt_model":          s1.model_used,
        "llm_model":          s2.model_used,
        "stt_time":           s1.processing_time_seconds,
        "llm_time":           s2.processing_time_seconds,
        "decision_time":      d.processing_time_seconds,
    }


def store_history(resp: dict) -> None:
    _history.insert(0, resp)
    if len(_history) > MAX_HISTORY:
        _history.pop()


# ==============================================================================
#  ROUTES
# ==============================================================================

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/triage/text", methods=["POST"])
def triage_text():
    """
    Triage from a text transcript.
    Body: { "transcript": "...", "language": "hi" (optional), "case_id": "..." (optional) }
    """
    data = request.get_json(force=True, silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    if not transcript:
        return jsonify({"success": False, "error": "No transcript provided"}), 400

    case_id  = data.get("case_id") or f"API_{uuid.uuid4().hex[:8].upper()}"
    language = data.get("language") or "hi"

    try:
        result = run_pipeline(PatientInput(
            case_id=case_id,
            raw_transcript=transcript,
            language_hint=language,
        ))
        resp = result_to_response(result)
        store_history(resp)
        return jsonify(resp)
    except Exception as e:
        logger.error(f"Triage error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/triage/audio", methods=["POST"])
def triage_audio():
    """
    Triage from audio (recorded via microphone or uploaded file).
    Form-data: file=<audio>, language=hi (optional), transcript=<optional>
    """
    if "file" not in request.files and "transcript" not in request.form:
        return jsonify({"success": False, "error": "No voice recording or transcript provided"}), 400

    language = request.form.get("language", "hi")
    client_transcript = (request.form.get("transcript") or "").strip()

    # If audio file is provided
    if "file" in request.files and request.files["file"].filename:
        audio_file = request.files["file"]
        ext = Path(audio_file.filename).suffix.lower() if audio_file.filename else ".wav"
        if not ext:
            ext = ".wav"

        if ext not in (".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"):
            ext = ".wav"

        PATHS["audio_dir"].mkdir(parents=True, exist_ok=True)
        case_id = f"VOICE_{uuid.uuid4().hex[:8].upper()}"
        save_path = PATHS["audio_dir"] / f"{case_id}{ext}"

        try:
            audio_file.save(str(save_path))

            # Run full voice pipeline with Whisper STT
            try:
                result = run_pipeline(PatientInput(
                    case_id=case_id,
                    audio_file=str(save_path),
                    language_hint=language,
                ))
            except Exception as stt_err:
                # If audio loading fails (e.g. webm format without ffmpeg) and we have client_transcript, fallback gracefully!
                if client_transcript:
                    logger.warning(f"Audio STT error ({stt_err}), falling back to live speech transcript: {client_transcript}")
                    result = run_pipeline(PatientInput(
                        case_id=case_id,
                        raw_transcript=client_transcript,
                        language_hint=language,
                    ))
                else:
                    raise stt_err

            resp = result_to_response(result)
            store_history(resp)
            return jsonify(resp)
        except Exception as e:
            logger.error(f"Voice triage error: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    # If only client speech transcript is provided
    if client_transcript:
        case_id = f"VOICE_{uuid.uuid4().hex[:8].upper()}"
        try:
            result = run_pipeline(PatientInput(
                case_id=case_id,
                raw_transcript=client_transcript,
                language_hint=language,
            ))
            resp = result_to_response(result)
            resp["stt_model"] = "Direct Voice (Live Speech)"
            store_history(resp)
            return jsonify(resp)
        except Exception as e:
            logger.error(f"Live voice triage error: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500

    return jsonify({"success": False, "error": "Invalid voice data"}), 400


@app.route("/api/triage/voice", methods=["POST"])
def triage_voice():
    """Alias for direct voice triage endpoint."""
    return triage_audio()


@app.route("/api/demo/<case_id>", methods=["GET"])
def demo_case(case_id: str):
    """Run a specific demo case by ID."""
    try:
        result = run_demo(case_id=case_id.upper())
        resp = result_to_response(result)
        store_history(resp)
        return jsonify(resp)
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/demo", methods=["GET"])
def demo_default():
    """Run the default demo case."""
    try:
        result = run_demo()
        resp = result_to_response(result)
        store_history(resp)
        return jsonify(resp)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/cases", methods=["GET"])
def get_cases():
    """Return list of all demo cases."""
    all_cases = []
    for path in [PATHS["cases_hindi"], PATHS["cases_regional"]]:
        if path.exists():
            all_cases.extend(_load_json(path).get("cases", []))

    cases_summary = []
    for c in all_cases:
        gt = c.get("ground_truth", {})
        cases_summary.append({
            "case_id":        c["case_id"],
            "language":       c["language"],
            "language_name":  c["language_name"],
            "urgency":        gt.get("urgency", "?"),
            "specialty":      gt.get("specialty", "?"),
            "symptoms":       gt.get("symptoms", [])[:4],
            "transcript_preview": c.get("transcript_english", c.get("transcript", ""))[:80] + "...",
        })
    return jsonify({"success": True, "count": len(cases_summary), "cases": cases_summary})


@app.route("/api/history", methods=["GET"])
def get_history():
    """Return recent triage results."""
    limit = min(int(request.args.get("limit", 20)), 50)
    return jsonify({"success": True, "count": len(_history), "results": _history[:limit]})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Return aggregate stats over recent history."""
    if not _history:
        return jsonify({"success": True, "total": 0, "stats": {}})

    urgency_counts = {"EMERGENCY": 0, "URGENT": 0, "ROUTINE": 0}
    specialty_counts: dict = {}
    total_confidence = 0.0

    for r in _history:
        urg = r.get("urgency_level", "ROUTINE")
        urgency_counts[urg] = urgency_counts.get(urg, 0) + 1
        spec = r.get("suggested_specialty", "GeneralMedicine")
        specialty_counts[spec] = specialty_counts.get(spec, 0) + 1
        total_confidence += r.get("confidence_score", 0)

    return jsonify({
        "success":      True,
        "total":        len(_history),
        "urgency":      urgency_counts,
        "top_specialties": dict(sorted(specialty_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
        "avg_confidence": round(total_confidence / len(_history), 3),
    })


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "pipeline_version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "ollama_model": CONFIG["ollama_model"],
        "whisper_model": CONFIG["whisper_model"],
    })


if __name__ == "__main__":
    port = 5000
    host = "127.0.0.1"
    print()
    print("=" * 60)
    print(f"  VoiceTriage Web App")
    print(f"  http://{host}:{port}")
    print("=" * 60)
    print()
    app.run(debug=True, host=host, port=port, use_reloader=False)
