"""Stage 5: format a completed triage run for operators and save JSON output."""

from __future__ import annotations

import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("Stage5-Output")

URGENCY_DISPLAY = {
    "EMERGENCY": {"icon": "[EMERGENCY]", "border": "!"},
    "URGENT": {"icon": "[URGENT]", "border": "*"},
    "ROUTINE": {"icon": "[ROUTINE]", "border": "-"},
}


def _field(obj: Any, name: str, default: Any = None) -> Any:
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def result_to_dict(result: Any) -> dict[str, Any]:
    """Convert the pipeline result to its public JSON schema."""
    if hasattr(result, "to_dict"):
        return result.to_dict()
    if isinstance(result, dict):
        return result
    raise TypeError("result must provide to_dict() or be a dictionary")


def render_summary(result: Any, *, width: int = 70) -> str:
    """Render a readable terminal summary without changing the result object."""
    decision = _field(result, "stage4_decision")
    extraction = _field(result, "stage2_extraction")
    stt = _field(result, "stage1_stt")
    if decision is None or extraction is None or stt is None:
        raise ValueError("Full pipeline result is missing a required stage output")
    urgency = str(_field(decision, "urgency_level", "ROUTINE")).upper()
    display = URGENCY_DISPLAY.get(urgency, URGENCY_DISPLAY["ROUTINE"])
    case_id = _field(result, "case_id", "UNKNOWN")
    timestamp = _field(result, "timestamp", "")
    transcript = _field(stt, "transcript", "") or ""
    symptoms = _field(extraction, "symptoms", []) or []
    diseases = _field(extraction, "existing_diseases", []) or []
    red_flags = _field(decision, "red_flags_triggered", []) or []
    rule_ids = _field(decision, "rule_ids_matched", []) or []
    explanation = _field(decision, "explanation", "") or ""
    age = _field(extraction, "age")
    transcript_lines = textwrap.wrap(transcript, width=width - 4) if transcript else ["  (No transcript available)"]
    explanation_lines = textwrap.wrap(explanation, width=width - 4) if explanation else ["  No explanation available."]

    lines = [
        "=" * width,
        "VOICETRIAGE - ADVISORY TRIAGE SUMMARY",
        f"Case ID: {case_id}    Timestamp: {timestamp}",
        "=" * width,
        f"{display['icon']}  {urgency}  ->  {_field(decision, 'suggested_specialty', 'GeneralMedicine')}",
        "",
        f"Transcript [{_field(stt, 'detected_language_name', 'Unknown')} | {_field(stt, 'model_used', 'unknown')}]:",
        *transcript_lines,
        "",
        "EXTRACTED INFORMATION",
        f"Symptoms: {', '.join(symptoms) if symptoms else 'None extracted'}",
        f"Duration: {_field(extraction, 'duration', 'unknown')}",
        f"Severity: {_field(extraction, 'severity', 'unknown')}",
        f"Existing conditions: {', '.join(diseases) if diseases else 'None mentioned'}",
        f"Age: {age if age is not None else 'Not mentioned'}",
        f"Gender: {_field(extraction, 'gender', 'unknown')}",
        "",
        "DECISION EVIDENCE",
        f"Confidence: {_field(decision, 'confidence_score', 0.0):.0%}",
        f"Red-flag rules: {', '.join(red_flags) if red_flags else 'None matched'}",
        f"Rules matched: {', '.join(rule_ids) if rule_ids else 'None'}",
        "",
        "EXPLANATION",
        *explanation_lines,
        "",
        "Advisory only: a clinician or operator must review and make the final decision.",
        "=" * width,
    ]
    return "\n".join(lines)


def save_result(result: Any, output_dir: str | Path) -> Path:
    """Write the stable JSON result and return its path."""
    data = result_to_dict(result)
    case_id = str(data.get("case_id", "CASE"))
    timestamp = str(data.get("timestamp", ""))
    safe_case_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id).strip("._") or "CASE"
    safe_timestamp = re.sub(r"[^A-Za-z0-9_.-]+", "-", timestamp).strip(".-") or "run"
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{safe_case_id}_{safe_timestamp}.json"
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Triage JSON saved to %s", output_path)
    return output_path


def present_result(
    result: Any,
    *,
    output_dir: str | Path,
    save_json: bool = True,
    print_summary: bool = True,
) -> Optional[Path]:
    """Display and optionally save the final Stage 5 result."""
    if print_summary:
        print(render_summary(result))
    if save_json:
        return save_result(result, output_dir)
    return None
