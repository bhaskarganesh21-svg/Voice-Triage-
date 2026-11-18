"""Stage 4: apply deterministic triage rules to extraction and RAG evidence.

This module produces an advisory suggestion for human review. It is not a
diagnosis and does not recommend treatment.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("Stage4-Decision")
URGENCY_RANK = {"ROUTINE": 0, "URGENT": 1, "EMERGENCY": 2}
URGENCY_COLOR = {"EMERGENCY": "RED", "URGENT": "YELLOW", "ROUTINE": "GREEN"}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.lower().replace("_", " ").replace("-", " "), re.UNICODE))


def _matches(query: str, phrase: str) -> bool:
    query, phrase = _norm(query), _norm(phrase)
    if not query or not phrase:
        return False
    # Prevent generic one-word fragments such as "pain" matching every rule.
    if len(query.split()) == 1:
        return query == phrase or (len(query) >= 8 and f" {query} " in f" {phrase} ")
    return f" {query} " in f" {phrase} " or f" {phrase} " in f" {query} "


def _explanation(urgency: str, specialty: str, extraction: Any,
                 red_flags: list[str], composites: list[str], evidence: list[dict]) -> str:
    symptoms = _get(extraction, "symptoms", []) or []
    parts = []
    if symptoms:
        parts.append(f"Patient described: {', '.join(symptoms[:5])}.")
    duration = _get(extraction, "duration", "unknown")
    severity = _get(extraction, "severity", "unknown")
    if duration and duration != "unknown":
        parts.append(f"Reported duration: {duration}.")
    if severity and severity != "unknown":
        parts.append(f"Reported severity: {severity}.")
    diseases = _get(extraction, "existing_diseases", []) or []
    if diseases:
        parts.append(f"Reported existing conditions: {', '.join(diseases[:5])}.")
    if red_flags:
        parts.append(f"Red-flag rules matched: {', '.join(red_flags[:3])}.")
    if composites:
        parts.append(f"Combined symptom rules matched: {', '.join(composites[:3])}.")
    if evidence and not red_flags and not composites:
        names = [rule.get("rule_name", rule.get("rule_id", "rule")) for rule in evidence[:2]]
        parts.append(f"Knowledge-base guidance matched: {', '.join(names)}.")
    if not symptoms and not red_flags:
        parts.append("No symptoms or red flags were extracted; this suggestion needs human review.")
    if urgency == "EMERGENCY":
        parts.append(f"Suggested EMERGENCY review with {specialty} immediately.")
    elif urgency == "URGENT":
        parts.append(f"Suggested URGENT review with {specialty}.")
    else:
        parts.append(f"Suggested ROUTINE review with {specialty}.")
    parts.append("This is an advisory triage suggestion; a clinician or operator must make the final decision.")
    return " ".join(parts)


def decide(
    extraction: Any,
    rag_result: Any,
    *,
    red_flags_path: str | Path,
    emergency_confidence_boost: float = 0.10,
    composite_rule_boost: float = 0.08,
) -> dict[str, Any]:
    """Return Stage 4's stable decision schema from Stage 2 and Stage 3 data."""
    started = time.perf_counter()
    kb_path = Path(red_flags_path)
    with kb_path.open("r", encoding="utf-8") as handle:
        kb = json.load(handle)
    if not isinstance(kb, dict):
        raise ValueError(f"Red-flag knowledge base must be an object: {kb_path}")

    symptoms = [s for s in (_get(extraction, "symptoms", []) or []) if isinstance(s, str)]
    extracted_flags = [s for s in (_get(extraction, "red_flags_mentioned", []) or []) if isinstance(s, str)]
    queries = symptoms + extracted_flags
    evidence = _get(rag_result, "retrieved_rules", []) or []
    urgency_hints = _get(rag_result, "retrieved_urgency_hints", []) or []
    specialties = _get(rag_result, "retrieved_specialties", []) or []

    urgency = "ROUTINE"
    specialty = specialties[0] if specialties else "GeneralMedicine"
    matched_rule_ids: list[str] = []
    red_flags_found: list[str] = []
    composite_found: list[str] = []
    confidence = _get(extraction, "extraction_confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0

    def raise_urgency(level: str, suggested_specialty: str = "") -> None:
        nonlocal urgency, specialty
        level = str(level).upper()
        if level not in URGENCY_RANK:
            return
        old_rank = URGENCY_RANK[urgency]
        new_rank = URGENCY_RANK[level]
        if new_rank > old_rank:
            urgency = level
        if suggested_specialty and new_rank >= old_rank and specialty in {"", "GeneralMedicine"}:
            specialty = suggested_specialty

    for hint in urgency_hints:
        raise_urgency(hint)
    for rule in evidence:
        rule_id = rule.get("rule_id") if isinstance(rule, dict) else None
        if rule_id and rule_id not in matched_rule_ids:
            matched_rule_ids.append(rule_id)

    # Recheck red flags from source data so Stage 4 remains the safety layer,
    # even if retrieval omitted a rule. Match explicit phrases only.
    for rule_name, rule in kb.get("red_flags", {}).items():
        triggers = rule.get("symptoms_en", []) + rule.get("symptoms_hi", [])
        hits = [query for query in queries if any(_matches(query, trigger) for trigger in triggers)]
        if hits:
            label = f"{rule_name} (triggered by: {', '.join(dict.fromkeys(hits[:2]))})"
            red_flags_found.append(label)
            rule_id = f"RF-{rule_name.upper()}"
            if rule_id not in matched_rule_ids:
                matched_rule_ids.append(rule_id)
            level = rule.get("urgency", "ROUTINE")
            raise_urgency(level, rule.get("specialty", ""))
            confidence = min(1.0, confidence + emergency_confidence_boost)

    for rule in kb.get("composite_rules", {}).get("rules", []):
        triggers = rule.get("trigger_symptoms", [])
        hits = [trigger for trigger in triggers if any(_matches(query, trigger) for query in symptoms)]
        required = max(1, int(rule.get("required_count", 2)))
        if len(hits) >= required:
            name = rule.get("name", rule.get("rule_id", "Combined symptom rule"))
            composite_found.append(name)
            rule_id = rule.get("rule_id")
            if rule_id and rule_id not in matched_rule_ids:
                matched_rule_ids.append(rule_id)
            raise_urgency(rule.get("escalate_to", "ROUTINE"), rule.get("specialty", ""))
            confidence = min(1.0, confidence + composite_rule_boost)

    # Clinical rules are allowed to raise urgency; reported mild severity must
    # never downgrade an explicit urgent or emergency rule.
    if str(_get(extraction, "severity", "unknown")).lower() == "severe" and urgency == "ROUTINE":
        urgency = "URGENT"
        confidence = min(1.0, confidence + 0.05)
    if urgency == "EMERGENCY" and specialty == "GeneralMedicine":
        specialty = "Emergency"
    if len(symptoms) >= 3:
        confidence = min(0.99, confidence + 0.05)
    if red_flags_found:
        confidence = min(0.99, confidence + 0.08)
    confidence = round(min(max(confidence, 0.30), 0.99), 2)

    return {
        "urgency_level": urgency,
        "urgency_color": URGENCY_COLOR[urgency],
        "suggested_specialty": specialty,
        "confidence_score": confidence,
        "red_flags_triggered": red_flags_found,
        "composite_rules_triggered": composite_found,
        "explanation": _explanation(urgency, specialty, extraction, red_flags_found, composite_found, evidence),
        "rule_ids_matched": matched_rule_ids,
        "processing_time_seconds": round(time.perf_counter() - started, 3),
    }
