"""Stage 3: retrieve triage knowledge relevant to extracted symptoms.

Phase 1 deliberately uses transparent local JSON keyword retrieval. This layer
does not decide urgency; it returns matching rule evidence for Stage 4.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger("Stage3-RAG")
VALID_URGENCY = {"ROUTINE", "URGENT", "EMERGENCY"}


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Knowledge base must contain a JSON object: {source}")
    return value


def _terms(values: Iterable[Any]) -> list[str]:
    terms = []
    for value in values:
        if isinstance(value, str) and value.strip():
            term = _normalize(value)
            if term and term not in terms:
                terms.append(term)
    return terms


def _normalize(text: str) -> str:
    text = text.lower().replace("_", " ").replace("-", " ")
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def _phrase_matches(query: str, candidate: str) -> bool:
    """Match whole normalized phrases; avoid accidental partial-word hits."""
    if not query or not candidate:
        return False
    return f" {query} " in f" {candidate} " or f" {candidate} " in f" {query} "


def _make_rule(rule_id: str, name: str, urgency: str, specialty: str,
               reasoning: str, matched_on: list[str]) -> dict[str, Any]:
    urgency = urgency.upper()
    return {
        "rule_id": rule_id,
        "rule_name": name,
        "urgency": urgency if urgency in VALID_URGENCY else "ROUTINE",
        "specialty": specialty or "GeneralMedicine",
        "reasoning": reasoning,
        "matched_on": matched_on[:5],
    }


def retrieve_knowledge(
    extraction: Any,
    *,
    red_flags_path: str | Path,
    specialty_path: str | Path,
    guidelines_path: str | Path,
    top_k: int = 3,
) -> dict[str, Any]:
    """Return matched rules, specialties, urgency hints, and readable context.

    ``extraction`` may be a Stage2 dataclass or a mapping with ``symptoms``,
    ``red_flags_mentioned``, and ``success`` fields.
    """
    if isinstance(extraction, dict):
        get_value = extraction.get
    else:
        get_value = lambda key, default=None: getattr(extraction, key, default)
    if not get_value("success", False):
        return {
            "retrieved_rules": [], "retrieved_specialties": [],
            "retrieved_urgency_hints": [], "rag_context_text": "No successful extraction to query.",
            "rag_enabled": True, "success": False,
        }

    symptoms = _terms(get_value("symptoms", []) or [])
    flags = _terms(get_value("red_flags_mentioned", []) or [])
    queries = list(dict.fromkeys(symptoms + flags))
    red_flag_kb = _read_json(red_flags_path)
    specialty_kb = _read_json(specialty_path)
    guideline_kb = _read_json(guidelines_path)

    rules: list[dict[str, Any]] = []
    urgency_hints: list[str] = []
    retrieved_specialties: list[str] = []

    # Explicit emergency/urgent rules are retrieved from their trigger phrases.
    for rule_name, rule in red_flag_kb.get("red_flags", {}).items():
        triggers = _terms(rule.get("symptoms_en", []))
        matched = [q for q in queries if any(_phrase_matches(q, trigger) for trigger in triggers)]
        if matched:
            item = _make_rule(
                f"RF-{rule_name.upper()}", rule_name,
                rule.get("urgency", "ROUTINE"), rule.get("specialty", "GeneralMedicine"),
                rule.get("reasoning", ""), matched,
            )
            rules.append(item)
            urgency_hints.append(item["urgency"])
            if item["specialty"] not in retrieved_specialties:
                retrieved_specialties.append(item["specialty"])

    # Rank likely specialties by matched English/Hinglish keywords.
    scores: dict[str, float] = {}
    for specialty, data in specialty_kb.get("specialties", {}).items():
        keywords = _terms(data.get("keywords_en", []) + data.get("keywords_hi", []))
        score = 0.0
        for query in queries:
            if any(_phrase_matches(query, keyword) for keyword in keywords):
                score += float(data.get("urgency_weight", 0.5))
        if score:
            scores[specialty] = score
    for specialty, _score in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:max(1, top_k)]:
        if specialty not in retrieved_specialties:
            retrieved_specialties.append(specialty)

    # Guidelines are prose, so match extracted phrases against explicit criteria.
    for protocol in guideline_kb.get("protocols", []):
        criteria = _normalize(protocol.get("criteria", ""))
        matched = [q for q in queries if _phrase_matches(q, criteria)]
        if matched:
            item = _make_rule(
                protocol.get("protocol_id", ""), protocol.get("title", ""),
                protocol.get("urgency", "ROUTINE"), protocol.get("specialty", "GeneralMedicine"),
                protocol.get("criteria", ""), matched,
            )
            rules.append(item)
            if item["urgency"] != "ROUTINE" and item["urgency"] not in urgency_hints:
                urgency_hints.append(item["urgency"])
            if item["specialty"] not in retrieved_specialties:
                retrieved_specialties.append(item["specialty"])

    # Keep evidence ordering deterministic and avoid duplicate rules.
    unique_rules = []
    seen_ids = set()
    for rule in rules:
        if rule["rule_id"] not in seen_ids:
            unique_rules.append(rule)
            seen_ids.add(rule["rule_id"])
    context_lines = [
        f"Rule {r['rule_id']}: {r['rule_name']} -> {r['urgency']} ({r['specialty']}); "
        f"matched: {', '.join(r['matched_on'])}. {r['reasoning']}"
        for r in unique_rules[:max(1, top_k + 2)]
    ]
    return {
        "retrieved_rules": unique_rules[:max(1, top_k + 2)],
        "retrieved_specialties": retrieved_specialties[:max(1, top_k)],
        "retrieved_urgency_hints": urgency_hints,
        "rag_context_text": "\n".join(context_lines) if context_lines else "No specific rules matched.",
        "rag_enabled": True,
        "success": True,
    }
