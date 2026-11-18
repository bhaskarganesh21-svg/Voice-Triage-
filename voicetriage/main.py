"""
╔══════════════════════════════════════════════════════════════════════════╗
║       VoiceTriage -- AI-Powered Pre-Call Triage for eSanjeevani        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  STAGE 1 -- Speech Processing  (Noise Reduction + VAD + Whisper STT)  ║
║  STAGE 2 -- NLP Extraction     (Ollama LLM / keyword fallback)        ║
║  STAGE 3 -- RAG Cross-Check    (Knowledge Base rule retrieval)        ║
║  STAGE 4 -- Decision Engine    (Urgency + Specialty + Confidence)     ║
║  STAGE 5 -- Output             (JSON + Terminal Dashboard)            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  CLI USAGE:                                                            ║
║    python main.py --demo                 (run on synthetic case)      ║
║    python main.py --demo --case HI_001   (specific case)              ║
║    python main.py --list-cases           (show all demo cases)        ║
║    python main.py --audio file.wav       (real audio file)            ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import sys
import json
import time
import logging
import argparse
import textwrap
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

# ── Project Root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
#  STAGE 0 -- CONFIGURATION & SETUP
# ==============================================================================

# ── Logging (UTF-8 safe on Windows) ──────────────────────────────────────────
log_handlers = [
    logging.FileHandler(PROJECT_ROOT / "voicetriage.log", encoding="utf-8"),
]
# Add console handler with UTF-8 encoding to avoid cp1252 issues on Windows
try:
    console_handler = logging.StreamHandler(
        open(os.devnull, "w") if False else
        __import__("io").TextIOWrapper(
            __import__("sys").stdout.buffer, encoding="utf-8", errors="replace"
        )
    )
    log_handlers.append(console_handler)
except Exception:
    log_handlers.append(logging.StreamHandler(sys.stdout))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s -- %(message)s",
    datefmt="%H:%M:%S",
    handlers=log_handlers,
)
logger = logging.getLogger("VoiceTriage")


# ── File Paths ────────────────────────────────────────────────────────────────
PATHS = {
    "knowledge_base":        PROJECT_ROOT / "knowledge_base",
    "symptom_map":           PROJECT_ROOT / "knowledge_base" / "symptom_specialty_mapping.json",
    "red_flags":             PROJECT_ROOT / "knowledge_base" / "red_flag_symptoms.json",
    "guidelines":            PROJECT_ROOT / "knowledge_base" / "triage_guidelines.json",
    "cases_hindi":           PROJECT_ROOT / "data" / "synthetic_transcripts" / "cases_hindi.json",
    "cases_regional":        PROJECT_ROOT / "data" / "synthetic_transcripts" / "cases_regional.json",
    "output_dir":            PROJECT_ROOT / "output",
    "audio_dir":             PROJECT_ROOT / "audio",
}

# ── Pipeline Config ───────────────────────────────────────────────────────────
CONFIG = {
    # Stage 1
    "whisper_model":          "base",        # tiny | base | small | medium | large-v3
    "whisper_device":         "cpu",
    "noise_reduction":        True,
    "vad_enabled":            True,
    "noise_strength":         0.75,
    # Stage 2
    "ollama_model":           "llama3.2:3b",
    "ollama_host":            "http://localhost:11434",
    "llm_temperature":        0.1,
    "llm_timeout_seconds":    60,
    # Stage 3
    "rag_top_k":              3,
    # Stage 4
    "emergency_boost":        0.10,
    "composite_boost":        0.08,
    # Output
    "save_output_json":       True,
}


# ==============================================================================
#  DATA STRUCTURES  (shared across all stages)
# ==============================================================================

@dataclass
class PatientInput:
    case_id:         str            = "LIVE"
    audio_file:      Optional[str] = None
    raw_transcript:  Optional[str] = None   # Pre-supplied text (demo/API mode)
    language_hint:   Optional[str] = None   # "hi", "ta", etc.


@dataclass
class Stage1_STT:
    transcript:              str           = ""
    detected_language:       str           = "unknown"
    detected_language_name:  str           = "Unknown"
    confidence:              float         = 0.0
    audio_duration_seconds:  float         = 0.0
    speech_duration_seconds: float         = 0.0
    model_used:              str           = ""
    noise_reduction_applied: bool          = False
    vad_applied:             bool          = False
    processing_time_seconds: float         = 0.0
    segments:                List[dict]    = field(default_factory=list)
    success:                 bool          = False
    error_message:           Optional[str] = None


@dataclass
class Stage2_Extraction:
    symptoms:                  List[str]    = field(default_factory=list)
    duration:                  str          = "unknown"
    severity:                  str          = "unknown"
    existing_diseases:         List[str]    = field(default_factory=list)
    age:                       Optional[int]= None
    gender:                    str          = "unknown"
    patient_complaint_summary: str          = ""
    red_flags_mentioned:       List[str]    = field(default_factory=list)
    extraction_confidence:     float        = 0.0
    model_used:                str          = ""
    processing_time_seconds:   float        = 0.0
    success:                   bool         = False
    error_message:             Optional[str]= None


@dataclass
class Stage3_RAG:
    retrieved_rules:          List[dict]   = field(default_factory=list)
    retrieved_specialties:    List[str]    = field(default_factory=list)
    retrieved_urgency_hints:  List[str]    = field(default_factory=list)
    rag_context_text:         str          = ""
    rag_enabled:              bool         = False
    success:                  bool         = False


@dataclass
class Stage4_Decision:
    urgency_level:             str          = "ROUTINE"
    urgency_color:             str          = "GREEN"
    suggested_specialty:       str          = "GeneralMedicine"
    confidence_score:          float        = 0.0
    red_flags_triggered:       List[str]    = field(default_factory=list)
    composite_rules_triggered: List[str]    = field(default_factory=list)
    explanation:               str          = ""
    rule_ids_matched:          List[str]    = field(default_factory=list)
    processing_time_seconds:   float        = 0.0


@dataclass
class TriageResult:
    case_id:            str
    timestamp:          str
    patient_input:      PatientInput
    stage1_stt:         Stage1_STT
    stage2_extraction:  Stage2_Extraction
    stage3_rag:         Stage3_RAG
    stage4_decision:    Stage4_Decision
    total_time_seconds: float = 0.0
    pipeline_version:   str   = "1.0.0"

    def to_dict(self) -> dict:
        return {
            "case_id":            self.case_id,
            "timestamp":          self.timestamp,
            "pipeline_version":   self.pipeline_version,
            "total_time_seconds": self.total_time_seconds,
            "transcript":         self.stage1_stt.transcript,
            "detected_language":  self.stage1_stt.detected_language_name,
            "urgency_level":      self.stage4_decision.urgency_level,
            "urgency_color":      self.stage4_decision.urgency_color,
            "suggested_specialty":self.stage4_decision.suggested_specialty,
            "symptoms":           self.stage2_extraction.symptoms,
            "duration":           self.stage2_extraction.duration,
            "severity":           self.stage2_extraction.severity,
            "existing_diseases":  self.stage2_extraction.existing_diseases,
            "age":                self.stage2_extraction.age,
            "gender":             self.stage2_extraction.gender,
            "confidence_score":   self.stage4_decision.confidence_score,
            "explanation":        self.stage4_decision.explanation,
            "red_flags_triggered":self.stage4_decision.red_flags_triggered,
            "rule_ids_matched":   self.stage4_decision.rule_ids_matched,
            "rag_rules_retrieved":[r.get("rule_id","") for r in self.stage3_rag.retrieved_rules],
            "stt_model":          self.stage1_stt.model_used,
            "llm_model":          self.stage2_extraction.model_used,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ==============================================================================
#  STAGE 1 -- SPEECH PROCESSING
#  (Noise Reduction -> VAD -> Whisper)
# ==============================================================================

def run_stage1_stt(patient_input: PatientInput) -> Stage1_STT:
    stage_start = time.time()
    logger.info("=" * 68)
    logger.info("  STAGE 1 -- SPEECH PROCESSING")
    logger.info("=" * 68)

    # Demo/API mode: transcript already provided
    if patient_input.raw_transcript:
        logger.info("  [BYPASS] Pre-supplied transcript -- skipping audio pipeline")
        return Stage1_STT(
            transcript=patient_input.raw_transcript,
            detected_language=patient_input.language_hint or "hi",
            detected_language_name=_lang_name(patient_input.language_hint or "hi"),
            confidence=1.0,
            model_used="bypass",
            processing_time_seconds=round(time.time() - stage_start, 3),
            success=True,
        )

    # Live audio mode
    if not patient_input.audio_file or not Path(patient_input.audio_file).exists():
        err = f"Audio file not found: {patient_input.audio_file}"
        logger.error(err)
        return Stage1_STT(success=False, error_message=err)

    try:
        from pipeline.stage1_stt import process_audio
        res = process_audio(
            audio_file_path=patient_input.audio_file,
            whisper_model_size=CONFIG["whisper_model"],
            force_language=patient_input.language_hint,
            apply_noise_reduce=CONFIG["noise_reduction"],
            apply_vad=CONFIG["vad_enabled"],
            noise_reduction_strength=CONFIG["noise_strength"],
            device=CONFIG["whisper_device"],
        )
        logger.info(f"  Transcribed ({res.detected_language_name}): {res.transcript[:80]}...")
        return Stage1_STT(**{k: getattr(res, k) for k in Stage1_STT.__dataclass_fields__})
    except Exception as e:
        logger.error(f"Stage 1 error: {e}")
        return Stage1_STT(success=False, error_message=str(e))


# ==============================================================================
#  STAGE 2 -- NLP & SYMPTOM EXTRACTION
#  (Ollama LLM -> structured fields, keyword fallback if unavailable)
# ==============================================================================

_SYS_PROMPT = """You are a medical triage assistant for eSanjeevani India.
Extract structured data from a patient description. Return ONLY valid JSON.
Do NOT diagnose. Do NOT recommend treatment. Extract only what the patient said."""

_USR_PROMPT = """Extract medical fields from this patient description:

PATIENT: {transcript}

Return ONLY a JSON object with these exact fields:
{{
  "symptoms": ["list of symptoms mentioned"],
  "duration": "how long (e.g. '2 hours', '3 days', 'unknown')",
  "severity": "mild | moderate | severe | unknown",
  "existing_diseases": ["pre-existing conditions mentioned"],
  "age": null or integer,
  "gender": "male | female | unknown",
  "patient_complaint_summary": "one sentence summary in English",
  "red_flags_mentioned": ["alarming symptoms: chest pain, unconscious, suicidal etc."],
  "extraction_confidence": 0.0 to 1.0
}}

Return ONLY the JSON. No markdown. No explanation."""


def run_stage2_extraction(stage1: Stage1_STT) -> Stage2_Extraction:
    stage_start = time.time()
    logger.info("=" * 68)
    logger.info("  STAGE 2 -- NLP & SYMPTOM EXTRACTION")
    logger.info("=" * 68)

    if not stage1.success or not stage1.transcript:
        return Stage2_Extraction(success=False, error_message="No transcript from Stage 1")

    transcript = stage1.transcript
    logger.info(f"  Input ({len(transcript)} chars): {transcript[:100]}")

    # Try Ollama LLM
    llm_result = _call_ollama(transcript)
    if llm_result:
        llm_result["processing_time_seconds"] = round(time.time() - stage_start, 3)
        llm_result["model_used"] = CONFIG["ollama_model"]
        llm_result["success"] = True
        logger.info(f"  LLM extracted {len(llm_result.get('symptoms',[]))} symptoms")
        valid_fields = {k: llm_result[k] for k in Stage2_Extraction.__dataclass_fields__ if k in llm_result}
        return Stage2_Extraction(**valid_fields)

    # Keyword fallback
    logger.warning("  Ollama unavailable -- using keyword fallback")
    return _keyword_fallback(transcript, stage_start)


_OLLAMA_ACTIVE: Optional[bool] = None

def _is_ollama_online() -> bool:
    global _OLLAMA_ACTIVE
    # Re-check every 30 seconds if was offline
    if _OLLAMA_ACTIVE is False and hasattr(_is_ollama_online, "_last_check"):
        if time.time() - _is_ollama_online._last_check < 30:
            return False
    _is_ollama_online._last_check = time.time()

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", 11434))
        s.close()
        _OLLAMA_ACTIVE = True
        return True
    except Exception:
        _OLLAMA_ACTIVE = False
        return False

def _call_ollama(transcript: str) -> Optional[dict]:
    if not _is_ollama_online():
        return None
    try:
        import urllib.request
        prompt = _USR_PROMPT.format(transcript=transcript)
        payload = json.dumps({
            "model": CONFIG["ollama_model"],
            "prompt": f"<|system|>\n{_SYS_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>",
            "stream": False,
            "options": {"temperature": CONFIG["llm_temperature"], "num_predict": 512}
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{CONFIG['ollama_host']}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=min(CONFIG["llm_timeout_seconds"], 10)) as resp:
            raw = json.loads(resp.read())["response"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s == -1:
            raise ValueError("No JSON in response")
        return json.loads(raw[s:e])
    except Exception as e:
        logger.warning(f"  Ollama: {e}")
        return None


def _keyword_fallback(transcript: str, stage_start: float) -> Stage2_Extraction:
    """Rule-based keyword extraction -- works without any external model."""
    t = transcript.lower()

    SYMPTOMS = {
        "chest pain":        ["chest pain","seene mein dard","seeene mein dard","seena dard","chhati mein dard","chhati dard"],
        "left arm pain":     ["left arm pain","bayi banh","left arm mein dard","left arm"],
        "breathlessness":    ["breathless","breathing difficulty","saans nahi","sans lena mushkil","difficulty breathing","saans lene mein dikkat","breath"],
        "sweating":          ["sweat","paseena","paseene","ghaam","diaphoresis"],
        "fever":             ["fever","bukhar","bukhaar","temperature","taap"],
        "high fever":        ["high fever","tez bukhar","very high fever","104","103","105"],
        "cough":             ["cough","khansi","khaansi"],
        "headache":          ["headache","sar dard","sardard","sir dard"],
        "vomiting":          ["vomit","ulti","nausea","nauseous"],
        "diarrhea":          ["diarrhea","dast","loose stool","loose motion"],
        "body ache":         ["body ache","body pain","badan dard","sharir dard","limb pain"],
        "dizziness":         ["dizzy","dizziness","chakkar","vertigo"],
        "weakness":          ["weakness","weak","kamzori","thakan","fatigue","tired"],
        "abdominal pain":    ["abdominal pain","stomach pain","pet mein dard","pet dard","tummy"],
        "seizure":           ["seizure","convulsion","fitting","jhatkam","daure","mirgi","jhatke"],
        "unconscious":       ["unconscious","behosh","unresponsive","no response","passed out","fainted"],
        "bleeding":          ["bleed","khoon","blood","rakt","haemorrhage","hemorrhage"],
        "suicidal":          ["suicidal","want to die","marne","suicide","overdose","pills","khud ko","self harm"],
        "facial drooping":   ["facial droop","face droop","chehra tedha","chehra ek taraf","face drooped","drooped to one side","drooped","one side face","tedha"],
        "arm weakness":      ["arm weakness","haath kamzori","arm weak","haath uthana mushkil","difficulty lifting","lifting his arm","lifting arm","arm numb","cannot lift"],
        "speech difficulty": ["speech problem","cannot speak","bol nahi","speech difficult","slurred","cannot speak","cannot talk","not speaking","speech suddenly"],
        "jaundice":          ["jaundice","yellow","peeliya","aankhein peeli","yellow eyes"],
        "weight loss":       ["weight loss","wajan ghatna","wajan kam","losing weight"],
        "night sweats":      ["night sweat","raat ko paseena","night sweating"],
        "hemoptysis":        ["cough blood","blood in cough","coughing blood","khansi mein khoon","blood cough"],
        "skin rash":         ["rash","skin rash","daane","red spots","red daane","spots on skin"],
        "cold":              ["cold","sardi","runny nose","naak behna","stuffy nose"],
        "sore throat":       ["sore throat","gala dard","throat pain","throat hurt"],
        "knee pain":         ["knee pain","ghutne mein dard","ghutne dard","knee hurt"],
        "back pain":         ["back pain","kamar dard","spine pain","pith dard"],
        "hypoglycemia":      ["sugar low","blood sugar low","sugar drop","hypoglycemia","sugar bahut gir","sugar giri"],
        "heavy bleeding":    ["heavy bleeding","bahut khoon","heavy blood loss","khoon bahut aa"],
        "palpitations":      ["palpitation","heart racing","dhadkan tez","heart beat fast","dil tez"],
        "trembling":         ["trembling","shaking","haath kaanp","kaanpna","shivering hands"],
    }

    found = []
    for symptom, keywords in SYMPTOMS.items():
        if any(kw in t for kw in keywords):
            found.append(symptom)

    RED_FLAG_MAP = {
        "chest pain": "chest_pain", "breathlessness": "breathlessness",
        "unconscious": "loss_of_consciousness", "seizure": "seizure_active",
        "suicidal": "suicidal_ideation", "heavy bleeding": "severe_bleeding",
        "facial drooping": "facial_drooping", "arm weakness": "arm_weakness",
        "speech difficulty": "speech_difficulty", "hemoptysis": "cough_blood",
        "hypoglycemia": "diabetic_emergency",
    }
    red_flags = [RED_FLAG_MAP[s] for s in found if s in RED_FLAG_MAP]

    # Severity
    if any(w in t for w in ["very severe","bahut zyada","bahut tez","excruciating","unbearable","very high","severe","tez dard"]):
        severity = "severe"
    elif any(w in t for w in ["moderate","zyada","quite"]):
        severity = "moderate"
    elif found:
        severity = "mild"
    else:
        severity = "unknown"

    # Duration
    DURATION_MAP = {
        "2 hours":   ["do ghante","2 hours","two hours","2 ghante"],
        "3 hours":   ["teen ghante","3 hours","three hours"],
        "few hours": ["kuch ghante","few hours"],
        "1 day":     ["ek din","1 day","one day","kal se"],
        "2 days":    ["do din","2 days","two days"],
        "3 days":    ["teen din","3 days","three days"],
        "4 days":    ["chaar din","4 days","four days"],
        "5 days":    ["paanch din","5 days"],
        "1 week":    ["ek hafte","1 week","one week"],
        "2 weeks":   ["do hafte","2 weeks","two weeks"],
        "3 weeks":   ["teen hafte","3 weeks","three weeks"],
        "1 month":   ["ek mahine","1 month","one month"],
        "2 months":  ["do mahine","2 months","two months"],
        "few minutes":["kuch minute","few minutes","minute pehle"],
        "this morning":["subah","this morning","aaj subah"],
    }
    duration = "unknown"
    for dur, hints in DURATION_MAP.items():
        if any(h in t for h in hints):
            duration = dur
            break

    # Age
    age = None
    m = re.search(r"\b(\d{1,2})\s*(saal|year|sal|varsha|varsh)\b", t)
    if m:
        age = int(m.group(1))

    # Gender
    gender = "unknown"
    if any(w in t for w in ["my husband","mera pati","my son","mera beta","male","he ","his "]):
        gender = "male"
    elif any(w in t for w in ["my wife","meri patni","my daughter","meri beti","female","she ","her ","i am pregnant","main pregnant"]):
        gender = "female"

    # Existing diseases
    diseases = []
    if any(w in t for w in ["diabetes","sugar ki problem","sugar patient","sugar ki"]):
        diseases.append("diabetes")
    if any(w in t for w in ["hypertension","bp high","blood pressure"]):
        diseases.append("hypertension")
    if any(w in t for w in ["asthma","inhaler","asthma ka"]):
        diseases.append("asthma")
    if any(w in t for w in ["thyroid","thyroid ka"]):
        diseases.append("thyroid")

    summary = f"Patient reports: {', '.join(found[:5])}." if found else "No clear symptoms extracted."
    confidence = min(0.55 + (len(found) * 0.05), 0.80) if found else 0.20

    logger.info(f"  Keyword extracted: {found}")
    return Stage2_Extraction(
        symptoms=found,
        duration=duration,
        severity=severity,
        existing_diseases=diseases,
        age=age,
        gender=gender,
        patient_complaint_summary=summary,
        red_flags_mentioned=red_flags,
        extraction_confidence=confidence,
        model_used="keyword_fallback",
        processing_time_seconds=round(time.time() - stage_start, 3),
        success=bool(found),
        error_message=None if found else "No symptoms extracted",
    )


# ==============================================================================
#  STAGE 3 -- MEDICAL KNOWLEDGE LAYER (RAG)
#  (Retrieve relevant rules from knowledge base JSON files)
# ==============================================================================

def _load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {path.name}: {e}")
        return {}

# Load knowledge base at import time
_KB_SYMPTOM_MAP = _load_json(PATHS["symptom_map"])
_KB_RED_FLAGS   = _load_json(PATHS["red_flags"])
_KB_GUIDELINES  = _load_json(PATHS["guidelines"])


def run_stage3_rag(stage2: Stage2_Extraction) -> Stage3_RAG:
    stage_start = time.time()
    logger.info("=" * 68)
    logger.info("  STAGE 3 -- MEDICAL KNOWLEDGE LAYER (RAG)")
    logger.info("=" * 68)

    if not stage2.success:
        return Stage3_RAG(rag_enabled=False, success=False)

    all_symptoms = [s.lower() for s in stage2.symptoms + stage2.red_flags_mentioned]
    retrieved_rules   = []
    retrieved_specs   = []
    retrieved_urgency = []

    # Match red flag rules
    for rule_name, rule in _KB_RED_FLAGS.get("red_flags", {}).items():
        triggers = [s.lower() for s in rule.get("symptoms_en", [])]
        matched  = [s for s in all_symptoms if any(t in s or s in t for t in triggers)]
        if matched:
            retrieved_rules.append({
                "rule_id":   f"RF-{rule_name.upper()}",
                "urgency":   rule.get("urgency", "ROUTINE"),
                "specialty": rule.get("specialty", "GeneralMedicine"),
                "reasoning": rule.get("reasoning", ""),
                "matched_on":matched[:2],
            })
            retrieved_urgency.append(rule.get("urgency", "ROUTINE"))
            spec = rule.get("specialty", "")
            if spec and spec not in retrieved_specs:
                retrieved_specs.append(spec)

    # Match specialties by keyword score
    specialty_scores: Dict[str, float] = {}
    for spec_name, spec_data in _KB_SYMPTOM_MAP.get("specialties", {}).items():
        kws = [k.lower() for k in spec_data.get("keywords_en", [])]
        score = sum(
            spec_data.get("urgency_weight", 0.5)
            for s in all_symptoms
            for kw in kws
            if kw in s or s in kw
        )
        if score > 0:
            specialty_scores[spec_name] = score

    for spec, _ in sorted(specialty_scores.items(), key=lambda x: x[1], reverse=True)[:CONFIG["rag_top_k"]]:
        if spec not in retrieved_specs:
            retrieved_specs.append(spec)

    # Match triage guidelines
    for p in _KB_GUIDELINES.get("protocols", []):
        criteria = p.get("criteria", "").lower()
        if any(s in criteria for s in all_symptoms):
            retrieved_rules.append({
                "rule_id":   p.get("protocol_id", ""),
                "urgency":   p.get("urgency", "ROUTINE"),
                "specialty": p.get("specialty", "GeneralMedicine"),
                "reasoning": p.get("criteria", ""),
                "matched_on":all_symptoms[:2],
            })

    context_lines = [
        f"Rule {r['rule_id']}: {r['urgency']} ({r['specialty']})"
        for r in retrieved_rules[:5]
    ]
    logger.info(f"  Retrieved {len(retrieved_rules)} rules | Specialties: {retrieved_specs[:3]}")

    return Stage3_RAG(
        retrieved_rules=retrieved_rules[:CONFIG["rag_top_k"] + 3],
        retrieved_specialties=retrieved_specs[:CONFIG["rag_top_k"]],
        retrieved_urgency_hints=list(set(retrieved_urgency)),
        rag_context_text="\n".join(context_lines) or "No rules matched.",
        rag_enabled=True,
        success=True,
    )


# ==============================================================================
#  STAGE 4 -- DECISION ENGINE
#  (Rule Validation -> Urgency -> Specialist -> Score -> Explanation)
# ==============================================================================

_URGENCY_RANK  = {"ROUTINE": 0, "URGENT": 1, "EMERGENCY": 2}
_URGENCY_COLOR = {"EMERGENCY": "RED", "URGENT": "YELLOW", "ROUTINE": "GREEN"}
_COMPOSITE_RULES = _KB_RED_FLAGS.get("composite_rules", {}).get("rules", [])


def run_stage4_decision(stage2: Stage2_Extraction, stage3: Stage3_RAG) -> Stage4_Decision:
    stage_start = time.time()
    logger.info("=" * 68)
    logger.info("  STAGE 4 -- DECISION ENGINE")
    logger.info("=" * 68)

    symptoms_lower = [s.lower() for s in stage2.symptoms]
    red_flags_found  = []
    composite_found  = []
    matched_ids      = []
    urgency          = "ROUTINE"
    specialty        = "GeneralMedicine"
    confidence       = stage2.extraction_confidence

    # 4.1 Apply RAG urgency hints
    for hint in stage3.retrieved_urgency_hints:
        if _URGENCY_RANK.get(hint, 0) > _URGENCY_RANK[urgency]:
            urgency = hint
    if stage3.retrieved_specialties:
        specialty = stage3.retrieved_specialties[0]
    for r in stage3.retrieved_rules:
        matched_ids.append(r.get("rule_id", ""))

    # 4.2 Hard red flag rules
    for rule_name, rule in _KB_RED_FLAGS.get("red_flags", {}).items():
        triggers = [s.lower() for s in rule.get("symptoms_en", [])]
        all_checks = symptoms_lower + [s.lower() for s in stage2.red_flags_mentioned]
        hit = [s for s in all_checks if any(t in s or s in t for t in triggers)]
        if hit:
            ru = rule.get("urgency", "ROUTINE")
            if _URGENCY_RANK.get(ru, 0) > _URGENCY_RANK[urgency]:
                urgency  = ru
                specialty = rule.get("specialty", specialty)
            red_flags_found.append(f"{rule_name} [triggered: {', '.join(hit[:2])}]")
            matched_ids.append(f"RF-{rule_name.upper()}")
            confidence = min(confidence + CONFIG["emergency_boost"], 1.0)

    # 4.3 Composite multi-symptom rules
    for rule in _COMPOSITE_RULES:
        triggers = [t.lower() for t in rule.get("trigger_symptoms", [])]
        hits = sum(1 for t in triggers if any(t in s or s in t for s in symptoms_lower))
        if hits >= rule.get("required_count", 2):
            cu = rule.get("escalate_to", "ROUTINE")
            if _URGENCY_RANK.get(cu, 0) > _URGENCY_RANK[urgency]:
                urgency   = cu
                specialty = rule.get("specialty", specialty)
            composite_found.append(rule.get("name", rule.get("rule_id", "")))
            matched_ids.append(rule.get("rule_id", ""))
            confidence = min(confidence + CONFIG["composite_boost"], 1.0)

    # 4.4 Severity tuning
    if stage2.severity == "severe" and urgency == "ROUTINE":
        urgency = "URGENT"
        confidence = min(confidence + 0.05, 1.0)
    if stage2.severity == "mild" and urgency == "URGENT" and not red_flags_found and not composite_found:
        urgency = "ROUTINE"

    # 4.5 Final specialty
    if specialty == "GeneralMedicine" and stage3.retrieved_specialties:
        specialty = stage3.retrieved_specialties[0]
    if urgency == "EMERGENCY" and specialty == "GeneralMedicine":
        specialty = "Emergency"

    # 4.6 Confidence calculation
    confidence = round(min(max(confidence, 0.30), 0.99), 2)
    if len(stage2.symptoms) >= 3:
        confidence = min(confidence + 0.05, 0.99)
    if red_flags_found:
        confidence = min(confidence + 0.08, 0.99)

    explanation = _build_explanation(urgency, specialty, stage2, red_flags_found, composite_found)
    logger.info(f"  Decision: {urgency} -> {specialty} | Confidence: {confidence:.1%}")

    return Stage4_Decision(
        urgency_level=urgency,
        urgency_color=_URGENCY_COLOR[urgency],
        suggested_specialty=specialty,
        confidence_score=confidence,
        red_flags_triggered=red_flags_found,
        composite_rules_triggered=composite_found,
        explanation=explanation,
        rule_ids_matched=list(set(matched_ids)),
        processing_time_seconds=round(time.time() - stage_start, 3),
    )


def _build_explanation(urgency, specialty, s2, red_flags, composite):
    parts = []
    if s2.symptoms:
        parts.append(f"Patient described: {', '.join(s2.symptoms[:5])}.")
    if s2.duration and s2.duration != "unknown":
        parts.append(f"Duration: {s2.duration}.")
    if s2.severity and s2.severity != "unknown":
        parts.append(f"Severity: {s2.severity}.")
    if s2.existing_diseases:
        parts.append(f"Known conditions: {', '.join(s2.existing_diseases)}.")
    if red_flags:
        parts.append(f"Red flags: {', '.join(f.split(' [')[0] for f in red_flags[:2])}.")
    if composite:
        parts.append(f"Multi-symptom rules: {', '.join(composite[:2])}.")
    if urgency == "EMERGENCY":
        parts.append(f"Classified EMERGENCY -- {specialty} required immediately.")
    elif urgency == "URGENT":
        parts.append(f"Classified URGENT -- {specialty} within 60 minutes.")
    else:
        parts.append(f"Classified ROUTINE -- {specialty} standard queue.")
    return " ".join(parts)


# ==============================================================================
#  STAGE 5 -- OUTPUT & DASHBOARD
#  (Terminal display + JSON file save)
# ==============================================================================

def run_stage5_output(result: TriageResult) -> None:
    logger.info("=" * 68)
    logger.info("  STAGE 5 -- OUTPUT & DASHBOARD")
    logger.info("=" * 68)

    d, s2, s1 = result.stage4_decision, result.stage2_extraction, result.stage1_stt
    BORDER = {"EMERGENCY": "!", "URGENT": "*", "ROUTINE": "-"}[d.urgency_level]

    lines = [
        "",
        "=" * 68,
        f"  VOICETRIAGE RESULT | Case: {result.case_id} | {result.timestamp}",
        "=" * 68,
        "",
        BORDER * 68,
        f"  [{d.urgency_level}]  >>  {d.suggested_specialty}",
        BORDER * 68,
        "",
        f"  TRANSCRIPT  [{s1.detected_language_name} | {s1.model_used}]",
        "  " + "-" * 64,
    ]
    for line in textwrap.wrap(s1.transcript, 62):
        lines.append(f"    {line}")

    lines += [
        "",
        "  PATIENT INFORMATION",
        "  " + "-" * 64,
        f"    Symptoms   : {', '.join(s2.symptoms) or 'None extracted'}",
        f"    Duration   : {s2.duration}",
        f"    Severity   : {s2.severity}",
        f"    Diseases   : {', '.join(s2.existing_diseases) or 'None'}",
        f"    Age/Gender : {s2.age or 'N/A'} / {s2.gender}",
        "",
        "  DECISION",
        "  " + "-" * 64,
        f"    Urgency    : {d.urgency_level}",
        f"    Specialty  : {d.suggested_specialty}",
        f"    Confidence : {d.confidence_score:.0%}",
    ]
    for rf in d.red_flags_triggered[:3]:
        lines.append(f"    Red Flag   : {rf}")
    for cr in d.composite_rules_triggered[:2]:
        lines.append(f"    Rule Match : {cr}")

    lines += [
        "",
        "  REASON",
        "  " + "-" * 64,
    ]
    for line in textwrap.wrap(d.explanation, 62):
        lines.append(f"    {line}")

    lines += [
        "",
        "  PERFORMANCE",
        "  " + "-" * 64,
        f"    Total: {result.total_time_seconds:.2f}s | STT: {s1.processing_time_seconds:.2f}s | LLM: {s2.processing_time_seconds:.2f}s | Decision: {d.processing_time_seconds:.2f}s",
        "",
        "=" * 68,
    ]

    output = "\n".join(lines)
    # Write to stdout safely (encode/decode for Windows)
    sys.stdout.buffer.write((output + "\n").encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()

    # Save JSON
    if CONFIG["save_output_json"]:
        PATHS["output_dir"].mkdir(parents=True, exist_ok=True)
        ts = result.timestamp.replace(":", "-")
        out_file = PATHS["output_dir"] / f"{result.case_id}_{ts}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(result.to_json())
        logger.info(f"  Saved: {out_file}")


# ==============================================================================
#  FULL PIPELINE RUNNER
# ==============================================================================

def run_pipeline(patient_input: PatientInput) -> TriageResult:
    """Run all 5 stages end-to-end. Returns a TriageResult."""
    t0 = time.time()
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    logger.info(f"\n{'#'*68}")
    logger.info(f"#  VoiceTriage Pipeline | Case: {patient_input.case_id} | {ts}")
    logger.info(f"{'#'*68}")

    s1 = run_stage1_stt(patient_input)
    s2 = run_stage2_extraction(s1)
    s3 = run_stage3_rag(s2)
    s4 = run_stage4_decision(s2, s3)

    result = TriageResult(
        case_id=patient_input.case_id,
        timestamp=ts,
        patient_input=patient_input,
        stage1_stt=s1,
        stage2_extraction=s2,
        stage3_rag=s3,
        stage4_decision=s4,
        total_time_seconds=round(time.time() - t0, 3),
    )
    run_stage5_output(result)
    return result


# ==============================================================================
#  DEMO MODE
# ==============================================================================

def run_demo(case_id: Optional[str] = None) -> TriageResult:
    """Run on a synthetic case (no audio file required)."""
    all_cases = []
    for path in [PATHS["cases_hindi"], PATHS["cases_regional"]]:
        if path.exists():
            all_cases.extend(_load_json(path).get("cases", []))

    if not all_cases:
        logger.warning("No dataset files found -- using built-in fallback case")
        return run_pipeline(PatientInput(
            case_id="FALLBACK",
            raw_transcript="I have severe chest pain and difficulty breathing for the past 2 hours. I am also sweating.",
            language_hint="en",
        ))

    case = next((c for c in all_cases if c["case_id"] == case_id), None) if case_id else None
    if not case:
        case = all_cases[0]

    gt = case.get("ground_truth", {})
    logger.info(f"  Demo: {case['case_id']} | {case['language_name']} | Expected: {gt.get('urgency')} - {gt.get('specialty')}")

    return run_pipeline(PatientInput(
        case_id=case["case_id"],
        raw_transcript=case.get("transcript_english", case.get("transcript", "")),
        language_hint=case.get("language", "hi"),
    ))


def list_cases() -> None:
    """Print all available demo cases."""
    all_cases = []
    for path in [PATHS["cases_hindi"], PATHS["cases_regional"]]:
        if path.exists():
            all_cases.extend(_load_json(path).get("cases", []))

    if not all_cases:
        print("No dataset files found.")
        return

    header = f"\n  {'ID':<18} {'LANG':<10} {'URGENCY':<12} {'SPECIALTY':<20} SYMPTOMS"
    print(header)
    print("  " + "-" * 85)
    for c in all_cases:
        gt = c.get("ground_truth", {})
        syms = ", ".join(gt.get("symptoms", [])[:3])
        print(f"  {c['case_id']:<18} {c['language_name']:<10} {gt.get('urgency','?'):<12} {gt.get('specialty','?'):<20} {syms}")
    print()


# ==============================================================================
#  HELPERS
# ==============================================================================

def _lang_name(code: str) -> str:
    return {
        "hi": "Hindi", "mr": "Marathi", "ta": "Tamil", "te": "Telugu",
        "kn": "Kannada", "ml": "Malayalam", "gu": "Gujarati", "pa": "Punjabi",
        "bn": "Bengali", "en": "English", "ur": "Urdu",
    }.get(code, f"Unknown ({code})")


# ==============================================================================
#  CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="VoiceTriage -- Voice-Based Symptom Triage for eSanjeevani",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python main.py --demo                      (first synthetic case)
          python main.py --demo --case CASE_HI_002   (stroke case)
          python main.py --list-cases                (show all demo cases)
          python main.py --audio patient.wav         (real audio)
          python main.py --audio patient.wav --language hi --model medium
        """),
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--audio",       "-a", help="Path to audio file (WAV/MP3/M4A)")
    grp.add_argument("--demo",        "-d", action="store_true", help="Run on synthetic demo case")
    grp.add_argument("--list-cases",        action="store_true", help="List all demo cases")

    parser.add_argument("--case",     "-c", default=None, help="Case ID for demo (e.g. CASE_HI_001)")
    parser.add_argument("--language", "-l", default=None, help="Language code: hi, ta, mr, te ...")
    parser.add_argument("--model",    "-m", default="base",
                        choices=["tiny","base","small","medium","large-v2","large-v3"],
                        help="Whisper model size (default: base)")
    parser.add_argument("--no-save",        action="store_true", help="Do not save output JSON")
    parser.add_argument("--ollama-model",   default=None, help="Override Ollama model name")

    args = parser.parse_args()

    CONFIG["whisper_model"] = args.model
    if args.no_save:      CONFIG["save_output_json"] = False
    if args.ollama_model: CONFIG["ollama_model"] = args.ollama_model

    if args.list_cases:
        list_cases()
    elif args.demo or (not args.audio):
        run_demo(case_id=args.case)
    elif args.audio:
        run_pipeline(PatientInput(
            case_id=f"LIVE_{Path(args.audio).stem.upper()}",
            audio_file=args.audio,
            language_hint=args.language,
        ))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
