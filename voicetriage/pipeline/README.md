# Pipeline stages

Each stage has one job and passes a structured result to the next stage.

| File | Responsibility | Input | Output |
| --- | --- | --- | --- |
| `stage1_stt.py` | Audio cleanup, speech detection, transcription | Audio file | Transcript, detected language, audio metadata |
| `stage2_extraction.py` | Extract stated symptoms and patient details with Ollama JSON mode; validate schema | Transcript text | Symptoms, duration, severity, known diseases, age, gender, red flags, confidence |
| `stage3_rag.py` | Retrieve matching local triage rules and specialty keywords; assemble evidence context | Stage 2 extraction + knowledge base JSON | Matched rules, specialties, urgency hints, context text |
| `stage4_decision.py` | Apply deterministic decision rules to Stage 2 and Stage 3 evidence | Structured extraction + retrieved evidence + red-flag rules | Advisory urgency, suggested specialty, confidence, matched rules, explanation |
| `stage5_output.py` | Format the completed run for terminal display and save JSON | Full pipeline result | Human-readable summary + JSON file |
| `../main.py` | Connect the stages and display/save the final pipeline result | Stage outputs | Integrated run |

## Stage 3 knowledge sources

- `knowledge_base/red_flag_symptoms.json`: emergency/urgent trigger phrases and rule reasoning.
- `knowledge_base/symptom_specialty_mapping.json`: specialty keyword lists and ranking weights.
- `knowledge_base/triage_guidelines.json`: protocol criteria and their urgency/specialty labels.

Stage 3 currently performs transparent phrase matching against these local JSON
files. It does not call an LLM, change the knowledge files, or make the final
triage decision. The result's `rag_context_text` records which rules matched
and why, so later stages and reviewers can trace the evidence. It does not
require ChromaDB, LlamaIndex, or network access.

The structured result contains `retrieved_rules` (each with `rule_id`,
`rule_name`, `urgency`, `specialty`, `reasoning`, and `matched_on`),
`retrieved_specialties`, `retrieved_urgency_hints`, `rag_context_text`,
`rag_enabled`, and `success`.

## Stage 4 decision layer

`stage4_decision.py` owns the urgency hierarchy, direct red-flag rule check,
composite symptom checks, specialty selection, confidence calculation, and
plain-language explanation. It rechecks the red-flag JSON directly so a safety
rule can still be considered if Stage 3 retrieval misses it. A rule can raise
urgency; a reported mild severity cannot lower urgency already supported by a
rule. The result is still an advisory suggestion for clinician/operator review.

Its structured result contains `urgency_level`, `urgency_color`,
`suggested_specialty`, `confidence_score`, `red_flags_triggered`,
`composite_rules_triggered`, `explanation`, `rule_ids_matched`, and
`processing_time_seconds`.

## Stage 5 output layer

`stage5_output.py` owns terminal rendering and JSON file creation. It preserves
the public JSON field names provided by `TriageResult.to_dict()` and writes
case-specific output under `output/`. The terminal summary labels the result as
advisory. The interactive Streamlit dashboard and staff override remain a
separate later task under `app/`.

## Run the integrated demo

From the repository root, run `python main.py --demo`. Stage 3 reads its
knowledge sources from the paths configured in `main.py` and passes its result
to Stage 4.
