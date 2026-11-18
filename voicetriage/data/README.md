# VoiceTriage — Data Directory

This directory contains all datasets used for **testing**, **evaluation**, and **RAG knowledge grounding**.
No real patient data is stored here. All transcripts are **synthetically generated** based on realistic clinical presentations.

---

## 📁 Folder Structure

```
data/
├── synthetic_transcripts/
│   ├── cases_hindi.json          ← 20 Hindi patient voice transcript cases
│   ├── cases_regional.json       ← 10 cases in Marathi, Tamil, Telugu, Bengali
│   └── cases_english.json        ← 10 English cases (baseline / fallback testing)
│
├── evaluation/
│   ├── ground_truth.json         ← Expert-annotated urgency + specialty labels
│   ├── evaluation_config.json    ← Metrics config, thresholds, evaluation rules
│   └── results/                  ← (auto-generated) AI output vs ground truth
│
└── README.md                     ← This file
```

---

## 📋 Case Format

Each synthetic case has these fields:

| Field | Type | Description |
|-------|------|-------------|
| `case_id` | string | Unique identifier (e.g. `CASE_HI_001`) |
| `language` | string | Language code (`hi`, `mr`, `ta`, `te`, `bn`, `en`) |
| `language_name` | string | Full language name |
| `transcript` | string | Patient's spoken description (in native script) |
| `transcript_romanized` | string | Same text in Roman/Latin script |
| `transcript_english` | string | English translation |
| `ground_truth.urgency` | string | `EMERGENCY` / `URGENT` / `ROUTINE` |
| `ground_truth.specialty` | string | Expected specialist department |
| `ground_truth.symptoms` | list | Key symptoms mentioned |
| `ground_truth.duration` | string | How long symptoms present |
| `ground_truth.severity` | string | `mild` / `moderate` / `severe` |
| `ground_truth.existing_diseases` | list | Comorbidities mentioned |
| `ground_truth.age` | int | Patient age |
| `ground_truth.gender` | string | `male` / `female` / `unknown` |
| `ground_truth.red_flags` | list | Emergency triggers present |
| `ground_truth.confidence_expected` | float | Expected AI confidence range |
| `clinical_notes` | string | Why this urgency level was assigned |

---

## 🎯 Case Distribution

| Urgency | Hindi | Regional | English | Total |
|---------|-------|----------|---------|-------|
| EMERGENCY | 7 | 4 | 3 | 14 |
| URGENT | 7 | 4 | 4 | 15 |
| ROUTINE | 6 | 2 | 3 | 11 |
| **Total** | **20** | **10** | **10** | **40** |

---

## ✅ How to Use

### Run evaluation on all cases
```bash
python evaluation/run_eval.py --input data/synthetic_transcripts/ --ground-truth data/evaluation/ground_truth.json
```

### Test Stage 1 STT on synthetic audio
```bash
# First generate audio from text (TTS), then run STT
python data/utils/text_to_audio.py --input data/synthetic_transcripts/cases_hindi.json
python pipeline/stage1_stt.py audio/synthetic/CASE_HI_001.wav
```

---

## ⚠️ Important Notes

- All patient data is **100% synthetic** — no real patients, no PII
- Clinical presentations are based on published symptom-disease mappings
- Ground truth labels were reviewed against MoHFW triage protocols
- Regional language transcripts are modeled on authentic speech patterns
