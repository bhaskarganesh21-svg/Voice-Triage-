# VoiceTriage — AI-Powered Voice Symptom Triage for eSanjeevani

A voice-based symptom triage assistant designed for **eSanjeevani**, India's national telemedicine service. It processes patient complaints across Indian regional languages (Hindi, Marathi, Tamil, Telugu, Bengali, etc.), extracts symptoms, matches against clinical protocols, and assigns an urgency level and specialist routing.

---

## 🚀 Quick Start in VS Code

### 1. Open Project in VS Code
Open VS Code and navigate to:
```
File -> Open Folder -> D:\voicetriage
```

### 2. Install Requirements (one-time)
Open the VS Code Terminal (`Ctrl + ` `):
```powershell
pip install -r requirements.txt
```

### 3. Launch Web App
```powershell
python run.py
```
> The dashboard will automatically open at **`http://localhost:5000`**.

---

## 🩺 System Architecture

```
Patient (Voice / Text)
      │
      ▼
[Stage 1: Speech Processing]
Noise Reduction -> Voice Activity Detection -> Whisper Speech-to-Text
      │
      ▼
[Stage 2: NLP & Symptom Extraction]
Llama 3.2 (3B) via Ollama / Deterministic Medical Extraction Fallback
Extracts: Symptoms, Duration, Severity, Pre-existing conditions, Age, Gender
      │
      ▼
[Stage 3: Medical Knowledge Layer (RAG)]
Protocol Retrieval: Red Flag Registry + 14 Specialties + Clinical Protocols
      │
      ▼
[Stage 4: Decision Engine]
Composite Rule Evaluation -> Urgency Classification (Emergency / Urgent / Routine) -> Specialist Selection -> Confidence Scoring -> Clinical Reasoning
      │
      ▼
[Stage 5: Delivery & Web Dashboard]
Web UI for Telemedicine Operators & Clinicians + JSON Audit Trails
```

---

## 📂 Project Directory Structure

```
D:\voicetriage\
├── run.py                                 # One-click web application launcher
├── main.py                                # End-to-end 5-stage triage pipeline & CLI
├── requirements.txt                       # Project dependencies
│
├── webapp/                                # Full-stack Web Application
│   ├── app.py                             # Flask backend API & routing
│   ├── templates/
│   │   └── index.html                     # Responsive dark-theme triage dashboard
│   └── static/
│       ├── css/style.css                  # UI styling & animations
│       └── js/app.js                      # Client controller, API client & state management
│
├── knowledge_base/                        # Curated Clinical Reference Data
│   ├── symptom_specialty_mapping.json     # 14 specialties with bilingual multilingual keywords
│   ├── red_flag_symptoms.json             # Hard emergency triggers & composite clinical rules
│   └── triage_guidelines.json            # Ministry of Health teleconsultation protocols
│
├── pipeline/                              # Core processing modules
│   ├── stage1_stt.py                      # Speech processing (Noise reduction, VAD, Whisper)
│   ├── stage2_extraction.py               # NLP symptom extraction
│   ├── stage3_rag.py                      # Rule retrieval
│   ├── stage4_decision.py                 # Urgency & specialty decision engine
│   └── stage5_output.py                   # Output formatting & serialization
│
├── data/                                  # Datasets & evaluation
│   └── synthetic_transcripts/
│       ├── cases_hindi.json               # 20 Hindi clinical cases with ground truth
│       └── cases_regional.json            # Marathi, Tamil, Telugu, and Bengali test cases
│
├── audio/                                 # Audio inputs (WAV, MP3, M4A)
├── output/                                # Audit JSON records for every triage consultation
└── tests/
    └── test_stage1.py                     # Unit tests
```

---

## 🖥️ Command-Line Usage

You can also run triage directly from the CLI without starting the web server:

```powershell
# Run synthetic demo cases
python main.py --demo
python main.py --demo --case CASE_HI_001
python main.py --demo --case CASE_TA_001

# List all available demo cases
python main.py --list-cases

# Run triage on an audio file
python main.py --audio audio/patient_sample.wav --language hi
```

---

## ⚠️ Clinical Disclaimer
This system is an assistive triage tool designed for pre-call routing and prioritization. It does **not** diagnose medical conditions, prescribe pharmaceuticals, or replace qualified medical personnel.
