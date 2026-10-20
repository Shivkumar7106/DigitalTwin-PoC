# 🏥 Digital Twin — Predictive Health Engine for Type 2 Diabetes & Cardiovascular Risk

> A proof-of-concept Digital Twin system that fuses synthetic EHR data with simulated wearable sensor streams to predict glucose spikes (2 hours ahead) and flag cardiovascular anomalies.

**Built for the Happiest Health — Reimagining & Reforming Healthcare in India Summit 2026, Bengaluru**

---

## 📋 Team Details

| Role | Name | College/Incubator | Email |
|------|------|-------------------|-------|
| **Team Leader** |Shivkumar G| SJCE | shivumys2698@gmail.com |
| Member 2 |Sharada SR | SJCE | sharadasanjay7410@gmail.com |


> **Folder Name:** `twinfr_SJCE`

---

## 🎯 Project Title

**Digital Twin PoC: Predictive Health Engine for Type 2 Diabetes & Cardiovascular Risk Assessment**

---

## 📝 Problem Statement & Healthcare Use Case

### The Challenge
India faces a dual epidemic of Type 2 Diabetes and Cardiovascular Disease (CVD). With over **101 million diabetics** and **315 million** suffering from hypertension (ICMR-INDIAB study), India is the diabetes capital of the world. South Asians exhibit unique metabolic characteristics — the "thin-fat" phenotype — leading to diabetes onset at lower BMI cutoffs and younger ages compared to Western populations.

### The Problem
Current healthcare in India is **reactive** — patients visit doctors only after symptoms appear. By then, significant organ damage may have already occurred. What if we could **predict adverse health events before they happen**?

### Our Solution: The Digital Twin
We build a **Digital Twin** — a dynamic virtual replica of a patient that continuously ingests data from two streams:

1. **Static/Historical Data (EHR):** Demographics, diagnoses, lab results, medications, genetic risk markers
2. **Dynamic/Real-time Data (Wearables/IoT):** Continuous Glucose Monitoring (CGM), Heart Rate & HRV, Sleep patterns, Physical Activity

The Digital Twin processes these streams through AI/ML models to:
- **Predict glucose spikes 2 hours in advance** — giving patients and doctors time to intervene
- **Score cardiovascular risk** with explainable AI — showing which factors contribute most
- **Simulate "What-If" interventions** — "What if the patient exercises 30 min/day?"

### Healthcare Impact
- **Preventing diabetic emergencies** through early glucose spike detection
- **Reducing cardiovascular events** through proactive risk management
- **Personalizing treatment** based on individual patient data patterns
- **Empowering doctors** with an intelligent dashboard showing the patient's "digital mirror"

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│   ┌─────────────────┐          ┌──────────────────────────┐     │
│   │  EHR Generator   │          │  Wearable Sensor Generator│    │
│   │  (500 Indian     │          │  (CGM, HR/HRV, Sleep,     │    │
│   │   Patients)      │          │   Steps - 30 days)        │    │
│   └────────┬─────────┘          └────────────┬──────────────┘    │
└────────────┼─────────────────────────────────┼──────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION LAYER                          │
│   ┌──────────────────┐          ┌────────────────────────┐      │
│   │  EHR Loader       │          │  Sensor Loader &        │     │
│   │  (CSV Parser +    │          │  Stream Simulator       │     │
│   │   Validator)      │          │  (Real-time simulation) │     │
│   └────────┬──────────┘          └────────────┬─────────── ┘     │
└────────────┼─────────────────────────────────┼──────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PROCESSING & FEATURE STORE                       │
│   ┌──────────────────┐     ┌────────────────────────────┐       │
│   │ EHR Preprocessor  │     │  Sensor Preprocessor        │      │
│   │ (Demographics,    │     │  (HRV extraction, CGM       │      │
│   │  Labs, Conditions,│     │   features, Sleep metrics,  │      │
│   │  Medications)     │     │   Activity aggregation)     │      │
│   └────────┬──────────┘     └────────────┬───────────────┘       │
│            └──────────┬──────────────────┘                       │
│                       ▼                                          │
│            ┌────────────────────┐                                │
│            │ Feature Engineering │                                │
│            │ (Fusion + Splits)   │                                │
│            └────────┬───────────┘                                │
└─────────────────────┼────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DIGITAL TWIN CORE                             │
│   ┌──────────────────┐  ┌────────────────────────────────┐      │
│   │ Patient State     │  │  Glucose Spike Predictor        │     │
│   │ Manager           │  │  (BiLSTM + Attention)           │     │
│   │ (Virtual Patient) │  │  2-hour ahead prediction        │     │
│   └──────────────────┘  └────────────────────────────────┘      │
│   ┌──────────────────┐  ┌────────────────────────────────┐      │
│   │ Alert Engine      │  │  Cardiovascular Risk Scorer     │     │
│   │ (Severity-based   │  │  (XGBoost + SHAP Explainability)│    │
│   │  with cooldowns)  │  └────────────────────────────────┘      │
│   └──────────────────┘  ┌────────────────────────────────┐      │
│                          │  Multi-Modal Fusion Network     │     │
│                          │  (LSTM + MLP Hybrid)            │     │
│                          └────────────────────────────────┘      │
└─────────────────────┬────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STREAMLIT DASHBOARD (Doctor UI)                   │
│   ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌─────────────┐     │
│   │ Patient     │ │ Glucose   │ │ Cardio   │ │ What-If      │    │
│   │ Overview    │ │ Monitor   │ │ Risk     │ │ Simulator    │    │
│   └────────────┘ └──────────┘ └──────────┘ └─────────────┘     │
│                        ┌──────────┐                              │
│                        │ Alerts    │                              │
│                        └──────────┘                              │
└──────────────────────────────────────────────────────────────────┘
```

> 📄 **Detailed architecture diagram:** See [docs/architecture_diagram.md](docs/architecture_diagram.md)

---

## 🔧 Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core development language |
| **EHR Data** | Faker + Custom Generators | Synthetic Indian patient records |
| **Sensor Data** | NumPy + SciPy | Synthetic CGM, HR, Sleep, Activity signals |
| **ML — Glucose** | TensorFlow/Keras | BiLSTM + Attention for 2-hour prediction |
| **ML — Cardiovascular** | XGBoost + SHAP | Gradient boosting with explainability |
| **ML — Fusion** | TensorFlow/Keras | Multi-modal hybrid neural network |
| **Preprocessing** | Pandas, Scikit-learn, NeuroKit2 | Feature engineering & HRV analysis |
| **Dashboard** | Streamlit + Plotly | Interactive doctor-facing UI |
| **Configuration** | PyYAML | Centralized settings management |
| **Testing** | pytest | Unit and integration tests |

---

## 🤖 AI/ML Model Details

### 1. Glucose Spike Predictor (Bidirectional LSTM + Attention)
- **Input:** 6-hour sliding window of CGM data (72 timesteps × 8 features)
- **Features:** Glucose level, rate of change, acceleration, meal flag, exercise flag, rolling mean/std, medication timing
- **Output:** Predicted glucose for next 2 hours (24 × 5-minute intervals)
- **Spike Detection:** Flags when any predicted value exceeds 180 mg/dL
- **Uncertainty:** Monte Carlo Dropout for confidence intervals

### 2. Cardiovascular Risk Scorer (XGBoost + SHAP)
- **Input:** 35+ features combining static EHR (age, BMI, labs, conditions) and dynamic sensor aggregates (avg resting HR, HRV metrics, sleep efficiency, daily steps)
- **Output:** CVD event probability (0-1) with risk category (Low/Moderate/High/Critical)
- **Explainability:** SHAP waterfall charts showing per-patient feature contributions

### 3. Multi-Modal Fusion Network (Hybrid NN)
- **Architecture:** Dual-branch network
  - Temporal Branch: LSTM processing wearable time-series
  - Static Branch: MLP processing EHR tabular features
  - Fusion: Concatenation → Dense layers → Sigmoid output
- **Purpose:** Unified health risk score combining both data streams

### Evaluation Metrics
| Model | Metric | Target |
|-------|--------|--------|
| Glucose Predictor | RMSE | < 25 mg/dL |
| Glucose Predictor | Clarke Zone A+B | > 90% |
| Cardiovascular Risk | AUC-ROC | > 0.82 |
| Cardiovascular Risk | Sensitivity | > 0.85 |

---

## 📊 Data Strategy (Sandbox Compliance)

> ⚠️ **No real patient data is used.** All data is 100% synthetic, compliant with DPDP Act (India) and HIPAA.

### Electronic Health Records (EHR)
- **Generator:** Custom Python generator with Indian demographics (Faker en_IN locale)
- **Volume:** 500 synthetic patients with Indian names, cities, and demographic distributions
- **Clinical Realism:** Conditions, lab values, and medications are statistically correlated (e.g., diabetic patients have elevated HbA1c and corresponding medications)
- **Indian-specific:** BMI cutoffs per ICMR guidelines (overweight ≥23, obese ≥25), prevalence rates matching Indian urban populations

### Wearable/IoT Sensor Data
- **CGM:** 5-minute interval glucose readings (30 days) with postprandial spikes, dawn phenomenon, exercise effects
- **Heart Rate & HRV:** 1-minute interval HR with RMSSD/SDNN metrics
- **Sleep:** Nightly sleep staging (Wake, Light, Deep, REM) with efficiency metrics
- **Steps/Activity:** Hourly step counts with circadian activity patterns

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- 8GB+ RAM recommended for model training

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/digital-twin-poc.git
cd digital-twin-poc

# 2. Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Step 1: Generate synthetic data (500 patients, 30 days)
python scripts/generate_data.py --patients 500 --days 30

# Step 2: Train all models
python -m src.models.train

# Step 3: Launch the dashboard
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
Twin/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── config/
│   └── settings.yaml                 # Centralized configuration
├── data/
│   ├── raw/
│   │   ├── ehr/                      # Generated EHR CSVs
│   │   └── sensors/                  # Generated sensor CSVs
│   └── processed/                    # Feature-engineered datasets
├── src/
│   ├── data_generation/
│   │   ├── ehr_generator.py          # Synthetic EHR generator (Indian patients)
│   │   └── sensor_generator.py       # Synthetic wearable data generator
│   ├── data_ingestion/
│   │   ├── ehr_loader.py             # EHR data loader & validator
│   │   └── sensor_loader.py          # Sensor loader & stream simulator
│   ├── preprocessing/
│   │   ├── ehr_preprocessor.py       # EHR feature extraction
│   │   ├── sensor_preprocessor.py    # Sensor feature extraction & windowing
│   │   └── feature_engineering.py    # Feature fusion & data splitting
│   ├── models/
│   │   ├── glucose_predictor.py      # BiLSTM + Attention
│   │   ├── cardio_risk_scorer.py     # XGBoost + SHAP
│   │   ├── fusion_model.py           # Multi-modal fusion network
│   │   └── train.py                  # Training orchestrator
│   ├── digital_twin/
│   │   ├── patient_state.py          # PatientDigitalTwin class
│   │   ├── twin_engine.py            # Twin orchestration engine
│   │   └── alert_system.py           # Alert management system
│   ├── evaluation/
│   │   ├── metrics.py                # Clarke Error Grid + custom metrics
│   │   └── evaluate.py               # Evaluation pipeline
│   └── utils/
│       └── helpers.py                # Shared utilities
├── models/                           # Saved model weights
├── results/                          # Evaluation results & plots
├── dashboard/
│   ├── app.py                        # Streamlit main entry point
│   ├── views/
│   │   ├── patient_overview.py       # Patient profile & vitals
│   │   ├── glucose_monitoring.py     # Glucose prediction & alerts
│   │   ├── cardio_risk.py            # Cardiovascular risk analysis
│   │   ├── what_if.py                # Intervention simulator
│   │   └── alerts.py                 # Alert management
│   └── components/
│       ├── charts.py                 # Reusable Plotly charts
│       └── widgets.py                # Reusable UI widgets
├── scripts/
│   └── generate_data.py              # Master data generation script
├── docs/
│   └── architecture_diagram.md       # Architecture documentation
└── tests/
    ├── test_data_generation.py
    ├── test_preprocessing.py
    └── test_models.py
```

---

## 🎥 Video Demo & Presentation Script

> **Video Presentation (20+ minutes):**https://youtu.be/XtqGpoZ1pvc
>
> 📜 **Complete 20-Minute Verbatim Script & Walkthrough Guide:** https://1drv.ms/w/c/05c46bebc511535c/IQBX2S0BZyMkSKnlRHJXRXc9AXFAKGfVDfzMCBrFnVxUjOc?e=tbz0Dj(docs/video_presentation_script.md)

The video covers:
1. Problem statement and Indian chronic healthcare context
2. Architecture walkthrough & multi-modal data fusion
3. Synthetic data generation demonstration
4. Model training and quantitative evaluation results (Clarke Error Grid, ROC-AUC)
5. Dashboard live clinical demonstration
6. "What-If" counterfactual intervention simulation
7. Future roadmap and clinical impact

---

## 📄 Documentation & Presentation Deliverables

All required competition deliverables are generated and ready in the repository:
- **Architecture Diagram (PDF):** [docs/architecture_diagram.pdf](docs/architecture_diagram.pdf)
- **Architecture Diagram (Markdown & Mermaid):** [docs/architecture_diagram.md](docs/architecture_diagram.md)
- **Summit Presentation (PowerPoint PPTX):** [docs/presentation.pptx]https://1drv.ms/p/c/05c46bebc511535c/IQB4RfBetWd1R7_EquAIyCTbAWRERG-YS7kfbwNvSRecbnY?e=Fdcv6G
- **Summit Presentation (PDF):** [docs/presentation.pdf](docs/presentation.pdf)
- **Clinical Evaluation Report:** [results/evaluation_report.md](results/evaluation_report.md)

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Digital Twin PoC Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 🙏 Acknowledgments

- **Happiest Health** for organizing the Reimagining & Reforming Healthcare in India Summit 2026
- **Synthea** community for inspiration on synthetic patient data generation
- **PhysioNet** for open-source physiological datasets and clinical data standards
- **NeuroKit2** for HRV analysis tools
- **SHAP** library for model explainability

---

## 📚 References

1. ICMR-INDIAB Study — Prevalence of diabetes and prediabetes in India
2. Clarke, W.L. et al. (1987) — "Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose"
3. Yoon et al. (NeurIPS 2019) — "Time-series Generative Adversarial Networks"
4. WHO Asia-Pacific BMI Classification — Indian obesity cutoffs
5. HL7 FHIR R4 Specification — Healthcare interoperability standard

---

> **Contact:** For questions about this project, please reach out to the team leader at _[email]_.
