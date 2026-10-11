# Digital Twin Architecture Documentation

## 1. System Overview

```mermaid
flowchart TD
    subgraph Data Sources
        EHR[EHR Data]
        Sensors[Wearable Sensors / CGM]
    end

    subgraph Ingestion Pipeline
        EHR_L[EHR Loader]
        Sens_L[Sensor Loader]
    end

    subgraph Processing & Storage
        PreProc[Data Preprocessor]
        FS[Feature Store]
    end

    subgraph Digital Twin Core
        GlucMod[Glucose Predictor - BiLSTM]
        CardioMod[Cardio Risk - XGBoost]
        Fusion[Multimodal Fusion]
    end

    subgraph Presentation
        Dash[Clinical Dashboard]
        Alerts[Alerting System]
    end

    EHR --> EHR_L
    Sensors --> Sens_L
    EHR_L --> PreProc
    Sens_L --> PreProc
    PreProc --> FS
    FS --> GlucMod
    FS --> CardioMod
    GlucMod --> Fusion
    CardioMod --> Fusion
    Fusion --> Dash
    Fusion --> Alerts
```

## 2. Data Pipeline

```mermaid
sequenceDiagram
    participant EHRGen as EHR Generator
    participant SensGen as Sensor Generator
    participant CSV as Raw CSV
    participant Loader as Data Loader
    participant Prep as Preprocessor
    participant FS as Feature Store

    EHRGen->>CSV: Generate Patient Records
    SensGen->>CSV: Generate CGM/Vitals
    CSV->>Loader: Read raw data
    Loader->>Prep: Clean & Align Timestamps
    Prep->>FS: Save engineered features
```

## 3. Model Architecture

```mermaid
flowchart LR
    subgraph Inputs
        T[Time-Series Data]
        S[Static Patient Data]
    end

    subgraph BiLSTM Model
        L1[BiLSTM Layer]
        Att[Attention Mechanism]
    end

    subgraph XGBoost Model
        XGB[Gradient Boosting Trees]
        SHAP[SHAP Explainer]
    end

    T --> L1
    L1 --> Att
    Att --> GlucPred[Glucose Prediction]

    S --> XGB
    XGB --> CardioRisk[Cardio Risk Score]
    XGB --> SHAP
```

## 4. Digital Twin State Machine

```mermaid
stateDiagram-v2
    [*] --> Initializing
    Initializing --> Monitoring : Data Loaded
    Monitoring --> Predicting : Scheduled Interval
    Predicting --> Alerting : Risk Threshold Exceeded
    Predicting --> Idle : Normal Conditions
    Alerting --> Monitoring : Acknowledged
    Idle --> Monitoring : New Data Arrived
```

## 5. Dashboard Wireframe Description

- **Header:** Patient demographic info, current Digital Twin status (Active/Idle).
- **Left Panel (Vitals):** Real-time CGM graph, heart rate, and activity levels.
- **Center Panel (Predictions):** Next 2-hour glucose forecast with confidence intervals.
- **Right Panel (Risk):** Cardiovascular risk gauge with top SHAP feature contributors.
- **Bottom Panel (Alerts):** Log of generated alerts and clinical recommendations.

## 6. Technology Stack

| Component | Technology |
| --- | --- |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Deep Learning | PyTorch (BiLSTM + Attention) |
| Machine Learning | XGBoost |
| Interpretability | SHAP |
| Web Application | Streamlit / FastAPI |

## 7. Data Flow Description

The data flow starts with synthetic data generators simulating continuous data streams. The pipeline loads this into the preprocessor to handle missing values and align high-frequency sensor data with static EHR data. Features are extracted and saved to a Feature Store. The Digital Twin core utilizes models tuned on this store to continuously generate near-term (glucose) and long-term (cardio) predictions, driving the interactive dashboard and alerting mechanisms.
