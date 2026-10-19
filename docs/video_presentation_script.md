# 20-Minute Video Presentation Script & Demonstration Walkthrough

**Summit:** Happiest Health Reimagining & Reforming Healthcare in India Summit – 2026, Bengaluru  
**Project:** Digital Twin PoC – Predictive Health Engine for Type 2 Diabetes & Cardiovascular Risk  
**Target Video Duration:** 20 to 25 minutes  
**Format:** Slide Deck (PPTX/PDF) + Live System Demo of Dashboard  

---

## ⏱️ Video Presentation Timeline Outline

| Section | Timestamp | Topic / Focus | Visuals / Artifacts |
| :--- | :--- | :--- | :--- |
| **Part 1** | 00:00 – 03:00 | Introduction, Team Details & The Indian Chronic Epidemic | Slide 1 & 2 (`docs/presentation.pptx`) |
| **Part 2** | 03:00 – 06:30 | Digital Twin Paradigm & Multi-Modal Data Ingestion | Slide 3 & Architecture Diagram |
| **Part 3** | 06:30 – 11:00 | AI/ML Modeling: BiLSTM Attention, XGBoost SHAP, Fusion | Slide 4 & Technical Equations |
| **Part 4** | 11:00 – 15:00 | Clinical Validation, Clarke Error Grid & Metric Benchmarks | Slide 5 & `results/` plots |
| **Part 5** | 15:00 – 21:00 | Live System Demonstration: Clinician Dashboard & What-If | Live Browser (`http://localhost:8501`) |
| **Part 6** | 21:00 – 23:30 | Regulatory Privacy (DPDP Act), Scaling & Future Roadmap | Slide 6 & 7 |
| **Part 7** | 23:30 – 24:30 | Conclusion & Q&A Wrap-Up | Final Slide & GitHub Repository Link |

---

## 🎙️ Verbatim Speaker Script

### Part 1: Introduction, Team Details & The Indian Healthcare Challenge (00:00 – 03:00)
> *"Good morning, esteemed jury members, technical evaluators, and delegates of the Happiest Health Reimagining and Reforming Healthcare in India Summit 2026.*
>
> *Today, our team is proud to present our foundational prototype: the **Personalized Healthcare Digital Twin Engine**, specifically targeted at the dual chronic epidemics dominating India today: **Type 2 Diabetes** and **Cardiovascular Disease**.*
>
> *As stated in recent ICMR-INDIAB epidemiology studies, India is home to over 101 million individuals diagnosed with Type 2 Diabetes and over 315 million suffering from Hypertension. Crucially, the South Asian demographic exhibits a unique metabolic profile often termed the 'thin-fat phenotype'—where high visceral adiposity and insulin resistance occur at lower BMI thresholds. Consequently, Indians develop diabetes and cardiovascular complications up to a decade earlier than Western counterparts.*
>
> *The fundamental limitation of modern medicine is that it is reactive. Patients only consult clinicians when symptoms manifest or catastrophic events—such as acute myocardial infarction or diabetic hyperosmolar crises—have already occurred. Our challenge was to ask: **Can we create a dynamic, virtual replica of a patient that simulates their unique physiology in real time, predicting adverse events hours ahead and prescribing personalized interventions?** Let us explore how we built this foundational proof of concept."*

---

### Part 2: Digital Twin Architecture & Dual Data Streams (03:00 – 06:30)
> *(Transition to Architecture Diagram / Slide 3)*
>
> *"A true Healthcare Digital Twin cannot function on static records alone, nor can it rely exclusively on noisy wearables. Our system achieves physiological fidelity through the fusion of two heterogeneous data streams:*
>
> *1. **The Static/Historical Stream (EHR)**: Modeled using synthetic Indian patient profiles complying with HL7 FHIR standards. It encompasses demographics, chronic condition problem lists, active medications (Metformin, Statins, ACE inhibitors), and longitudinal quarterly laboratory panels (HbA1c, fasting glucose, lipid panels, eGFR, creatinine).*
>
> *2. **The Dynamic/Real-Time Stream (Wearables & Medical IoT)**: Continuous high-frequency telemetry including Continuous Glucose Monitoring (CGM) sampled at 5-minute intervals, continuous Heart Rate and Heart Rate Variability (HRV indices: RMSSD, SDNN), sleep stage architecture (Wake, Light, Deep, REM), and diurnal step counts.*
>
> *Under India's Digital Personal Data Protection (DPDP) Act and international HIPAA regulations, real patient data access is tightly restricted. To guarantee 100% regulatory compliance within the challenge sandbox, our entire system operates on clinically realistic synthetic data generators. We parameterized our generators with authentic physiological equations: modeling circadian dawn phenomena, postprandial glucose dynamics, nocturnal autonomic dipping, and exercise response curves."*

---

### Part 3: Algorithmic Formulations & AI Models (06:30 – 11:00)
> *(Transition to Model Architecture / Slide 4)*
>
> *"At the core of the Digital Twin are three intelligent predictive models:*
>
> *First, for **Continuous Glucose Spike Forecasting**, we deployed a **Bidirectional Long Short-Term Memory (BiLSTM) network equipped with a Self-Attention mechanism**. While standard recurrent networks suffer from forgetting distant temporal dynamics, our attention layer dynamically computes a normalized context weight across a 6-hour sliding window (72 timesteps). This allows the model to predict the continuous trajectory across the next 2-hour window (24 timesteps). If interstitial glucose is projected to exceed 180 mg/dL or rise faster than 3 mg/dL/min, a predictive alert is triggered.*
>
> *Second, for **Cardiovascular Risk Assessment**, we developed an **XGBoost gradient boosting classifier coupled with SHAP (SHapley Additive exPlanations)**. Rather than relying solely on static Framingham scores, our model ingests 35 multi-modal features—incorporating dynamic nocturnal HRV suppression, resting heart rate spikes, and sleep fragmentation alongside systolic BP and total cholesterol.*
>
> *Third, we constructed a **Multi-Modal Intermediate Fusion Neural Network**, featuring a temporal LSTM branch fused with a static clinical dense branch via a gated cross-modal layer. Finally, our **Clinical Alert Manager** applies debouncing and hysteresis logic across four triage tiers (Critical, High, Moderate, Info) to prevent clinician alarm fatigue."*

---

### Part 4: Clinical Validation & Quantitative Benchmarks (11:00 – 15:00)
> *(Transition to Results Slide 5 & `results/` Figures)*
>
> *"In clinical AI, standard mathematical metrics like mean squared error do not capture clinical danger. A model that underestimates glucose by 40 mg/dL in a hypoglycemic patient is fatal.*
>
> *Therefore, we validated our glucose forecasting engine using **Clarke Error Grid Analysis (CEGA)**. As displayed on the plot from our evaluation pipeline:*
> - *Our model achieved **99.60% of predictions in Clarke Zones A and B**, far exceeding the 95% clinical standard.*
> - *The **Mean Absolute Relative Difference (MARD)** is **7.47%**, satisfying the strict <10% threshold required for clinical-grade continuous glucose monitors.*
> - *The Root Mean Square Error (RMSE) is **10.31 mg/dL** over the 2-hour forecasting window.*
>
> *For our Cardiovascular Risk Scorer:*
> - *The model attained an **AUC-ROC of 0.965** and a **Precision-Recall AUC of 0.968**.*
> - *Crucially for medical triage, **Sensitivity (Recall) is 89.80%**, minimizing life-threatening false negatives, while maintaining a **Specificity of 89.39%** to avoid overburdening emergency staff.*
> - *Our complete automated test suite executed with 100% pass rates."*

---

### Part 5: Live Clinician Dashboard Demonstration (15:00 – 21:00)
> *(Screen Share: Open web browser to `http://localhost:8501`)*
>
> *"Now let us demonstrate how an attending physician or cardiologist interacts with this virtual patient through our live clinician dashboard:*
>
> **1. Patient Overview Screen (0:00 - 1:30 of demo):**
> *Notice the sidebar: we can select any verified patient in the cohort—such as Daksh Bakshi or Mahika Konda. Here on the Patient Overview, the clinician is immediately greeted by real-time vital summary cards computed directly from that individual's sensor stream—current interstitial glucose, resting heart rate, blood pressure, and oxygen saturation. Below, the system renders the patient's verified chronic condition problem list (Type 2 Diabetes, Stage 1 Hypertension), active medications, and quarterly lab panels with color-coded flags.*
>
> **2. Continuous Glucose Monitoring (1:30 - 3:00 of demo):**
> *Switching to the Glucose Monitoring tab: Here is the predictive twin in action. The solid blue trace displays the patient's observed CGM telemetry over the past 6 hours. Connected to this stream is our trained BiLSTM model, which generates a 2-hour forward forecast trajectory (dashed red line) with 95% confidence intervals.*
> *The alert banner above is not a static placeholder—it evaluates the forecast array in real time. If any predicted timestep exceeds 180 mg/dL, it calculates the exact minutes until spike onset and warns the clinician to initiate glycemic or nutritional interventions.*
>
> **3. Cardiovascular Risk & Explainability (3:00 - 4:30 of demo):**
> *Next, on the Cardiovascular Risk tab: We observe an ACC/AHA ASCVD 10-year risk gauge computed from the patient's actual age, systolic blood pressure, cholesterol ratio, smoking status, and diabetes diagnosis, combined with their wearable autonomic markers (nocturnal HRV RMSSD and resting heart rate). The waterfall plot breaks down the exact point contribution of each biomarker.*
>
> **4. 'What-If' Counterfactual Intervention Simulator (4:30 - 6:00 of demo):**
> *Finally, our What-If simulator demonstrates counterfactual clinical reasoning. Using the sliders on the left, we can adjust parameters—for example, adding 30 minutes of daily aerobic exercise, reducing HbA1c by 0.5%, losing 4 kg, or adding an SGLT2 inhibitor. Clicking 'Run Simulation' recalculates projected ASCVD hazard and systolic BP using clinical trial risk reduction factors, presenting a side-by-side baseline-versus-scenario comparison.*
>
> **5. Clinical Alert Management (6:00 of demo):**
> *Under the Alerts tab, the system reviews active alerts evaluated dynamically from the patient's live telemetry and forecast values, applying hysteresis debouncing to prevent alarm fatigue.*"

---

### Part 6: DPDP Regulatory Compliance & Future Roadmap (21:00 – 23:30)
> *(Transition to Slide 6 & 7)*
>
> *"As we look toward real-world deployment across hospital networks in India, four pillars guide our roadmap:*
>
> *1. **Regulatory Alignment**: The architecture is designed from the ground up for privacy-by-design under India's DPDP Act 2023, featuring local edge tokenization and federated learning compatibility.*
> *2. **Device Interoperability**: Seamless integration with ISO/IEEE 11073 standards for consumer wearables (smartwatches, commercial CGMs).*
> *3. **Closed-Loop Feedback**: Moving beyond predictive alerts to closed-loop automated insulin delivery systems and automated clinician communication.*
> *4. **Summit Showcase 2026**: We are prepared to bring this functional prototype to the summit in Bengaluru on November 20, 2026, demonstrating live wearable ingestion on stage."*

---

### Part 7: Conclusion & Summary (23:30 – 24:30)
> *"In conclusion, our Digital Twin proof of concept proves that proactive, predictive medicine in India is not a distant vision—it is technically feasible today.*
>
> *Our complete codebase, training pipelines, evaluation reports, and live dashboard are fully reproducible and accessible in our public GitHub repository under the MIT open-source license.*
>
> *Thank you to Happiest Health for spearheading this initiative. We look forward to answering your questions."*

---

## 📹 Recording & Production Checklist

1. **Resolution & Audio:** Record in 1080p (1920x1080) at 30/60 fps with a dedicated noise-canceling microphone.
2. **Dashboard Setup:** Ensure `py -3.11 -m streamlit run dashboard/app.py` is running smoothly at `http://localhost:8501`.
3. **Screen Layout:** Present slides in full-screen (`F5` in PowerPoint) and toggle to browser for Section 5 without window borders.
4. **Export Format:** Save video as `.mp4`, upload to YouTube as **Unlisted**, and paste the link into `README.md`.
