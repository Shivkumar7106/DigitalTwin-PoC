import streamlit as st
import pandas as pd
import numpy as np
from components.charts import create_risk_gauge, create_shap_waterfall, create_risk_trend_chart, create_heart_rate_chart

def render_cardio_risk(engine, patient_id: str):
    st.title("Cardiovascular Risk Assessment")
    
    # Calculate real patient-specific risk based on clinical guidelines
    cardio_res = engine.run_cardio_risk_assessment(patient_id)
    risk_pct = cardio_res['cardiovascular_risk']
    risk_cat = cardio_res['risk_category']
    contributions = cardio_res['contributions']
    
    twin = engine.initialize_twin(patient_id)
    demo = twin.demographics
    vitals = twin.current_vitals
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(create_risk_gauge(risk_pct, "10-Year ASCVD Risk (%)"), use_container_width=True)
        color = "green" if "Low" in risk_cat else ("orange" if "Moderate" in risk_cat else "red")
        st.markdown(f"<h3 style='text-align: center; color: {color};'>{risk_cat}</h3>", unsafe_allow_html=True)
        st.caption("Calculated using ACC/AHA ASCVD Risk Guidelines fused with nocturnal wearable autonomic telemetry.")
    
    with c2:
        st.subheader("Explainable Feature Attribution (SHAP-style Hazard Contribution)")
        features = list(contributions.keys())
        shaps = list(contributions.values())
        vals = [
            f"{demo.get('age', 45)} yrs",
            f"{vitals.systolic_bp:.0f} mmHg",
            "Elevated" if contributions.get('Cholesterol Ratio', 0) > 0 else "Normal",
            demo.get('smoking_status', 'Never'),
            "Yes" if any('diabetes' in str(c).lower() for c in twin.medical_history.get('conditions', [])) else "No",
            f"{vitals.hrv_rmssd_ms:.1f} ms",
            f"{vitals.heart_rate_bpm:.0f} bpm"
        ]
        st.plotly_chart(create_shap_waterfall(shaps, features, vals, base_value=5.0), use_container_width=True)
        
    st.markdown("---")
    
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Modifiable Risk Factors & Clinical Recommendations")
        if vitals.systolic_bp and vitals.systolic_bp > 120:
            drop_bp = round((vitals.systolic_bp - 120) * 0.3, 1)
            st.info(f"💡 **Blood Pressure Optimization:** Current Systolic BP is {vitals.systolic_bp:.0f} mmHg. Titrating antihypertensives to target < 120 mmHg can reduce ASCVD hazard by **~{drop_bp}%**.")
        if demo.get('bmi', 25) > 23.0:
            st.info(f"💡 **Weight & Metabolic Reduction:** Current BMI is {demo.get('bmi', 25):.1f} (Indian overweight threshold $\ge 23$). A 5% body mass reduction projects a **~2.5%** risk mitigation.")
        if demo.get('smoking_status', 'Never').lower() == 'current':
            st.warning("⚠️ **Smoking Cessation:** Active smoking contributes an estimated **+8.0%** absolute risk burden. Cessation therapy recommended.")
        if vitals.hrv_rmssd_ms < 25.0:
            st.info(f"💡 **Autonomic Health:** Nocturnal HRV RMSSD is suppressed ({vitals.hrv_rmssd_ms:.1f} ms). Structured aerobic exercise is advised to restore vagal tone.")
            
    with c4:
        st.subheader("Non-Modifiable Clinical Risk Factors")
        st.markdown(f"- **Patient Age:** {demo.get('age', 45)} years")
        st.markdown(f"- **Biological Sex:** {demo.get('gender', 'Unknown')}")
        st.markdown(f"- **Family History of Premature CVD:** {'Positive' if demo.get('family_history_cvd') else 'Negative'}")
        st.markdown(f"- **Family History of Diabetes:** {'Positive' if demo.get('family_history_diabetes') else 'Negative'}")
        st.markdown(f"- **Calculated Polygenic Risk Score:** {demo.get('genetic_risk_score', 0.2):.3f}")
        
    st.markdown("---")
    
    # Longitudinal telemetry traces
    st.subheader("Continuous Hemodynamic & Wearable Telemetry")
    hr_df = engine.sensor_loader.load_heart_rate_data(patient_id)
    if not hr_df.empty:
        recent_hr = hr_df.tail(288) # Last 24-48 hours of 1-minute data subsampled
        sampled_hr = recent_hr.iloc[::5] # Subsample every 5 mins for responsive chart
        
        times = list(sampled_hr['timestamp'])
        hr_vals = list(sampled_hr['heart_rate_bpm'])
        hrv_vals = list(sampled_hr['hrv_rmssd_ms']) if 'hrv_rmssd_ms' in sampled_hr.columns else None
        
        st.plotly_chart(create_heart_rate_chart(times, hr_vals, hrv_vals), use_container_width=True)
    else:
        st.info("No continuous heart rate telemetry available.")
