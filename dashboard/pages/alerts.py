import streamlit as st
import pandas as pd
from components.widgets import render_alert_badge

def render_alerts(engine, patient_id: str):
    st.title("Patient Alerts")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Critical", 0)
    with c2: st.metric("High", 1)
    with c3: st.metric("Moderate", 2)
    with c4: st.metric("Info", 5)
    
    st.markdown("---")
    st.subheader("Active Alerts")
    
    render_alert_badge("HIGH", "Predicted glucose spike > 180 mg/dL in 45 minutes", "10:30 AM")
    render_alert_badge("MODERATE", "Missed medication window: Metformin", "09:00 AM")
    render_alert_badge("MODERATE", "Resting heart rate elevated (88 bpm) over 2 hours", "08:15 AM")
    
    st.markdown("---")
    st.subheader("Alert History")
    
    hist = pd.DataFrame([
        {"Date": "2023-10-24 15:00", "Severity": "INFO", "Message": "Completed daily exercise goal"},
        {"Date": "2023-10-23 22:00", "Severity": "MODERATE", "Message": "Poor sleep quality detected"},
        {"Date": "2023-10-22 14:30", "Severity": "HIGH", "Message": "Hypoglycemia risk (<70 mg/dL)"}
    ])
    st.dataframe(hist, use_container_width=True)
    
    with st.expander("Alert Configuration"):
        st.markdown("**Glucose Thresholds**")
        st.slider("Hypoglycemia Threshold", 50, 80, 70)
        st.slider("Hyperglycemia Threshold", 150, 250, 180)
        
        st.markdown("**Heart Rate Thresholds**")
        st.slider("Bradycardia", 30, 60, 50)
        st.slider("Tachycardia", 100, 150, 110)
