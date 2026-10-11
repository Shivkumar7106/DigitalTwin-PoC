import streamlit as st
import numpy as np
import pandas as pd
from components.charts import create_glucose_timeline
from components.widgets import render_vital_card

def render_glucose_monitoring(engine, patient_id: str):
    st.title("Continuous Glucose Monitoring")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_vital_card("Current", "142 mg/dL", "5", "inverse")
    with col2:
        render_vital_card("Pred (30m)", "160 mg/dL", "18", "inverse")
    with col3:
        render_vital_card("Pred (2h)", "130 mg/dL", "-12", "normal")
    with col4:
        render_vital_card("TIR %", "85%", "2%", "normal", "🎯")
        
    # Alert banner demo
    st.warning("⚠️ Predicted spike > 180 mg/dL in 45 mins based on recent meal.")
    
    st.markdown("---")
    
    # Generate mock data
    times = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min')
    actual = np.random.normal(130, 15, 100)
    pred_times = pd.date_range(start=times[-1], periods=24, freq='5min')
    pred = np.random.normal(140, 20, 24)
    conf_l = pred - 15
    conf_u = pred + 15
    
    fig = create_glucose_timeline(
        timestamps=list(times) + list(pred_times),
        actual_glucose=list(actual) + [None]*24,
        predicted_glucose=[None]*100 + list(pred),
        confidence_lower=[None]*100 + list(conf_l),
        confidence_upper=[None]*100 + list(conf_u),
        meal_flags=[False]*90 + [True] + [False]*33
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Glucose Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", "135 mg/dL")
    c2.metric("CV %", "15%")
    c3.metric("TAR (>180)", "10%")
    c4.metric("TBR (<70)", "5%")
