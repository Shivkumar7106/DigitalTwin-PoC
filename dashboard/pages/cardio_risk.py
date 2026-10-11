import streamlit as st
import pandas as pd
import numpy as np
from components.charts import create_risk_gauge, create_shap_waterfall, create_risk_trend_chart, create_heart_rate_chart

def render_cardio_risk(engine, patient_id: str):
    st.title("Cardiovascular Risk Assessment")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(create_risk_gauge(45, "10-Year ASCVD Risk"), use_container_width=True)
        st.markdown("<h3 style='text-align: center; color: orange;'>Moderate Risk</h3>", unsafe_allow_html=True)
    
    with c2:
        features = ["Age", "Systolic BP", "Total Chol", "HDL", "Smoking", "Diabetes", "BMI"]
        vals = [55, 135, 210, 40, "No", "Yes", 28.5]
        shaps = [3.5, 2.1, 1.5, -1.0, 0, 4.0, 1.2]
        st.plotly_chart(create_shap_waterfall(shaps, features, vals, 20), use_container_width=True)
        
    st.markdown("---")
    
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Modifiable Risk Factors")
        st.info("💡 **Recommendation:** Lowering Systolic BP to < 120 can reduce risk by 15%.")
        st.info("💡 **Recommendation:** Reducing BMI to 25 can reduce risk by 5%.")
    with c4:
        st.subheader("Non-modifiable Factors")
        st.markdown("- **Age**: 55")
        st.markdown("- **Family History**: Positive for early CAD")
        
    st.markdown("---")
    
    # Trend
    dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
    trend = np.linspace(50, 45, 30) + np.random.normal(0, 1, 30)
    st.plotly_chart(create_risk_trend_chart(dates, trend, "ASCVD"), use_container_width=True)
    
    st.markdown("---")
    
    times = pd.date_range(end=pd.Timestamp.now(), periods=24, freq='H')
    hr = np.random.normal(70, 5, 24)
    hrv = np.random.normal(45, 10, 24)
    st.plotly_chart(create_heart_rate_chart(times, hr, hrv), use_container_width=True)
