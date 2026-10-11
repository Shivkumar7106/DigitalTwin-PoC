import streamlit as st
import sys
import os
import random
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pages.patient_overview import render_patient_overview
from pages.glucose_monitoring import render_glucose_monitoring
from pages.cardio_risk import render_cardio_risk
from pages.what_if import render_what_if
from pages.alerts import render_alerts

st.set_page_config(page_title='Digital Twin - Healthcare PoC', layout='wide', page_icon='🏥')

class DemoMode:
    def __init__(self):
        self.patients = ["P001", "P002", "P003"]
    
    def get_patient_info(self, pid):
        return {"name": f"Patient {pid}", "age": 45, "gender": "Male", "bmi": 26.5, "blood_type": "O+"}
    
    def get_vitals(self, pid):
        return {"glucose": "110", "hr": "72", "bp": "120/80", "spo2": "98%"}
    
    def get_glucose_data(self, pid):
        times = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min')
        base = 100
        noise = np.random.normal(0, 5, 100)
        return {"timestamps": times, "actual": base + noise, "predicted": base + noise + 2}

@st.cache_resource
def get_engine():
    try:
        from src.digital_twin.twin_engine import DigitalTwinEngine
        return DigitalTwinEngine()
    except Exception as e:
        return DemoMode()

engine = get_engine()

# Load patients dynamically if available
patients_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'ehr', 'patients.csv')
if os.path.exists(patients_path):
    try:
        pts_df = pd.read_csv(patients_path)
        available_patients = pts_df['patient_id'].tolist() if 'patient_id' in pts_df.columns else pts_df['id'].tolist()
    except Exception:
        available_patients = ["P001", "P002", "P003"]
else:
    available_patients = ["P001", "P002", "P003"]

# Session State
if "selected_patient" not in st.session_state or st.session_state.selected_patient not in available_patients:
    st.session_state.selected_patient = available_patients[0]
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False

st.sidebar.title("🏥 Digital Twin PoC")
patient = st.sidebar.selectbox("Select Patient", available_patients, index=available_patients.index(st.session_state.selected_patient))
st.session_state.selected_patient = patient

page = st.sidebar.radio("Navigation", ["Patient Overview", "Glucose Monitoring", "Cardiovascular Risk", "What-If Simulator", "Alerts"])

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation")
if st.sidebar.button("Toggle Simulation"):
    st.session_state.simulation_running = not st.session_state.simulation_running
st.sidebar.write("Running: ", st.session_state.simulation_running)
st.sidebar.slider("Speed Factor", 1, 10, 1)

st.sidebar.markdown("---")
st.sidebar.info("Digital Twin PoC v1.0")

if page == "Patient Overview":
    render_patient_overview(engine, st.session_state.selected_patient)
elif page == "Glucose Monitoring":
    render_glucose_monitoring(engine, st.session_state.selected_patient)
elif page == "Cardiovascular Risk":
    render_cardio_risk(engine, st.session_state.selected_patient)
elif page == "What-If Simulator":
    render_what_if(engine, st.session_state.selected_patient)
elif page == "Alerts":
    render_alerts(engine, st.session_state.selected_patient)
