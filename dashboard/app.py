import streamlit as st
import sys
import os
import random
import pandas as pd
import numpy as np

# Add project root and dashboard to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from views.patient_overview import render_patient_overview
from views.glucose_monitoring import render_glucose_monitoring
from views.cardio_risk import render_cardio_risk
from views.what_if import render_what_if
from views.alerts import render_alerts

st.set_page_config(page_title='Digital Twin - Healthcare PoC', layout='wide', page_icon='🏥')

# Hide default multi-page navigation links at the top of the sidebar
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

@st.cache_data
def load_available_patients():
    patients_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'ehr', 'patients.csv')
    patient_label_map = {}
    if os.path.exists(patients_path):
        try:
            pts_df = pd.read_csv(patients_path)
            id_col = 'patient_id' if 'patient_id' in pts_df.columns else 'id'
            available_patients = pts_df[id_col].tolist()
            for _, r in pts_df.iterrows():
                pid = str(r[id_col])
                name = f"{r.get('first_name', 'Patient')} {r.get('last_name', '')}".strip()
                gender = str(r.get('gender', ''))[:1]
                patient_label_map[pid] = f"{name} ({gender}) - {pid[:8]}"
            return available_patients, patient_label_map
        except Exception:
            return ["P001", "P002", "P003"], {}
    return ["P001", "P002", "P003"], {}

available_patients, patient_label_map = load_available_patients()

@st.cache_resource
def get_engine():
    try:
        from src.digital_twin.twin_engine import DigitalTwinEngine
        eng = DigitalTwinEngine()
        if available_patients:
            eng.warmup(available_patients[0])
        return eng
    except Exception as e:
        return DemoMode()

engine = get_engine()

# Session State
if "selected_patient" not in st.session_state or st.session_state.selected_patient not in available_patients:
    st.session_state.selected_patient = available_patients[0]
if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False

st.sidebar.title("🏥 Digital Twin PoC")
patient = st.sidebar.selectbox(
    "Select Patient", 
    available_patients, 
    index=available_patients.index(st.session_state.selected_patient),
    format_func=lambda pid: patient_label_map.get(pid, pid[:12])
)
if patient != st.session_state.selected_patient:
    st.session_state.selected_patient = patient
    if hasattr(engine, 'warmup'):
        engine.warmup(patient)

page = st.sidebar.radio("Navigation", ["Patient Overview", "Glucose Monitoring", "Cardiovascular Risk", "What-If Simulator", "Alerts"])

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
