import streamlit as st
import pandas as pd
from components.widgets import render_vital_card, render_patient_card, render_condition_tags, render_medication_list

def render_patient_overview(engine, patient_id: str):
    st.title("Patient Overview")
    
    # Mock data if using demo
    patient_info = {"name": f"John Doe ({patient_id})", "age": 55, "gender": "Male", "bmi": 28.5, "blood_type": "A+"}
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_vital_card("Glucose", "125 mg/dL", "15", "inverse", "🩸")
    with col2:
        render_vital_card("Heart Rate", "72 bpm", "-2", "normal", "❤️")
    with col3:
        render_vital_card("Blood Pressure", "128/82", None, "normal", "🫀")
    with col4:
        render_vital_card("SpO2", "98%", "1%", "normal", "🫁")
        
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        render_patient_card(patient_info)
        st.subheader("Conditions")
        render_condition_tags(["Type 2 Diabetes", "Hypertension", "Hyperlipidemia"])
        
    with col2:
        st.subheader("Medical History")
        st.markdown("- **2021**: Diagnosed with T2D")
        st.markdown("- **2019**: Diagnosed with Hypertension")
        
    st.markdown("---")
    st.subheader("Latest Lab Results")
    labs = pd.DataFrame([
        {"Test": "HbA1c", "Value": "7.2%", "Reference": "< 5.7%"},
        {"Test": "LDL", "Value": "115 mg/dL", "Reference": "< 100 mg/dL"},
        {"Test": "Triglycerides", "Value": "160 mg/dL", "Reference": "< 150 mg/dL"}
    ])
    st.table(labs)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Active Medications")
        render_medication_list([
            {"name": "Metformin", "dosage": "1000mg BID"},
            {"name": "Lisinopril", "dosage": "10mg QD"},
            {"name": "Atorvastatin", "dosage": "20mg QD"}
        ])
    with col2:
        st.subheader("Active Alerts")
        st.info("No critical alerts at this time.")
