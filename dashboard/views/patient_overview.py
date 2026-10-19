import streamlit as st
import pandas as pd
from components.widgets import render_vital_card, render_patient_card, render_condition_tags, render_medication_list

def render_patient_overview(engine, patient_id: str):
    st.title("Patient Overview")
    
    # Ingest from real DigitalTwinEngine
    twin = engine.initialize_twin(patient_id)
    demo = twin.demographics
    vitals = twin.current_vitals
    med_hist = twin.medical_history
    
    full_name = f"{demo.get('first_name', 'Patient')} {demo.get('last_name', patient_id[:6])}"
    patient_info = {
        "name": f"{full_name} ({patient_id[:8]})",
        "age": demo.get('age', 45),
        "gender": demo.get('gender', 'Unknown'),
        "bmi": demo.get('bmi', 25.0),
        "blood_type": demo.get('blood_type', 'O+')
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_vital_card("Glucose", f"{vitals.glucose_mg_dl:.1f} mg/dL", None, "normal", "🩸")
    with col2:
        render_vital_card("Heart Rate", f"{vitals.heart_rate_bpm:.0f} bpm", None, "normal", "❤️")
    with col3:
        render_vital_card("Blood Pressure", f"{vitals.systolic_bp:.0f}/{vitals.diastolic_bp:.0f}", None, "normal", "🫀")
    with col4:
        render_vital_card("SpO2", f"{vitals.spo2_pct:.0f}%", None, "normal", "🫁")
        
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        render_patient_card(patient_info)
        st.subheader("Active Diagnosed Conditions")
        conds = med_hist.get('conditions', [])
        if conds:
            render_condition_tags(conds)
        else:
            st.info("No active chronic conditions recorded.")
        
    with col2:
        st.subheader("Patient Clinical Profile")
        st.markdown(f"- **City**: {demo.get('city', 'Urban India')}")
        st.markdown(f"- **Smoking Status**: {demo.get('smoking_status', 'Never')}")
        st.markdown(f"- **Physical Activity Level**: {demo.get('physical_activity_level', 'Moderate')}")
        st.markdown(f"- **Family History T2D**: {'Yes' if demo.get('family_history_diabetes') else 'No'}")
        st.markdown(f"- **Family History CVD**: {'Yes' if demo.get('family_history_cvd') else 'No'}")
        st.markdown(f"- **Genetic Risk Factor**: {demo.get('genetic_risk_score', 0.2):.3f}")
        
    st.markdown("---")
    st.subheader("Latest Longitudinal Laboratory Panels")
    obs_list = med_hist.get('observations', [])
    if obs_list:
        # Group by observation_name and show latest
        obs_df = pd.DataFrame(obs_list)
        name_col = 'observation_name' if 'observation_name' in obs_df.columns else 'name'
        date_col = 'observation_date' if 'observation_date' in obs_df.columns else 'date'
        val_col = 'value'
        unit_col = 'unit' if 'unit' in obs_df.columns else None
        
        latest_obs = obs_df.sort_values(date_col).groupby(name_col).last().reset_index()
        display_rows = []
        for _, row in latest_obs.iterrows():
            unit_str = f" {row[unit_col]}" if unit_col and pd.notna(row[unit_col]) else ""
            display_rows.append({
                "Test / Biomarker": row[name_col],
                "Latest Value": f"{row[val_col]}{unit_str}",
                "Date": str(row[date_col])[:10]
            })
        st.table(pd.DataFrame(display_rows))
    else:
        st.info("No laboratory panel records available for this patient.")
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Active Prescriptions & Dosages")
        meds = twin.medications
        if meds:
            render_medication_list([{"name": m, "dosage": "Active Regimen"} for m in meds])
        else:
            st.info("No active pharmacological medications recorded.")
    with col2:
        st.subheader("Active Evaluated Alerts")
        alerts = engine.get_patient_alerts(patient_id)
        if alerts:
            for a in alerts[:3]:
                st.warning(f"**[{a.severity}]** {a.message}")
        else:
            st.success("✅ All evaluated telemetry within safe clinical parameters.")
