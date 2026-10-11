import streamlit as st

def render_vital_card(label: str, value: str, delta: str = None, delta_color: str = 'normal', icon: str = '❤️'):
    st.metric(label=f"{icon} {label}", value=value, delta=delta, delta_color=delta_color)

def render_patient_card(patient_info: dict):
    with st.container(border=True):
        st.markdown(f"### {patient_info.get('name', 'Unknown')}")
        st.markdown(f"**Age:** {patient_info.get('age', '--')} | **Gender:** {patient_info.get('gender', '--')}")
        st.markdown(f"**BMI:** {patient_info.get('bmi', '--')} | **Blood Type:** {patient_info.get('blood_type', '--')}")

def render_alert_badge(severity: str, message: str, timestamp: str):
    severity = severity.upper()
    if severity == 'CRITICAL':
        st.error(f"🚨 **{timestamp} - CRITICAL:** {message}")
    elif severity == 'HIGH':
        st.warning(f"⚠️ **{timestamp} - HIGH:** {message}")
    elif severity == 'MODERATE':
        st.info(f"ℹ️ **{timestamp} - MODERATE:** {message}")
    else:
        st.success(f"✅ **{timestamp} - INFO:** {message}")

def render_condition_tags(conditions: list):
    tags = "".join([f"<span style='display:inline-block; padding: 2px 8px; margin: 2px; border-radius: 12px; background-color: #e0e0e0; color: #333; font-size: 0.9em;'>{c}</span>" for c in conditions])
    st.markdown(tags, unsafe_allow_html=True)

def render_medication_list(medications: list):
    for med in medications:
        st.markdown(f"- **{med.get('name', '')}**: {med.get('dosage', '')}")

def render_risk_summary(risk_scores: dict):
    for name, score in risk_scores.items():
        color = "red" if score > 60 else "orange" if score > 40 else "yellow" if score > 20 else "green"
        st.markdown(f"**{name}:** {score}% <div style='width: 100%; background-color: #f0f0f0; border-radius: 4px;'><div style='width: {score}%; background-color: {color}; height: 8px; border-radius: 4px;'></div></div>", unsafe_allow_html=True)
