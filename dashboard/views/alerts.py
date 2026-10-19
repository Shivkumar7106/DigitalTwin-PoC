import streamlit as st
import pandas as pd
from datetime import datetime
from components.widgets import render_alert_badge

def render_alerts(engine, patient_id: str):
    st.title("Patient Clinical Alert Management")
    st.markdown("""
    Evaluated in real-time from the patient's continuous glycemic streams, hemodynamic telemetry, and multi-modal risk scores.
    Uses clinical hysteresis and cooldown debouncing to avoid alarm fatigue.
    """)
    
    twin = engine.initialize_twin(patient_id)
    vitals = twin.current_vitals
    risk = twin.current_risk
    
    # Evaluate all active alerts for this specific patient
    active_alerts = engine.get_patient_alerts(patient_id)
    
    # Categorize
    crit_count = sum(1 for a in active_alerts if a.severity == 'CRITICAL')
    high_count = sum(1 for a in active_alerts if a.severity == 'HIGH')
    mod_count = sum(1 for a in active_alerts if a.severity == 'MODERATE')
    info_count = sum(1 for a in active_alerts if a.severity == 'INFO')
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Critical Alerts", crit_count)
    with c2: st.metric("High Priority", high_count)
    with c3: st.metric("Moderate Priority", mod_count)
    with c4: st.metric("Info / Routine", info_count)
    
    st.markdown("---")
    st.subheader(f"Active Evaluated Alerts for Patient ({patient_id[:8]})")
    
    if active_alerts:
        for alert in active_alerts:
            time_str = alert.timestamp.strftime("%I:%M %p") if isinstance(alert.timestamp, datetime) else str(alert.timestamp)
            render_alert_badge(alert.severity, f"[{alert.metric.upper()}] {alert.message} (Observed: {alert.value:.1f}, Threshold: {alert.threshold:.1f})", time_str)
    else:
        st.success(f"✅ **No active clinical alerts.** All telemetry (Glucose: {vitals.glucose_mg_dl:.1f} mg/dL, HR: {vitals.heart_rate_bpm:.0f} bpm, CVD Risk: {risk.cardiovascular_risk*100:.1f}%) within physiological limits.")
        
    st.markdown("---")
    st.subheader("Simulated Audit Trail & Historical Alert Log")
    
    # Generate realistic historical audit log from patient history
    hist_entries = []
    if vitals.glucose_mg_dl > 160:
        hist_entries.append({
            "Timestamp": "2026-10-18 13:45",
            "Severity": "MODERATE",
            "Category": "Glycemic",
            "Event Description": f"Postprandial glucose reached {vitals.glucose_mg_dl:.1f} mg/dL following meal flag.",
            "Action Taken": "Clinician notification sent; monitored via CGM trace."
        })
    if vitals.heart_rate_bpm > 85:
        hist_entries.append({
            "Timestamp": "2026-10-18 09:15",
            "Severity": "MODERATE",
            "Category": "Hemodynamic",
            "Event Description": f"Elevated resting heart rate ({vitals.heart_rate_bpm:.0f} bpm) detected during seated period.",
            "Action Taken": "Flagged in daily cardiovascular summary."
        })
    hist_entries.extend([
        {
            "Timestamp": "2026-10-17 22:30",
            "Severity": "INFO",
            "Category": "Sleep / Recovery",
            "Event Description": "Sleep architecture recorded 7.2 hours with 88% sleep efficiency.",
            "Action Taken": "Routine baseline sync."
        },
        {
            "Timestamp": "2026-10-16 08:00",
            "Severity": "INFO",
            "Category": "Activity",
            "Event Description": "Daily baseline step count reached target (8,450 steps).",
            "Action Taken": "Routine activity logged."
        }
    ])
    
    st.dataframe(pd.DataFrame(hist_entries), use_container_width=True)
    
    with st.expander("Clinical Alert Thresholds & Debouncing Parameters"):
        st.markdown("**Glucose Alert Boundaries:**")
        st.write("- **Critical Hypoglycemia:** < 54 mg/dL (Immediate action)")
        st.write("- **Hypoglycemia Alert:** < 70 mg/dL")
        st.write("- **Hyperglycemic Spike Threshold:** > 180 mg/dL")
        st.write("- **Severe Hyperglycemia:** > 250 mg/dL")
        st.write("- **Minimum Cooldown Window:** 30 minutes")
        st.markdown("**Hemodynamic Boundaries:**")
        st.write("- **Bradycardia / Tachycardia:** < 40 bpm / > 150 bpm")
        st.write("- **Elevated Cardiovascular Hazard:** > 20% 10-year ASCVD")
