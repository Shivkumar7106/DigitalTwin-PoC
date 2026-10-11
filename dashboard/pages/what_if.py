import streamlit as st
from components.charts import create_risk_gauge, create_metrics_comparison

def render_what_if(engine, patient_id: str):
    st.title("What-If Intervention Simulator")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Interventions")
        exercise = st.slider("Exercise (min/day)", 0, 60, 0)
        hba1c_red = st.slider("HbA1c Target Reduction (%)", 0.0, 3.0, 0.0, 0.5)
        weight_loss = st.slider("Weight Loss (kg)", 0, 15, 0)
        stop_smoking = st.checkbox("Stop Smoking")
        meds = st.multiselect("Add Medication", ["Metformin", "Statin", "ACE Inhibitor", "SGLT2 Inhibitor"])
        
        run = st.button("Run Simulation", type="primary", use_container_width=True)
        
    with col2:
        st.subheader("Results")
        if not run:
            st.info("Adjust the interventions on the left and click 'Run Simulation' to see the potential impact.")
        else:
            base_risk = 45
            new_risk = base_risk - (exercise * 0.1) - (hba1c_red * 2) - (weight_loss * 0.5)
            if stop_smoking: new_risk -= 10
            if "Statin" in meds: new_risk -= 8
            new_risk = max(5, new_risk)
            
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(create_risk_gauge(base_risk, "Current Risk"), use_container_width=True)
            with c2:
                st.plotly_chart(create_risk_gauge(new_risk, "Simulated Risk"), use_container_width=True)
                
            metrics = ["HbA1c", "Sys BP", "Weight", "ASCVD Risk"]
            before = {"HbA1c": 7.2, "Sys BP": 135, "Weight": 85, "ASCVD Risk": 45}
            after = {
                "HbA1c": 7.2 - hba1c_red, 
                "Sys BP": 135 - (5 if "ACE Inhibitor" in meds else 0) - weight_loss*0.2, 
                "Weight": 85 - weight_loss, 
                "ASCVD Risk": new_risk
            }
            
            st.plotly_chart(create_metrics_comparison(before, after, metrics), use_container_width=True)
            
            st.subheader("Personalized Recommendations")
            st.success(f"These interventions could reduce overall risk by {base_risk - new_risk:.1f}%.")
