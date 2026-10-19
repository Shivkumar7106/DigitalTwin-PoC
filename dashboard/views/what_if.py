import streamlit as st
from components.charts import create_risk_gauge, create_metrics_comparison

def render_what_if(engine, patient_id: str):
    st.title("What-If Counterfactual Intervention Simulator")
    st.markdown("""
    Simulate patient-specific lifestyle and pharmacological interventions. 
    The engine computes counterfactual projections from the patient's current baseline using validated clinical relative risk models.
    """)
    
    twin = engine.initialize_twin(patient_id)
    demo = twin.demographics
    vitals = twin.current_vitals
    current_bmi = float(demo.get('bmi', 26.0))
    is_smoker = str(demo.get('smoking_status', 'Never')).lower() == 'current'
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Intervention Parameters")
        exercise = st.slider("Aerobic Exercise (minutes/day)", min_value=0, max_value=90, value=30, step=5,
                             help="Target: 150 min/week moderate-intensity exercise per AHA guidelines.")
        hba1c_red = st.slider("Target HbA1c Reduction (%)", min_value=0.0, max_value=3.0, value=0.5, step=0.1,
                              help="Based on intensive glycemic control trials (UKPDS).")
        weight_loss = st.slider("Target Weight Loss (kg)", min_value=0.0, max_value=20.0, value=4.0, step=0.5,
                                help="Reduces visceral adiposity, systolic BP, and insulin resistance.")
        
        stop_smoking = st.checkbox("Smoking Cessation Protocol", value=is_smoker, disabled=not is_smoker,
                                   help="Simulates 1-year cardiovascular hazard reduction post-cessation.")
        
        meds = st.multiselect(
            "Pharmacological Additions / Titrations",
            options=["Statin (Atorvastatin 20mg)", "ACE Inhibitor / ARB (Telmisartan 40mg)", "SGLT2 Inhibitor (Dapagliflozin 10mg)", "Metformin (1000mg BID)"],
            default=["Statin (Atorvastatin 20mg)"] if twin.current_risk.cardiovascular_risk > 0.15 else []
        )
        
        # Format med names
        clean_meds = []
        for m in meds:
            if "Statin" in m: clean_meds.append("Statin")
            if "ACE" in m or "Telmisartan" in m: clean_meds.append("ACE Inhibitor")
            if "SGLT2" in m: clean_meds.append("SGLT2 Inhibitor")
            if "Metformin" in m: clean_meds.append("Metformin")
            
        run = st.button("Run Simulation", type="primary", use_container_width=True)
        
    with col2:
        st.subheader("Counterfactual Simulation Results")
        
        # Calculate simulation
        changes = {
            'exercise_min': exercise,
            'hba1c_reduction': hba1c_red,
            'weight_loss_kg': weight_loss,
            'stop_smoking': stop_smoking,
            'meds_added': clean_meds
        }
        res = engine.run_what_if(patient_id, changes)
        
        base_risk = res['base_risk']
        sim_risk = res['simulated_risk']
        risk_drop = res['risk_reduction_pct']
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_risk_gauge(base_risk, "Baseline ASCVD Risk (%)"), use_container_width=True)
            st.caption(f"Current Status: SBP {vitals.systolic_bp:.0f} mmHg, BMI {current_bmi:.1f}")
        with c2:
            st.plotly_chart(create_risk_gauge(sim_risk, "Simulated Future Risk (%)"), use_container_width=True)
            st.caption(f"Projected Status: SBP {res['metrics_after']['Systolic BP']:.0f} mmHg, BMI {res['metrics_after']['BMI']:.1f}")
            
        st.markdown("---")
        st.subheader("Baseline vs. Counterfactual Biomarker Comparison")
        metrics_list = list(res['metrics_before'].keys())
        st.plotly_chart(create_metrics_comparison(res['metrics_before'], res['metrics_after'], metrics_list), use_container_width=True)
        
        st.subheader("Personalized Clinical Impact")
        if risk_drop > 0:
            st.success(f"""
            🎉 **Projected Absolute Risk Reduction:** **-{risk_drop:.1f}%** (from {base_risk:.1f}% down to {sim_risk:.1f}%).  
            - **Estimated Number Needed to Treat (NNT):** {max(1, int(100 / max(0.1, risk_drop)))} patients to prevent 1 major adverse cardiovascular event over 10 years.  
            - **Key Drivers:** {'Physical activity, ' if exercise > 0 else ''}{'Weight management, ' if weight_loss > 0 else ''}{'Statin therapy, ' if 'Statin' in clean_meds else ''}{'Smoking cessation.' if stop_smoking else ''}
            """)
        else:
            st.info("Adjust the sliders on the left and click 'Run Simulation' to evaluate alternative therapy paths.")
