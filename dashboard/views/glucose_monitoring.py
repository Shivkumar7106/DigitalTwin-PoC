import streamlit as st
import numpy as np
import pandas as pd
from datetime import timedelta
from components.charts import create_glucose_timeline
from components.widgets import render_vital_card

def render_glucose_monitoring(engine, patient_id: str):
    st.title("Continuous Glucose Monitoring & Predictive Forecasting")
    
    twin = engine.initialize_twin(patient_id)
    cgm_df = engine.sensor_loader.load_cgm_data(patient_id)
    
    if cgm_df.empty:
        st.error(f"No continuous glucose monitoring sensor data found for patient {patient_id}.")
        return

    # Run trained model forecast (2-hour trajectory = 24 timesteps ahead)
    pred_res = engine.run_glucose_prediction(patient_id)
    predictions = pred_res['predicted_values']
    conf_l = pred_res['confidence_lower']
    conf_u = pred_res['confidence_upper']
    
    # Calculate patient-specific glycemic metrics from real CGM stream
    all_glucose = cgm_df['glucose_mg_dl'].values
    current_glucose = float(all_glucose[-1])
    mean_glucose = float(np.mean(all_glucose))
    std_glucose = float(np.std(all_glucose))
    cv_pct = float((std_glucose / mean_glucose) * 100.0) if mean_glucose > 0 else 0.0
    tir_pct = float(np.mean((all_glucose >= 70) & (all_glucose <= 180)) * 100.0)
    tar_pct = float(np.mean(all_glucose > 180) * 100.0)
    tbr_pct = float(np.mean(all_glucose < 70) * 100.0)
    
    pred_30m = predictions[5] if len(predictions) > 5 else predictions[-1]
    pred_2h = predictions[-1] if len(predictions) > 0 else current_glucose
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        delta_val = f"{current_glucose - all_glucose[-2]:+.1f}" if len(all_glucose) > 1 else None
        render_vital_card("Current Glucose", f"{current_glucose:.1f} mg/dL", delta_val, "inverse", "🩸")
    with col2:
        render_vital_card("Forecast (30m Ahead)", f"{pred_30m:.1f} mg/dL", f"{pred_30m - current_glucose:+.1f}", "inverse", "⏱️")
    with col3:
        render_vital_card("Forecast (2h Horizon)", f"{pred_2h:.1f} mg/dL", f"{pred_2h - current_glucose:+.1f}", "normal" if pred_2h <= 180 else "inverse", "🔮")
    with col4:
        render_vital_card("Time-In-Range (TIR)", f"{tir_pct:.1f}%", f"{tar_pct:.1f}% TAR", "normal" if tir_pct >= 70 else "inverse", "🎯")
        
    # Dynamic evaluated alert banner
    spikes = [p for p in predictions if p > 180.0]
    hypos = [p for p in predictions if p < 70.0]
    if spikes:
        max_spike = max(spikes)
        first_step = next(i for i, p in enumerate(predictions) if p > 180.0)
        time_to_spike = (first_step + 1) * 5
        st.error(f"🚨 **PREDICTED HYPERGLYCEMIC SPIKE**: Forecast projects glucose reaching **{max_spike:.1f} mg/dL** in approx **{time_to_spike} minutes**. Recommend early nutritional/insulin titration.")
    elif hypos:
        st.error(f"⚠️ **HYPOGLYCEMIA WARNING**: Forecast projects interstitial glucose dropping below safe threshold (< 70 mg/dL). Immediate carbohydrate intake advised.")
    else:
        st.success(f"✅ **GLYCEMIC STABILITY**: 2-hour predictive forecast indicates glucose staying within target range (70–180 mg/dL).")
        
    st.markdown("---")
    
    # Display recent actual trace (last 72 readings = 6 hours) paired with 2-hour forecast (24 readings)
    recent_actual = cgm_df.tail(72).copy()
    actual_times = list(recent_actual['timestamp'])
    actual_vals = list(recent_actual['glucose_mg_dl'])
    meal_flags = list(recent_actual['meal_flag'] if 'meal_flag' in recent_actual.columns else [False]*len(actual_vals))
    
    # Future prediction timestamps (5-minute intervals following latest actual reading)
    last_dt = actual_times[-1]
    pred_times = [last_dt + timedelta(minutes=5 * (i + 1)) for i in range(len(predictions))]
    
    combined_times = actual_times + pred_times
    aligned_actual = actual_vals + [None] * len(predictions)
    aligned_pred = [None] * (len(actual_vals) - 1) + [actual_vals[-1]] + list(predictions)
    aligned_lower = [None] * (len(actual_vals) - 1) + [actual_vals[-1]] + list(conf_l)
    aligned_upper = [None] * (len(actual_vals) - 1) + [actual_vals[-1]] + list(conf_u)
    aligned_meals = [bool(m) for m in meal_flags] + [False] * len(predictions)
    
    fig = create_glucose_timeline(
        timestamps=combined_times,
        actual_glucose=aligned_actual,
        predicted_glucose=aligned_pred,
        confidence_lower=aligned_lower,
        confidence_upper=aligned_upper,
        meal_flags=aligned_meals
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Statistical Glycemic Summary (Past 30 Days)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("30-Day Mean Glucose", f"{mean_glucose:.1f} mg/dL")
    c2.metric("Coefficient of Variation (CV)", f"{cv_pct:.1f}%", "Target < 36%" if cv_pct <= 36 else "Elevated Variability")
    c3.metric("Time Above Range (>180)", f"{tar_pct:.1f}%")
    c4.metric("Time Below Range (<70)", f"{tbr_pct:.1f}%")
