"""
Digital Twin Engine orchestrator.
Manages the full pipeline of the Digital Twin for multiple patients.
"""
from typing import Dict, List, Any, Generator, Optional
import time
from datetime import datetime

from .patient_state import PatientDigitalTwin, VitalSigns, RiskScores
from .alert_system import AlertSystem

class DigitalTwinEngine:
    """Orchestrates the Digital Twin pipeline, managing models and active twins."""
    def __init__(self, config_path: str = 'config/settings.yaml'):
        self.config_path = config_path
        # Mock loaders / preprocessors / models for this engine layout
        self.ehr_loader = None
        self.sensor_loader = None
        self.glucose_predictor = None
        self.cardio_risk_scorer = None
        
        self.active_twins: Dict[str, PatientDigitalTwin] = {}
        self.alert_system = AlertSystem()

    def initialize_twin(self, patient_id: str) -> PatientDigitalTwin:
        """Load patient data and create a new active PatientDigitalTwin."""
        # Mocking data loading
        demographics = {'age': 45, 'gender': 'M'}
        medical_history = {'diabetes_type': 2, 'smoker': False}
        medications = ['Metformin']
        
        twin = PatientDigitalTwin(patient_id, demographics, medical_history, medications)
        
        # Initial risk assessment mock
        initial_risk = RiskScores(
            glucose_spike_risk=0.1,
            cardiovascular_risk=0.15,
            composite_risk=0.12,
            risk_category='Low',
            last_updated=datetime.now()
        )
        twin.update_risk_scores(initial_risk)
        self.active_twins[patient_id] = twin
        return twin

    def update_twin(self, patient_id: str, new_sensor_data: dict) -> dict:
        """Update a specific twin with new sensor data and evaluate pipelines."""
        twin = self.active_twins.get(patient_id)
        if not twin:
            raise ValueError(f"Twin for patient {patient_id} not initialized.")

        vital_signs = VitalSigns(
            glucose_mg_dl=new_sensor_data.get('glucose_mg_dl', twin.current_vitals.glucose_mg_dl),
            heart_rate_bpm=new_sensor_data.get('heart_rate_bpm', twin.current_vitals.heart_rate_bpm),
            hrv_rmssd_ms=new_sensor_data.get('hrv_rmssd_ms', twin.current_vitals.hrv_rmssd_ms),
            hrv_sdnn_ms=new_sensor_data.get('hrv_sdnn_ms', twin.current_vitals.hrv_sdnn_ms),
            systolic_bp=new_sensor_data.get('systolic_bp', twin.current_vitals.systolic_bp),
            diastolic_bp=new_sensor_data.get('diastolic_bp', twin.current_vitals.diastolic_bp),
            spo2_pct=new_sensor_data.get('spo2_pct', twin.current_vitals.spo2_pct),
            timestamp=new_sensor_data.get('timestamp', datetime.now())
        )
        twin.update_vitals(vital_signs)
        
        # Mock logic: run predictions if we have enough data
        if len(twin.glucose_history) > 10:
            self.run_glucose_prediction(patient_id)
            
        self.run_cardio_risk_assessment(patient_id)
        
        new_alerts = self.alert_system.check_all_alerts(patient_id, twin.current_vitals, twin.current_risk)
        for alert in new_alerts:
            twin.add_alert(alert)
            
        return twin.get_current_state()

    def run_glucose_prediction(self, patient_id: str) -> dict:
        """Run glucose predictor model and update twin risk scores."""
        twin = self.active_twins[patient_id]
        
        # Mock prediction output
        current_glucose = twin.current_vitals.glucose_mg_dl
        predictions = [current_glucose + (i * 2.5) for i in range(1, 13)] # next 12 predictions
        
        new_risk = twin.current_risk
        new_risk.glucose_predicted_values = predictions
        new_risk.glucose_spike_risk = 0.8 if max(predictions) > 200 else 0.2
        new_risk.last_updated = datetime.now()
        
        twin.update_risk_scores(new_risk)
        return {
            'predicted_values': predictions,
            'spike_risk': new_risk.glucose_spike_risk
        }

    def run_cardio_risk_assessment(self, patient_id: str) -> dict:
        """Run cardio risk scorer model and update twin risk scores."""
        twin = self.active_twins[patient_id]
        
        # Mock cardio logic
        hr = twin.current_vitals.heart_rate_bpm
        cvd_risk = 0.2 + (0.1 if hr > 100 else 0.0)
        
        new_risk = twin.current_risk
        new_risk.cardiovascular_risk = cvd_risk
        new_risk.risk_category = 'High' if cvd_risk > 0.5 else 'Low'
        new_risk.last_updated = datetime.now()
        
        twin.update_risk_scores(new_risk)
        return {
            'cardiovascular_risk': cvd_risk,
            'explanations': {'heart_rate_contribution': 0.1 if hr > 100 else 0.0}
        }

    def run_what_if(self, patient_id: str, changes: dict) -> dict:
        """Delegate to twin's what_if_scenario method."""
        twin = self.active_twins.get(patient_id)
        if not twin:
            raise ValueError(f"Twin {patient_id} not initialized.")
        return twin.what_if_scenario(changes)

    def get_twin_state(self, patient_id: str) -> dict:
        """Return current state of specified twin."""
        twin = self.active_twins.get(patient_id)
        if not twin:
            raise ValueError(f"Twin {patient_id} not found.")
        return twin.get_current_state()

    def get_all_active_twins(self) -> List[str]:
        """Return list of active twin patient IDs."""
        return list(self.active_twins.keys())

    def simulate_realtime(self, patient_id: str, speed_factor: float = 60.0) -> Generator:
        """Yield updated states simulating realtime sensor streaming."""
        twin = self.active_twins.get(patient_id)
        if not twin:
            raise ValueError(f"Twin {patient_id} not initialized.")
            
        base_glucose = twin.current_vitals.glucose_mg_dl or 100.0
        
        for i in range(100):
            sensor_data = {
                'glucose_mg_dl': base_glucose + (i % 10),
                'heart_rate_bpm': 70.0 + (i % 5),
                'timestamp': datetime.now()
            }
            updated_state = self.update_twin(patient_id, sensor_data)
            yield updated_state
            time.sleep(1.0 / speed_factor)
