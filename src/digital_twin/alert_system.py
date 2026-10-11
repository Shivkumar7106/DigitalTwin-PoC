"""
Alert management system for the Digital Twin.
Handles severity levels and deduplication of alerts.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import uuid

from .patient_state import Alert, VitalSigns, RiskScores

class AlertSystem:
    """Manages alert generation, severity mapping, and deduplication/cooldowns."""
    def __init__(self, config: dict = None):
        self.config = config or {}
        # Cooldown periods in minutes to prevent alert fatigue
        self.cooldown_periods: Dict[str, int] = {
            'glucose_critical': 60,
            'glucose_high': 30,
            'glucose_moderate': 30,
            'cvd_critical': 15,
            'cvd_high': 60,
            'cvd_moderate': 120,
            'cvd_info': 1440 # daily
        }
        # Maps (patient_id, alert_type) -> last_triggered_time
        self.alert_history: Dict[Tuple[str, str], datetime] = {}

    def _should_alert(self, patient_id: str, alert_type: str, cooldown_minutes: int) -> bool:
        """Check if enough time has passed since last alert of this type."""
        key = (patient_id, alert_type)
        last_time = self.alert_history.get(key)
        now = datetime.now()
        
        if last_time is None or now - last_time > timedelta(minutes=cooldown_minutes):
            self.alert_history[key] = now
            return True
        return False

    def check_glucose_alerts(self, patient_id: str, current_glucose: float, predicted_glucose: List[float]) -> List[Alert]:
        """Check for glucose related alerts."""
        alerts = []
        now = datetime.now()
        max_pred = max(predicted_glucose) if predicted_glucose else current_glucose
        
        # CRITICAL
        if current_glucose < 54 or current_glucose > 300:
            if self._should_alert(patient_id, 'glucose_critical', self.cooldown_periods['glucose_critical']):
                alerts.append(Alert(
                    alert_id=str(uuid.uuid4()),
                    severity='CRITICAL',
                    metric='glucose',
                    value=current_glucose,
                    threshold=54.0 if current_glucose < 54 else 300.0,
                    message=f"Critical glucose level: {current_glucose} mg/dL",
                    timestamp=now
                ))
        # HIGH
        elif current_glucose < 70 or current_glucose > 250 or max_pred > 250:
            if self._should_alert(patient_id, 'glucose_high', self.cooldown_periods['glucose_high']):
                alerts.append(Alert(
                    alert_id=str(uuid.uuid4()),
                    severity='HIGH',
                    metric='glucose',
                    value=max(current_glucose, max_pred),
                    threshold=250.0,
                    message=f"High glucose level or predicted risk",
                    timestamp=now
                ))
        # MODERATE
        elif current_glucose > 180 or max_pred > 180:
            if self._should_alert(patient_id, 'glucose_moderate', self.cooldown_periods['glucose_moderate']):
                alerts.append(Alert(
                    alert_id=str(uuid.uuid4()),
                    severity='MODERATE',
                    metric='glucose',
                    value=max(current_glucose, max_pred),
                    threshold=180.0,
                    message=f"Moderate glucose elevation",
                    timestamp=now
                ))
        return alerts

    def check_cardiovascular_alerts(self, patient_id: str, cvd_risk: float, heart_rate: float, hrv_rmssd: float) -> List[Alert]:
        """Check for cardiovascular related alerts."""
        alerts = []
        now = datetime.now()
        
        # CRITICAL
        if heart_rate > 150 or heart_rate < 40:
            if self._should_alert(patient_id, 'cvd_critical', self.cooldown_periods['cvd_critical']):
                alerts.append(Alert(
                    alert_id=str(uuid.uuid4()),
                    severity='CRITICAL',
                    metric='heart_rate',
                    value=heart_rate,
                    threshold=150.0 if heart_rate > 150 else 40.0,
                    message=f"Critical heart rate: {heart_rate} BPM",
                    timestamp=now
                ))
        # HIGH
        if cvd_risk > 0.80:
            if self._should_alert(patient_id, 'cvd_high', self.cooldown_periods['cvd_high']):
                alerts.append(Alert(
                    alert_id=str(uuid.uuid4()),
                    severity='HIGH',
                    metric='cvd_risk',
                    value=cvd_risk,
                    threshold=0.80,
                    message=f"High Cardiovascular Risk (>80%)",
                    timestamp=now
                ))
        # MODERATE
        elif cvd_risk > 0.60 or (hrv_rmssd > 0 and hrv_rmssd < 15):
            if self._should_alert(patient_id, 'cvd_moderate', self.cooldown_periods['cvd_moderate']):
                alerts.append(Alert(
                    alert_id=str(uuid.uuid4()),
                    severity='MODERATE',
                    metric='cvd_risk' if cvd_risk > 0.60 else 'hrv',
                    value=cvd_risk if cvd_risk > 0.60 else hrv_rmssd,
                    threshold=0.60 if cvd_risk > 0.60 else 15.0,
                    message=f"Moderate Cardiovascular Risk or low HRV",
                    timestamp=now
                ))
        return alerts

    def check_all_alerts(self, patient_id: str, vitals: VitalSigns, risk_scores: RiskScores) -> List[Alert]:
        """Run all alert checks, deduplicate, sort by severity, return new alerts."""
        alerts = []
        
        alerts.extend(self.check_glucose_alerts(
            patient_id, 
            vitals.glucose_mg_dl, 
            risk_scores.glucose_predicted_values
        ))
        
        alerts.extend(self.check_cardiovascular_alerts(
            patient_id,
            risk_scores.cardiovascular_risk,
            vitals.heart_rate_bpm,
            vitals.hrv_rmssd_ms
        ))
        
        # Sort by severity (CRITICAL first)
        severity_order = {'CRITICAL': 0, 'HIGH': 1, 'MODERATE': 2, 'INFO': 3}
        alerts.sort(key=lambda x: severity_order.get(x.severity, 4))
        
        return alerts

    def get_alert_summary(self, patient_id: str) -> dict:
        """Count alerts by severity level based on recent history."""
        # This is a simplified summary based on local cache
        return {
            'recent_alerts': sum(1 for (pid, _), dt in self.alert_history.items() if pid == patient_id and datetime.now() - dt < timedelta(hours=24)),
            'message': 'Alert summary tracked by history cache.'
        }
