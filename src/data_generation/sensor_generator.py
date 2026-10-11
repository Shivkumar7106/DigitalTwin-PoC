"""
Synthetic Sensor Data Generator for Digital Twin PoC.

This module generates realistic physiological time-series data for patients, including:
1. Continuous Glucose Monitor (CGM) data
2. Heart Rate & Heart Rate Variability (HRV) data
3. Sleep data (stages, efficiency, etc.)
4. Step Count & Activity data

Data is parameterized based on patient demographics and conditions.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

class SensorDataGenerator:
    """Generates synthetic sensor data for patients based on their profiles."""
    
    def __init__(self, patients_df, observations_df, start_date, num_days=30, seed=42):
        self.patients_df = patients_df
        self.observations_df = observations_df
        self.start_date = start_date
        self.num_days = num_days
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def _extract_patient_profile(self, row):
        """Extract or infer patient profile for parameterizing sensor data."""
        profile = {}
        
        # Age
        try:
            if 'date_of_birth' in row:
                birthdate = pd.to_datetime(row['date_of_birth'])
                profile['age'] = (self.start_date - birthdate).days / 365.25
            elif 'BIRTHDATE' in row:
                birthdate = pd.to_datetime(row['BIRTHDATE'])
                profile['age'] = (self.start_date - birthdate).days / 365.25
            elif 'age' in row:
                profile['age'] = float(row['age'])
            else:
                profile['age'] = 45.0
        except:
            profile['age'] = 45.0
            
        # Activity level (check from row if present)
        if 'physical_activity_level' in row:
            lvl_map = {'Sedentary': 0, 'Light': 1, 'Moderate': 2, 'Active': 3}
            profile['activity_level'] = lvl_map.get(str(row['physical_activity_level']), 1)
        else:
            # 0: Sedentary, 1: Light, 2: Moderate, 3: Active
            prob_active = max(0.05, 0.3 - (profile['age'] / 200))
            prob_sed = min(0.5, 0.1 + (profile['age'] / 100))
            prob_light = (1 - prob_active - prob_sed) * 0.6
            prob_mod = (1 - prob_active - prob_sed) * 0.4
            profile['activity_level'] = np.random.choice([0, 1, 2, 3], p=[prob_sed, prob_light, prob_mod, prob_active])
        
        # Diabetic status (infer from observations if available, otherwise random)
        # 0: Normal, 1: Prediabetes, 2: Mild, 3: Moderate, 4: Severe
        profile['diabetic_status'] = np.random.choice([0, 1, 2, 3, 4], p=[0.70, 0.15, 0.08, 0.05, 0.02])
        
        return profile

    def generate_cgm_for_patient(self, patient_id, profile):
        """Generate CGM data (5 min sampling) for a single patient."""
        # 30 days * 24 hours * 12 readings/hour = 8640 readings
        timestamps = [self.start_date + timedelta(minutes=5 * i) for i in range(self.num_days * 24 * 12)]
        
        status = profile['diabetic_status']
        if status == 0:
            base_mean = 95
            base_std = 5
            spike_mag = 30
        elif status == 1:
            base_mean = 115
            base_std = 10
            spike_mag = 45
        elif status == 2:
            base_mean = 140
            base_std = 15
            spike_mag = 60
        elif status == 3:
            base_mean = 170
            base_std = 20
            spike_mag = 70
        else:
            base_mean = 200
            base_std = 25
            spike_mag = 80
            
        cgm_data = []
        
        # To make it realistic, we use a daily circadian baseline and add meals
        for t in timestamps:
            hour = t.hour
            minute = t.minute
            time_in_hours = hour + minute / 60.0
            
            # Dawn phenomenon (rise around 4-8 AM)
            dawn_effect = 0
            if 4 <= time_in_hours <= 8:
                dawn_effect = (time_in_hours - 4) / 4 * (10 + status * 5)
            elif 8 < time_in_hours <= 10:
                dawn_effect = (10 - time_in_hours) / 2 * (10 + status * 5)
                
            # Nocturnal dip
            nocturnal_dip = 0
            if 2 <= time_in_hours <= 4:
                nocturnal_dip = - (1 + status) * 2
                
            # Meals (approx 8AM, 1PM, 8PM)
            meal_flag = 0
            meal_effect = 0
            for meal_time in [8, 13, 20]:
                if meal_time <= time_in_hours <= meal_time + 2.5:
                    # Peak at 1 hour after meal
                    diff = time_in_hours - meal_time
                    if diff <= 1:
                        meal_effect += (diff) * spike_mag
                    else:
                        meal_effect += (1 - (diff - 1)/1.5) * spike_mag
                    
                    if 0 <= diff <= 0.5:
                        meal_flag = 1
            
            # Exercise effect (randomly drop if active)
            exercise_flag = 0
            exercise_effect = 0
            if profile['activity_level'] > 0 and 17 <= time_in_hours <= 18:
                exercise_flag = 1
                exercise_effect = -20 * profile['activity_level']
            
            # Base noise
            noise = np.random.normal(0, base_std * 0.3)
            
            glucose = base_mean + dawn_effect + nocturnal_dip + meal_effect + exercise_effect + noise
            glucose = max(40, min(400, glucose)) # Clamp realistic values
            
            cgm_data.append({
                'patient_id': patient_id,
                'timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'glucose_mg_dl': round(glucose, 1),
                'meal_flag': meal_flag,
                'exercise_flag': exercise_flag
            })
            
        return cgm_data

    def generate_hr_for_patient(self, patient_id, profile):
        """Generate Heart Rate and HRV data (1 min sampling)."""
        # 30 days * 24 hours * 60 readings/hour = 43200 readings
        timestamps = [self.start_date + timedelta(minutes=i) for i in range(self.num_days * 24 * 60)]
        
        act = profile['activity_level']
        if act == 0:
            rest_hr = np.random.uniform(72, 90)
        elif act == 1:
            rest_hr = np.random.uniform(65, 80)
        elif act == 2:
            rest_hr = np.random.uniform(60, 75)
        else:
            rest_hr = np.random.uniform(55, 68)
            
        hr_data = []
        for t in timestamps:
            hour = t.hour
            minute = t.minute
            time_in_hours = hour + minute / 60.0
            
            # Sleep drop
            sleep_drop = 0
            if 0 <= time_in_hours <= 6:
                sleep_drop = -15
                
            # Exercise elevation
            ex_elev = 0
            if act > 0 and 17 <= time_in_hours <= 18:
                ex_elev = 30 + act * 15
                
            # Random stress
            stress = np.random.choice([0, 10, 20], p=[0.95, 0.04, 0.01])
            
            noise = np.random.normal(0, 3)
            hr = rest_hr + sleep_drop + ex_elev + stress + noise
            hr = max(40, min(200, hr))
            
            # HRV inversely correlated
            hrv_rmssd = max(10, 100 - (hr - 50) * 0.8 + np.random.normal(0, 5))
            hrv_sdnn = hrv_rmssd * 1.2 + np.random.normal(0, 3)
            
            hr_data.append({
                'patient_id': patient_id,
                'timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'heart_rate_bpm': round(hr, 1),
                'hrv_rmssd_ms': round(hrv_rmssd, 1),
                'hrv_sdnn_ms': round(hrv_sdnn, 1)
            })
            
        return hr_data

    def generate_sleep_for_patient(self, patient_id, profile):
        """Generate nightly sleep data."""
        sleep_data = []
        age = profile['age']
        
        for i in range(self.num_days):
            date = (self.start_date + timedelta(days=i)).date()
            
            # Sleep duration 5-9h
            base_sleep_h = np.random.normal(7, 1)
            base_sleep_h = max(5, min(9, base_sleep_h))
            total_sleep_min = int(base_sleep_h * 60)
            
            # Stages
            # Light: 40-55%, Deep: 13-25%, REM: 20-25%, Wake: 5-15%
            deep_pct = max(0.05, np.random.uniform(0.13, 0.25) - (age / 1000))
            rem_pct = np.random.uniform(0.20, 0.25)
            wake_pct = np.random.uniform(0.05, 0.15) + (age / 1000)
            light_pct = 1.0 - (deep_pct + rem_pct + wake_pct)
            
            deep_sleep_min = int(total_sleep_min * deep_pct)
            rem_sleep_min = int(total_sleep_min * rem_pct)
            wake_min = int(total_sleep_min * wake_pct)
            light_sleep_min = total_sleep_min - deep_sleep_min - rem_sleep_min - wake_min
            
            sleep_efficiency = round((total_sleep_min - wake_min) / total_sleep_min * 100, 1)
            sol = int(np.random.uniform(5, 45))
            awakenings = int(np.random.uniform(0, 4) + (age / 20))
            
            bedtime_hour = int(np.random.normal(23, 1)) % 24
            bedtime = datetime.combine(date, datetime.min.time()) + timedelta(hours=bedtime_hour, minutes=sol)
            if bedtime_hour < 12:
                bedtime += timedelta(days=1)
                
            wake_time = bedtime + timedelta(minutes=total_sleep_min)
            
            sleep_data.append({
                'patient_id': patient_id,
                'date': date.strftime('%Y-%m-%d'),
                'bedtime': bedtime.strftime('%Y-%m-%d %H:%M:%S'),
                'wake_time': wake_time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_sleep_min': total_sleep_min,
                'sleep_efficiency_pct': sleep_efficiency,
                'deep_sleep_min': deep_sleep_min,
                'light_sleep_min': light_sleep_min,
                'rem_sleep_min': rem_sleep_min,
                'wake_min': wake_min,
                'sleep_onset_latency_min': sol,
                'awakenings': awakenings
            })
            
        return sleep_data

    def generate_steps_for_patient(self, patient_id, profile):
        """Generate hourly step counts."""
        # 30 days * 24 hours = 720 readings
        timestamps = [self.start_date + timedelta(hours=i) for i in range(self.num_days * 24)]
        
        act = profile['activity_level']
        if act == 0:
            daily_mean = 3000
        elif act == 1:
            daily_mean = 5500
        elif act == 2:
            daily_mean = 8500
        else:
            daily_mean = 12500
            
        steps_data = []
        for t in timestamps:
            hour = t.hour
            
            # Distribution: very low during sleep
            if 0 <= hour <= 6:
                hourly_steps = np.random.uniform(0, 50)
                active_mins = 0
            else:
                # Peaks around 8-9 and 17-18
                if hour in [8, 9, 17, 18]:
                    hourly_steps = np.random.normal(daily_mean * 0.15, daily_mean * 0.05)
                else:
                    hourly_steps = np.random.normal(daily_mean * 0.02, daily_mean * 0.01)
                    
                hourly_steps = max(0, hourly_steps)
                active_mins = int(hourly_steps / 100)
                active_mins = min(60, active_mins)
                
            steps_data.append({
                'patient_id': patient_id,
                'timestamp': t.strftime('%Y-%m-%d %H:%M:%S'),
                'steps': int(hourly_steps),
                'active_minutes': active_mins
            })
            
        return steps_data


def generate_sensor_data(patients_csv, observations_csv, output_dir='data/raw/sensors', num_days=30, seed=42):
    print(f"Loading patient data from {patients_csv}")
    
    try:
        patients_df = pd.read_csv(patients_csv)
    except FileNotFoundError:
        print(f"Warning: {patients_csv} not found. Generating dummy patients.")
        patients_df = pd.DataFrame({'Id': [f'P{i:03d}' for i in range(10)], 'BIRTHDATE': ['1980-01-01']*10})
        
    try:
        observations_df = pd.read_csv(observations_csv)
    except FileNotFoundError:
        print(f"Warning: {observations_csv} not found. Using empty observations.")
        observations_df = pd.DataFrame()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    start_date = datetime.now() - timedelta(days=num_days)
    
    generator = SensorDataGenerator(patients_df, observations_df, start_date, num_days, seed)
    
    all_cgm = []
    all_hr = []
    all_sleep = []
    all_steps = []
    
    if 'patient_id' in patients_df.columns:
        patient_ids = patients_df['patient_id'].tolist()
    elif 'Id' in patients_df.columns:
        patient_ids = patients_df['Id'].tolist()
    else:
        patient_ids = patients_df.iloc[:, 0].tolist()
    total_patients = len(patient_ids)
    
    print(f"Starting generation for {total_patients} patients...")
    for i, pid in enumerate(patient_ids):
        row = patients_df.iloc[i].to_dict()
        profile = generator._extract_patient_profile(row)
        
        all_cgm.extend(generator.generate_cgm_for_patient(pid, profile))
        all_hr.extend(generator.generate_hr_for_patient(pid, profile))
        all_sleep.extend(generator.generate_sleep_for_patient(pid, profile))
        all_steps.extend(generator.generate_steps_for_patient(pid, profile))
        
        if (i + 1) % 50 == 0 or (i + 1) == total_patients:
            print(f"Processed {i + 1}/{total_patients} patients...")
            
    print("Saving to CSV files...")
    pd.DataFrame(all_cgm).to_csv(out_path / 'cgm_data.csv', index=False)
    pd.DataFrame(all_hr).to_csv(out_path / 'hr_data.csv', index=False)
    pd.DataFrame(all_sleep).to_csv(out_path / 'sleep_data.csv', index=False)
    pd.DataFrame(all_steps).to_csv(out_path / 'steps_data.csv', index=False)
    
    print(f"Successfully generated sensor data in {output_dir}")


if __name__ == '__main__':
    # Default paths assuming running from workspace root
    workspace_dir = Path(r"C:\Users\shivu\OneDrive\Desktop\Twin")
    patients_file = workspace_dir / "data" / "raw" / "patients.csv"
    obs_file = workspace_dir / "data" / "raw" / "observations.csv"
    out_dir = workspace_dir / "data" / "raw" / "sensors"
    
    generate_sensor_data(
        patients_csv=str(patients_file),
        observations_csv=str(obs_file),
        output_dir=str(out_dir)
    )
