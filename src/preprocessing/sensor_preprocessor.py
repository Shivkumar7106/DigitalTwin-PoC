"""
Sensor Preprocessor for Digital Twin PoC.
Processes time-series sensor data for ML models.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict

class SensorPreprocessor:
    """
    Processes time-series sensor data (CGM, HRV, Sleep, Activity) for ML models.
    """
    def __init__(self, config: dict = None):
        """
        Initializes the SensorPreprocessor.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        self.window_size = self.config.get('window_size', 72)
        self.prediction_horizon = self.config.get('prediction_horizon', 24)
        self.stride = self.config.get('stride', 6)
        self.scalers = {}

    def preprocess_cgm(self, cgm_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocesses CGM data to create sliding windows and targets.
        
        Args:
            cgm_df (pd.DataFrame): DataFrame containing CGM readings.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                X_windows (N, window_size, features), 
                y_targets (N, prediction_horizon),
                patient_ids (N,)
        """
        df = cgm_df.copy()
        
        # Ensure datetime and sort
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['patient_id', 'timestamp'])
        
        X_all, y_all, pids_all = [], [], []
        
        for pid, group in df.groupby('patient_id'):
            # Interpolate missing values
            group = group.set_index('timestamp').resample('5T').mean()
            group['glucose_value'] = group['glucose_value'].interpolate(method='linear', limit_direction='both')
            
            # Derived features
            group['glucose_velocity'] = group['glucose_value'].diff()
            group['glucose_acceleration'] = group['glucose_velocity'].diff()
            group['rolling_mean_30min'] = group['glucose_value'].rolling(window=6, min_periods=1).mean()
            group['rolling_std_30min'] = group['glucose_value'].rolling(window=6, min_periods=1).std().fillna(0)
            
            group = group.fillna(0) # Fill any remaining NaNs
            
            # Normalize
            features = ['glucose_value', 'glucose_velocity', 'glucose_acceleration', 'rolling_mean_30min', 'rolling_std_30min']
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(group[features])
            self.scalers[pid] = scaler
            
            # Sliding windows
            X_pid, y_pid = [], []
            n_samples = len(scaled_features)
            
            for i in range(0, n_samples - self.window_size - self.prediction_horizon + 1, self.stride):
                X_pid.append(scaled_features[i : i + self.window_size])
                # Target is actual glucose values, not scaled
                y_pid.append(group['glucose_value'].iloc[i + self.window_size : i + self.window_size + self.prediction_horizon].values)
                
            X_all.extend(X_pid)
            y_all.extend(y_pid)
            pids_all.extend([pid] * len(X_pid))
            
        return np.array(X_all), np.array(y_all), np.array(pids_all)

    def extract_cgm_features(self, cgm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts aggregate CGM features per patient.
        """
        df = cgm_df.copy()
        features = []
        
        for pid, group in df.groupby('patient_id'):
            glucose = group['glucose_value']
            mean_glu = glucose.mean()
            std_glu = glucose.std()
            cv_glu = std_glu / mean_glu if mean_glu > 0 else 0
            
            tir_pct = ((glucose >= 70) & (glucose <= 180)).mean() * 100
            tbr_pct = (glucose < 70).mean() * 100
            tar_pct = (glucose > 180).mean() * 100
            
            num_spikes = (glucose > 180).sum()
            num_hypo_events = (glucose < 70).sum()
            
            # Approximation for mean_postprandial_peak without meal data
            spikes = glucose[glucose > 180]
            mean_postprandial_peak = spikes.mean() if not spikes.empty else mean_glu
            
            glucose_range = glucose.max() - glucose.min()
            
            features.append({
                'patient_id': pid,
                'mean_glucose': mean_glu,
                'std_glucose': std_glu,
                'cv_glucose': cv_glu,
                'time_in_range_pct': tir_pct,
                'time_below_range_pct': tbr_pct,
                'time_above_range_pct': tar_pct,
                'num_spikes': num_spikes,
                'num_hypo_events': num_hypo_events,
                'mean_postprandial_peak': mean_postprandial_peak,
                'glucose_range': glucose_range
            })
            
        return pd.DataFrame(features).set_index('patient_id')

    def extract_hrv_features(self, hr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts aggregate HRV features per patient.
        """
        df = hr_df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        features = []
        for pid, group in df.groupby('patient_id'):
            # Assuming 'hr', 'rmssd', 'sdnn' columns exist
            hr = group.get('hr', pd.Series(dtype=float))
            rmssd = group.get('rmssd', pd.Series(dtype=float))
            sdnn = group.get('sdnn', pd.Series(dtype=float))
            
            # Filter nighttime data (approx 10 PM to 6 AM)
            night_mask = (group['timestamp'].dt.hour >= 22) | (group['timestamp'].dt.hour < 6)
            day_mask = ~night_mask
            
            mean_resting_hr = hr[night_mask].mean()
            std_resting_hr = hr[night_mask].std()
            
            mean_hrv_rmssd = rmssd.mean()
            std_hrv_rmssd = rmssd.std()
            mean_hrv_sdnn = sdnn.mean()
            std_hrv_sdnn = sdnn.std()
            
            max_hr = hr.max()
            min_hr = hr.min()
            hr_range = max_hr - min_hr
            
            mean_daytime_hr = hr[day_mask].mean()
            mean_nighttime_hr = mean_resting_hr
            
            features.append({
                'patient_id': pid,
                'mean_resting_hr': mean_resting_hr,
                'std_resting_hr': std_resting_hr,
                'mean_hrv_rmssd': mean_hrv_rmssd,
                'std_hrv_rmssd': std_hrv_rmssd,
                'mean_hrv_sdnn': mean_hrv_sdnn,
                'std_hrv_sdnn': std_hrv_sdnn,
                'max_hr': max_hr,
                'min_hr': min_hr,
                'hr_range': hr_range,
                'mean_daytime_hr': mean_daytime_hr,
                'mean_nighttime_hr': mean_nighttime_hr
            })
            
        return pd.DataFrame(features).set_index('patient_id')

    def extract_sleep_features(self, sleep_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts aggregate sleep features per patient.
        """
        df = sleep_df.copy()
        features = []
        for pid, group in df.groupby('patient_id'):
            mean_sleep_duration = group.get('duration_minutes', pd.Series([0])).mean()
            std_sleep_duration = group.get('duration_minutes', pd.Series([0])).std()
            mean_sleep_efficiency = group.get('efficiency', pd.Series([0])).mean()
            mean_deep_sleep_pct = group.get('deep_sleep_pct', pd.Series([0])).mean()
            mean_rem_sleep_pct = group.get('rem_sleep_pct', pd.Series([0])).mean()
            mean_awakenings = group.get('awakenings_count', pd.Series([0])).mean()
            mean_sleep_onset_latency = group.get('latency_minutes', pd.Series([0])).mean()
            
            # Sleep regularity index (std of bedtime)
            if 'bedtime' in group.columns:
                bedtimes = pd.to_datetime(group['bedtime']).dt.hour + pd.to_datetime(group['bedtime']).dt.minute / 60.0
                sleep_regularity_index = bedtimes.std()
            else:
                sleep_regularity_index = 0
                
            features.append({
                'patient_id': pid,
                'mean_sleep_duration': mean_sleep_duration,
                'std_sleep_duration': std_sleep_duration,
                'mean_sleep_efficiency': mean_sleep_efficiency,
                'mean_deep_sleep_pct': mean_deep_sleep_pct,
                'mean_rem_sleep_pct': mean_rem_sleep_pct,
                'mean_awakenings': mean_awakenings,
                'mean_sleep_onset_latency': mean_sleep_onset_latency,
                'sleep_regularity_index': sleep_regularity_index
            })
            
        return pd.DataFrame(features).set_index('patient_id')

    def extract_activity_features(self, step_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts aggregate activity features per patient.
        """
        df = step_df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        features = []
        for pid, group in df.groupby('patient_id'):
            # Group by day
            daily = group.set_index('timestamp').resample('D')
            daily_steps = daily['steps'].sum()
            
            mean_daily_steps = daily_steps.mean()
            std_daily_steps = daily_steps.std()
            
            # Active minutes (assuming 'active_minutes' column exists or estimated)
            mean_active_minutes = group.get('active_minutes', pd.Series([0])).mean()
            
            sedentary_days_pct = (daily_steps < 4000).mean() * 100
            
            # Peak activity hour
            hourly_steps = group.groupby(group['timestamp'].dt.hour)['steps'].mean()
            peak_activity_hour = hourly_steps.idxmax() if not hourly_steps.empty else 0
            
            features.append({
                'patient_id': pid,
                'mean_daily_steps': mean_daily_steps,
                'std_daily_steps': std_daily_steps,
                'mean_active_minutes': mean_active_minutes,
                'sedentary_days_pct': sedentary_days_pct,
                'peak_activity_hour': peak_activity_hour
            })
            
        return pd.DataFrame(features).set_index('patient_id')

    def get_dynamic_features(self, cgm_df: pd.DataFrame, hr_df: pd.DataFrame, 
                             sleep_df: pd.DataFrame, step_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combines all sensor aggregate features per patient.
        
        Args:
            cgm_df: DataFrame of CGM data.
            hr_df: DataFrame of HRV data.
            sleep_df: DataFrame of sleep data.
            step_df: DataFrame of activity data.
            
        Returns:
            pd.DataFrame: Combined dynamic features indexed by patient_id.
        """
        cgm_feats = self.extract_cgm_features(cgm_df)
        hrv_feats = self.extract_hrv_features(hr_df)
        sleep_feats = self.extract_sleep_features(sleep_df)
        act_feats = self.extract_activity_features(step_df)
        
        combined_df = pd.concat([cgm_feats, hrv_feats, sleep_feats, act_feats], axis=1)
        # Handle cases where some patients lack certain sensor data
        combined_df = combined_df.fillna(0)
        
        return combined_df
