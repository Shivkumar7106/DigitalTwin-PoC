"""
Sensor Data Loader and Simulator for Digital Twin PoC.
Loads time-series sensor data and provides streaming simulation.
"""

import os
import logging
import time
from typing import Dict, Generator, Any
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorLoader:
    """Loads time-series sensor data from CSV files."""

    def __init__(self, data_dir: str = 'data/raw/sensors'):
        """Initialize the loader with the directory containing sensor data."""
        self.data_dir = data_dir

    def _load_sensor_file(self, filename: str, patient_id: str = None) -> pd.DataFrame:
        """Helper to load sensor data and optionally filter by patient."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Sensor file not found: {filepath}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(filepath)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.sort_values('timestamp')
            if patient_id and 'patient_id' in df.columns:
                df = df[df['patient_id'] == patient_id]
            return df
        except Exception as e:
            logger.error(f"Error loading sensor file {filepath}: {e}")
            return pd.DataFrame()

    def load_cgm_data(self, patient_id: str = None) -> pd.DataFrame:
        """Load CGM data, optionally filter by patient."""
        df = self._load_sensor_file('cgm_data.csv', patient_id)
        if df.empty:
            df = self._load_sensor_file('cgm.csv', patient_id)
        return df

    def load_heart_rate_data(self, patient_id: str = None) -> pd.DataFrame:
        """Load heart rate data, optionally filter by patient."""
        df = self._load_sensor_file('hr_data.csv', patient_id)
        if df.empty:
            df = self._load_sensor_file('heart_rate.csv', patient_id)
        return df

    def load_sleep_data(self, patient_id: str = None) -> pd.DataFrame:
        """Load sleep data, optionally filter by patient."""
        df = self._load_sensor_file('sleep_data.csv', patient_id)
        if df.empty:
            df = self._load_sensor_file('sleep.csv', patient_id)
        return df

    def load_step_data(self, patient_id: str = None) -> pd.DataFrame:
        """Load step data, optionally filter by patient."""
        df = self._load_sensor_file('steps_data.csv', patient_id)
        if df.empty:
            df = self._load_sensor_file('steps.csv', patient_id)
        return df

    def load_all_sensors(self, patient_id: str) -> Dict[str, pd.DataFrame]:
        """Return dict with all sensor dataframes for one patient."""
        return {
            'cgm': self.load_cgm_data(patient_id),
            'heart_rate': self.load_heart_rate_data(patient_id),
            'sleep': self.load_sleep_data(patient_id),
            'steps': self.load_step_data(patient_id)
        }


class SensorStreamSimulator:
    """Provides a streaming simulation of sensor data for a patient."""

    def __init__(self, sensor_loader: SensorLoader, patient_id: str, speed_factor: float = 60.0):
        """
        Initialize the simulator.
        :param sensor_loader: An instance of SensorLoader
        :param patient_id: The ID of the patient to simulate data for
        :param speed_factor: How much faster than real-time to stream data. Default 60.0 means 1 second = 1 minute of data.
        """
        self.sensor_loader = sensor_loader
        self.patient_id = patient_id
        self.speed_factor = speed_factor
        self.data = self.sensor_loader.load_all_sensors(self.patient_id)
        self._initialize_streams()

    def _initialize_streams(self):
        """Set initial timestamps for streams."""
        pass  # Reset state if needed

    def _stream_generator(self, sensor_type: str) -> Generator[Dict[str, Any], None, None]:
        """Generic stream generator."""
        df = self.data.get(sensor_type, pd.DataFrame())
        if df.empty or 'timestamp' not in df.columns:
            return

        last_ts = None
        for _, row in df.iterrows():
            current_ts = row['timestamp']
            if last_ts is not None:
                delta_sec = (current_ts - last_ts).total_seconds()
                sleep_time = delta_sec / self.speed_factor
                if sleep_time > 0:
                    time.sleep(sleep_time)
            last_ts = current_ts
            yield row.to_dict()

    def stream_cgm(self) -> Generator[Dict[str, Any], None, None]:
        """Yields CGM readings one at a time with simulated timing."""
        return self._stream_generator('cgm')

    def stream_heart_rate(self) -> Generator[Dict[str, Any], None, None]:
        """Yields HR readings one at a time with simulated timing."""
        return self._stream_generator('heart_rate')

    def get_latest_window(self, sensor_type: str, window_minutes: int = 360) -> pd.DataFrame:
        """Get last N minutes of data from the available set."""
        df = self.data.get(sensor_type, pd.DataFrame())
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        max_time = df['timestamp'].max()
        cutoff = max_time - pd.Timedelta(minutes=window_minutes)
        return df[df['timestamp'] >= cutoff]

    def reset(self):
        """Reset stream to beginning."""
        self._initialize_streams()
        logger.info(f"Reset sensor streams for patient {self.patient_id}")
