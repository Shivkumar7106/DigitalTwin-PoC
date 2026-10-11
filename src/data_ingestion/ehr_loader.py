"""
EHR Data Loader for Digital Twin PoC.
Loads and validates patient Electronic Health Record data from CSV files.
"""

import os
import logging
from typing import List, Dict, Any
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EHRLoader:
    """Loads and validates EHR data from generated CSVs."""

    def __init__(self, data_dir: str = 'data/raw/ehr'):
        """Initialize the loader with the directory containing EHR CSV files."""
        self.data_dir = data_dir

    def _load_csv(self, filename: str) -> pd.DataFrame:
        """Helper method to load a CSV file, with error handling."""
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            raise

    def load_patients(self) -> pd.DataFrame:
        """Load patients.csv, parse dates, validate required columns."""
        df = self._load_csv('patients.csv')
        if 'id' not in df.columns and 'patient_id' in df.columns:
            df['id'] = df['patient_id']
        if 'birth_date' not in df.columns and 'date_of_birth' in df.columns:
            df['birth_date'] = df['date_of_birth']
            
        required_cols = ['id', 'gender']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col} in patients.csv")
        if 'birth_date' in df.columns:
            df['birth_date'] = pd.to_datetime(df['birth_date'], errors='coerce')
        return df

    def load_conditions(self) -> pd.DataFrame:
        """Load conditions.csv, filter to active conditions."""
        df = self._load_csv('conditions.csv')
        if 'status' in df.columns:
            df = df[df['status'].str.lower() == 'active']
        return df

    def load_observations(self) -> pd.DataFrame:
        """Load observations.csv, parse dates, convert numeric values."""
        df = self._load_csv('observations.csv')
        if 'date' not in df.columns and 'observation_date' in df.columns:
            df['date'] = df['observation_date']
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df

    def load_medications(self) -> pd.DataFrame:
        """Load medications.csv, filter active medications."""
        df = self._load_csv('medications.csv')
        if 'status' in df.columns:
            df = df[df['status'] == 'active']
        return df

    def get_patient_profile(self, patient_id: str) -> Dict[str, Any]:
        """Return complete profile for one patient."""
        try:
            patients = self.load_patients()
            conditions = self.load_conditions()
            observations = self.load_observations()
            medications = self.load_medications()

            patient_data = patients[patients['id'] == patient_id].to_dict('records')
            if not patient_data:
                return {}

            profile = {
                'demographics': patient_data[0],
                'conditions': conditions[conditions['patient_id'] == patient_id].to_dict('records') if 'patient_id' in conditions.columns else [],
                'observations': observations[observations['patient_id'] == patient_id].to_dict('records') if 'patient_id' in observations.columns else [],
                'medications': medications[medications['patient_id'] == patient_id].to_dict('records') if 'patient_id' in medications.columns else []
            }
            return profile
        except Exception as e:
            logger.error(f"Error getting patient profile for {patient_id}: {e}")
            return {}

    def get_all_patient_ids(self) -> List[str]:
        """Return list of all patient IDs."""
        try:
            patients = self.load_patients()
            return patients['id'].dropna().unique().tolist()
        except Exception as e:
            logger.error(f"Error getting patient IDs: {e}")
            return []

    def validate_data(self) -> Dict[str, Any]:
        """Check data integrity, return validation report."""
        report = {'missing_values': {}, 'referential_integrity': {}}
        try:
            patients = self.load_patients()
            report['missing_values']['patients'] = patients.isnull().sum().to_dict()
            patient_ids = set(patients['id'].dropna())
            
            for file, func, ref_col in [('conditions', self.load_conditions, 'patient_id'), 
                                        ('observations', self.load_observations, 'patient_id'), 
                                        ('medications', self.load_medications, 'patient_id')]:
                try:
                    df = func()
                    report['missing_values'][file] = df.isnull().sum().to_dict()
                    if ref_col in df.columns:
                        invalid_refs = df[~df[ref_col].isin(patient_ids)]
                        report['referential_integrity'][file] = f"{len(invalid_refs)} records with invalid {ref_col}"
                except FileNotFoundError:
                    report['missing_values'][file] = "File not found"
        except Exception as e:
            report['error'] = str(e)
            
        return report
