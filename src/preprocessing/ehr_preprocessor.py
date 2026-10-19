"""
EHR Preprocessor for Digital Twin PoC.
Extracts static feature vectors from EHR data for ML models.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Any

class EHRPreprocessor:
    """
    Extracts and scales static feature vectors from EHR data.
    """
    def __init__(self, config: dict = None):
        """
        Initializes the EHRPreprocessor.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        self.scaler = StandardScaler()

    def preprocess_demographics(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses demographic data from patients.
        
        Args:
            patients_df (pd.DataFrame): DataFrame containing patient demographic data.
            
        Returns:
            pd.DataFrame: Processed demographics DataFrame indexed by patient_id.
        """
        df = patients_df.copy()
        
        # Binary encode gender (Assuming 'M', 'F')
        if 'gender' in df.columns:
            df['gender_binary'] = df['gender'].apply(lambda x: 1 if str(x).upper() == 'M' else 0)
            
        # Calculate age from date_of_birth
        if 'date_of_birth' in df.columns:
            df['date_of_birth'] = pd.to_datetime(df['date_of_birth'])
            df['age'] = (pd.Timestamp.now() - df['date_of_birth']).dt.days / 365.25
            
        # Normalize BMI (if present)
        # Note: we are not fitting scaler here, we will do it at the end for all numerics.
        
        # One-hot encode categorical columns
        categorical_cols = ['smoking_status', 'alcohol_use', 'physical_activity_level', 'blood_type']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                
        # Binary encode family history
        binary_cols = ['family_history_diabetes', 'family_history_cvd']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
                
        # Ensure patient_id is index
        if 'patient_id' in df.columns:
            df.set_index('patient_id', inplace=True)
            
        # Select numeric and encoded columns
        cols_to_keep = ['age', 'bmi', 'gender_binary', 'genetic_risk_score'] + \
                       [c for c in df.columns if any(c.startswith(prefix) for prefix in categorical_cols)] + \
                       binary_cols
        
        return df[[c for c in cols_to_keep if c in df.columns]]

    def preprocess_conditions(self, conditions_df: pd.DataFrame, patient_ids: List[str]) -> pd.DataFrame:
        """
        Preprocesses condition data to create binary flags and comorbidity indices.
        
        Args:
            conditions_df (pd.DataFrame): DataFrame containing patient conditions.
            patient_ids (list): List of patient IDs.
            
        Returns:
            pd.DataFrame: Processed conditions DataFrame indexed by patient_id.
        """
        df = conditions_df.copy()
        
        # Target conditions
        target_conditions = ['diabetes', 'prediabetes', 'hypertension', 'dyslipidemia', 'obesity', 'cad']
        
        # Initialize output dataframe
        out_df = pd.DataFrame(index=patient_ids)
        out_df.index.name = 'patient_id'
        
        for cond in target_conditions:
            # Assuming 'condition_name' column exists and contains the condition string
            cond_patients = df[df['condition_name'].str.lower().str.contains(cond, na=False)]['patient_id'].unique()
            out_df[f'has_{cond}'] = out_df.index.isin(cond_patients).astype(int)
            
        # Total number of active conditions (Assuming all in df are active or filtered before)
        condition_counts = df.groupby('patient_id').size()
        out_df['total_active_conditions'] = condition_counts.reindex(out_df.index).fillna(0)
        
        # Calculate comorbidity index (simple sum of target conditions for PoC)
        out_df['comorbidity_index'] = out_df[[f'has_{c}' for c in target_conditions]].sum(axis=1)
        
        return out_df

    def preprocess_observations(self, observations_df: pd.DataFrame, patient_ids: List[str]) -> pd.DataFrame:
        """
        Preprocesses observations (lab values) to extract latest values and derived features.
        
        Args:
            observations_df (pd.DataFrame): DataFrame containing patient observations.
            patient_ids (list): List of patient IDs.
            
        Returns:
            pd.DataFrame: Processed observations DataFrame indexed by patient_id.
        """
        df = observations_df.copy()
        
        # Ensure datetime
        date_col = 'observation_date' if 'observation_date' in df.columns else ('date' if 'date' in df.columns else None)
        if date_col:
            df['obs_dt'] = pd.to_datetime(df[date_col])
        else:
            df['obs_dt'] = pd.Timestamp.now()
            
        # Sort by date to get latest easily
        df = df.sort_values(by=['patient_id', 'obs_dt'])
        
        name_col = 'observation_name' if 'observation_name' in df.columns else ('observation_type' if 'observation_type' in df.columns else 'name')
        
        target_map = {
            'hba1c': ['hba1c'],
            'fasting_glucose': ['fasting blood glucose', 'fasting_glucose', 'glucose'],
            'total_cholesterol': ['total cholesterol', 'total_cholesterol'],
            'hdl': ['hdl cholesterol', 'hdl'],
            'ldl': ['ldl cholesterol', 'ldl'],
            'triglycerides': ['triglycerides'],
            'systolic_bp': ['systolic blood pressure', 'systolic_bp'],
            'diastolic_bp': ['diastolic blood pressure', 'diastolic_bp'],
            'creatinine': ['creatinine'],
            'egfr': ['egfr'],
            'bmi_obs': ['bmi']
        }
                       
        out_df = pd.DataFrame(index=patient_ids)
        out_df.index.name = 'patient_id'
        
        for feature_key, aliases in target_map.items():
            mask = df[name_col].astype(str).str.lower().apply(lambda x: any(a in x for a in aliases))
            lab_df = df[mask]
            if not lab_df.empty:
                latest_lab = lab_df.groupby('patient_id').tail(1).set_index('patient_id')['value']
                out_df[feature_key] = latest_lab.reindex(out_df.index)
            else:
                out_df[feature_key] = np.nan
            
        # Median imputation for missing values
        out_df = out_df.fillna(out_df.median())
        
        # Derived features
        out_df['pulse_pressure'] = out_df['systolic_bp'] - out_df['diastolic_bp']
        out_df['cholesterol_ratio'] = out_df['total_cholesterol'] / out_df['hdl'].replace(0, np.nan)
        out_df['cholesterol_ratio'] = out_df['cholesterol_ratio'].fillna(0)
        out_df['non_hdl_cholesterol'] = out_df['total_cholesterol'] - out_df['hdl']
        
        return out_df

    def preprocess_medications(self, medications_df: pd.DataFrame, patient_ids: List[str]) -> pd.DataFrame:
        """
        Preprocesses medication data into binary flags.
        
        Args:
            medications_df (pd.DataFrame): DataFrame containing patient medications.
            patient_ids (list): List of patient IDs.
            
        Returns:
            pd.DataFrame: Processed medications DataFrame indexed by patient_id.
        """
        df = medications_df.copy()
        
        med_classes = ['metformin', 'insulin', 'sulfonylurea', 'dpp4i', 'antihypertensive', 'statin']
        
        out_df = pd.DataFrame(index=patient_ids)
        out_df.index.name = 'patient_id'
        
        for med in med_classes:
            med_patients = df[df['medication_name'].str.lower().str.contains(med, na=False)]['patient_id'].unique()
            out_df[f'on_{med}'] = out_df.index.isin(med_patients).astype(int)
            
        med_counts = df.groupby('patient_id').size()
        out_df['total_medications'] = med_counts.reindex(out_df.index).fillna(0)
        
        return out_df

    def get_static_features(self, patients_df: pd.DataFrame, conditions_df: pd.DataFrame, 
                            observations_df: pd.DataFrame, medications_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Combines all static features into a unified feature matrix and scales numeric features.
        
        Args:
            patients_df (pd.DataFrame): Demographics dataframe.
            conditions_df (pd.DataFrame): Conditions dataframe.
            observations_df (pd.DataFrame): Observations dataframe.
            medications_df (pd.DataFrame): Medications dataframe.
            
        Returns:
            Tuple[pd.DataFrame, StandardScaler]: Scaled unified feature matrix and the fitted scaler.
        """
        demo_feats = self.preprocess_demographics(patients_df)
        patient_ids = demo_feats.index.tolist()
        
        cond_feats = self.preprocess_conditions(conditions_df, patient_ids)
        obs_feats = self.preprocess_observations(observations_df, patient_ids)
        med_feats = self.preprocess_medications(medications_df, patient_ids)
        
        # Combine all features
        unified_df = pd.concat([demo_feats, cond_feats, obs_feats, med_feats], axis=1)
        
        # Separate numeric and non-numeric for scaling
        # Assuming binary flags (0, 1) and counts/continuous are all numeric
        # We might not want to scale binary flags, but typically StandardScaler is fine.
        # Let's scale continuous variables specifically
        continuous_cols = ['age', 'bmi', 'genetic_risk_score', 'total_active_conditions', 'comorbidity_index',
                           'hba1c', 'fasting_glucose', 'total_cholesterol', 'hdl', 'ldl', 'triglycerides', 
                           'systolic_bp', 'diastolic_bp', 'creatinine', 'egfr', 'bmi_obs', 'pulse_pressure', 
                           'cholesterol_ratio', 'non_hdl_cholesterol', 'total_medications']
                           
        cols_to_scale = [c for c in continuous_cols if c in unified_df.columns]
        
        if cols_to_scale:
            unified_df[cols_to_scale] = self.scaler.fit_transform(unified_df[cols_to_scale])
            
        return unified_df, self.scaler
