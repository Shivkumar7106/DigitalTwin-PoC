"""
Feature Engineering for Digital Twin PoC.
Combines static EHR features with dynamic sensor features.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import pickle
from typing import Dict, Tuple

class FeatureEngineer:
    """
    Combines static and dynamic features, prepares datasets for ML models.
    """
    def __init__(self, config: dict = None):
        """
        Initializes the FeatureEngineer.
        
        Args:
            config (dict, optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}

    def create_unified_features(self, static_features: pd.DataFrame, dynamic_features: pd.DataFrame) -> pd.DataFrame:
        """
        Combines static EHR features with dynamic sensor features.
        
        Args:
            static_features (pd.DataFrame): EHR static features.
            dynamic_features (pd.DataFrame): Sensor dynamic features.
            
        Returns:
            pd.DataFrame: Merged feature matrix.
        """
        merged_df = static_features.merge(dynamic_features, left_index=True, right_index=True, how='outer')
        # Handle remaining NaN values with median imputation
        merged_df = merged_df.fillna(merged_df.median())
        return merged_df

    def create_glucose_prediction_dataset(self, cgm_windows: np.ndarray, cgm_targets: np.ndarray, 
                                          patient_ids: np.ndarray, static_features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Creates dataset for glucose prediction model pairing windows with static features.
        
        Args:
            cgm_windows: Time series windows.
            cgm_targets: Glucose targets.
            patient_ids: Corresponding patient IDs for windows.
            static_features: DataFrame of static features.
            
        Returns:
            Dict: Dictionary containing X_temporal, X_static, y, patient_ids.
        """
        X_static_list = []
        
        for pid in patient_ids:
            if pid in static_features.index:
                X_static_list.append(static_features.loc[pid].values)
            else:
                # Fallback if static features missing for patient
                X_static_list.append(np.zeros(static_features.shape[1]))
                
        X_static = np.array(X_static_list)
        
        return {
            'X_temporal': cgm_windows,
            'X_static': X_static,
            'y': cgm_targets,
            'patient_ids': patient_ids
        }

    def create_cardio_risk_labels(self, conditions_df: pd.DataFrame, patient_ids: list) -> pd.Series:
        """
        Creates binary cardio risk labels for patients.
        
        Args:
            conditions_df (pd.DataFrame): DataFrame of conditions.
            patient_ids (list): List of patient IDs.
            
        Returns:
            pd.Series: Binary labels indexed by patient_id.
        """
        df = conditions_df.copy()
        
        labels = pd.Series(0, index=patient_ids, name='cardio_risk')
        labels.index.name = 'patient_id'
        
        for pid, group in df.groupby('patient_id'):
            if pid not in labels.index:
                continue
                
            conds = group['condition_name'].str.lower().tolist()
            
            has_cad = any('cad' in c for c in conds)
            has_htn = any('hypertension' in c for c in conds)
            has_dm = any('diabetes' in c for c in conds)
            
            # Risk criteria
            if has_cad or (has_htn and has_dm):
                labels[pid] = 1
                
        return labels

    def split_data(self, features: pd.DataFrame, labels: pd.Series, test_size: float = 0.15, 
                   val_size: float = 0.15, random_state: int = 42) -> Dict[str, Any]:
        """
        Splits data into train/val/test and applies SMOTE to training set.
        
        Args:
            features: Feature DataFrame.
            labels: Labels Series.
            test_size: Proportion of test data.
            val_size: Proportion of validation data (relative to total).
            random_state: Random seed.
            
        Returns:
            Dict: Train, val, test datasets.
        """
        # Ensure alignment
        common_idx = features.index.intersection(labels.index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx]
        
        # Calculate val_size relative to the remaining data after test split
        val_ratio = val_size / (1.0 - test_size)
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, 
                                                          stratify=y, random_state=random_state)
                                                          
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, 
                                                          stratify=y_temp, random_state=random_state)
                                                          
        # Handle class imbalance on training set
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        return {
            'X_train': X_train_res,
            'y_train': y_train_res,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }

    def save_processed_data(self, data_dict: Dict, output_dir: str = 'data/processed') -> None:
        """
        Saves processed dataset to disk.
        
        Args:
            data_dict (Dict): Dictionary containing data.
            output_dir (str): Output directory path.
        """
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'processed_data.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(data_dict, f)

    def load_processed_data(self, input_dir: str = 'data/processed') -> Dict:
        """
        Loads processed dataset from disk.
        
        Args:
            input_dir (str): Input directory path.
            
        Returns:
            Dict: Dictionary containing loaded data.
        """
        in_path = os.path.join(input_dir, 'processed_data.pkl')
        if not os.path.exists(in_path):
            raise FileNotFoundError(f"Processed data not found at {in_path}")
            
        with open(in_path, 'rb') as f:
            data_dict = pickle.load(f)
            
        return data_dict
