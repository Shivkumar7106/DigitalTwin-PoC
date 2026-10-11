import os
import pytest
import pandas as pd
import numpy as np

from src.data_ingestion.ehr_loader import EHRLoader
from src.data_ingestion.sensor_loader import SensorLoader
from src.evaluation.metrics import clarke_error_grid, glucose_prediction_metrics, cardiovascular_metrics

def test_ehr_loader_loads_data():
    loader = EHRLoader(data_dir='data/raw/ehr')
    pts = loader.load_patients()
    assert not pts.empty
    assert 'gender' in pts.columns
    assert len(pts) > 0

    conds = loader.load_conditions()
    assert not conds.empty

    obs = loader.load_observations()
    assert not obs.empty

    meds = loader.load_medications()
    assert not meds.empty

def test_sensor_loader_loads_streams():
    loader = SensorLoader(data_dir='data/raw/sensors')
    cgm = loader.load_cgm_data()
    assert not cgm.empty
    assert 'glucose_mg_dl' in cgm.columns

    hr = loader.load_heart_rate_data()
    assert not hr.empty
    assert 'heart_rate_bpm' in hr.columns

    sleep = loader.load_sleep_data()
    assert not sleep.empty

    steps = loader.load_step_data()
    assert not steps.empty

def test_clarke_error_grid():
    ref = np.array([100.0, 150.0, 200.0, 80.0, 250.0])
    pred = np.array([102.0, 148.0, 205.0, 82.0, 245.0])
    zones = clarke_error_grid(ref, pred)
    assert 'A' in zones
    assert 'B' in zones
    assert zones['A+B'] >= 90.0

def test_cardiovascular_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_pred_proba = np.array([0.1, 0.2, 0.85, 0.9, 0.15, 0.75, 0.3, 0.88])
    metrics = cardiovascular_metrics(y_true, y_pred_proba)
    assert metrics['AUC-ROC'] > 0.8
    assert metrics['F1_Score'] > 0.7
