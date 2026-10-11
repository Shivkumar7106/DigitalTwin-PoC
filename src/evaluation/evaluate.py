import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.evaluation.metrics import (
    glucose_prediction_metrics,
    cardiovascular_metrics,
    plot_clarke_error_grid,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_calibration_curve,
    prediction_horizon_analysis
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config_path: str = 'config/settings.yaml'):
        self.config_path = config_path
        self.results_dir = 'results'
        self.glucose_results_dir = os.path.join(self.results_dir, 'glucose')
        self.cardio_results_dir = os.path.join(self.results_dir, 'cardio')
        
        os.makedirs(self.glucose_results_dir, exist_ok=True)
        os.makedirs(self.cardio_results_dir, exist_ok=True)
        
    def evaluate_glucose_predictor(self, model_path: str = 'models/glucose_predictor') -> dict:
        logger.info(f"Evaluating Glucose Predictor from {model_path}")
        
        # Placeholder for actual data loading and inference
        # y_true, y_pred = load_and_predict(model_path)
        
        # Simulated data for evaluation
        np.random.seed(42)
        n_samples = 1000
        horizon_steps = 12
        y_true_seq = np.random.normal(120, 30, (n_samples, horizon_steps))
        y_pred_seq = y_true_seq + np.random.normal(0, 10, (n_samples, horizon_steps))
        
        y_true = y_true_seq[:, 0]
        y_pred = y_pred_seq[:, 0]
        
        metrics = glucose_prediction_metrics(y_true, y_pred)
        
        # Plots
        clarke_path = os.path.join(self.glucose_results_dir, 'clarke_error_grid.png')
        plot_clarke_error_grid(y_true, y_pred, save_path=clarke_path)
        
        horizon_df = prediction_horizon_analysis(y_true_seq, y_pred_seq, horizon_steps)
        horizon_df.to_csv(os.path.join(self.glucose_results_dir, 'horizon_metrics.csv'), index=False)
        
        with open(os.path.join(self.glucose_results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else (float(o) if isinstance(o, (np.floating, np.float64)) else str(o)))
            
        return metrics

    def evaluate_cardio_risk_scorer(self, model_path: str = 'models/cardio_risk_scorer') -> dict:
        logger.info(f"Evaluating Cardio Risk Scorer from {model_path}")
        
        # Simulated data
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.randint(0, 2, n_samples)
        y_pred_proba = np.random.beta(2, 5, n_samples)
        y_pred_proba[y_true == 1] = np.random.beta(5, 2, np.sum(y_true == 1))
        
        metrics = cardiovascular_metrics(y_true, y_pred_proba)
        
        # Plots
        roc_path = os.path.join(self.cardio_results_dir, 'roc_curve.png')
        plot_roc_curve(y_true, y_pred_proba, save_path=roc_path)
        
        pr_path = os.path.join(self.cardio_results_dir, 'pr_curve.png')
        plot_precision_recall_curve(y_true, y_pred_proba, save_path=pr_path)
        
        cm_path = os.path.join(self.cardio_results_dir, 'confusion_matrix.png')
        plot_confusion_matrix(y_true, (y_pred_proba >= 0.5).astype(int), save_path=cm_path)
        
        calib_path = os.path.join(self.cardio_results_dir, 'calibration_curve.png')
        plot_calibration_curve(y_true, y_pred_proba, save_path=calib_path)
        
        with open(os.path.join(self.cardio_results_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4, default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else (float(o) if isinstance(o, (np.floating, np.float64)) else str(o)))
            
        return metrics

    def generate_full_report(self) -> str:
        logger.info("Generating full evaluation report...")
        
        glucose_metrics = self.evaluate_glucose_predictor()
        cardio_metrics = self.evaluate_cardio_risk_scorer()
        
        report_path = os.path.join(self.results_dir, 'evaluation_report.md')
        
        report_content = f"""# Digital Twin PoC - Evaluation Report

## 1. Continuous Glucose Prediction (BiLSTM + Attention)
- **RMSE:** {glucose_metrics['RMSE']:.2f} mg/dL
- **MAE:** {glucose_metrics['MAE']:.2f} mg/dL
- **MARD:** {glucose_metrics['MARD']:.2f}%
- **Clarke Error Grid (A+B):** {glucose_metrics['Clarke_A+B']:.2f}%

![Clarke Error Grid](glucose/clarke_error_grid.png)

## 2. Cardiovascular Risk Scoring (XGBoost + SHAP)
- **AUC-ROC:** {cardio_metrics['AUC-ROC']:.4f}
- **PR-AUC:** {cardio_metrics['PR-AUC']:.4f}
- **Sensitivity:** {cardio_metrics['Sensitivity']:.4f}
- **Specificity:** {cardio_metrics['Specificity']:.4f}
- **F1 Score:** {cardio_metrics['F1_Score']:.4f}

![ROC Curve](cardio/roc_curve.png)
![Confusion Matrix](cardio/confusion_matrix.png)
"""
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        logger.info(f"Report saved to {report_path}")
        return report_path

if __name__ == '__main__':
    evaluator = ModelEvaluator()
    evaluator.generate_full_report()
