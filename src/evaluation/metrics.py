import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, brier_score_loss, confusion_matrix, classification_report, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import seaborn as sns
import os

def clarke_error_grid(reference: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Implement the full Clarke Error Grid Analysis for blood glucose prediction.
    
    Zones:
    - Zone A: Clinically accurate (within 20% of reference, or both <70)
    - Zone B: Benign errors (deviate from reference but would not lead to inappropriate treatment)
    - Zone C: Overcorrection (would lead to unnecessary treatment)
    - Zone D: Failure to detect (failure to detect hypoglycemia or hyperglycemia)
    - Zone E: Erroneous treatment (opposite treatment)
    """
    ref = np.asarray(reference)
    pred = np.asarray(predicted)
    
    zones = np.zeros_like(ref, dtype=str)
    
    # Zone A
    zone_a_mask = (abs(ref - pred) <= 0.2 * ref) | ((ref < 70) & (pred < 70))
    zones[zone_a_mask] = 'A'
    
    # Zone E
    zone_e_mask = ((ref <= 70) & (pred >= 180)) | ((ref >= 180) & (pred <= 70))
    zones[zone_e_mask] = 'E'
    
    # Zone C
    zone_c_mask = ((ref >= 70) & (ref <= 290) & (pred >= ref + 110)) | ((ref >= 130) & (ref <= 180) & (pred <= (7/5)*ref - 182))
    zones[zone_c_mask & ~zone_a_mask & ~zone_e_mask] = 'C'
    
    # Zone D
    zone_d_mask = ((ref < 70) & (pred > 70) & (pred <= 180)) | ((ref > 240) & (pred >= 70) & (pred <= 180))
    zones[zone_d_mask & ~zone_a_mask & ~zone_e_mask & ~zone_c_mask] = 'D'
    
    # Zone B (everything else)
    zones[zones == ''] = 'B'
    
    total = len(ref)
    results = {
        'A': np.sum(zones == 'A') / total * 100,
        'B': np.sum(zones == 'B') / total * 100,
        'C': np.sum(zones == 'C') / total * 100,
        'D': np.sum(zones == 'D') / total * 100,
        'E': np.sum(zones == 'E') / total * 100,
    }
    results['A+B'] = results['A'] + results['B']
    
    return results

def plot_clarke_error_grid(reference, predicted, title='Clarke Error Grid', save_path=None) -> plt.Figure:
    """
    Plot the Clarke Error Grid.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot
    ax.scatter(reference, predicted, alpha=0.5, marker='.')
    
    # Plot boundaries
    ax.plot([0, 400], [0, 400], 'k--')
    ax.plot([0, 175/3], [70, 70], 'k-')
    ax.plot([175/3, 400], [70, 480], 'k-')
    ax.plot([70, 70], [84, 400], 'k-')
    ax.plot([0, 70], [180, 180], 'k-')
    ax.plot([70, 290], [180, 400], 'k-')
    ax.plot([70, 70], [0, 56], 'k-')
    ax.plot([70, 400], [56, 320], 'k-')
    ax.plot([180, 180], [0, 70], 'k-')
    ax.plot([180, 400], [70, 70], 'k-')
    ax.plot([240, 240], [70, 180], 'k-')
    ax.plot([240, 400], [180, 180], 'k-')
    ax.plot([130, 180], [0, 70], 'k-')
    
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Reference Glucose (mg/dL)')
    ax.set_ylabel('Predicted Glucose (mg/dL)')
    ax.set_title(title)
    
    # Add text for zones
    ax.text(30, 30, 'A', fontsize=12)
    ax.text(370, 260, 'B', fontsize=12)
    ax.text(280, 370, 'B', fontsize=12)
    ax.text(160, 370, 'C', fontsize=12)
    ax.text(160, 15, 'C', fontsize=12)
    ax.text(30, 140, 'D', fontsize=12)
    ax.text(370, 120, 'D', fontsize=12)
    ax.text(30, 370, 'E', fontsize=12)
    ax.text(370, 15, 'E', fontsize=12)
    
    res = clarke_error_grid(reference, predicted)
    stats_text = f"Zone A: {res['A']:.1f}%\nZone B: {res['B']:.1f}%\nZone C: {res['C']:.1f}%\nZone D: {res['D']:.1f}%\nZone E: {res['E']:.1f}%"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def glucose_prediction_metrics(y_true, y_pred) -> dict:
    """Calculate regression metrics and domain-specific metrics for glucose."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, a_min=1e-5, a_max=None))) * 100
    mard = np.mean(np.abs(y_true - y_pred) / y_true) * 100
    
    clarke = clarke_error_grid(y_true, y_pred)
    
    # Time lag analysis (cross-correlation)
    corr = np.correlate(y_true - np.mean(y_true), y_pred - np.mean(y_pred), mode='full')
    lags = np.arange(-len(y_true) + 1, len(y_true))
    best_lag = lags[np.argmax(corr)]
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'MARD': mard,
        'Clarke_A': clarke['A'],
        'Clarke_B': clarke['B'],
        'Clarke_A+B': clarke['A+B'],
        'Time_Lag': best_lag
    }

def cardiovascular_metrics(y_true, y_pred_proba, threshold=0.5) -> dict:
    """Calculate classification metrics for cardio risk."""
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    f1 = f1_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    
    return {
        'AUC-ROC': roc_auc,
        'PR-AUC': pr_auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1_Score': f1,
        'Brier_Score': brier,
        'Optimal_Threshold_Youden': optimal_threshold
    }

def plot_roc_curve(y_true, y_pred_proba, title='ROC Curve', save_path=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.plot(fpr[best_idx], tpr[best_idx], marker='o', markersize=8, color='red', label=f'Optimal Threshold ({thresholds[best_idx]:.2f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def plot_precision_recall_curve(y_true, y_pred_proba, title='PR Curve', save_path=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def plot_confusion_matrix(y_true, y_pred, labels=['Low Risk', 'High Risk'], title='Confusion Matrix', save_path=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def plot_calibration_curve(y_true, y_pred_proba, n_bins=10, title='Calibration Curve', save_path=None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    
    ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    return fig

def prediction_horizon_analysis(y_true_sequences, y_pred_sequences, horizon_steps, interval_minutes=5) -> pd.DataFrame:
    """Analyze model accuracy over prediction horizon."""
    results = []
    
    for i in range(horizon_steps):
        true_step = y_true_sequences[:, i]
        pred_step = y_pred_sequences[:, i]
        
        rmse = np.sqrt(np.mean((true_step - pred_step)**2))
        mae = np.mean(np.abs(true_step - pred_step))
        mape = np.mean(np.abs((true_step - pred_step) / np.clip(true_step, a_min=1e-5, a_max=None))) * 100
        
        results.append({
            'horizon_minutes': (i + 1) * interval_minutes,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        })
        
    return pd.DataFrame(results)
