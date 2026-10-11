import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, brier_score_loss, confusion_matrix
from typing import List, Dict, Any

class CardioRiskScorer:
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        self.params = {
            'n_estimators': config.get('n_estimators', 300),
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.05),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
            'eval_metric': config.get('eval_metric', 'auc'),
            'early_stopping_rounds': 50
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.feature_names = None

    def train(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        self.feature_names = X_train.columns.tolist() if isinstance(X_train, pd.DataFrame) else [f'feature_{i}' for i in range(X_train.shape[1])]
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        preds_proba = self.model.predict_proba(X_val)[:, 1]
        preds = self.model.predict(X_val)
        
        auc_roc = roc_auc_score(y_val, preds_proba)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        brier_score = brier_score_loss(y_val, preds_proba)
        
        return {
            'auc_roc': auc_roc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'brier_score': brier_score
        }

    def predict_risk(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def classify_risk(self, probabilities) -> List[str]:
        categories = []
        for p in probabilities:
            if p < 0.2:
                categories.append('Low')
            elif p <= 0.4:
                categories.append('Moderate')
            elif p <= 0.6:
                categories.append('High')
            else:
                categories.append('Critical')
        return categories

    def explain_prediction(self, X_single) -> Dict[str, Any]:
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_single)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            
        feature_vals = X_single.iloc[0] if isinstance(X_single, pd.DataFrame) else X_single[0]
        
        importance_dict = {
            'feature': self.feature_names,
            'shap_value': shap_values[0],
            'value': feature_vals
        }
        
        df_imp = pd.DataFrame(importance_dict)
        df_imp['abs_shap'] = df_imp['shap_value'].abs()
        df_imp = df_imp.sort_values(by='abs_shap', ascending=False)
        
        return {
            'shap_values': shap_values[0],
            'feature_names': df_imp['feature'].tolist(),
            'feature_values': df_imp['value'].tolist(),
            'base_value': explainer.expected_value
        }

    def get_feature_importance(self) -> pd.DataFrame:
        booster = self.model.get_booster()
        gain = booster.get_score(importance_type='gain')
        weight = booster.get_score(importance_type='weight')
        cover = booster.get_score(importance_type='cover')
        
        df = pd.DataFrame({
            'gain': pd.Series(gain),
            'weight': pd.Series(weight),
            'cover': pd.Series(cover)
        }).fillna(0)
        return df

    def evaluate(self, X_test, y_test) -> Dict[str, Any]:
        preds_proba = self.model.predict_proba(X_test)[:, 1]
        preds = self.model.predict(X_test)
        
        auc_roc = roc_auc_score(y_test, preds_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        brier_score = brier_score_loss(y_test, preds_proba)
        
        fpr, tpr, thresholds = roc_curve(y_test, preds_proba)
        
        return {
            'auc_roc': auc_roc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1': f1,
            'brier_score': brier_score,
            'classification_report': classification_report(y_test, preds, output_dict=True),
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        }

    def save_model(self, path):
        joblib.dump({'model': self.model, 'feature_names': self.feature_names}, path)

    def load_model(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
