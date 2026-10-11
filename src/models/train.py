import os
import numpy as np
import pandas as pd
from .glucose_predictor import GlucosePredictor
from .cardio_risk_scorer import CardioRiskScorer
from .fusion_model import MultiModalFusionModel

class ModelTrainer:
    def __init__(self, config_path: str = 'config/settings.yaml'):
        self.config_path = config_path
        os.makedirs('results', exist_ok=True)
        os.makedirs('saved_models', exist_ok=True)

    def load_and_preprocess_data(self) -> dict:
        return {
            'glucose': {
                'X_train': np.random.rand(100, 72, 8),
                'y_train': np.random.rand(100, 24),
                'X_val': np.random.rand(20, 72, 8),
                'y_val': np.random.rand(20, 24)
            },
            'cardio': {
                'X_train': pd.DataFrame(np.random.rand(100, 35)),
                'y_train': np.random.randint(0, 2, 100),
                'X_val': pd.DataFrame(np.random.rand(20, 35)),
                'y_val': np.random.randint(0, 2, 20)
            },
            'fusion': {
                'X_temp_train': np.random.rand(100, 72, 8),
                'X_stat_train': np.random.rand(100, 35),
                'y_train': np.random.randint(0, 2, 100),
                'X_temp_val': np.random.rand(20, 72, 8),
                'X_stat_val': np.random.rand(20, 35),
                'y_val': np.random.randint(0, 2, 20)
            }
        }

    def train_glucose_predictor(self, data: dict) -> dict:
        print("Training Glucose Predictor...")
        model = GlucosePredictor()
        d = data['glucose']
        history = model.train(d['X_train'], d['y_train'], d['X_val'], d['y_val'], epochs=5, batch_size=16)
        model.save_model('saved_models/glucose_model.keras')
        return {'history': history.history}

    def train_cardio_risk_scorer(self, data: dict) -> dict:
        print("Training Cardio Risk Scorer...")
        model = CardioRiskScorer({'n_estimators': 10})
        d = data['cardio']
        metrics = model.train(d['X_train'], d['y_train'], d['X_val'], d['y_val'])
        eval_metrics = model.evaluate(d['X_val'], d['y_val'])
        model.save_model('saved_models/cardio_model.joblib')
        return {'metrics': metrics, 'eval': eval_metrics}

    def train_fusion_model(self, data: dict) -> dict:
        print("Training Fusion Model...")
        model = MultiModalFusionModel()
        d = data['fusion']
        history = model.train(
            d['X_temp_train'], d['X_stat_train'], d['y_train'],
            d['X_temp_val'], d['X_stat_val'], d['y_val'],
            epochs=5, batch_size=16
        )
        model.save_model('saved_models/fusion_model.keras')
        return {'history': history.history}

    def train_all(self) -> dict:
        data = self.load_and_preprocess_data()
        
        glucose_res = self.train_glucose_predictor(data)
        cardio_res = self.train_cardio_risk_scorer(data)
        fusion_res = self.train_fusion_model(data)
        
        results = {
            'glucose': glucose_res,
            'cardio': cardio_res,
            'fusion': fusion_res
        }
        
        self.generate_training_report(results)
        print("Training complete.")
        return results

    def generate_training_report(self, results: dict):
        print("Generating report in results/")
        with open('results/report.txt', 'w') as f:
            f.write(str(results))

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.train_all()
