from __future__ import annotations

import numpy as np

from ml.model_trainer import ModelTrainer
from ml.predictor import MLPredictor


def _candles():
    return [
        {
            "timestamp": "2024-01-01 09:15:00",
            "open": 100.0,
            "close": 100.5,
            "volume": 1200,
            "buyers_pct": 52.0,
            "force_score": 0.65,
            "f1": 0.60,
            "f2": 0.40,
            "f3": 0.50,
            "f4": 0.70,
            "f5": 0.55,
            "f6": 0.45,
            "f7": 0.80,
            "f8": 0.90,
            "f9": 0.65,
            "lead_value": 0.1,
            "catchup_time": 5,
            "lead_progress_pct": 0.42,
            "is_trending": 1,
            "hurst_exponent": 0.62,
            "volatility_ratio": 1.1,
        },
        {
            "timestamp": "2024-01-01 09:16:00",
            "open": 100.5,
            "close": 101.0,
            "volume": 1300,
            "buyers_pct": 53.0,
            "force_score": 0.72,
            "f1": 0.70,
            "f2": 0.45,
            "f3": 0.60,
            "f4": 0.78,
            "f5": 0.62,
            "f6": 0.50,
            "f7": 0.83,
            "f8": 0.95,
            "f9": 0.68,
            "lead_value": 0.15,
            "catchup_time": 4,
            "lead_progress_pct": 0.48,
            "is_trending": 1,
            "hurst_exponent": 0.7,
            "volatility_ratio": 1.2,
        },
        {
            "timestamp": "2024-01-01 09:17:00",
            "open": 101.0,
            "close": 101.4,
            "volume": 1500,
            "buyers_pct": 54.0,
            "force_score": 0.81,
            "f1": 0.75,
            "f2": 0.55,
            "f3": 0.65,
            "f4": 0.82,
            "f5": 0.67,
            "f6": 0.55,
            "f7": 0.85,
            "f8": 1.0,
            "f9": 0.73,
            "lead_value": 0.22,
            "catchup_time": 3,
            "lead_progress_pct": 0.54,
            "is_trending": 1,
            "hurst_exponent": 0.78,
            "volatility_ratio": 1.3,
        },
        {
            "timestamp": "2024-01-01 09:18:00",
            "open": 101.4,
            "close": 101.8,
            "volume": 1700,
            "buyers_pct": 55.0,
            "force_score": 0.9,
            "f1": 0.8,
            "f2": 0.6,
            "f3": 0.7,
            "f4": 0.87,
            "f5": 0.72,
            "f6": 0.60,
            "f7": 0.88,
            "f8": 1.1,
            "f9": 0.8,
            "lead_value": 0.27,
            "catchup_time": 3,
            "lead_progress_pct": 0.6,
            "is_trending": 1,
            "hurst_exponent": 0.8,
            "volatility_ratio": 1.4,
        },
        {
            "timestamp": "2024-01-01 09:19:00",
            "open": 101.8,
            "close": 102.1,
            "volume": 1800,
            "buyers_pct": 56.0,
            "force_score": 0.85,
            "f1": 0.82,
            "f2": 0.58,
            "f3": 0.75,
            "f4": 0.9,
            "f5": 0.75,
            "f6": 0.62,
            "f7": 0.9,
            "f8": 1.15,
            "f9": 0.83,
            "lead_value": 0.31,
            "catchup_time": 2,
            "lead_progress_pct": 0.67,
            "is_trending": 1,
            "hurst_exponent": 0.81,
            "volatility_ratio": 1.45,
        },
    ]


def test_prepare_features_builds_matrix_and_labels():
    trainer = ModelTrainer()
    X, y = trainer.prepare_features(_candles())
    assert X.shape[0] == len(_candles()) - 1
    assert X.shape[1] >= 30
    assert set(np.unique(y)).issubset({0, 1})


def test_train_ensemble_returns_models_and_scores():
    trainer = ModelTrainer()
    X, y = trainer.prepare_features(_candles())
    models, scores = trainer.train_ensemble(X, y)
    assert isinstance(models, dict)
    assert set(models).issuperset({"random_forest", "gradient_boosting", "xgboost", "logistic"})
    assert isinstance(scores, dict)


def test_predict_live_returns_signal_dictionary():
    predictor = MLPredictor()
    result = predictor.predict_live({"f1": 0.8, "f2": 0.7, "f3": 0.6, "f4": 0.9, "f5": 0.8, "f6": 0.5, "f7": 0.7, "f8": 0.8, "f9": 0.6}, {"lead_value": 0.3}, {"is_trending": 1, "hurst_exponent": 0.7, "volatility_ratio": 1.2})
    assert set(result.keys()) >= {"win_probability", "individual_models", "confidence", "signal"}
    assert 0.0 <= result["win_probability"] <= 1.0
    assert result["signal"] in {"BUY", "SELL", "WAIT"}
