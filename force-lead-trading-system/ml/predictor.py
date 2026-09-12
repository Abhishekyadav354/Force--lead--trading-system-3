from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import joblib
import numpy as np


class MLPredictor:
    """Predict a directional trade probability using the saved ensemble models."""

    MODEL_WEIGHTS = {
        "random_forest": 0.35,
        "xgboost": 0.35,
        "gradient_boosting": 0.20,
        "logistic": 0.10,
    }

    def __init__(self, model_dir: str | None = None):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent
        self.models = {}
        self.scaler = None
        self._load_saved_models()

    def _load_saved_models(self):
        models_path = self.model_dir / "saved_models.pkl"
        scaler_path = self.model_dir / "scaler.pkl"

        if models_path.exists():
            self.models = joblib.load(models_path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)

    def _coerce_feature_dict(self, current_factors: Dict[str, Any]) -> Dict[str, float]:
        normalized = {f"f{i}": 0.0 for i in range(1, 10)}
        if isinstance(current_factors, dict):
            for key, value in current_factors.items():
                if key.startswith("f") and key[1:].isdigit():
                    normalized[key] = float(value)
        return normalized

    def build_feature_vector(self, current_factors: Dict[str, Any], lead_status: Dict[str, Any], regime: Dict[str, Any]) -> List[float]:
        """Build the same feature vector layout expected by model training."""
        factors = self._coerce_feature_dict(current_factors)
        raw = [factors[f"f{i}"] for i in range(1, 10)]

        lead_status = lead_status or {}
        regime = regime or {}

        time_features = [
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
        ]

        rolling_stats = [
            float(lead_status.get("vol_mean_5", 0.0)),
            float(lead_status.get("vol_std_5", 0.0)),
            float(lead_status.get("price_momentum_5", 0.0)),
            float(lead_status.get("buyers_pct_trend", 0.0)),
            float(lead_status.get("force_score_mean_5", 0.0)),
            float(lead_status.get("force_score_std_5", 0.0)),
            float(lead_status.get("vol_mean_20", 0.0)),
            float(lead_status.get("vol_std_20", 0.0)),
            float(lead_status.get("price_momentum_20", 0.0)),
            float(lead_status.get("buyers_pct_trend_20", 0.0)),
            float(lead_status.get("force_score_mean_20", 0.0)),
        ]

        lead = [
            float(lead_status.get("lead_value", 0.0)),
            float(lead_status.get("catchup_time", 0.0)),
            float(lead_status.get("lead_progress_pct", 0.0)),
            float(regime.get("is_trending", 0.0)),
            float(regime.get("hurst_exponent", 0.5)),
            float(regime.get("volatility_ratio", 1.0)),
        ]

        return raw + time_features + rolling_stats + lead

    def predict_live(self, current_factors: Dict[str, Any], lead_status: Dict[str, Any], regime: Dict[str, Any]):
        """Load the saved ensemble, score a current feature vector, and return a directional probability."""
        if not self.models:
            return {
                "win_probability": 0.5,
                "individual_models": {},
                "confidence": 0.0,
                "signal": "WAIT",
            }

        feature_vector = self.build_feature_vector(current_factors, lead_status, regime)
        if self.scaler is not None:
            feature_vector = self.scaler.transform([feature_vector])[0]

        probabilities = []
        model_names = []
        for name, model in self.models.items():
            if hasattr(model, "predict_proba"):
                model_names.append(name)
                prob = float(model.predict_proba([feature_vector])[0][1])
                probabilities.append(prob)

        if not probabilities:
            return {
                "win_probability": 0.5,
                "individual_models": {},
                "confidence": 0.0,
                "signal": "WAIT",
            }

        weights = [self.MODEL_WEIGHTS.get(name, 1.0 / len(model_names)) for name in model_names]
        total_weight = sum(weights)
        ensemble_prob = sum(w * p for w, p in zip(weights, probabilities)) / total_weight
        confidence = max(probabilities) - min(probabilities)

        signal = "BUY" if ensemble_prob > 0.6 else "SELL" if ensemble_prob < 0.4 else "WAIT"
        return {
            "win_probability": float(np.clip(ensemble_prob, 0.0, 1.0)),
            "individual_models": dict(zip(model_names, probabilities)),
            "confidence": float(confidence),
            "signal": signal,
        }


class Predictor(MLPredictor):
    """Backward-compatible alias for the live predictor."""

    pass
