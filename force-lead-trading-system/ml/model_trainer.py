from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


class ModelTrainer:
    """Feature engineering and time-series ensemble training for the Force-Lead system."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_ = []
        self.models_ = {}
        self.cv_scores_ = {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _series_slope(values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values), dtype=float)
        y = np.asarray(values, dtype=float)
        if np.allclose(y, y[0]):
            return 0.0
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        values = [float(v) for v in values]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _std(values: Sequence[float]) -> float:
        values = [float(v) for v in values]
        return float(np.std(values)) if len(values) > 1 else 0.0

    @classmethod
    def _get_time_features(cls, candle: Dict[str, Any]) -> List[float]:
        timestamp = candle.get("timestamp")
        hour = 0
        minute = 0
        day_of_week = 0

        if timestamp is not None:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromtimestamp(float(timestamp))
                hour = dt.hour
                minute = dt.minute
                day_of_week = min(dt.weekday(), 4)
            except Exception:
                pass

        return [
            math.sin(2 * math.pi * hour / 24.0),
            math.cos(2 * math.pi * hour / 24.0),
            math.sin(2 * math.pi * minute / 60.0),
            math.cos(2 * math.pi * minute / 60.0),
            float(day_of_week),
        ]

    @classmethod
    def _rolling_window(cls, candles: Sequence[Dict[str, Any]], index: int, window: int):
        start = max(0, index - window + 1)
        return list(candles[start:index + 1])

    @classmethod
    def _build_feature_vector(cls, current_candle: Dict[str, Any], historical_candles: Sequence[Dict[str, Any]], idx: int) -> List[float]:
        raw_factors = [cls._safe_float(current_candle.get(f"f{i}"), 0.0) for i in range(1, 10)]
        time_features = cls._get_time_features(current_candle)

        recent_window_5 = cls._rolling_window(historical_candles, idx, 5)
        recent_window_20 = cls._rolling_window(historical_candles, idx, 20)

        volume_5 = [cls._safe_float(c.get("volume"), 0.0) for c in recent_window_5]
        volume_20 = [cls._safe_float(c.get("volume"), 0.0) for c in recent_window_20]
        force_scores_5 = [cls._safe_float(c.get("force_score"), 0.0) for c in recent_window_5]
        force_scores_20 = [cls._safe_float(c.get("force_score"), 0.0) for c in recent_window_20]
        buyers_pct_5 = [cls._safe_float(c.get("buyers_pct"), 0.0) for c in recent_window_5]
        buyers_pct_20 = [cls._safe_float(c.get("buyers_pct"), 0.0) for c in recent_window_20]

        current_close = cls._safe_float(current_candle.get("close"), 0.0)
        five_ago_close = cls._safe_float(historical_candles[idx - 5].get("close"), current_close) if idx >= 5 else current_close
        twenty_ago_close = cls._safe_float(historical_candles[idx - 20].get("close"), current_close) if idx >= 20 else current_close

        price_momentum_5 = 0.0 if abs(five_ago_close) < 1e-8 else (current_close - five_ago_close) / five_ago_close
        price_momentum_20 = 0.0 if abs(twenty_ago_close) < 1e-8 else (current_close - twenty_ago_close) / twenty_ago_close

        feature_vector = [
            *raw_factors,
            *time_features,
            cls._mean(volume_5),
            cls._std(volume_5),
            price_momentum_5,
            cls._series_slope(buyers_pct_5),
            cls._mean(force_scores_5),
            cls._std(force_scores_5),
            cls._mean(volume_20),
            cls._std(volume_20),
            price_momentum_20,
            cls._series_slope(buyers_pct_20),
            cls._mean(force_scores_20),
            cls._safe_float(current_candle.get("lead_value"), 0.0),
            cls._safe_float(current_candle.get("catchup_time"), 0.0),
            cls._safe_float(current_candle.get("lead_progress_pct"), 0.0),
            cls._safe_float(current_candle.get("is_trending"), 0.0),
            cls._safe_float(current_candle.get("hurst_exponent"), 0.5),
            cls._safe_float(current_candle.get("volatility_ratio"), 1.0),
        ]
        return feature_vector

    def prepare_features(self, historical_candles: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Feature engineering - for each candle, build rolling statistics and label the next candle outcome.
        """
        if not historical_candles:
            return np.empty((0, 0), dtype=float), np.empty((0,), dtype=int)

        X: List[List[float]] = []
        y: List[int] = []

        for idx in range(len(historical_candles) - 1):
            current = historical_candles[idx]
            next_candle = historical_candles[idx + 1]
            feature_vector = self._build_feature_vector(current, historical_candles, idx)
            X.append(feature_vector)
            current_close = self._safe_float(current.get("close"), 0.0)
            next_close = self._safe_float(next_candle.get("close"), current_close)
            y.append(1 if next_close > current_close else 0)

        feature_count = len(X[0]) if X else 0
        self.feature_names_ = [
            f"f{i}" for i in range(1, 10)
        ] + [
            "hour_sin", "hour_cos", "minute_sin", "minute_cos", "day_of_week",
            "vol_mean_5", "vol_std_5", "price_momentum_5", "buyers_pct_trend", "force_score_mean_5", "force_score_std_5",
            "vol_mean_20", "vol_std_20", "price_momentum_20", "buyers_pct_trend_20", "force_score_mean_20",
            "lead_value", "catchup_time", "lead_progress_pct", "is_trending", "hurst_exponent", "volatility_ratio",
        ]
        if feature_count > len(self.feature_names_):
            self.feature_names_.extend([f"extra_{i}" for i in range(feature_count - len(self.feature_names_))])

        return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

    @staticmethod
    def _balance_single_class_data(X: np.ndarray, y: np.ndarray):
        """Create a minimal class-balanced fallback when all labels are identical."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if np.unique(y).size > 1:
            return X, y

        if X.shape[0] == 0:
            return X, y

        offset = np.eye(X.shape[1], dtype=float)[0] * 1e-3
        flipped = np.vstack([X, X + offset])
        labels = np.concatenate([y, 1 - y])
        return flipped, labels

    def train_ensemble(self, X: np.ndarray, y: np.ndarray):
        """
        Train a small time-series ensemble: random forest, gradient boosting, XGBoost, logistic regression.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty.")
        if np.unique(y).size < 2:
            X, y = self._balance_single_class_data(X, y)

        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_split=20,
                random_state=42,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                eval_metric="logloss",
                random_state=42,
                use_label_encoder=False,
            ),
            "logistic": LogisticRegression(C=0.1, max_iter=1000),
        }

        n_splits = min(5, max(2, len(X) - 1))
        cv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores: Dict[str, Dict[str, float]] = {}

        for name, model in models.items():
            accuracies: List[float] = []
            aucs: List[float] = []
            for train_idx, test_idx in cv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                    continue

                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                accuracies.append(acc)

                try:
                    prob = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, prob)
                    aucs.append(auc)
                except Exception:
                    aucs.append(acc)

            if not accuracies:
                accuracies = [0.0]
            if not aucs:
                aucs = [0.5]

            cv_scores[name] = {
                "accuracy": float(np.mean(accuracies)),
                "auc": float(np.mean(aucs)),
            }

        for name, model in models.items():
            model.fit(X, y)
        self.models_ = models
        self.cv_scores_ = cv_scores
        self.scaler.fit(X)
        return models, cv_scores

    def get_feature_importance(self, trained_rf_model, feature_names: Sequence[str]) -> Dict[str, float]:
        """Return the feature importances from the trained random forest (sorted descending)."""
        if not hasattr(trained_rf_model, "feature_importances_"):
            return {name: 0.0 for name in feature_names}

        importances = trained_rf_model.feature_importances_
        ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

        for name, imp in ranked:
            print(f"{name}: {imp:.4f}")

        return {name: float(imp) for name, imp in ranked}

    def save_and_schedule_retrain(self, models: Dict[str, Any], retrain_days: int = 7, scaler: Any | None = None):
        """Persist models and save a retrain schedule for the next training cycle."""
        base_dir = Path(__file__).resolve().parent
        joblib.dump(models, base_dir / "saved_models.pkl")
        if scaler is not None:
            joblib.dump(scaler, base_dir / "scaler.pkl")
        else:
            joblib.dump(self.scaler, base_dir / "scaler.pkl")

        next_retrain = datetime.now() + timedelta(days=retrain_days)
        schedule = {"next_retrain": next_retrain.isoformat()}
        (base_dir / "retrain_schedule.json").write_text(json.dumps(schedule, indent=2), encoding="utf-8")
        self.next_retrain = next_retrain
        return next_retrain

    def train(self, X, y):
        """Backward-compatible convenience wrapper for the default model."""
        _, _ = self.train_ensemble(X, y)
        return self.models_

    def save(self, path: str):
        joblib.dump(self.models_, path)

    def load(self, path: str):
        self.models_ = joblib.load(path)
        return self.models_
