from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize


class FactorModel:
    """Factor aggregation with mathematically derived optimal weights."""

    FACTOR_NAMES = [
        "F1_Volume",
        "F2_Wall",
        "F3_Position",
        "F4_Accel",
        "F5_Institutional",
        "F6_OI",
        "F7_Spread",
        "F8_Nifty",
        "F9_Candle",
    ]

    def __init__(self, weights: Optional[Mapping[str, float]] = None):
        self.weights = {
            "momentum": 0.30,
            "volume": 0.25,
            "trend": 0.25,
            "volatility": 0.20,
        }
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})

    @staticmethod
    def _normalize(value: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric > 1.0:
            return 1.0
        if numeric < -1.0:
            return -1.0
        return numeric

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def aggregate_score(self, factors: Mapping[str, float]) -> float:
        """Combine multiple factor scores into a final normalized score out of 100."""
        if not factors:
            return 0.0

        total_weight = 0.0
        weighted_score = 0.0
        for name, weight in self.weights.items():
            value = factors.get(name, 0.0)
            normalized = self._normalize(value)
            weighted_score += normalized * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        score = (weighted_score / total_weight) * 100.0
        return max(0.0, min(100.0, float(score)))

    def score_series(self, factors: Iterable[Mapping[str, float]]) -> list[float]:
        return [self.aggregate_score(factor_map) for factor_map in factors]

    def build_factor_matrix(self, historical_candles: Sequence[Mapping[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
        """Build a matrix of 9 factor scores against the future return of each candle."""
        if not historical_candles:
            return np.empty((0, 9), dtype=float), np.empty((0,), dtype=float)

        rows = []
        returns = []

        for index in range(len(historical_candles) - 1):
            current = historical_candles[index]
            nxt = historical_candles[index + 1]

            current_close = self._safe_float(current.get("close", current.get("ltp", 0.0)))
            next_close = self._safe_float(nxt.get("close", nxt.get("ltp", 0.0)))
            pct_return = 0.0 if current_close == 0 else ((next_close - current_close) / current_close) * 100.0

            volume = self._safe_float(current.get("volume", 0.0))
            prev_volume = self._safe_float(historical_candles[index - 1].get("volume", volume) if index > 0 else volume)
            volume_ratio = 1.0 if prev_volume == 0 else volume / prev_volume
            momentum = 1.0 if current_close >= next_close else -1.0
            trend = 1.0 if next_close >= current_close else -1.0
            spread = self._safe_float(current.get("spread", 0.0))
            nifty_direction = self._safe_float(current.get("nifty_direction", 0.0))
            body = abs(self._safe_float(current.get("close", 0.0)) - self._safe_float(current.get("open", 0.0)))
            total_range = max(self._safe_float(current.get("high", 0.0)) - self._safe_float(current.get("low", 0.0)), 1e-6)
            candle_conviction = body / total_range

            f1 = min(max(volume_ratio - 1.0, -1.0), 1.0)
            bid_total = self._safe_float(current.get("bid_total", 0.0))
            ask_total = self._safe_float(current.get("ask_total", 0.0))
            f2_value = (bid_total - ask_total) / max(bid_total + ask_total, 1e-6)
            f2 = min(max(f2_value, -1.0), 1.0)
            day_low = self._safe_float(current.get("low", 0.0))
            day_high = self._safe_float(current.get("high", 0.0))
            f3_value = (current_close - day_low) / max(day_high - day_low, 1e-6)
            f3 = min(max(f3_value, 0.0), 1.0)
            f4 = min(max(momentum * volume_ratio, -1.0), 1.0)
            buyers_pct = self._safe_float(current.get("buyers_pct", 0.0))
            sellers_pct = self._safe_float(current.get("sellers_pct", 0.0))
            f5 = min(max((buyers_pct - sellers_pct) / 100.0, -1.0), 1.0)
            current_call_oi = self._safe_float(current.get("current_call_oi", 0.0))
            prev_call_oi = self._safe_float(current.get("prev_call_oi", 0.0))
            current_put_oi = self._safe_float(current.get("current_put_oi", 0.0))
            prev_put_oi = self._safe_float(current.get("prev_put_oi", 0.0))
            oi_delta = (current_call_oi - prev_call_oi) - (current_put_oi - prev_put_oi)
            f6 = min(max(oi_delta / max(abs(oi_delta) + 1.0, 1e-6), -1.0), 1.0)
            avg_spread = self._safe_float(current.get("avg_spread", spread or 1.0))
            f7 = min(max(1.0 - (spread / max(avg_spread, 1e-6)), -1.0), 1.0)
            f8 = min(max(nifty_direction, -1.0), 1.0)
            f9 = min(max(candle_conviction * trend, -1.0), 1.0)

            factor_row = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

            rows.append(factor_row)
            returns.append(pct_return)

        matrix = np.asarray(rows, dtype=float)
        if matrix.size == 0:
            return np.empty((0, 9), dtype=float), np.empty((0,), dtype=float)
        if matrix.shape[1] != 9:
            matrix = np.pad(matrix, ((0, 0), (0, 9 - matrix.shape[1])), mode='constant', constant_values=0.0)
        return matrix, np.asarray(returns, dtype=float)

    def calculate_correlation_matrix(self, factor_matrix: np.ndarray):
        """Compute a numerically stable correlation matrix and condition number."""
        matrix = np.asarray(factor_matrix, dtype=float)
        if matrix.size == 0:
            return np.empty((0, 0), dtype=float), 0.0

        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)

        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        feature_std = matrix.std(axis=0)
        if np.any(feature_std == 0):
            zero_variance = np.where(feature_std == 0)[0]
            for idx in zero_variance:
                matrix[:, idx] = 0.0
            feature_std = matrix.std(axis=0)

        if matrix.shape[0] <= 1:
            corr_matrix = np.eye(matrix.shape[1], dtype=float)
            return corr_matrix, 1.0

        covariance = np.cov(matrix, rowvar=False)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        diagonal = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
        diagonal[diagonal == 0.0] = 1.0
        corr_matrix = covariance / np.outer(diagonal, diagonal)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        eigvals = np.linalg.eigvalsh(corr_matrix)
        positive_eigvals = eigvals[eigvals > 0]
        if positive_eigvals.size == 0:
            condition_number = 0.0
        else:
            condition_number = float(positive_eigvals.max() / positive_eigvals.min())

        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > 0.8:
                    print(f"WARNING: F{i + 1} and F{j + 1} highly correlated")
                    print("Consider removing one to reduce redundancy")
        if condition_number > 30:
            print("High multicollinearity detected, use Ridge regression")
        return corr_matrix, float(condition_number)

    def pca_analysis(self, factor_matrix: np.ndarray, variance_threshold: float = 0.90):
        """Reduce dimension using PCA and report the full variance-explained spectrum."""
        if factor_matrix.size == 0:
            return np.empty((0, 0), dtype=float), 0, np.empty((0,), dtype=float)

        pca = PCA()
        pca.fit(factor_matrix)
        explained = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained)
        n_components = next((i + 1 for i, v in enumerate(cumulative_variance) if v >= variance_threshold), len(explained))

        print(f"{n_components} factors explain {variance_threshold * 100}% variance")
        print(f"Factor importance ranking: {explained}")

        pca_final = PCA(n_components=min(n_components, factor_matrix.shape[1]))
        reduced_matrix = pca_final.fit_transform(factor_matrix)
        return reduced_matrix, n_components, explained

    def derive_optimal_weights_regression(self, factor_matrix: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Use Ridge regression to derive weights from historical data, then normalize them."""
        if factor_matrix.size == 0 or returns.size == 0:
            return {name: 0.0 for name in self.FACTOR_NAMES}

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(factor_matrix)
        y = np.asarray(returns, dtype=float)

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)

        raw_weights = np.abs(model.coef_)
        positive_weights = raw_weights if np.sum(raw_weights) > 0 else np.ones_like(raw_weights)
        optimal_weights = positive_weights / positive_weights.sum()

        factor_names = self.FACTOR_NAMES
        output = dict(zip(factor_names, optimal_weights))

        for name, weight in sorted(output.items(), key=lambda x: x[1], reverse=True):
            print(f"{name}: {weight:.3f}")

        config_path = Path(__file__).resolve().parents[1] / "config" / "optimal_weights.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        return output

    def portfolio_optimization(self, stock_signals: Sequence[float], covariance_matrix: np.ndarray):
        """Optimize a Markowitz portfolio when trading multiple stocks concurrently."""
        n_stocks = len(stock_signals)
        if n_stocks == 0:
            return np.array([], dtype=float)

        expected_returns = np.asarray(stock_signals, dtype=float)
        cov_matrix = np.asarray(covariance_matrix, dtype=float)

        if cov_matrix.shape != (n_stocks, n_stocks):
            cov_matrix = np.eye(n_stocks, dtype=float)

        def portfolio_variance(weights):
            return float(weights @ cov_matrix @ weights)

        def neg_sharpe(weights):
            ret = expected_returns @ weights
            risk = np.sqrt(portfolio_variance(weights))
            if risk == 0:
                return -ret
            return float(-ret / risk)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]
        bounds = [(0.0, 0.3)] * n_stocks
        initial_guess = np.full(n_stocks, 1.0 / n_stocks, dtype=float)

        result = minimize(neg_sharpe, x0=initial_guess, bounds=bounds, constraints=constraints)
        if not result.success:
            return initial_guess
        return np.asarray(result.x, dtype=float)


def calculate_factor_score(factors: Mapping[str, float], weights: Optional[Mapping[str, float]] = None) -> float:
    """Convenience function for a single-factor aggregate score."""
    model = FactorModel(weights=weights)
    return model.aggregate_score(factors)
