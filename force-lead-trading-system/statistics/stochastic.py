from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller


class StochasticPriceModel:
    """Geometric Brownian motion and regime-detection utilities for price processes."""

    @staticmethod
    def estimate_gbm_parameters(price_history):
        """Estimate drift and volatility for a GBM model from log returns."""
        prices = np.asarray(price_history, dtype=float)
        if prices.size < 2:
            return {"mu": 0.0, "sigma": 0.0, "annualized_volatility": 0.0, "current_regime_vol": 0.0}

        log_returns = np.diff(np.log(np.clip(prices, 1e-12, None)))
        if log_returns.size == 0:
            return {"mu": 0.0, "sigma": 0.0, "annualized_volatility": 0.0, "current_regime_vol": 0.0}

        mu = float(np.mean(log_returns))
        sigma = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else float(np.std(log_returns))
        annualized_vol = sigma * np.sqrt(252 * 375)
        return {
            "mu": mu,
            "sigma": sigma,
            "annualized_volatility": annualized_vol,
            "current_regime_vol": sigma,
        }

    @staticmethod
    def simulate_gbm_paths(current_price, mu, sigma, horizon_minutes=10, n_paths=1000):
        """Simulate multiple GBM paths over a future horizon."""
        if horizon_minutes < 0:
            raise ValueError("horizon_minutes must be non-negative")
        if n_paths <= 0:
            return {
                "all_paths": np.empty((0, 0), dtype=float),
                "expected_price": float(current_price),
                "median_price": float(current_price),
                "price_5th_pct": float(current_price),
                "price_95th_pct": float(current_price),
                "prob_up": 0.0,
                "prob_down": 0.0,
            }

        dt = 1.0
        rng = np.random.default_rng()
        all_paths = []

        for _ in range(int(n_paths)):
            path = [float(current_price)]
            for _ in range(int(horizon_minutes)):
                dW = rng.normal(0.0, np.sqrt(dt))
                next_price = path[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * dW)
                path.append(float(next_price))
            all_paths.append(path)

        paths_array = np.asarray(all_paths, dtype=float)
        final_prices = paths_array[:, -1]

        return {
            "all_paths": paths_array,
            "expected_price": float(np.mean(final_prices)),
            "median_price": float(np.median(final_prices)),
            "price_5th_pct": float(np.percentile(final_prices, 5)),
            "price_95th_pct": float(np.percentile(final_prices, 95)),
            "prob_up": float(np.mean(final_prices > current_price)),
            "prob_down": float(np.mean(final_prices < current_price)),
        }

    @staticmethod
    def _hurst_exponent(ts, max_lag=20):
        ts = np.asarray(ts, dtype=float)
        if ts.size < 3:
            return 0.5

        lags = np.arange(2, min(max_lag, ts.size // 2) + 1)
        tau = []
        for lag in lags:
            x = ts[:-lag]
            y = ts[lag:]
            if x.size == 0 or y.size == 0:
                continue
            tau.append(np.std(y - x))
        if len(lags) != len(tau) or len(tau) < 2:
            return 0.5
        return float(np.polyfit(np.log(lags[: len(tau)]), np.log(tau), 1)[0])

    @staticmethod
    def _calculate_variance_ratio(ts, lag=10):
        ts = np.asarray(ts, dtype=float)
        if ts.size <= lag + 1:
            return 1.0
        returns = np.diff(ts)
        if returns.size == 0:
            return 1.0
        variance_of_lag = np.var(returns[lag:], ddof=1) if returns[lag:].size > 1 else 0.0
        variance_of_single = np.var(returns[:-lag], ddof=1) if returns[:-lag].size > 1 else 0.0
        if variance_of_single == 0:
            return 1.0
        return float(variance_of_lag / variance_of_single)

    @staticmethod
    def get_regime_weights(regime: str):
        regime_key = str(regime).upper()
        if regime_key == "TRENDING":
            return {
                "f1": 1.2,
                "f2": 1.0,
                "f3": 0.9,
                "f4": 1.3,
                "f5": 1.1,
                "f6": 1.0,
                "f7": 1.0,
                "f8": 1.1,
                "f9": 1.2,
            }
        if regime_key == "MEAN_REVERTING":
            return {
                "f1": 0.9,
                "f2": 1.3,
                "f3": 1.4,
                "f4": 0.9,
                "f5": 1.1,
                "f6": 1.0,
                "f7": 1.2,
                "f8": 1.0,
                "f9": 0.8,
            }
        return {
            "f1": 1.0,
            "f2": 1.0,
            "f3": 1.0,
            "f4": 1.0,
            "f5": 1.0,
            "f6": 1.0,
            "f7": 1.0,
            "f8": 1.0,
            "f9": 1.0,
        }

    @staticmethod
    def detect_market_regime(price_history, volume_history=None):
        """Detect trending, mean-reverting, or random-walk price behavior."""
        prices = np.asarray(price_history, dtype=float)
        if prices.size < 2:
            return {
                "regime": "RANDOM_WALK",
                "hurst": 0.5,
                "is_stationary": False,
                "volatility_regime": "NORMAL",
                "recommended_strategy": "Neutral, equal weights, be cautious",
                "weight_adjustments": StochasticPriceModel.get_regime_weights("RANDOM_WALK"),
            }

        adf_stat, p_value, *_ = adfuller(prices)
        is_stationary = bool(p_value < 0.05)
        H = StochasticPriceModel._hurst_exponent(prices)
        vr_ratio = StochasticPriceModel._calculate_variance_ratio(prices)

        if H > 0.6 and not is_stationary:
            regime = "TRENDING"
            strategy = "Follow momentum, higher F1/F4 weights"
        elif H < 0.4 and is_stationary:
            regime = "MEAN_REVERTING"
            strategy = "Counter-trend, higher F2/F3 weights"
        else:
            regime = "RANDOM_WALK"
            strategy = "Neutral, equal weights, be cautious"

        current_vol = float(np.std(prices[-20:])) if prices.size >= 20 else float(np.std(prices))
        avg_vol = float(np.std(prices[-min(100, prices.size):])) if prices.size > 1 else 0.0
        vol_regime = "HIGH" if avg_vol > 0 and current_vol > avg_vol * 1.5 else "NORMAL"

        return {
            "regime": regime,
            "hurst": float(H),
            "is_stationary": bool(is_stationary),
            "variance_ratio": float(vr_ratio),
            "volatility_regime": vol_regime,
            "recommended_strategy": strategy,
            "weight_adjustments": StochasticPriceModel.get_regime_weights(regime),
        }

    @staticmethod
    def ornstein_uhlenbeck_mean_reversion(price_history):
        """Estimate OU parameters: dX = θ(μ - X)dt + σdW."""
        prices = np.asarray(price_history, dtype=float)
        if prices.size < 2:
            return 0.0, 0.0, 0.0, 0.0

        x_t = prices[:-1]
        delta_x = np.diff(prices)
        slope, intercept, r_value, _, _ = stats.linregress(x_t, delta_x)

        theta = float(-slope)
        if abs(theta) < 1e-12:
            theta = 0.0
            mu = float(np.mean(prices))
        else:
            mu = float(intercept / theta)

        if x_t.size == 0 or delta_x.size == 0:
            sigma_ou = 0.0
        else:
            residuals = delta_x - (slope * x_t + intercept)
            sigma_ou = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(residuals))

        r_squared = float(r_value ** 2)
        return theta, mu, sigma_ou, r_squared


def geometric_brownian_motion(s0: float, mu: float, sigma: float, steps: int, dt: float = 1.0, rng=None):
    """Generate a simple geometric Brownian motion price path."""
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    path = [float(s0)]
    generator = np.random.default_rng() if rng is None else rng

    for _ in range(int(steps)):
        z = generator.normal(0, 1)
        next_value = path[-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
        path.append(float(next_value))
    return path


def stochastic_oscillator(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14):
    """Compute the stochastic oscillator %K and %D values for a price series."""
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, and close sequences must have the same length.")
    if period <= 0:
        raise ValueError("period must be positive.")
    if not high:
        return {"%k": 0.0, "%d": 0.0, "k": 0.0, "d": 0.0}

    k_values = []
    for idx in range(len(close)):
        start = max(0, idx - period + 1)
        window_high = high[start : idx + 1]
        window_low = low[start : idx + 1]
        if not window_high or not window_low:
            k_values.append(0.0)
            continue

        current_close = float(close[idx])
        lowest_low = min(window_low)
        highest_high = max(window_high)
        span = highest_high - lowest_low
        k_value = 50.0 if span == 0 else ((current_close - lowest_low) / span) * 100.0
        k_values.append(float(k_value))

    d_values = []
    for idx in range(len(k_values)):
        start = max(0, idx - period + 1)
        chunk = k_values[start : idx + 1]
        d_value = sum(chunk) / len(chunk) if chunk else 0.0
        d_values.append(float(d_value))

    return {
        "%k": float(k_values[-1]),
        "%d": float(d_values[-1]),
        "k": float(k_values[-1]),
        "d": float(d_values[-1]),
    }
