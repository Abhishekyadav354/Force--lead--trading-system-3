from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller


class SignificanceTester:
    """Signal significance and statistical robustness checks for trading systems."""

    @staticmethod
    def z_test(sample_a, sample_b):
        a = np.asarray(sample_a, dtype=float)
        b = np.asarray(sample_b, dtype=float)
        mean_diff = np.mean(a) - np.mean(b)
        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
        if pooled_std == 0:
            return 0.0
        return float(mean_diff / pooled_std)

    @staticmethod
    def test_signal_significance(force_scores_history: List[float], current_signal: float) -> Dict[str, Any]:
        """Null hypothesis: current signal is random noise."""
        if not force_scores_history:
            return {
                "z_score": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "confidence_pct": 0.0,
            }

        history = np.asarray(force_scores_history, dtype=float)
        mean = np.mean(history)
        std_dev = np.std(history)
        if std_dev == 0:
            return {
                "z_score": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "confidence_pct": 0.0,
            }

        z = (current_signal - mean) / std_dev
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        is_significant = p_value < 0.05
        return {
            "z_score": float(z),
            "p_value": float(p_value),
            "is_significant": bool(is_significant),
            "confidence_pct": float((1 - p_value) * 100.0),
        }

    @staticmethod
    def rolling_significance_streak(signal_history: List[float], window: int = 20) -> Dict[str, Any]:
        """Track consecutive significant signal events."""
        if not signal_history:
            return {"streak_count": 0, "confidence_boost": 1.0}

        streak = 0
        for signal in reversed(signal_history[-window:]):
            # use a simple one-sample t-test versus zero mean
            sample = np.asarray(signal_history[max(0, len(signal_history) - window):], dtype=float)
            if len(sample) < 2:
                break
            try:
                _, p_val = stats.ttest_1samp(sample, popmean=0)
            except Exception:
                p_val = 1.0
            if p_val < 0.05:
                streak += 1
            else:
                break

        if streak >= 5:
            confidence_boost = 1.5
        elif streak >= 3:
            confidence_boost = 1.3
        else:
            confidence_boost = 1.0

        return {
            "streak_count": int(streak),
            "confidence_boost": float(confidence_boost),
        }

    @staticmethod
    def signal_to_noise_ratio(signal_history: List[float]) -> Dict[str, Any]:
        """SNR = abs(mean signal) / std dev of signal."""
        if not signal_history:
            return {"snr_value": 0.0, "quality_label": "NO_DATA"}

        arr = np.asarray(signal_history, dtype=float)
        mean_signal = np.mean(arr)
        std_dev = np.std(arr)
        if std_dev == 0:
            return {"snr_value": 0.0, "quality_label": "NO_NOISE"}

        snr_value = abs(mean_signal) / std_dev
        if snr_value > 2.0:
            label = "TRUST_FULLY"
        elif snr_value >= 1.0:
            label = "MODERATE_SIGNAL"
        else:
            label = "NOISY_SIGNAL"

        return {
            "snr_value": float(snr_value),
            "quality_label": label,
        }

    @staticmethod
    def confidence_interval_check(score_history: List[float], current_score: float) -> Dict[str, Any]:
        """Check whether current score is unusual relative to recent history."""
        if len(score_history) < 2:
            return {"ci_lower": current_score, "ci_upper": current_score, "is_unusual": False, "signal_type": "NORMAL"}

        recent = np.asarray(score_history[-20:], dtype=float)
        mean = np.mean(recent)
        std = np.std(recent)
        n = len(recent)

        if std == 0 or n == 0:
            return {"ci_lower": mean, "ci_upper": mean, "is_unusual": False, "signal_type": "NORMAL"}

        ci_lower, ci_upper = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(n))
        is_unusual = current_score < ci_lower or current_score > ci_upper

        if is_unusual and abs(current_score) > 0.7:
            signal_type = "EXTREME_SIGNAL"
        elif is_unusual:
            signal_type = "UNUSUAL"
        else:
            signal_type = "NORMAL"

        return {
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "is_unusual": bool(is_unusual),
            "signal_type": signal_type,
        }

    @staticmethod
    def autocorrelation_test(price_returns: List[float], lag: int = 5) -> Dict[str, Any]:
        """Identify trend vs mean-reverting regime and test stationarity."""
        if len(price_returns) < 2:
            return {
                "dw_stat": 0.0,
                "market_type": "UNKNOWN",
                "is_trending": False,
                "is_mean_reverting": False,
                "adf_p_value": 1.0,
            }

        returns = np.asarray(price_returns, dtype=float)
        dw_stat = durbin_watson(returns)

        if dw_stat < 1.5:
            regime = "STRONG_POSITIVE_AUTOCORRELATION_TRENDING"
            is_trending = True
            is_mean_reverting = False
        elif dw_stat > 2.5:
            regime = "STRONG_NEGATIVE_AUTOCORRELATION_MEAN_REVERTING"
            is_trending = False
            is_mean_reverting = True
        else:
            regime = "RANDOM_OR_WEAK_AUTOCORRELATION"
            is_trending = False
            is_mean_reverting = False

        adf_result = adfuller(returns)
        adf_p_value = adf_result[1]
        if adf_p_value < 0.05:
            # Stationary mean-reverting behavior
            if regime == "RANDOM_OR_WEAK_AUTOCORRELATION":
                regime = "STATIONARY_MEAN_REVERTING"

        return {
            "dw_stat": float(dw_stat),
            "market_type": regime,
            "is_trending": bool(is_trending),
            "is_mean_reverting": bool(is_mean_reverting),
            "adf_p_value": float(adf_p_value),
        }


def z_test(sample_a, sample_b):
    return SignificanceTester.z_test(sample_a, sample_b)
