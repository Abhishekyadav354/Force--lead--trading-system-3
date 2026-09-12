from __future__ import annotations

from statistics.backtester import Backtester


def _historical_data():
    return [
        {
            "timestamp": "2024-01-01 09:15:00",
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 2000,
            "buyers_pct": 52,
            "force_score": 0.7,
            "lead_value": 0.2,
            "catchup_time": 3,
            "lead_progress_pct": 0.4,
            "is_trending": 1,
            "hurst_exponent": 0.6,
            "volatility_ratio": 1.0,
            "f1": 0.8,
            "f2": 0.6,
            "f3": 0.5,
            "f4": 0.9,
            "f5": 0.7,
            "f6": 0.4,
            "f7": 0.8,
            "f8": 0.9,
            "f9": 0.7,
        },
        {
            "timestamp": "2024-01-01 09:16:00",
            "open": 100.5,
            "high": 101.5,
            "low": 100.2,
            "close": 101.1,
            "volume": 2100,
            "buyers_pct": 53,
            "force_score": 0.8,
            "lead_value": 0.25,
            "catchup_time": 2,
            "lead_progress_pct": 0.45,
            "is_trending": 1,
            "hurst_exponent": 0.7,
            "volatility_ratio": 1.1,
            "f1": 0.7,
            "f2": 0.7,
            "f3": 0.6,
            "f4": 0.85,
            "f5": 0.75,
            "f6": 0.5,
            "f7": 0.82,
            "f8": 1.0,
            "f9": 0.8,
        },
        {
            "timestamp": "2024-01-01 09:17:00",
            "open": 101.1,
            "high": 102.2,
            "low": 100.8,
            "close": 102.0,
            "volume": 2200,
            "buyers_pct": 54,
            "force_score": 0.9,
            "lead_value": 0.3,
            "catchup_time": 2,
            "lead_progress_pct": 0.5,
            "is_trending": 1,
            "hurst_exponent": 0.75,
            "volatility_ratio": 1.2,
            "f1": 0.82,
            "f2": 0.8,
            "f3": 0.68,
            "f4": 0.9,
            "f5": 0.8,
            "f6": 0.6,
            "f7": 0.85,
            "f8": 1.1,
            "f9": 0.85,
        },
        {
            "timestamp": "2024-01-01 09:18:00",
            "open": 102.0,
            "high": 102.8,
            "low": 101.3,
            "close": 102.4,
            "volume": 2300,
            "buyers_pct": 55,
            "force_score": 0.95,
            "lead_value": 0.35,
            "catchup_time": 2,
            "lead_progress_pct": 0.6,
            "is_trending": 1,
            "hurst_exponent": 0.8,
            "volatility_ratio": 1.25,
            "f1": 0.85,
            "f2": 0.82,
            "f3": 0.72,
            "f4": 0.95,
            "f5": 0.85,
            "f6": 0.63,
            "f7": 0.88,
            "f8": 1.2,
            "f9": 0.9,
        },
        {
            "timestamp": "2024-01-01 09:19:00",
            "open": 102.4,
            "high": 103.1,
            "low": 101.8,
            "close": 103.0,
            "volume": 2400,
            "buyers_pct": 56,
            "force_score": 1.0,
            "lead_value": 0.4,
            "catchup_time": 1,
            "lead_progress_pct": 0.7,
            "is_trending": 1,
            "hurst_exponent": 0.82,
            "volatility_ratio": 1.3,
            "f1": 0.9,
            "f2": 0.86,
            "f3": 0.78,
            "f4": 1.0,
            "f5": 0.9,
            "f6": 0.7,
            "f7": 0.9,
            "f8": 1.25,
            "f9": 0.95,
        },
    ]


def test_run_full_backtest():
    backtester = Backtester()
    metrics = backtester.run_full_backtest(_historical_data(), {"f1": 0.3, "f2": 0.2, "f3": 0.15, "f4": 0.15, "f5": 0.1, "f6": 0.05, "f7": 0.03, "f8": 0.01, "f9": 0.01}, capital=10000)
    assert isinstance(metrics, dict)
    assert metrics["total_trades"] >= 0
    assert "final_capital" in metrics
    assert "total_return" in metrics


def test_calculate_all_metrics():
    trades = [{"outcome": "WIN", "pnl": 200.0}, {"outcome": "LOSS", "pnl": -100.0}, {"outcome": "WIN", "pnl": 150.0}]
    metrics = Backtester.calculate_all_metrics(trades, [10000, 10200, 10100, 10350], 10000)
    assert isinstance(metrics, dict)
    assert metrics["total_trades"] == 3
    assert "win_rate" in metrics and "sharpe_ratio" in metrics


def test_best_time_analysis():
    trades = [
        {"timestamp": "2024-01-01 09:15:00", "outcome": "WIN", "pnl": 120.0, "force_score": 0.7},
        {"timestamp": "2024-01-01 09:15:00", "outcome": "LOSS", "pnl": -50.0, "force_score": 0.8},
        {"timestamp": "2024-01-01 10:15:00", "outcome": "WIN", "pnl": 200.0, "force_score": 0.9},
        {"timestamp": "2024-01-01 10:15:00", "outcome": "WIN", "pnl": 180.0, "force_score": 0.6},
    ]
    insights = Backtester.best_time_analysis(trades)
    assert isinstance(insights, dict)
    assert "by_hour" in insights
    assert "by_score_range" in insights
