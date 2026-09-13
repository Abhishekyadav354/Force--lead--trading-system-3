from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from statistics.backtester import Backtester


class RecordingPredictionEngine:
    def __init__(self):
        self.calls = []

    def predict(self, live_data, historical_data, **kwargs):
        self.calls.append({"live": live_data, "history": list(historical_data), "kwargs": kwargs})
        return {
            "direction": "UP" if historical_data[-1]["close"] >= historical_data[-2]["close"] else "DOWN",
            "confidence": 0.75,
            "adaptive_lead_time": {
                "normal_eta_minutes": 10.0,
                "speed_regime": "NORMAL",
            },
            "move": {"type": "UNAVAILABLE"},
            "reversal": {"type": "UNAVAILABLE"},
        }


def _rows(symbol="ABC", segment="STOCKS", timeframe="5m"):
    start = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    closes = [100.0, 101.0, 102.0, 101.0, 103.0, 104.0]
    return [
        {
            "timestamp": (start + timedelta(minutes=index * 5)).isoformat(),
            "market": "US",
            "symbol": symbol,
            "segment": segment,
            "timeframe": timeframe,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100 + index,
            "regime": "TRENDING",
            "period": "OPENING",
        }
        for index, close in enumerate(closes)
    ]


def test_walk_forward_is_prior_only_and_isolates_identity():
    engine = RecordingPredictionEngine()
    rows = _rows() + _rows(symbol="XYZ", segment="OPTIONS", timeframe="15m")
    result = Backtester.walk_forward_validate(rows, engine, min_train_samples=2, data_source="synthetic")

    assert result["synthetic_data"] is True
    assert result["paper_simulation_only"] is True
    assert result["samples"] == 8
    assert len(result["reports"]) == 2
    assert result["leakage_checks"]["future_rows_used_for_prediction"] == 0
    for call in engine.calls:
        assert call["history"]
        assert call["history"][-1]["timestamp"] < call["live"]["timestamp"] or call["live"] == call["history"][-1]
        assert all(call["history"][index]["timestamp"] <= call["history"][index + 1]["timestamp"] for index in range(len(call["history"]) - 1))
    json.dumps(result)


def test_walk_forward_reports_direction_and_condition_breakdowns():
    result = Backtester.validate_walk_forward(_rows(), RecordingPredictionEngine(), min_train_samples=2, data_source="synthetic")
    report = result["reports"]["US/ABC/STOCKS/5m"]

    assert report["overall"]["direction"]["samples"] == 4
    assert report["overall"]["direction"]["accuracy"] is not None
    assert "OPENING" in report["by_period"]
    assert "TRENDING" in report["by_regime"]
    assert "NORMAL" in report["by_speed_state"]
    assert report["overall"]["lead_time"]["samples"] == 0
    assert report["overall"]["move_size"]["samples"] == 0


def test_rolling_window_and_simulation_parameters_are_reported_without_orders():
    result = Backtester.run_walk_forward(
        _rows(), RecordingPredictionEngine(), min_train_samples=2,
        rolling_window=2, transaction_cost_bps=8, slippage_bps=3,
    )

    assert result["rolling_window"] == 2
    assert result["expanding_window"] is False
    assert result["transaction_cost_bps"] == 8.0
    assert result["slippage_bps"] == 3.0
    assert result["paper_simulation_only"] is True