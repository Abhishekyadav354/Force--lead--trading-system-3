from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from core.adaptive_intelligence import AdaptiveIntelligence, AdaptiveIntelligenceEngine
from data.historical_data_manager import HistoricalDataManager


def _row(timestamp, *, market="US", segment="STOCKS", symbol="ABC", instrument="ABC-EQ", timeframe="5m", volume=50, **extra):
    return {
        "timestamp": timestamp.isoformat(), "market": market, "segment": segment,
        "symbol": symbol, "instrument": instrument, "timezone": "UTC",
        "timeframe": timeframe, "open": 100.0, "high": 101.0,
        "low": 99.0, "close": 100.0 + volume / 1000.0, "volume": volume,
        **extra,
    }


def test_analyze_isolates_markets_symbols_segments_instruments_and_timeframes():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [
        _row(base + timedelta(days=index), market="US", symbol="ABC", instrument="ABC-EQ", timeframe="5m", volume=10 + index)
        for index in range(6)
    ]
    rows += [
        _row(base + timedelta(days=index), market="EUROPE", segment="STOCKS", symbol="ABC", instrument="ABC-EU", timeframe="15m", volume=100 + index)
        for index in range(6)
    ]
    rows += [
        _row(base + timedelta(days=index), market="INDIA", segment="DELIVERY", symbol="ABC", instrument="ABC-IN", timeframe="1d", volume=200 + index)
        for index in range(6)
    ]
    manager = HistoricalDataManager()
    manager.register("us", rows[:6], data_source="synthetic")
    manager.register("europe", rows[6:12], data_source="synthetic")
    manager.register("india", rows[12:], data_source="synthetic")
    intelligence = AdaptiveIntelligence(manager, min_samples=2)
    report = {"reports": {}}
    for dataset_id in ("us", "europe", "india"):
        report["reports"].update(intelligence.analyze(dataset_id=dataset_id, data_source="synthetic")["reports"])
    report["groups"] = len(report["reports"])
    report["synthetic_data"] = True

    assert report["groups"] == 3
    assert "US/STOCKS/ABC/ABC-EQ/5m/UTC" in report["reports"]
    assert "EUROPE/STOCKS/ABC/ABC-EU/15m/UTC" in report["reports"]
    assert "INDIA/DELIVERY/ABC/ABC-IN/1d/UTC" in report["reports"]
    assert report["synthetic_data"] is True
    json.dumps(report)
    assert AdaptiveIntelligenceEngine is AdaptiveIntelligence


def test_recent_medium_and_long_history_are_exposed_without_future_leakage():
    latest = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        _row(latest - timedelta(days=30), volume=10),
        _row(latest - timedelta(days=180), volume=20),
        _row(latest - timedelta(days=800), volume=40),
        _row(latest, volume=60),
        _row(latest + timedelta(days=30), volume=999999),
    ]
    manager = HistoricalDataManager()
    manager.register("periods", rows, data_source="synthetic")
    report = AdaptiveIntelligence(manager, min_samples=1).analyze(
        dataset_id="periods", current_timestamp=latest, data_source="synthetic"
    )
    group = report["reports"]["US/STOCKS/ABC/ABC-EQ/5m/UTC"]
    periods = group["profile"]["period_profiles"]

    assert group["historical_rows"] == 4
    assert group["future_rows_excluded"] == 1
    assert periods["recent"]["sample_count"] >= 1
    assert periods["medium_term"]["sample_count"] >= 1
    assert periods["long_term"]["sample_count"] >= 1
    assert 999999 not in [periods["full_history"].get("median_speed")]
    json.dumps(report)


def test_rare_extreme_speed_is_reported_but_does_not_define_normal_eta():
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = [_row(base + timedelta(days=index), volume=10 + index) for index in range(6)]
    rows.append(_row(base + timedelta(days=6), volume=1000000))
    rows.append(_row(base + timedelta(days=7), volume=20))
    manager = HistoricalDataManager()
    manager.register("extreme", rows, data_source="synthetic")
    group = AdaptiveIntelligence(manager, min_samples=2).analyze(
        dataset_id="extreme", data_source="synthetic"
    )["reports"]["US/STOCKS/ABC/ABC-EQ/5m/UTC"]

    profile = group["profile"]
    assert profile["outlier_count"] >= 1
    assert profile["maximum_reliable_speed"] < 1000000
    assert group["adaptive_estimate"]["confidence"] <= 1.0


def test_completion_validation_exposes_error_metrics_and_speed_states():
    base = datetime(2025, 1, 1, 9, 30, tzinfo=timezone.utc)
    rows = []
    for index, speed in enumerate((10, 12, 14, 16, 18, 100)):
        start = base + timedelta(days=index)
        gap = 100.0
        elapsed = gap / speed
        rows.extend([
            _row(start, volume=speed, event_id=f"event-{index}", remaining_gap=gap, regime="TREND"),
            _row(start + timedelta(minutes=elapsed), volume=speed, event_id=f"event-{index}", remaining_gap=0, completed=True, regime="TREND"),
        ])
    manager = HistoricalDataManager()
    manager.register("events", rows, data_source="synthetic")
    group = AdaptiveIntelligence(manager, min_samples=2).analyze(
        dataset_id="events", data_source="synthetic"
    )["reports"]["US/STOCKS/ABC/ABC-EQ/5m/UTC"]
    adaptive = group["validation"]["adaptive_lead"]

    report = next(iter(adaptive["reports"].values()))
    assert report["overall"]["mae_minutes"] is not None
    assert report["overall"]["median_error_minutes"] is not None
    assert report["overall"]["p90_error_minutes"] is not None
    assert report["overall"]["coverage"] is not None
    assert "NORMAL" in report["by_speed_state"]
    assert "SHOCK/EXPANSION" in report["by_speed_state"]


def test_prediction_wrapper_excludes_current_and_future_rows_and_handles_missing_fields():
    class RecordingEngine:
        def __init__(self):
            self.history_lengths = []

        def predict(self, live_data, historical_data, **kwargs):
            self.history_lengths.append(len(historical_data))
            return {"direction": "UP", "confidence": 0.4, "adaptive_lead_time": {}}

    current = _row(datetime(2026, 1, 3, tzinfo=timezone.utc), volume=20)
    rows = [
        _row(datetime(2026, 1, 1, tzinfo=timezone.utc), volume=10),
        _row(datetime(2026, 1, 2, tzinfo=timezone.utc), volume=11),
        _row(datetime(2026, 1, 4, tzinfo=timezone.utc), volume=999),
    ]
    recording = RecordingEngine()
    intelligence = AdaptiveIntelligence(prediction_engine=recording, min_samples=2)
    prediction = intelligence.predict(current, rows)

    assert recording.history_lengths == [2]
    assert prediction["historical_intelligence"]["future_rows_excluded"] == 1
    assert prediction["historical_intelligence"]["paper_simulation_only"] is True

    missing = dict(current)
    missing.pop("volume")
    report = intelligence.analyze([missing], data_source="synthetic")
    assert report["reports"]
    json.dumps(report)
