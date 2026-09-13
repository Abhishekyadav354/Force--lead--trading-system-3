from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from core.historical_lead_analyzer import HistoricalLeadAnalyzer


def _event_rows(speeds, *, market="US", symbol="ABC", segment="STOCKS", timeframe="5m"):
    base = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    rows = []
    for index, speed in enumerate(speeds):
        start = base + timedelta(hours=index)
        gap = 420.0
        elapsed = gap / speed
        rows.append({"event_id": f"{symbol}-{index}", "market": market, "symbol": symbol, "segment": segment, "timeframe": timeframe, "timestamp": start.isoformat(), "open": 100, "high": 101, "low": 99, "close": 100, "volume": speed, "remaining_gap": gap, "regime": "TRENDING"})
        rows.append({"event_id": f"{symbol}-{index}", "market": market, "symbol": symbol, "segment": segment, "timeframe": timeframe, "timestamp": (start + timedelta(minutes=elapsed)).isoformat(), "open": 100, "high": 101, "low": 99, "close": 100, "volume": speed, "remaining_gap": 0, "completed": True, "regime": "TRENDING"})
    return rows


def test_extract_events_calculates_progress_speed_and_completion_time():
    analyzer = HistoricalLeadAnalyzer(min_samples=2)
    extracted = analyzer.extract_events(analyzer.pipeline.normalize(_event_rows([4.0])))
    event = extracted["events"][0]
    assert event["initial_gap"] == 420.0
    assert event["final_gap"] == 0.0
    assert event["actual_progress"] == 420.0
    assert event["actual_elapsed_minutes"] == 105.0
    assert event["actual_speed"] == 4.0


def test_analysis_learns_eta_distributions_and_comparison_metrics_walk_forward():
    report = HistoricalLeadAnalyzer(min_samples=2).analyze(_event_rows([4.0, 6.0, 8.0, 10.0]), data_source="synthetic")
    group = report["groups"]["US/ABC/STOCKS/5m"]
    assert report["historical_only"] is True
    assert report["synthetic_data"] is True
    assert report["candidate_events"] == 4
    assert report["validated_events"] == 2
    assert group["samples"] == 2
    assert group["reliable"] is True
    assert group["normal"]["mae_minutes"] is not None
    assert group["fast"]["coverage"] is not None
    assert group["shock"]["p90_error_minutes"] is not None
    assert group["eta_distributions"]["normal"]
    assert "ACCELERATING" in group["speed_states"]
    json.dumps(report)


def test_market_symbol_segment_and_timeframe_groups_never_mix():
    rows = _event_rows([4.0, 6.0, 8.0], symbol="ABC", timeframe="1m")
    rows += _event_rows([20.0, 22.0, 24.0], symbol="XYZ", segment="OPTIONS", timeframe="15m")
    report = HistoricalLeadAnalyzer(min_samples=1).analyze(rows, data_source="synthetic")
    assert set(report["groups"]) == {"US/ABC/STOCKS/1m", "US/XYZ/OPTIONS/15m"}
    assert report["groups"]["US/ABC/STOCKS/1m"]["events"][0]["actual_speed"] < report["groups"]["US/XYZ/OPTIONS/15m"]["events"][0]["actual_speed"]


def test_small_history_is_safe_and_missing_event_metadata_is_reported():
    rows = _event_rows([4.0, 6.0])
    rows.append({"timestamp": "2026-01-02T00:00:00Z", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1, "remaining_gap": 20})
    report = HistoricalLeadAnalyzer(min_samples=3).analyze(rows, data_source="historical")
    assert report["unsupported_events"] >= 1
    assert report["groups"] == {}
    assert report["synthetic_data"] is False


def test_future_event_cannot_change_earlier_analysis():
    base = _event_rows([4.0, 6.0, 8.0, 10.0])
    future = _event_rows([999.0], symbol="ABC")[0:2]
    future[0]["event_id"] = "future"
    future[1]["event_id"] = "future"
    future[0]["timestamp"] = "2030-01-01T00:00:00Z"
    future[1]["timestamp"] = "2030-01-01T00:01:00Z"
    first = HistoricalLeadAnalyzer(min_samples=2).analyze(base, data_source="synthetic")
    second = HistoricalLeadAnalyzer(min_samples=2).analyze(base + future, data_source="synthetic")
    first_events = first["groups"]["US/ABC/STOCKS/5m"]["events"]
    second_events = second["groups"]["US/ABC/STOCKS/5m"]["events"]
    assert first_events == second_events