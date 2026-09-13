from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from data.multi_year_data_analyzer import MultiYearDataAnalyzer


def _rows(days, timeframe="1d", symbol="ABC"):
    start = datetime(2018, 1, 1, tzinfo=timezone.utc)
    return [{"timestamp": (start + timedelta(days=index * max(1, days // max(days, 1)))).isoformat(), "open": 100 + index, "high": 101 + index, "low": 99 + index, "close": 100 + index * (1 if index % 3 else 0), "volume": 100 + index, "market": "US", "symbol": symbol, "segment": "STOCKS", "timeframe": timeframe, "data_source": "synthetic"} for index in range(days)]


def test_discovers_full_multi_year_range_and_quality_gaps():
    rows = _rows(1500)
    rows.pop(700)
    report = MultiYearDataAnalyzer(min_samples=20).analyze(rows, data_source="synthetic")
    group = report["groups"]["US/ABC/STOCKS/1d"]
    assert report["earliest_timestamp"].startswith("2018-01-01")
    assert report["latest_timestamp"] > report["earliest_timestamp"]
    assert group["quality"]["coverage_days"] > 1400
    assert group["quality"]["gap_count"] >= 1
    assert group["quality"]["rows"] == 1499
    assert report["synthetic_data"] is True
    json.dumps(report)


def test_recent_medium_and_long_periods_are_separate():
    report = MultiYearDataAnalyzer(min_samples=2).analyze(_rows(800), data_source="synthetic")
    periods = report["groups"]["US/ABC/STOCKS/1d"]["periods"]
    assert periods["recent"]["samples"] > 0
    assert periods["medium_term"]["samples"] > 0
    assert periods["long_term"]["samples"] > 0
    assert periods["recent"]["speed_statistics"] != periods["long_term"]["speed_statistics"]


def test_timeframes_and_symbols_are_not_mixed():
    rows = _rows(100, "1d", "ABC") + _rows(100, "1d", "XYZ") + _rows(100, "5m", "ABC")
    report = MultiYearDataAnalyzer(min_samples=2).analyze(rows, data_source="synthetic")
    assert set(report["groups"]) == {"US/ABC/STOCKS/1d", "US/XYZ/STOCKS/1d", "US/ABC/STOCKS/5m"}


def test_short_history_reports_low_confidence_and_no_fabricated_groups():
    report = MultiYearDataAnalyzer(min_samples=20).analyze(_rows(3), data_source="historical")
    group = report["groups"]["US/ABC/STOCKS/1d"]
    assert group["sample_count"] == 3
    assert 0.0 < group["confidence"] < 0.2
    assert report["synthetic_data"] is False
    empty = MultiYearDataAnalyzer().analyze([], data_source="historical")
    assert empty["groups"] == {}
    assert "fabricated" in empty["note"]


def test_available_report_contains_adaptive_profiles_and_leakage_safe_order():
    rows = _rows(400)
    report = MultiYearDataAnalyzer(min_samples=10).analyze(rows, data_source="synthetic")
    group = report["groups"]["US/ABC/STOCKS/1d"]
    timestamps = [event["timestamp"] for event in group["events"]]
    assert timestamps == sorted(timestamps)
    assert group["adaptive_profiles"]
    assert report["availability"][0]["available"] is True