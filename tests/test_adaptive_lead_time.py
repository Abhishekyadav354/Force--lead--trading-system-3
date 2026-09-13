from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from config.markets import MARKETS
from core.adaptive_lead_time import AdaptiveLeadTimeEngine, TIMEFRAME_MINUTES


def _event_rows(speeds, timeframe="5m", market="US", segment="STOCKS"):
    start = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
    rows = []
    for index, speed in enumerate(speeds):
        event_start = start + timedelta(hours=index)
        gap = 420.0
        elapsed = gap / speed
        rows.extend([
            {"event_id": f"event-{index}", "market": market, "segment": segment, "timeframe": timeframe, "timestamp": event_start.isoformat(), "remaining_gap": gap, "period": "OPENING" if index == 0 else "MIDDAY", "regime": "TRENDING"},
            {"event_id": f"event-{index}", "market": market, "segment": segment, "timeframe": timeframe, "timestamp": (event_start + timedelta(minutes=elapsed)).isoformat(), "remaining_gap": 0, "completed": True, "period": "OPENING" if index == 0 else "MIDDAY", "regime": "TRENDING"},
        ])
    return rows


def test_profiles_keep_timeframes_separate_and_learn_from_data():
    engine = AdaptiveLeadTimeEngine(min_samples=2)
    rows = _event_rows([7.0, 11.0, 15.0], timeframe="1m") + _event_rows([3.0, 5.0, 9.0], timeframe="15m")
    profiles = engine.build_profiles([
        {"market": "US", "segment": "STOCKS", "timeframe": "1m", "timestamp": "2026-01-01T09:30:00Z", "volume": 7},
        {"market": "US", "segment": "STOCKS", "timeframe": "15m", "timestamp": "2026-01-01T09:31:00Z", "volume": 3},
    ])
    assert set(profiles["US"]["STOCKS"]) == {"1m", "15m"}
    profile_rows = [
        {"market": "US", "segment": "STOCKS", "timeframe": "1m", "timestamp": f"2026-01-01T09:{30 + index:02d}:00Z", "speed": speed}
        for index, speed in enumerate([7.0, 11.0, 15.0])
    ]
    result = engine.estimate(420, 7, profile_rows, "US", "STOCKS", "1m", "2026-01-01T12:00:00Z")
    assert result["timeframe"] == "1m"
    assert result["historical_speed_percentiles"]["median"] == 11.0


def test_walk_forward_validation_reports_errors_coverage_and_event_measurements():
    engine = AdaptiveLeadTimeEngine(min_samples=2)
    report = engine.validate_historical_events(_event_rows([7.0, 11.0, 15.0, 19.0]), data_source="synthetic")
    group = report["reports"]["US/STOCKS/5m"]
    assert report["historical_only"] is True
    assert report["synthetic_data"] is True
    assert group["samples"] == 2
    assert group["reliable"] is True
    assert group["overall"]["mae_minutes"] is not None
    assert group["overall"]["coverage"] is not None
    assert group["events"][0]["initial_gap"] == 420.0
    assert group["events"][0]["actual_progress"] == 420.0
    assert "by_speed_state" in group and "by_period" in group and "by_regime" in group
    json.dumps(report)


def test_future_rows_and_outlier_do_not_change_earlier_walk_forward_prediction():
    engine = AdaptiveLeadTimeEngine(min_samples=2)
    base = _event_rows([7.0, 11.0, 15.0, 19.0])
    future = dict(base[-1])
    future["timestamp"] = "2030-01-01T00:00:00Z"
    future["remaining_gap"] = 0
    future["volume"] = 999999999.0
    first = engine.validate_historical_events(base, data_source="synthetic")
    second = engine.validate_historical_events(base + [future], data_source="synthetic")
    assert first["reports"]["US/STOCKS/5m"]["samples"] == second["reports"]["US/STOCKS/5m"]["samples"]
    assert first["reports"]["US/STOCKS/5m"]["events"] == second["reports"]["US/STOCKS/5m"]["events"]


def test_insufficient_history_is_explicit_and_low_confidence():
    report = AdaptiveLeadTimeEngine(min_samples=3).validate_historical_events(_event_rows([7.0, 11.0]), data_source="synthetic")
    group = report["reports"]["US/STOCKS/5m"]
    assert group["reliable"] is False
    assert group["confidence"] == 0.0


def test_configured_segments_are_discoverable_without_live_api():
    configured = AdaptiveLeadTimeEngine.configured_timeframes()
    assert set(configured) == set(MARKETS)
    assert any("OPTIONS" in segment for market in configured.values() for segment in market)
    assert any("DELIVERY" in segment for market in configured.values() for segment in market)
    assert any("FUTURES" in segment for market in configured.values() for segment in market)


def test_all_supported_timeframes_keep_independent_profiles():
    rows = []
    for index, timeframe in enumerate(TIMEFRAME_MINUTES):
        rows.extend([
            {"market": "US", "segment": "STOCKS", "timeframe": timeframe, "timestamp": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index)).isoformat(), "speed": float(index + 3)},
            {"market": "US", "segment": "STOCKS", "timeframe": timeframe, "timestamp": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=index + 1)).isoformat(), "speed": float(index + 3)},
        ])
    profiles = AdaptiveLeadTimeEngine(min_samples=1).build_profiles(rows)
    assert set(profiles["US"]["STOCKS"]) == set(TIMEFRAME_MINUTES)
    assert len({profile["median_speed"] for profile in profiles["US"]["STOCKS"].values()}) == len(TIMEFRAME_MINUTES)


def test_each_configured_segment_can_be_validated_with_sufficient_synthetic_events():
    configured = AdaptiveLeadTimeEngine.configured_timeframes()
    engine = AdaptiveLeadTimeEngine(min_samples=1)
    for market, segments in configured.items():
        for segment, timeframes in segments.items():
            timeframe = timeframes[0]
            report = engine.validate_historical_events(_event_rows([7.0, 11.0], timeframe, market, segment), data_source="synthetic")
            key = f"{market}/{segment}/{timeframe}"
            assert report["reports"][key]["samples"] == 1
            assert report["reports"][key]["reliable"] is True


def test_outlier_is_reported_without_changing_normal_speed_median():
    speeds = [7.0, 8.0, 9.0, 10.0, 10000.0]
    report = AdaptiveLeadTimeEngine(min_samples=2).validate_historical_events(_event_rows(speeds), data_source="synthetic")
    stats = report["reports"]["US/STOCKS/5m"]["speed_statistics"]
    assert stats["outlier_count"] == 1
    baseline = AdaptiveLeadTimeEngine(min_samples=2).validate_historical_events(_event_rows([7.0, 8.0, 9.0, 10.0]), data_source="synthetic")
    assert stats["median"] == baseline["reports"]["US/STOCKS/5m"]["speed_statistics"]["median"]
    assert stats["maximum_reliable"] == 10.0


def test_profiles_preserve_history_periods_and_available_conditions():
    latest = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    for index, (timestamp, speed, volume, regime) in enumerate([
        (latest - timedelta(days=30), 10.0, 100.0, "TREND"),
        (latest - timedelta(days=180), 20.0, 300.0, "SIDEWAYS"),
        (latest - timedelta(days=800), 40.0, 20.0, "HIGH_VOLATILITY"),
    ]):
        rows.append({
            "market": "US", "segment": "STOCKS", "timeframe": "5m",
            "timestamp": timestamp.isoformat(), "speed": speed, "volume": volume,
            "regime": regime, "period": "OPENING",
        })
    engine = AdaptiveLeadTimeEngine(min_samples=1, lookback_days=36500)
    profiles = engine.build_profiles(rows, current_timestamp=latest)
    profile = profiles["US"]["STOCKS"]["5m"]
    assert profile["period_profiles"]["recent"]["sample_count"] == 1
    assert profile["period_profiles"]["medium_term"]["sample_count"] == 1
    assert profile["period_profiles"]["long_term"]["sample_count"] == 1
    assert profile["period_profiles"]["full_history"]["sample_count"] == 3
    assert "time_of_day:OPENING" in profile["condition_profiles"]
    assert "regime:HIGH_VOLATILITY" in profile["condition_profiles"]
    assert "volume:HIGH" in profile["condition_profiles"]
    json.dumps(profiles)