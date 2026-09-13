from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from data.historical_data_manager import HistoricalDataError, HistoricalDatasetManager


def _rows(timeframe="5m", symbol="ABC", segment="STOCKS"):
    return [{"timestamp": f"2026-01-01T00:{index * 5:02d}:00Z", "open": 100 + index, "high": 101 + index, "low": 99 + index, "close": 100.5 + index, "volume": 10 + index, "market": "US", "symbol": symbol, "segment": segment, "timeframe": timeframe, "data_source": "historical"} for index in range(4)]


def test_register_and_registry_preserve_metadata_and_quality():
    manager = HistoricalDatasetManager()
    metadata = manager.register("us-abc-5m", _rows() + [_rows()[1]])
    assert metadata["market"] == "US"
    assert metadata["symbol"] == "ABC"
    assert metadata["timeframe"] == "5m"
    assert metadata["duplicate_count"] == 1
    assert metadata["rows"] == 4
    assert metadata["time_gaps"]["gap_count"] == 0
    assert manager.registry()["us-abc-5m"]["date_range"]["start"].startswith("2026-01-01")


def test_merge_requires_compatible_identity_and_removes_duplicates():
    manager = HistoricalDatasetManager()
    manager.register("dataset", _rows())
    merged = manager.merge("dataset", _rows()[2:])
    assert merged["rows"] == 4
    with pytest.raises(HistoricalDataError, match="Incompatible timeframe"):
        manager.merge("dataset", _rows(timeframe="15m"))


def test_retrieval_filters_and_gap_reporting_do_not_fill_data():
    rows = _rows()
    rows.pop(2)
    manager = HistoricalDatasetManager()
    metadata = manager.register("gapped", rows)
    assert metadata["rows"] == 3
    assert metadata["time_gaps"]["gap_count"] == 1
    assert len(manager.get("gapped", timeframe="5m", start="2026-01-01T00:05:00Z", end="2026-01-01T00:15:00Z")) == 2
    assert len(manager.records("gapped", symbol="ABC")) == 3


def test_timeframe_isolation_and_invalid_mixed_dataset():
    manager = HistoricalDatasetManager()
    manager.register("five", _rows("5m"))
    manager.register("fifteen", _rows("15m"))
    assert manager.get("five")["timeframe"].unique().tolist() == ["5m"]
    assert manager.get("fifteen")["timeframe"].unique().tolist() == ["15m"]
    with pytest.raises(HistoricalDataError, match="mixes incompatible timeframe"):
        manager.register("mixed", _rows("5m") + _rows("15m"))


def test_chronological_split_cannot_leak_future_rows_and_is_json_safe():
    manager = HistoricalDatasetManager()
    manager.register("split", _rows())
    split = manager.split("split", train=0.5, validation=0.25, test=0.25)
    assert split["train"]["timestamp"].max() < split["validation"]["timestamp"].min()
    assert split["validation"]["timestamp"].max() < split["test"]["timestamp"].min()
    payload = {"registry": manager.registry(), "metadata": manager.quality("split"), "records": manager.records("split")}
    json.dumps(payload)


def test_csv_and_dataframe_inputs_are_supported_without_fabrication(tmp_path):
    path = tmp_path / "history.csv"
    pd.DataFrame(_rows()).to_csv(path, index=False)
    manager = HistoricalDatasetManager()
    manager.register("csv", path)
    manager.register("dataframe", pd.DataFrame(_rows(symbol="XYZ")))
    assert manager.get_metadata("csv")["source"].endswith("history.csv")
    assert manager.get_metadata("dataframe")["symbol"] == "XYZ"


def test_full_history_analysis_preserves_short_medium_long_periods_and_json_safety():
    rows = []
    for index in range(800):
        rows.append({"timestamp": f"2020-01-01T00:00:00Z", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10, "volume_per_minute": 10 + index % 9, "market": "US", "symbol": "ABC", "segment": "STOCKS", "timeframe": "1d"})
    timestamps = pd.date_range("2020-01-01", periods=800, freq="D", tz="UTC")
    for index, timestamp in enumerate(timestamps):
        rows[index]["timestamp"] = timestamp.isoformat()
    rows.pop(500)
    manager = HistoricalDatasetManager()
    metadata = manager.register("multi-year", rows, data_source="synthetic")
    report = manager.analyze("multi-year", min_samples=20)
    assert metadata["rows"] == 799
    assert metadata["time_gaps"]["gap_count"] == 1
    assert set(report["periods"]) == {"recent", "medium_term", "long_term", "full_history"}
    assert report["periods"]["full_history"]["samples"] == 799
    assert report["adaptive_profiles"]
    assert report["synthetic_data"] is True
    json.dumps(report)


def test_walk_forward_and_filters_never_expose_future_rows():
    manager = HistoricalDatasetManager()
    manager.register("history", _rows())
    windows = manager.walk_forward("history", min_train_rows=2)
    assert windows
    for window in windows:
        assert window["train"][-1]["timestamp"] < window["evaluation"]["timestamp"]
        assert all(row["timestamp"] < window["evaluation_timestamp"] for row in window["train"])
    assert len(manager.get("history", start="2026-01-01T00:10:00Z")) == 2


def test_time_of_day_regime_and_rare_extremes_are_reported_separately():
    rows = _rows()
    for index, row in enumerate(rows):
        row["timestamp"] = f"2026-01-01T{9 + index:02d}:00:00Z"
        row["volume_per_minute"] = [10, 11, 12, 1000][index]
    manager = HistoricalDatasetManager()
    manager.register("regimes", rows, data_source="historical")
    report = manager.analyze("regimes", min_samples=2)
    assert report["time_of_day"]
    assert report["rare_extreme_events"]
    assert report["data_source"] == "historical"


class _FakeAngelClient:
    def __init__(self, payloads=None, fail_first=False):
        self.payloads = payloads or []
        self.fail_first = fail_first
        self.requests = []
        self.calls = 0

    def getCandleData(self, request):
        self.calls += 1
        self.requests.append(dict(request))
        if self.fail_first and self.calls == 1:
            raise RuntimeError("temporary API failure")
        return {"status": True, "data": self.payloads[min(self.calls - 1, len(self.payloads) - 1)] if self.payloads else []}


class _FakeAngelFeed:
    def __init__(self, client, symbol="ABC"):
        self.client = client
        self.symbol = symbol

    def _ensure_session(self):
        return self.client

    def _resolve_instrument(self):
        return {"exchange": "NSE", "tradingsymbol": self.symbol, "symboltoken": "123", "instrumenttype": "EQ", "series": "EQ"}


def _angel_candle(timestamp, close=100, volume=10):
    return [timestamp, close - 1, close + 1, close - 2, close, volume]


def test_angel_history_chunks_one_month_and_preserves_normalized_schema():
    client = _FakeAngelClient([
        [_angel_candle("2026-01-20 09:15:00", 100)],
        [_angel_candle("2026-02-15 09:15:00", 101)],
    ])
    manager = HistoricalDatasetManager()
    metadata = manager.fetch_angel_history(
        "india-abc-1m", symbol="ABC", market="INDIA", segment="INTRADAY", timeframe="1m",
        history="1M", end="2026-02-15T23:59:00Z", live_feed=_FakeAngelFeed(client), rate_limit_seconds=0,
    )

    assert len(client.requests) == 2
    assert client.requests[0]["interval"] == "ONE_MINUTE"
    assert metadata["requested_range"]["start"].startswith("2026-01-15")
    assert metadata["actual_range"]["start"].startswith("2026-01-20")
    assert metadata["candle_count"] == 2
    assert manager.get("india-abc-1m")["timestamp"].is_monotonic_increasing
    assert all(column in manager.get("india-abc-1m") for column in ("open", "high", "low", "close", "volume", "symbol", "segment", "timeframe"))


def test_angel_history_resume_skips_completed_ranges_and_deduplicates():
    client = _FakeAngelClient([[_angel_candle("2026-01-02 09:15:00")]])
    manager = HistoricalDatasetManager()
    first = manager.fetch_angel_history(
        "resume", symbol="ABC", market="INDIA", segment="INTRADAY", timeframe="1d",
        start="2026-01-01", end="2026-01-03", live_feed=_FakeAngelFeed(client), rate_limit_seconds=0,
    )
    second = manager.fetch_angel_history(
        "resume", symbol="ABC", market="INDIA", segment="INTRADAY", timeframe="1d",
        start="2026-01-01", end="2026-01-03", live_feed=_FakeAngelFeed(client), rate_limit_seconds=0,
    )

    assert first["candle_count"] == 1
    assert second["skipped_chunks"] == 1
    assert len(client.requests) == 1
    assert len(manager.get("resume")) == 1
    assert second["skipped_ranges"]


def test_angel_history_isolates_dataset_identity_and_reports_missing_sessions():
    client = _FakeAngelClient([[_angel_candle("2026-01-05 09:15:00")]])
    manager = HistoricalDatasetManager()
    metadata = manager.fetch_angel_history(
        "abc-5m", symbol="ABC", market="INDIA", segment="INTRADAY", timeframe="5m",
        start="2026-01-05", end="2026-01-07", live_feed=_FakeAngelFeed(client), rate_limit_seconds=0,
    )
    other = manager.register("xyz-15m", _rows(timeframe="15m", symbol="XYZ", segment="OPTIONS"))

    assert metadata["market"] == "INDIA"
    assert metadata["missing_sessions"]
    assert manager.get("abc-5m")["symbol"].unique().tolist() == ["ABC"]
    assert other["symbol"] == "XYZ"
    json.dumps(metadata)


def test_angel_history_retries_and_returns_failed_ranges_without_secrets():
    client = _FakeAngelClient([
        [_angel_candle("2026-01-02 09:15:00")],
        [_angel_candle("2026-02-02 09:15:00")],
    ], fail_first=True)
    manager = HistoricalDatasetManager()
    metadata = manager.fetch_angel_history(
        "failure", symbol="ABC", market="INDIA", segment="INTRADAY", timeframe="1m",
        start="2026-01-01", end="2026-02-15", live_feed=_FakeAngelFeed(client), max_retries=0, rate_limit_seconds=0,
    )

    assert metadata["failed_ranges"]
    assert metadata["resumable"] is True
    assert metadata["retry_count"] == 0
    serialized = json.dumps(metadata)
    assert "password" not in serialized.lower()
    assert '"123"' not in serialized


class _UniversalProvider:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    def get_candles(self, request):
        self.requests.append(dict(request))
        return {"status": True, "data": self.payloads[min(len(self.requests) - 1, len(self.payloads) - 1)]}


def test_universal_provider_isolates_markets_instruments_timeframes_and_timezones():
    provider = _UniversalProvider([
        [_angel_candle("2026-01-02 09:15:00")],
        [_angel_candle("2026-01-03 09:15:00", 101)],
    ])
    manager = HistoricalDatasetManager()
    india = manager.fetch_historical(
        "india-etf", provider="mock_provider", client=provider, symbol="ETF1", market="INDIA",
        segment="DELIVERY", instrument="ETF1-EQ", timeframe="1d", timezone_name="Asia/Kolkata",
        start="2026-01-01", end="2026-01-03", interval_limits={"1d": ("DAILY", 2)}, rate_limit_seconds=0,
    )
    europe = manager.fetch_historical(
        "europe-stock", provider="mock_provider", client=provider, symbol="ETF1", market="EUROPE",
        segment="STOCKS", instrument="ETF1-EU", timeframe="1d", timezone_name="Europe/London",
        start="2026-01-01", end="2026-01-03", interval_limits={"1d": ("DAILY", 2)}, rate_limit_seconds=0,
    )

    assert india["market"] == "INDIA"
    assert india["instrument"] == "ETF1-EQ"
    assert india["timezone"] == "Asia/Kolkata"
    assert europe["market"] == "EUROPE"
    assert europe["instrument"] == "ETF1-EU"
    assert len(manager.get("india-etf")) == 2
    assert len(manager.get("europe-stock")) == 1
    assert provider.requests[0]["market"] == "INDIA"
    assert provider.requests[2]["market"] == "EUROPE"


def test_all_available_uses_provider_start_without_arbitrary_maximum_and_preserves_oi():
    provider = _UniversalProvider([[
        {"timestamp": "2020-01-02T00:00:00Z", "open": 1, "high": 2, "low": 1, "close": 2, "volume": 10, "open_interest": 50}
    ]])
    manager = HistoricalDatasetManager()
    metadata = manager.fetch_historical(
        "long-history", provider="mock_provider", client=provider, symbol="BTC", market="US",
        segment="CRYPTO", instrument="BTC-USD", timeframe="1d", history="ALL_AVAILABLE",
        available_start="2020-01-01", end="2020-01-03", interval_limits={"1d": ("DAY", 2000)}, rate_limit_seconds=0,
    )

    assert metadata["requested_range"]["start"].startswith("2020-01-01")
    assert metadata["candle_count"] == 1
    assert manager.get("long-history").iloc[0]["open_interest"] == 50
    assert metadata["provider"] == "mock_provider"
    json.dumps(metadata)


def test_all_available_requires_explicit_provider_availability_when_unstored():
    provider = _UniversalProvider([])
    manager = HistoricalDatasetManager()
    with pytest.raises(HistoricalDataError, match="ALL_AVAILABLE"):
        manager.fetch_historical(
            "unknown-history", provider="mock_provider", client=provider, symbol="ABC", market="US",
            segment="STOCKS", timeframe="1d", end="2026-01-03", interval_limits={"1d": ("DAY", 2000)}, rate_limit_seconds=0,
        )