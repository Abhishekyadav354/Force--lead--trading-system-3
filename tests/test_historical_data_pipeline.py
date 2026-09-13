from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from data.historical_data_pipeline import HistoricalDataError, HistoricalDataPipeline


def _rows():
    return [
        {"datetime": "2026-01-01T09:35:00+05:30", "open_price": 101, "high_price": 103, "low_price": 100, "close_price": 102, "vol": 20, "market": "INDIA", "segment": "INTRADAY", "symbol": "NIFTY", "interval": "5minute"},
        {"datetime": "2026-01-01T09:30:00+05:30", "open_price": 100, "high_price": 102, "low_price": 99, "close_price": 101, "vol": 10, "market": "INDIA", "segment": "INTRADAY", "symbol": "NIFTY", "interval": "5minute"},
    ]


def test_normalizes_project_aliases_timezone_metadata_and_deduplicates():
    pipeline = HistoricalDataPipeline()
    frame = pipeline.normalize(_rows() + [_rows()[0]])
    assert list(frame["timeframe"]) == ["5m", "5m"]
    assert frame.iloc[0]["timestamp"].tzinfo is not None
    assert frame.iloc[0]["timestamp_original"].startswith("2026-01-01T09:30")
    assert frame.iloc[0]["market"] == "INDIA"
    assert len(frame) == 2


def test_validation_separates_required_and_optional_columns():
    frame = HistoricalDataPipeline().normalize(_rows())
    report = HistoricalDataPipeline().validate(frame)
    assert report["valid"] is True
    assert report["missing_required_columns"] == []
    assert "open_interest" in report["missing_optional_columns"]


def test_missing_required_columns_and_invalid_values_fail_clearly():
    with pytest.raises(HistoricalDataError, match="Missing required"):
        HistoricalDataPipeline().normalize([{"timestamp": "2026-01-01T00:00:00Z", "close": 1}])
    bad = _rows()
    bad[0]["high_price"] = -1
    with pytest.raises(HistoricalDataError, match="cannot be negative"):
        HistoricalDataPipeline().normalize(bad)


def test_timeframes_do_not_mix_and_valid_resampling_aggregates_ohlcv():
    pipeline = HistoricalDataPipeline()
    frame = pipeline.normalize([
        {"timestamp": f"2026-01-01T00:{index * 5:02d}:00Z", "open": 100 + index, "high": 101 + index, "low": 99 + index, "close": 100.5 + index, "volume": 10, "timeframe": "5m"}
        for index in range(4)
    ])
    resampled = pipeline.resample(frame, "15m")
    assert len(resampled) == 2
    assert resampled.iloc[0]["open"] == 100
    assert resampled.iloc[0]["close"] == 102.5
    assert resampled.iloc[0]["volume"] == 30
    with pytest.raises(HistoricalDataError):
        pipeline.resample(frame, "1h", source_timeframe="7m")


def test_lookback_statistics_records_and_chronological_splits_prevent_leakage():
    pipeline = HistoricalDataPipeline()
    frame = pipeline.normalize([{**row, "timeframe": "1d"} for row in [
        {"timestamp": f"2026-01-0{index}T00:00:00Z", "open": index, "high": index + 1, "low": index - 1, "close": index, "volume": index * 10}
        for index in range(1, 6)
    ]])
    split = pipeline.split(frame, train=0.6, validation=0.2, test=0.2)
    assert split["train"]["timestamp"].max() < split["validation"]["timestamp"].min()
    assert split["validation"]["timestamp"].max() < split["test"]["timestamp"].min()
    assert split["train"]["timestamp"].max() < split["test"]["timestamp"].min()
    stats = pipeline.statistics(frame)
    assert stats["rows"] == 5
    assert stats["date_range"]["start"].startswith("2026-01-01")
    assert json.dumps({"stats": stats, "records": pipeline.records(frame), "metadata": split["metadata"]})
    assert len(pipeline.filter_lookback(frame, "1m", "2026-02-01T00:00:00Z")) == 5


def test_dataframe_and_project_mapping_inputs_are_supported_without_fabrication():
    pipeline = HistoricalDataPipeline()
    frame = pipeline.normalize(pd.DataFrame(_rows()))
    mapping_frame = pipeline.normalize({"candles": _rows()})
    assert len(frame) == len(mapping_frame) == 2
    assert all(value != "synthetic" for value in frame.get("data_source", pd.Series(dtype=str)))


def test_csv_path_input_is_supported(tmp_path):
    path = tmp_path / "candles.csv"
    pd.DataFrame(_rows()).to_csv(path, index=False)
    frame = HistoricalDataPipeline().normalize(path)
    assert len(frame) == 2
    assert frame["timeframe"].unique().tolist() == ["5m"]