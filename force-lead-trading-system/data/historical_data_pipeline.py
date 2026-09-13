"""Reusable, credential-free historical market-data preparation utilities."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd

from core.adaptive_lead_time import TIMEFRAME_MINUTES


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")
OPTIONAL_COLUMNS = (
    "symbol", "market", "segment", "timeframe", "buy_volume", "sell_volume",
    "trade_count", "avg_trade_size", "bid_total", "ask_total", "buyers_pct",
    "sellers_pct", "spread", "volume_per_minute", "open_interest", "oi",
    "delta", "order_flow", "data_source",
)
ALIASES = {
    "datetime": "timestamp", "date": "timestamp", "time": "timestamp",
    "open_price": "open", "high_price": "high", "low_price": "low",
    "close_price": "close", "last": "close", "ltp": "close",
    "vol": "volume", "qty": "volume", "interval": "timeframe", "tf": "timeframe",
    "ticker": "symbol", "instrument": "symbol", "market_segment": "segment",
}
LOOKBACK_DAYS = {"1m": 31, "1month": 31, "3m": 92, "3month": 92, "6m": 183, "6month": 183, "1y": 365, "1year": 365}


class HistoricalDataError(ValueError):
    """Raised when historical input cannot be safely normalized."""


class HistoricalDataPipeline:
    """Normalize historical candles without network access or data fabrication."""

    def __init__(self, timezone_name: str = "UTC") -> None:
        self.timezone_name = timezone_name

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, (datetime, pd.Timestamp)):
            return value.isoformat()
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, TypeError):
                pass
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _canonical_timeframe(value: Any) -> Optional[str]:
        if value is None or str(value).strip() == "":
            return None
        text = str(value).strip().lower().replace("minute", "m").replace("hour", "h").replace("day", "d")
        if text == "d":
            text = "1d"
        if text in TIMEFRAME_MINUTES:
            return text
        if text.endswith(("m", "h")):
            try:
                number = float(text[:-1])
                return f"{int(number) if number.is_integer() else number:g}{text[-1]}"
            except ValueError:
                return None
        return None

    def _timestamp_series(self, values: pd.Series) -> tuple[pd.Series, pd.Series]:
        original = values.map(lambda value: self._json_value(value))
        timestamps = pd.to_datetime(values, errors="coerce", utc=True, format="mixed")
        if timestamps.isna().any():
            raise HistoricalDataError(f"Invalid timestamp values: {int(timestamps.isna().sum())}")
        return timestamps, original

    def _as_frame(self, source: Any) -> pd.DataFrame:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() != ".csv":
                raise HistoricalDataError("Only CSV file inputs are supported")
            if not path.exists():
                raise HistoricalDataError(f"Historical data file does not exist: {path}")
            return pd.read_csv(path)
        if isinstance(source, pd.DataFrame):
            return source.copy()
        if isinstance(source, Mapping):
            if "candles" in source:
                source = source["candles"]
            elif "data" in source and isinstance(source["data"], list):
                source = source["data"]
        if isinstance(source, list):
            return pd.DataFrame([dict(item) for item in source if isinstance(item, Mapping)])
        raise HistoricalDataError("Input must be a CSV path, DataFrame, mapping, or candle record list")

    def normalize(self, source: Any, *, market: Optional[str] = None, segment: Optional[str] = None, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> pd.DataFrame:
        """Normalize project/CSV records and preserve source timestamp text."""
        frame = self._as_frame(source)
        frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
        frame = frame.rename(columns={column: ALIASES.get(column, column) for column in frame.columns})
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise HistoricalDataError(f"Missing required columns: {', '.join(missing)}")
        timestamps, original = self._timestamp_series(frame["timestamp"])
        frame["timestamp_original"] = original
        frame["timestamp"] = timestamps
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        invalid_ohlcv = frame[list(REQUIRED_COLUMNS)].isna().any(axis=1)
        if invalid_ohlcv.any():
            raise HistoricalDataError(f"Invalid required OHLCV values: {int(invalid_ohlcv.sum())}")
        if (frame[["high", "low", "volume"]] < 0).any().any():
            raise HistoricalDataError("High, low, and volume values cannot be negative")
        for key, value in (("market", market), ("segment", segment), ("symbol", symbol), ("timeframe", timeframe)):
            if value is not None:
                frame[key] = value
        if "timeframe" in frame:
            raw_timeframes = frame["timeframe"].copy()
            frame["timeframe"] = raw_timeframes.map(self._canonical_timeframe)
            invalid_tf = frame["timeframe"].isna() & raw_timeframes.notna() & raw_timeframes.astype(str).str.strip().ne("")
            if invalid_tf.any():
                raise HistoricalDataError(f"Unsupported or missing timeframe values: {int(invalid_tf.sum())}")
        for column in frame.columns:
            if column not in {"timestamp", "timestamp_original", "market", "segment", "symbol", "timeframe"}:
                converted = pd.to_numeric(frame[column], errors="coerce")
                if converted.notna().all() or frame[column].isna().all():
                    frame[column] = converted
                else:
                    continue
        return self.clean(frame)

    @staticmethod
    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        """Sort chronologically and remove exact candle duplicates by metadata key."""
        result = frame.copy()
        key = [column for column in ("market", "segment", "symbol", "timeframe", "timestamp") if column in result.columns]
        if not key:
            raise HistoricalDataError("Cannot clean data without timestamp")
        result = result.sort_values(key, kind="stable").drop_duplicates(key, keep="last").reset_index(drop=True)
        if not result["timestamp"].is_monotonic_increasing and len(key) == 1:
            raise HistoricalDataError("Historical timestamps are not chronological")
        return result

    def validate(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Return required/optional column and data-quality information."""
        columns = set(frame.columns)
        missing_required = [column for column in REQUIRED_COLUMNS if column not in columns]
        missing_optional = [column for column in OPTIONAL_COLUMNS if column not in columns]
        invalid_rows = int(frame[list(REQUIRED_COLUMNS)].isna().any(axis=1).sum()) if not missing_required else len(frame)
        return {
            "valid": not missing_required and invalid_rows == 0 and len(frame) > 0,
            "rows": int(len(frame)), "missing_required_columns": missing_required,
            "missing_optional_columns": missing_optional, "invalid_required_rows": invalid_rows,
            "timeframes": sorted({str(value) for value in frame.get("timeframe", pd.Series(dtype=str)).dropna()}),
        }

    def filter_lookback(self, frame: pd.DataFrame, period: str, as_of: Any = None) -> pd.DataFrame:
        key = str(period).strip().lower()
        if key not in LOOKBACK_DAYS:
            raise HistoricalDataError(f"Unsupported lookback period: {period}")
        end = pd.Timestamp(as_of, tz="UTC") if as_of is not None else frame["timestamp"].max()
        start = end - pd.Timedelta(days=LOOKBACK_DAYS[key])
        return frame[(frame["timestamp"] >= start) & (frame["timestamp"] <= end)].copy().reset_index(drop=True)

    @staticmethod
    def _minutes(timeframe: Any) -> Optional[float]:
        canonical = HistoricalDataPipeline._canonical_timeframe(timeframe)
        return TIMEFRAME_MINUTES.get(canonical) if canonical else None

    def resample(self, frame: pd.DataFrame, target_timeframe: str, source_timeframe: Optional[str] = None) -> pd.DataFrame:
        """Aggregate only when target duration is an integer multiple of source duration."""
        source = source_timeframe or frame["timeframe"].dropna().iloc[0] if "timeframe" in frame and not frame["timeframe"].dropna().empty else None
        source_minutes, target_minutes = self._minutes(source), self._minutes(target_timeframe)
        if source_minutes is None or target_minutes is None or target_minutes < source_minutes or target_minutes % source_minutes:
            raise HistoricalDataError("Resampling requires a known source timeframe and an integer multiple target")
        result = frame.copy().set_index("timestamp")
        aggregation = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        for column in ("buy_volume", "sell_volume", "trade_count"):
            if column in result:
                aggregation[column] = "sum"
        grouped = result.resample(f"{int(target_minutes)}min", label="left", closed="left").agg(aggregation).dropna(subset=["open", "high", "low", "close"])
        grouped["timeframe"] = self._canonical_timeframe(target_timeframe)
        for column in ("market", "segment", "symbol"):
            if column in frame:
                grouped[column] = frame[column].dropna().iloc[0] if frame[column].dropna().any() else None
        grouped["timestamp_original"] = grouped.index.map(lambda value: value.isoformat())
        return grouped.reset_index()

    def split(self, frame: pd.DataFrame, train: float = 0.6, validation: float = 0.2, test: float = 0.2) -> Dict[str, Any]:
        """Create chronological, non-overlapping train/validation/test datasets."""
        if min(train, validation, test) <= 0 or not math.isclose(train + validation + test, 1.0, rel_tol=1e-9):
            raise HistoricalDataError("Split ratios must be positive and sum to 1")
        ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
        first = int(len(ordered) * train)
        second = first + int(len(ordered) * validation)
        return {"train": ordered.iloc[:first].copy(), "validation": ordered.iloc[first:second].copy(), "test": ordered.iloc[second:].copy(), "metadata": {"chronological": True, "rows": {"train": first, "validation": second - first, "test": len(ordered) - second}, "boundaries": {"train_end": self._json_value(ordered.iloc[first - 1]["timestamp"]) if first else None, "validation_end": self._json_value(ordered.iloc[second - 1]["timestamp"]) if second else None}}}

    def statistics(self, frame: pd.DataFrame) -> Dict[str, Any]:
        missing = frame.isna().mean().mean() if len(frame) else 0.0
        return {"rows": int(len(frame)), "date_range": {"start": self._json_value(frame["timestamp"].min()) if len(frame) else None, "end": self._json_value(frame["timestamp"].max()) if len(frame) else None}, "timeframes": sorted({str(value) for value in frame.get("timeframe", pd.Series(dtype=str)).dropna()}), "markets": sorted({str(value) for value in frame.get("market", pd.Series(dtype=str)).dropna()}), "segments": sorted({str(value) for value in frame.get("segment", pd.Series(dtype=str)).dropna()}), "volume_rows": int(frame["volume"].notna().sum()) if "volume" in frame else 0, "missing_data_percentage": round(float(missing) * 100.0, 4)}

    @staticmethod
    def records(frame: pd.DataFrame) -> list[Dict[str, Any]]:
        return [{key: HistoricalDataPipeline._json_value(value) for key, value in row.items()} for row in frame.to_dict(orient="records")]
