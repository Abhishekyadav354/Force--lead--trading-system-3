"""Credential-free storage and retrieval for normalized historical datasets."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import json
import math
import time

import pandas as pd

from core.adaptive_lead_time import TIMEFRAME_MINUTES
from core.adaptive_lead_time import AdaptiveLeadTimeCalculator
from config.settings import settings
from config.markets import MARKETS
from data.historical_data_pipeline import ALIASES, HistoricalDataError, HistoricalDataPipeline


_ANGEL_INTERVALS = {
    "1m": ("ONE_MINUTE", 30),
    "3m": ("THREE_MINUTE", 60),
    "5m": ("FIVE_MINUTE", 100),
    "10m": ("TEN_MINUTE", 100),
    "15m": ("FIFTEEN_MINUTE", 200),
    "30m": ("THIRTY_MINUTE", 200),
    "1h": ("ONE_HOUR", 400),
    "1d": ("ONE_DAY", 2000),
}

_PROVIDER_INTERVALS = {"angel_one": _ANGEL_INTERVALS}

_HISTORY_DAYS = {
    "1M": 31,
    "3M": 92,
    "6M": 183,
    "1Y": 365,
    "3Y": 1095,
    "5Y": 1825,
}


class HistoricalDatasetManager:
    """Keep compatible historical datasets isolated and discoverable."""

    def __init__(self, pipeline: Optional[HistoricalDataPipeline] = None) -> None:
        self.pipeline = pipeline or HistoricalDataPipeline()
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._registry: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _value(frame: pd.DataFrame, column: str, fallback: str = "UNKNOWN") -> str:
        if column not in frame:
            return fallback
        values = [str(value).strip() for value in frame[column].dropna().unique() if str(value).strip()]
        return values[0] if values else fallback

    @staticmethod
    def _identities(frame: pd.DataFrame) -> Dict[str, list[str]]:
        result = {}
        for column in ("market", "symbol", "segment", "instrument", "timeframe", "timezone", "provider"):
            if column in frame:
                result[column] = sorted({str(value) for value in frame[column].dropna() if str(value).strip()})
        return result

    @staticmethod
    def _json(value: Any) -> Any:
        return HistoricalDataPipeline._json_value(value)

    @staticmethod
    def _source_without_instrument_alias(source: Any) -> Any:
        """Avoid the legacy pipeline's ``instrument`` -> ``symbol`` alias."""
        if isinstance(source, pd.DataFrame):
            result = source.copy()
            if "instrument" in result.columns:
                result = result.rename(columns={"instrument": "instrument_id"})
            return result
        if isinstance(source, list):
            return [{**{key: value for key, value in row.items() if key != "instrument"}, "instrument_id": row["instrument"]} if isinstance(row, dict) and "instrument" in row else row for row in source]
        if isinstance(source, dict):
            result = dict(source)
            for key in ("candles", "data"):
                if isinstance(result.get(key), list):
                    result[key] = HistoricalDatasetManager._source_without_instrument_alias(result[key])
            if "instrument" in result:
                result["instrument_id"] = result.pop("instrument")
            return result
        return source

    @staticmethod
    def _restore_instrument_column(frame: pd.DataFrame) -> pd.DataFrame:
        if "instrument_id" in frame.columns:
            frame = frame.rename(columns={"instrument_id": "instrument"})
        return frame

    def _raw_duplicate_count(self, source: Any, normalized: pd.DataFrame) -> int:
        try:
            raw = self.pipeline._as_frame(source)
            raw.columns = [str(column).strip().lower().replace(" ", "_") for column in raw.columns]
            raw = raw.rename(columns={column: ALIASES.get(column, column) for column in raw.columns})
            if "timestamp" not in raw:
                return 0
            timestamps = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True, format="mixed")
            keys = [column for column in ("market", "symbol", "segment", "timeframe") if column in raw]
            raw = raw.assign(timestamp=timestamps)
            return int(len(raw) - len(raw.drop_duplicates(keys + ["timestamp"], keep="last")))
        except Exception:
            return max(0, int(len(normalized) - len(normalized.drop_duplicates("timestamp"))))

    def _gap_report(self, frame: pd.DataFrame) -> Dict[str, Any]:
        minutes = self.pipeline._minutes(self._value(frame, "timeframe", ""))
        if minutes is None or len(frame) < 2:
            return {"expected_interval_minutes": minutes, "gap_count": 0, "gaps": []}
        ordered = frame.sort_values("timestamp")
        differences = ordered["timestamp"].diff().dt.total_seconds().div(60.0)
        gaps = []
        for index, difference in differences.items():
            if difference > minutes * 1.5:
                gaps.append({"after": self._json(ordered.loc[index - 1, "timestamp"]), "before": self._json(ordered.loc[index, "timestamp"]), "minutes": float(difference), "missing_intervals_estimate": max(0, int(round(difference / minutes)) - 1)})
        return {"expected_interval_minutes": minutes, "gap_count": len(gaps), "gaps": gaps}

    def _metadata(self, dataset_id: str, frame: pd.DataFrame, source: Any, duplicate_count: int) -> Dict[str, Any]:
        identities = self._identities(frame)
        gaps = self._gap_report(frame)
        optional = self.pipeline.validate(frame).get("missing_optional_columns", [])
        missing = float(frame.isna().mean().mean()) * 100.0 if len(frame) else 0.0
        return {
            "dataset_id": dataset_id,
            "source": str(source) if isinstance(source, (str, Path)) else "in_memory",
            "data_source": self._value(frame, "data_source", "historical"),
            "market": self._value(frame, "market"), "symbol": self._value(frame, "symbol"),
            "segment": self._value(frame, "segment"), "timeframe": self._value(frame, "timeframe"),
            "instrument": self._value(frame, "instrument"), "timezone": self._value(frame, "timezone", "UTC"),
            "provider": self._value(frame, "provider"),
            "date_range": {"start": self._json(frame["timestamp"].min()) if len(frame) else None, "end": self._json(frame["timestamp"].max()) if len(frame) else None},
            "rows": int(len(frame)), "duplicate_count": int(duplicate_count), "missing_data_percentage": round(missing, 4), "missing_optional_columns": optional, "identities": identities, "time_gaps": gaps,
        }

    def register(self, dataset_id: str, source: Any, *, market: Optional[str] = None, symbol: Optional[str] = None, segment: Optional[str] = None, instrument: Optional[str] = None, timeframe: Optional[str] = None, timezone_name: Optional[str] = None, timezone: Optional[str] = None, provider: Optional[str] = None, data_source: str = "historical") -> Dict[str, Any]:
        """Normalize, validate, and register one compatible dataset."""
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            raise HistoricalDataError("dataset_id must be a non-empty string")
        frame = self._restore_instrument_column(self.pipeline.normalize(self._source_without_instrument_alias(source), market=market, segment=segment, symbol=symbol, timeframe=timeframe))
        timezone_name = timezone_name or timezone
        for column, value in (("instrument", instrument), ("timezone", timezone_name), ("provider", provider)):
            if value is not None:
                frame[column] = value
        identities = self._identities(frame)
        for field in ("market", "symbol", "segment", "instrument", "timeframe", "timezone", "provider"):
            if len(identities.get(field, [])) > 1:
                raise HistoricalDataError(f"Dataset mixes incompatible {field} values: {identities[field]}")
        if not identities.get("timeframe"):
            raise HistoricalDataError("Dataset timeframe is required for isolated storage")
        if data_source and "data_source" not in frame:
            frame["data_source"] = data_source
        duplicate_count = self._raw_duplicate_count(self._source_without_instrument_alias(source), frame)
        self._datasets[dataset_id] = frame.sort_values("timestamp").reset_index(drop=True)
        self._registry[dataset_id] = self._metadata(dataset_id, self._datasets[dataset_id], source, duplicate_count)
        return deepcopy(self._registry[dataset_id])

    def merge(self, dataset_id: str, source: Any, **metadata: Any) -> Dict[str, Any]:
        """Merge another source only when its identity matches the registered dataset."""
        if dataset_id not in self._datasets:
            return self.register(dataset_id, source, **metadata)
        existing = self._registry[dataset_id]
        incoming = self._restore_instrument_column(self.pipeline.normalize(self._source_without_instrument_alias(source), market=metadata.get("market"), symbol=metadata.get("symbol"), segment=metadata.get("segment"), timeframe=metadata.get("timeframe")))
        for field in ("market", "symbol", "segment", "timeframe"):
            incoming_value = self._value(incoming, field)
            if existing.get(field, "UNKNOWN") != incoming_value and incoming_value != "UNKNOWN":
                raise HistoricalDataError(f"Incompatible {field} for dataset '{dataset_id}'")
        duplicate_count = self._raw_duplicate_count(self._source_without_instrument_alias(source), incoming)
        covered_ranges = list(existing.get("covered_ranges", []))
        combined = pd.concat([self._datasets[dataset_id], incoming], ignore_index=True)
        identity_columns = [column for column in ("market", "symbol", "segment", "instrument", "timeframe", "timezone", "provider") if column in combined]
        combined = combined.sort_values("timestamp", kind="stable").drop_duplicates(identity_columns + ["timestamp"], keep="last").reset_index(drop=True)
        self._datasets[dataset_id] = combined
        self._registry[dataset_id] = self._metadata(dataset_id, combined, existing.get("source", "in_memory"), existing.get("duplicate_count", 0) + duplicate_count)
        self._registry[dataset_id]["covered_ranges"] = covered_ranges
        return deepcopy(self._registry[dataset_id])

    @staticmethod
    def _angel_timestamp(value: Any) -> datetime:
        parsed = pd.to_datetime(value, errors="raise", utc=True)
        return parsed.to_pydatetime()

    @staticmethod
    def _angel_exchange(market: str, segment: str, exchange: Optional[str]) -> str:
        if exchange:
            return str(exchange).strip().upper()
        if str(market).strip().upper() == "INDIA":
            return "NSE"
        raise HistoricalDataError("Angel One historical data supports configured INDIA instruments only; exchange is required for other markets")

    @staticmethod
    def _angel_candle_rows(candles: Any, *, symbol: str, market: str, segment: str, timeframe: str) -> list[Dict[str, Any]]:
        if not isinstance(candles, list):
            return []
        rows = []
        for candle in candles:
            if isinstance(candle, dict):
                row = dict(candle)
            elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                row = {
                    "timestamp": candle[0], "open": candle[1], "high": candle[2],
                    "low": candle[3], "close": candle[4], "volume": candle[5],
                }
            else:
                continue
            row.update({"symbol": symbol, "market": market, "segment": segment, "timeframe": timeframe, "data_source": "angel_one"})
            rows.append(row)
        return rows

    def _angel_chunk_is_stored(self, dataset_id: str, start: datetime, end: datetime) -> bool:
        """Return true only when this isolated dataset already covers a chunk."""
        covered = self._registry.get(dataset_id, {}).get("covered_ranges", [])
        requested = {"start": self._json(start), "end": self._json(end)}
        return requested in covered

    def _record_angel_chunk(self, dataset_id: str, start: datetime, end: datetime) -> None:
        if dataset_id not in self._registry:
            return
        ranges = list(self._registry[dataset_id].get("covered_ranges", []))
        item = {"start": self._json(start), "end": self._json(end)}
        if item not in ranges:
            ranges.append(item)
        self._registry[dataset_id]["covered_ranges"] = ranges

    def _record_angel_range(self, dataset_id: str, field: str, start: datetime, end: datetime) -> None:
        if dataset_id not in self._registry:
            return
        ranges = list(self._registry[dataset_id].get(field, []))
        item = {"start": self._json(start), "end": self._json(end)}
        if item not in ranges:
            ranges.append(item)
        self._registry[dataset_id][field] = ranges

    @staticmethod
    def _history_window(history: Any, end: Optional[datetime], start: Any = None, available_start: Any = None) -> tuple[datetime, datetime]:
        end_ts = HistoricalDatasetManager._angel_timestamp(end or datetime.now(timezone.utc))
        if isinstance(end, str) and len(end.strip()) == 10:
            end_ts += timedelta(days=1) - timedelta(seconds=1)
        if start is not None:
            start_ts = HistoricalDatasetManager._angel_timestamp(start)
        else:
            key = str(history or "ALL_AVAILABLE").strip().upper().replace(" ", "")
            if key in {"ALL", "ALL_AVAILABLE", "MAX"}:
                if available_start is None:
                    raise HistoricalDataError("ALL_AVAILABLE requires a provider availability start or stored dataset history")
                start_ts = HistoricalDatasetManager._angel_timestamp(available_start)
            elif key not in _HISTORY_DAYS:
                raise HistoricalDataError(f"Unsupported history window: {history}")
            else:
                start_ts = end_ts - timedelta(days=_HISTORY_DAYS[key])
        if start_ts > end_ts:
            raise HistoricalDataError("Historical start must not be after end")
        return start_ts, end_ts

    @staticmethod
    def _provider_request(client: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call a provider's read-only candle method without owning authentication."""
        method = getattr(client, "get_candles", None) or getattr(client, "fetch_candles", None) or getattr(client, "getCandleData", None)
        if method is None:
            raise HistoricalDataError("Provider client must expose a read-only candle method")
        response = method(request)
        if isinstance(response, dict):
            if response.get("status") is False:
                raise HistoricalDataError("Historical provider rejected the candle request")
            return response
        if isinstance(response, list):
            return {"status": True, "data": response}
        raise HistoricalDataError("Historical provider returned an invalid candle response")

    @staticmethod
    def _missing_sessions(frame: pd.DataFrame, start: datetime, end: datetime) -> list[str]:
        if frame.empty or "timestamp" not in frame:
            return [day.date().isoformat() for day in pd.date_range(start.date(), end.date(), freq="D") if day.weekday() < 5]
        observed = {timestamp.date() for timestamp in pd.to_datetime(frame["timestamp"], utc=True)}
        return [day.date().isoformat() for day in pd.date_range(start.date(), end.date(), freq="D") if day.weekday() < 5 and day.date() not in observed]

    @staticmethod
    def _angel_request_with_retry(client: Any, request: Dict[str, Any], *, retries: int, backoff_seconds: float) -> tuple[Dict[str, Any], int]:
        attempts = 0
        last_error = None
        while attempts <= retries:
            try:
                response = client.getCandleData(request)
                if isinstance(response, dict) and response.get("status") is not False:
                    return response, attempts
                last_error = HistoricalDataError("Angel One historical API rejected a read-only candle request")
            except Exception as exc:
                last_error = exc
            if attempts == retries:
                break
            attempts += 1
            time.sleep(backoff_seconds * (2 ** (attempts - 1)))
        raise HistoricalDataError("Angel One historical request failed after bounded retries") from last_error

    def fetch_angel_history(
        self,
        dataset_id: str,
        *,
        symbol: Optional[str] = None,
        market: str = "INDIA",
        segment: str = "INTRADAY",
        timeframe: str = "5m",
        start: Any = None,
        end: Any = None,
        history: str = "ALL_AVAILABLE",
        exchange: Optional[str] = None,
        max_retries: int = 3,
        rate_limit_seconds: float = 0.25,
        live_feed: Any = None,
        instrument: Optional[str] = None,
        timezone_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch and store real Angel One candles in API-supported chunks.

        The requested date range is caller-controlled; no artificial overall
        history limit is applied. Empty API chunks remain absent and are
        reported by the returned quality metadata.
        """
        canonical = self.pipeline._canonical_timeframe(timeframe)
        if canonical not in _ANGEL_INTERVALS:
            raise HistoricalDataError(f"Unsupported Angel One timeframe: {timeframe}")
        start_ts, end_ts = self._history_window(history, end, start)
        if max_retries < 0 or rate_limit_seconds < 0:
            raise HistoricalDataError("Retry and rate-limit values cannot be negative")

        selected_symbol = (symbol or settings.stock_symbol).strip().upper()
        selected_market = str(market).strip().upper()
        selected_segment = str(segment).strip().upper()
        selected_exchange = self._angel_exchange(selected_market, selected_segment, exchange)

        from data.live_feed import AngelLiveFeed

        feed = live_feed or AngelLiveFeed(symbol=selected_symbol, exchange=selected_exchange)
        client = feed._ensure_session()
        resolved_instrument = feed._resolve_instrument()
        selected_instrument = instrument or resolved_instrument["tradingsymbol"]
        selected_timezone = timezone_name or MARKETS.get(selected_market, {}).get("timezone", "UTC")
        interval_name, chunk_days = _ANGEL_INTERVALS[canonical]
        rows: list[Dict[str, Any]] = []
        requests = 0
        retries_used = 0
        skipped_chunks = 0
        completed_chunks = 0
        missing_chunks = 0
        failed_ranges = []
        empty_ranges = []
        skipped_ranges = []
        seen_chunks = set()
        cursor = start_ts
        token = resolved_instrument["symboltoken"]
        alternate_token = None
        if (
            token.isdigit()
            and not token.startswith("999")
            and not resolved_instrument.get("instrumenttype")
            and not resolved_instrument.get("series")
        ):
            alternate_token = f"999{token}"
        while cursor <= end_ts:
            chunk_end = min(cursor + timedelta(days=chunk_days) - timedelta(seconds=1), end_ts)
            chunk_key = (cursor, chunk_end, canonical, selected_market, selected_segment, selected_symbol)
            if chunk_key in seen_chunks:
                cursor = chunk_end + timedelta(seconds=1)
                continue
            seen_chunks.add(chunk_key)
            if self._angel_chunk_is_stored(dataset_id, cursor, chunk_end):
                skipped_chunks += 1
                skipped_ranges.append({"start": self._json(cursor), "end": self._json(chunk_end)})
                cursor = chunk_end + timedelta(seconds=1)
                continue
            request = {
                "exchange": resolved_instrument["exchange"],
                "symboltoken": token,
                "interval": interval_name,
                "fromdate": cursor.strftime("%Y-%m-%d %H:%M"),
                "todate": chunk_end.strftime("%Y-%m-%d %H:%M"),
            }
            if requests:
                time.sleep(rate_limit_seconds)
            requests += 1
            try:
                response, attempts = self._angel_request_with_retry(client, request, retries=max_retries, backoff_seconds=max(rate_limit_seconds, 0.05))
                retries_used += attempts
                payload = response.get("data") if isinstance(response, dict) else None
                if not payload and alternate_token:
                    request["symboltoken"] = alternate_token
                    alternate_response, alternate_attempts = self._angel_request_with_retry(client, request, retries=max_retries, backoff_seconds=max(rate_limit_seconds, 0.05))
                    requests += 1
                    retries_used += alternate_attempts
                    alternate_payload = alternate_response.get("data") if isinstance(alternate_response, dict) else None
                    if alternate_payload:
                        token = alternate_token
                        resolved_instrument["symboltoken"] = token
                        payload = alternate_payload
                chunk_rows = self._angel_candle_rows(payload, symbol=selected_symbol, market=selected_market, segment=selected_segment, timeframe=canonical)
                bounded_rows = []
                for row in chunk_rows:
                    try:
                        candle_timestamp = self._angel_timestamp(row.get("timestamp"))
                    except (TypeError, ValueError):
                        continue
                    if cursor <= candle_timestamp <= chunk_end:
                        row.update({"instrument": selected_instrument, "timezone": selected_timezone, "provider": "angel_one"})
                        bounded_rows.append(row)
                chunk_rows = bounded_rows
                if chunk_rows:
                    rows.extend(chunk_rows)
                    self.merge(dataset_id, chunk_rows, market=selected_market, symbol=selected_symbol, segment=selected_segment, instrument=selected_instrument, timeframe=canonical, timezone=selected_timezone, provider="angel_one")
                    completed_chunks += 1
                    self._record_angel_chunk(dataset_id, cursor, chunk_end)
                else:
                    missing_chunks += 1
                    empty_ranges.append({"start": self._json(cursor), "end": self._json(chunk_end)})
            except Exception as exc:
                failed_ranges.append({"start": self._json(cursor), "end": self._json(chunk_end), "error_type": type(exc).__name__})
            cursor = chunk_end + timedelta(seconds=1)

        if not rows and dataset_id not in self._datasets:
            raise HistoricalDataError("Angel One returned no historical candles for the requested range")
        frame = self._datasets[dataset_id]
        metadata = self._metadata(dataset_id, frame, "in_memory", 0)
        metadata["angel_exchange"] = resolved_instrument["exchange"]
        metadata["angel_tradingsymbol"] = resolved_instrument["tradingsymbol"]
        metadata["angel_token_resolved"] = bool(resolved_instrument.get("symboltoken"))
        metadata["api_requests"] = requests
        metadata["retry_count"] = retries_used
        metadata["completed_chunks"] = completed_chunks
        metadata["skipped_chunks"] = skipped_chunks
        metadata["missing_chunks"] = missing_chunks
        metadata["failed_ranges"] = failed_ranges
        metadata["empty_ranges"] = empty_ranges
        metadata["skipped_ranges"] = skipped_ranges
        metadata["candle_count"] = int(len(frame))
        metadata["requested_range"] = {"start": self._json(start_ts), "end": self._json(end_ts)}
        metadata["actual_range"] = metadata["date_range"]
        metadata["instrument"] = selected_instrument
        metadata["timezone"] = selected_timezone
        metadata["provider"] = "angel_one"
        metadata["missing_sessions"] = self._missing_sessions(frame, start_ts, end_ts)
        metadata["missing_data_warnings"] = metadata.get("time_gaps", {}).get("gaps", [])
        metadata["missing_data_warnings"] += [{"type": "missing_session", "date": date} for date in metadata["missing_sessions"]]
        metadata["resumable"] = bool(missing_chunks or failed_ranges)
        metadata["covered_ranges"] = list(self._registry.get(dataset_id, {}).get("covered_ranges", []))
        self._registry[dataset_id].update(metadata)
        return metadata

    def fetch_historical(
        self,
        dataset_id: str,
        *,
        provider: str,
        client: Any,
        symbol: str,
        market: str,
        segment: str,
        timeframe: str,
        instrument: Optional[str] = None,
        timezone_name: str = "UTC",
        start: Any = None,
        end: Any = None,
        history: str = "ALL_AVAILABLE",
        available_start: Any = None,
        interval_limits: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        rate_limit_seconds: float = 0.25,
    ) -> Dict[str, Any]:
        """Collect candles through an authenticated, read-only provider adapter.

        ``client`` may be an authenticated provider client or an existing feed
        exposing ``_ensure_session`` and ``_resolve_instrument``. Authentication
        remains owned by that existing integration; this manager never stores it.
        """
        canonical = self.pipeline._canonical_timeframe(timeframe)
        limits = interval_limits or _PROVIDER_INTERVALS.get(str(provider).lower())
        if not limits or canonical not in limits:
            raise HistoricalDataError(f"Provider '{provider}' does not support timeframe '{timeframe}'")
        existing_start = None
        if dataset_id in self._datasets and not self._datasets[dataset_id].empty:
            existing_start = self._datasets[dataset_id]["timestamp"].min()
        start_ts, end_ts = self._history_window(history, end, start, available_start or existing_start)
        if max_retries < 0 or rate_limit_seconds < 0:
            raise HistoricalDataError("Retry and rate-limit values cannot be negative")
        selected_market, selected_symbol, selected_segment = str(market).upper(), str(symbol).upper(), str(segment).upper()
        auth_client = client._ensure_session() if hasattr(client, "_ensure_session") else client
        resolved = client._resolve_instrument() if hasattr(client, "_resolve_instrument") else {}
        selected_instrument = instrument or resolved.get("tradingsymbol") or selected_symbol
        provider_key = str(provider).lower()
        interval_value, chunk_days = limits[canonical]
        rows: list[Dict[str, Any]] = []
        api_calls = retries_used = completed_chunks = skipped_chunks = missing_chunks = 0
        failed_ranges: list[Dict[str, Any]] = []
        empty_ranges: list[Dict[str, Any]] = []
        skipped_ranges: list[Dict[str, Any]] = []
        cursor = start_ts
        while cursor <= end_ts:
            chunk_end = min(cursor + timedelta(days=int(chunk_days)) - timedelta(seconds=1), end_ts)
            if self._angel_chunk_is_stored(dataset_id, cursor, chunk_end):
                skipped_chunks += 1
                skipped_ranges.append({"start": self._json(cursor), "end": self._json(chunk_end)})
                cursor = chunk_end + timedelta(seconds=1)
                continue
            request = {
                "provider": provider_key, "market": selected_market, "segment": selected_segment,
                "symbol": selected_symbol, "instrument": selected_instrument, "timeframe": canonical,
                "interval": interval_value, "from": self._json(cursor), "to": self._json(chunk_end),
                "fromdate": cursor.strftime("%Y-%m-%d %H:%M"), "todate": chunk_end.strftime("%Y-%m-%d %H:%M"),
            }
            if resolved:
                request.update({"exchange": resolved.get("exchange"), "symboltoken": resolved.get("symboltoken")})
            if api_calls:
                time.sleep(rate_limit_seconds)
            api_calls += 1
            try:
                response, attempts = self._request_with_retry(auth_client, request, max_retries, rate_limit_seconds)
                retries_used += attempts
                payload = response.get("data", response.get("candles", []))
                chunk_rows = self._angel_candle_rows(payload, symbol=selected_symbol, market=selected_market, segment=selected_segment, timeframe=canonical)
                bounded = []
                for row in chunk_rows:
                    try:
                        timestamp = self._angel_timestamp(row["timestamp"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if cursor <= timestamp <= chunk_end:
                        row.update({"instrument": selected_instrument, "timezone": timezone_name, "provider": provider_key, "data_source": provider_key})
                        bounded.append(row)
                if bounded:
                    self.merge(dataset_id, bounded, market=selected_market, symbol=selected_symbol, segment=selected_segment, instrument=selected_instrument, timeframe=canonical, timezone=timezone_name, provider=provider_key)
                    rows.extend(bounded)
                    completed_chunks += 1
                    self._record_angel_chunk(dataset_id, cursor, chunk_end)
                else:
                    missing_chunks += 1
                    empty_ranges.append({"start": self._json(cursor), "end": self._json(chunk_end)})
            except Exception as exc:
                failed_ranges.append({"start": self._json(cursor), "end": self._json(chunk_end), "error_type": type(exc).__name__})
            cursor = chunk_end + timedelta(seconds=1)
        if not rows and dataset_id not in self._datasets:
            raise HistoricalDataError("Historical provider returned no candles for the requested range")
        frame = self._datasets[dataset_id]
        metadata = self._metadata(dataset_id, frame, "in_memory", 0)
        metadata.update({
            "provider": provider_key, "instrument": selected_instrument, "timezone": timezone_name,
            "requested_range": {"start": self._json(start_ts), "end": self._json(end_ts)},
            "actual_range": metadata["date_range"], "candle_count": int(len(frame)),
            "chunks": completed_chunks + skipped_chunks + missing_chunks + len(failed_ranges), "completed_chunks": completed_chunks,
            "api_calls": api_calls, "retry_count": retries_used, "skipped_chunks": skipped_chunks,
            "skipped_ranges": skipped_ranges, "failed_ranges": failed_ranges, "empty_ranges": empty_ranges,
            "missing_sessions": self._missing_sessions(frame, start_ts, end_ts),
            "resumable": bool(missing_chunks or failed_ranges),
            "limitations": [] if provider_key == "angel_one" else [f"Provider '{provider_key}' adapter limits and session behavior are caller-supplied"],
        })
        metadata["missing_data_warnings"] = metadata["time_gaps"]["gaps"] + [{"type": "missing_session", "date": date} for date in metadata["missing_sessions"]]
        self._registry[dataset_id].update(metadata)
        return deepcopy(metadata)

    @staticmethod
    def _request_with_retry(client: Any, request: Dict[str, Any], retries: int, backoff_seconds: float) -> tuple[Dict[str, Any], int]:
        attempts = 0
        while True:
            try:
                return HistoricalDatasetManager._provider_request(client, request), attempts
            except Exception as exc:
                if attempts >= retries:
                    raise HistoricalDataError("Historical provider request failed after bounded retries") from exc
                attempts += 1
                time.sleep(max(backoff_seconds, 0.05) * (2 ** (attempts - 1)))

    load_angel_history = fetch_angel_history
    load_history = fetch_historical

    def registry(self) -> Dict[str, Dict[str, Any]]:
        return deepcopy(self._registry)

    list_datasets = registry

    def get_metadata(self, dataset_id: str) -> Dict[str, Any]:
        if dataset_id not in self._registry:
            raise HistoricalDataError(f"Unknown dataset: {dataset_id}")
        return deepcopy(self._registry[dataset_id])

    def get(self, dataset_id: str, *, market: Optional[str] = None, segment: Optional[str] = None, symbol: Optional[str] = None, timeframe: Optional[str] = None, start: Any = None, end: Any = None, lookback: Optional[str] = None) -> pd.DataFrame:
        """Retrieve a chronological, metadata-filtered copy without future rows."""
        if dataset_id not in self._datasets:
            raise HistoricalDataError(f"Unknown dataset: {dataset_id}")
        frame = self._datasets[dataset_id].copy()
        for column, value in (("market", market), ("segment", segment), ("symbol", symbol), ("timeframe", timeframe)):
            if value is not None:
                canonical = self.pipeline._canonical_timeframe(value) if column == "timeframe" else str(value)
                frame = frame[frame[column].astype(str).str.upper() == str(canonical).upper()]
        if lookback:
            frame = self.pipeline.filter_lookback(frame, lookback, end)
        else:
            if start is not None:
                frame = frame[frame["timestamp"] >= pd.Timestamp(start, tz="UTC")]
            if end is not None:
                frame = frame[frame["timestamp"] <= pd.Timestamp(end, tz="UTC")]
        return frame.sort_values("timestamp").reset_index(drop=True)

    def records(self, dataset_id: str, **filters: Any) -> list[Dict[str, Any]]:
        return self.pipeline.records(self.get(dataset_id, **filters))

    def split(self, dataset_id: str, train: float = 0.6, validation: float = 0.2, test: float = 0.2, **filters: Any) -> Dict[str, Any]:
        """Expose pipeline chronological splits for backtesting consumers."""
        return self.pipeline.split(self.get(dataset_id, **filters), train=train, validation=validation, test=test)

    def quality(self, dataset_id: str) -> Dict[str, Any]:
        return self.get_metadata(dataset_id)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    @staticmethod
    def _regime(frame: pd.DataFrame) -> str:
        if len(frame) < 2:
            return "UNKNOWN"
        returns = pd.to_numeric(frame["close"], errors="coerce").pct_change().dropna()
        if returns.empty:
            return "UNKNOWN"
        volatility = float(returns.std(ddof=0))
        direction = abs(float(returns.mean()))
        if volatility >= 0.03:
            return "HIGH_VOLATILITY"
        if volatility <= 0.002:
            return "LOW_VOLATILITY"
        return "TREND" if direction >= volatility * 0.35 else "SIDEWAYS"

    def _speed_series(self, frame: pd.DataFrame) -> pd.Series:
        if "volume_per_minute" in frame and frame["volume_per_minute"].notna().any():
            return pd.to_numeric(frame["volume_per_minute"], errors="coerce")
        minutes = self.pipeline._minutes(self._value(frame, "timeframe", "")) or 1.0
        return pd.to_numeric(frame["volume"], errors="coerce").div(minutes)

    def _period_frames(self, frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if frame.empty:
            return {"recent": frame.copy(), "medium_term": frame.copy(), "long_term": frame.copy(), "full_history": frame.copy()}
        latest = frame["timestamp"].max()
        recent = latest - pd.Timedelta(days=90)
        medium = latest - pd.Timedelta(days=365)
        return {
            "recent": frame[frame["timestamp"] >= recent].copy(),
            "medium_term": frame[(frame["timestamp"] < recent) & (frame["timestamp"] >= medium)].copy(),
            "long_term": frame[frame["timestamp"] < medium].copy(),
            "full_history": frame.copy(),
        }

    def _period_report(self, frame: pd.DataFrame, calculator: AdaptiveLeadTimeCalculator) -> Dict[str, Any]:
        report = {}
        for name, subset in self._period_frames(frame).items():
            speed_values = [float(value) for value in self._speed_series(subset).dropna() if float(value) > 0]
            profile_rows = self.records_from_frame(subset)
            report[name] = {
                "rows": len(subset),
                "regime": self._regime(subset),
                "samples": len(speed_values),
                "speed_statistics": {
                    "median": self._percentile(speed_values, 0.5),
                    "p75": self._percentile(speed_values, 0.75),
                    "p90": self._percentile(speed_values, 0.9),
                    "p95": self._percentile(speed_values, 0.95),
                    "maximum": max(speed_values) if speed_values else None,
                },
                "adaptive_profiles": calculator.build_profiles(profile_rows) if speed_values else {},
                "confidence": round(min(1.0, len(speed_values) / max(calculator.min_samples, 1)) * (1.0 if len(speed_values) >= 2 else 0.0), 4),
            }
        return report

    @staticmethod
    def records_from_frame(frame: pd.DataFrame) -> list[Dict[str, Any]]:
        return HistoricalDataPipeline.records(frame)

    def analyze(self, dataset_id: str, *, min_samples: Optional[int] = None, calculator: Optional[AdaptiveLeadTimeCalculator] = None, data_source: Optional[str] = None) -> Dict[str, Any]:
        """Return full-history, period, regime, time-of-day, and adaptive quality analysis."""
        frame = self.get(dataset_id)
        calculator = calculator or AdaptiveLeadTimeCalculator(min_samples=min_samples or 5, lookback_days=36500)
        metadata = self.get_metadata(dataset_id)
        speed_values = [float(value) for value in self._speed_series(frame).dropna() if float(value) > 0]
        hour_groups = {}
        if not frame.empty:
            hours = frame["timestamp"].dt.hour
            for name, mask in (("OPENING", hours < 11), ("MIDDAY", (hours >= 11) & (hours < 14)), ("CLOSING", hours >= 14)):
                subset = frame[mask]
                hour_groups[name] = {"rows": len(subset), "regime": self._regime(subset), "median_speed": self._percentile([float(value) for value in self._speed_series(subset).dropna() if float(value) > 0], 0.5)}
        report = {
            "dataset_id": dataset_id,
            "historical_only": True,
            "data_source": data_source or metadata.get("data_source", "historical"),
            "synthetic_data": str(data_source or metadata.get("data_source", "")).lower() == "synthetic",
            "metadata": metadata,
            "availability": {"earliest": metadata["date_range"]["start"], "latest": metadata["date_range"]["end"], "rows": len(frame), "timeframe": metadata["timeframe"]},
            "periods": self._period_report(frame, calculator),
            "time_of_day": hour_groups,
            "regime": self._regime(frame),
            "speed_statistics": {"samples": len(speed_values), "median": self._percentile(speed_values, 0.5), "p75": self._percentile(speed_values, 0.75), "p90": self._percentile(speed_values, 0.9), "p95": self._percentile(speed_values, 0.95), "maximum": max(speed_values) if speed_values else None},
            "adaptive_profiles": calculator.build_profiles(self.records_from_frame(frame)) if speed_values else {},
            "rare_extreme_events": self._rare_events(frame),
            "confidence": round(min(1.0, len(speed_values) / max(calculator.min_samples, 1)) * (1.0 if len(speed_values) >= 2 else 0.0), 4),
            "note": "Synthetic results are correctness checks, not historical performance evidence." if str(data_source or metadata.get("data_source", "")).lower() == "synthetic" else "Missing periods are reported, never fabricated.",
        }
        json.dumps(report)
        return report

    analyze_history = analyze

    def _rare_events(self, frame: pd.DataFrame) -> list[Dict[str, Any]]:
        speeds = self._speed_series(frame)
        valid = speeds.dropna()
        if len(valid) < 4:
            return []
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        upper = float(q3 + 1.5 * (q3 - q1))
        return [
            {"timestamp": self.pipeline._json_value(timestamp), "speed": float(speed), "type": "EXTREME_SPEED"}
            for timestamp, speed in zip(frame.loc[valid.index, "timestamp"], valid)
            if speed > upper
        ]

    def walk_forward(self, dataset_id: str, *, min_train_rows: int = 1, step: int = 1, start: Any = None, end: Any = None) -> list[Dict[str, Any]]:
        """Return expanding chronological train/evaluation windows with no future rows."""
        if min_train_rows < 1 or step < 1:
            raise HistoricalDataError("min_train_rows and step must be positive")
        frame = self.get(dataset_id, start=start, end=end)
        rows = self.records_from_frame(frame)
        windows = []
        for index in range(min_train_rows, len(rows), step):
            evaluation = rows[index]
            train = rows[:index]
            windows.append({"train": train, "evaluation": evaluation, "train_end": train[-1]["timestamp"], "evaluation_timestamp": evaluation["timestamp"]})
        return windows

    chronological_access = walk_forward


HistoricalDataManager = HistoricalDatasetManager