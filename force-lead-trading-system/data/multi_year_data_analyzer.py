"""Multi-period historical analysis without a fixed maximum history window."""

from __future__ import annotations

from datetime import timedelta
import json
import math
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd

from core.adaptive_lead_time import AdaptiveLeadTimeCalculator
from data.historical_data_manager import HistoricalDataError, HistoricalDatasetManager
from data.historical_data_pipeline import HistoricalDataPipeline


class MultiYearDataAnalyzer:
    """Analyze every compatible historical identity and preserve period differences."""

    def __init__(self, manager: Optional[HistoricalDatasetManager] = None, calculator: Optional[AdaptiveLeadTimeCalculator] = None, min_samples: int = 5) -> None:
        self.manager = manager or HistoricalDatasetManager()
        self.pipeline = self.manager.pipeline
        self.calculator = calculator or AdaptiveLeadTimeCalculator(min_samples=min_samples, lookback_days=36500)
        self.min_samples = max(1, int(min_samples))

    @staticmethod
    def _number(value: Any, default: float = 0.0) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return default
        return result if math.isfinite(result) else default

    @staticmethod
    def _periods(frame: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        if frame.empty:
            return {"recent": frame.copy(), "medium_term": frame.copy(), "long_term": frame.copy()}
        latest = frame["timestamp"].max()
        recent_start = latest - pd.Timedelta(days=90)
        medium_start = latest - pd.Timedelta(days=365)
        return {
            "recent": frame[frame["timestamp"] >= recent_start].copy(),
            "medium_term": frame[(frame["timestamp"] < recent_start) & (frame["timestamp"] >= medium_start)].copy(),
            "long_term": frame[frame["timestamp"] < medium_start].copy(),
        }

    @staticmethod
    def _regime(frame: pd.DataFrame) -> str:
        if len(frame) < 2:
            return "UNKNOWN"
        returns = frame["close"].pct_change().dropna()
        if returns.empty:
            return "UNKNOWN"
        volatility = float(returns.std(ddof=0))
        direction = abs(float(returns.mean()))
        if volatility >= 0.03:
            return "HIGH_VOLATILITY"
        if direction >= max(volatility * 0.35, 1e-12):
            return "TREND"
        if volatility <= 0.002:
            return "LOW_VOLATILITY"
        return "SIDEWAYS"

    @staticmethod
    def _speed(frame: pd.DataFrame) -> pd.Series:
        if "volume_per_minute" in frame and frame["volume_per_minute"].notna().any():
            return pd.to_numeric(frame["volume_per_minute"], errors="coerce")
        minutes = frame["timeframe"].map(lambda value: AdaptiveLeadTimeCalculator._minutes(value) or 1.0) if "timeframe" in frame else 1.0
        return pd.to_numeric(frame["volume"], errors="coerce").div(minutes)

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    def _quality(self, frame: pd.DataFrame) -> Dict[str, Any]:
        timeframe = str(frame["timeframe"].iloc[0]) if not frame.empty and "timeframe" in frame else None
        minutes = self.pipeline._minutes(timeframe)
        gaps = []
        if minutes and len(frame) > 1:
            ordered = frame.sort_values("timestamp")
            differences = ordered["timestamp"].diff().dt.total_seconds().div(60.0)
            gaps = [float(value) for value in differences.dropna() if value > minutes * 1.5]
        return {"rows": int(len(frame)), "date_range": {"start": self.pipeline._json_value(frame["timestamp"].min()) if len(frame) else None, "end": self.pipeline._json_value(frame["timestamp"].max()) if len(frame) else None}, "coverage_days": float((frame["timestamp"].max() - frame["timestamp"].min()).total_seconds() / 86400.0) if len(frame) > 1 else 0.0, "missing_values": int(frame.isna().sum().sum()), "missing_data_percentage": round(float(frame.isna().mean().mean()) * 100.0, 4) if len(frame) else 0.0, "duplicate_records": 0, "gap_count": len(gaps), "gap_minutes": gaps}

    def _period_report(self, frame: pd.DataFrame) -> Dict[str, Any]:
        result = {}
        for name, subset in self._periods(frame).items():
            speeds = [float(value) for value in self._speed(subset).dropna() if value > 0]
            adaptive = self.calculator.build_profiles(self.pipeline.records(subset)) if speeds else {}
            result[name] = {"samples": len(speeds), "available": bool(speeds), "regime": self._regime(subset), "speed_statistics": {"median": self._percentile(speeds, .5), "p75": self._percentile(speeds, .75), "p90": self._percentile(speeds, .9), "p95": self._percentile(speeds, .95), "maximum": max(speeds) if speeds else None}, "adaptive_profiles": adaptive, "confidence": round(min(1.0, len(speeds) / self.min_samples) * (1.0 if len(speeds) >= 2 else 0.0), 4)}
        return result

    def _load(self, source: Any, **metadata: Any) -> pd.DataFrame:
        if isinstance(source, str) and source in self.manager.registry():
            return self.manager.get(source)
        if isinstance(source, (list, tuple)) and not source:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "market", "symbol", "segment", "timeframe"])
        return self.pipeline.normalize(source, **metadata)

    def analyze(self, source: Any, *, market: Optional[str] = None, symbol: Optional[str] = None, segment: Optional[str] = None, timeframe: Optional[str] = None, data_source: str = "historical") -> Dict[str, Any]:
        """Analyze all metadata groups in the supplied history, with no max-date truncation."""
        frame = self._load(source, market=market, symbol=symbol, segment=segment, timeframe=timeframe)
        if frame.empty:
            return {"historical_only": True, "data_source": data_source, "synthetic_data": str(data_source).lower() == "synthetic", "earliest_timestamp": None, "latest_timestamp": None, "groups": {}, "availability": [], "note": "No usable historical rows; no market behavior was fabricated."}
        groups: Dict[tuple[str, str, str, str], pd.DataFrame] = {}
        for key, subset in frame.groupby(["market", "symbol", "segment", "timeframe"], dropna=False):
            groups[tuple(str(value) for value in key)] = subset.sort_values("timestamp").reset_index(drop=True)
        reports: Dict[str, Any] = {}
        availability = []
        for key, subset in groups.items():
            group_name = "/".join(key)
            quality = self._quality(subset)
            speed = [float(value) for value in self._speed(subset).dropna() if value > 0]
            profile_rows = self.pipeline.records(subset)
            adaptive = self.calculator.build_profiles(profile_rows) if speed else {}
            regime = self._regime(subset)
            reports[group_name] = {"market": key[0], "symbol": key[1], "segment": key[2], "timeframe": key[3], "quality": quality, "periods": self._period_report(subset), "regimes": {regime: {"samples": len(subset)}}, "speed_statistics": {"samples": len(speed), "median": self._percentile(speed, .5), "p75": self._percentile(speed, .75), "p90": self._percentile(speed, .9), "p95": self._percentile(speed, .95), "maximum": max(speed) if speed else None}, "adaptive_profiles": adaptive, "sample_count": len(speed), "confidence": round(min(1.0, len(speed) / self.min_samples) * (1.0 if len(speed) >= 2 else 0.0), 4), "events": self.pipeline.records(subset)}
            availability.append({"market": key[0], "symbol": key[1], "segment": key[2], "timeframe": key[3], "available": bool(speed), "rows": len(subset), "earliest": quality["date_range"]["start"], "latest": quality["date_range"]["end"], "confidence": reports[group_name]["confidence"]})
        report = {"historical_only": True, "data_source": str(data_source), "synthetic_data": str(data_source).lower() == "synthetic", "earliest_timestamp": self.pipeline._json_value(frame["timestamp"].min()), "latest_timestamp": self.pipeline._json_value(frame["timestamp"].max()), "groups": reports, "availability": availability, "group_count": len(reports), "note": "Synthetic results are correctness checks, not historical performance evidence." if str(data_source).lower() == "synthetic" else "All reports use supplied historical rows only; missing periods are not filled."}
        json.dumps(report)
        return report

    analyze_historical_data = analyze