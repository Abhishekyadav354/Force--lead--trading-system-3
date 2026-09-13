"""Walk-forward historical analysis for adaptive lead-time observations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, Iterable, Mapping, Optional

from core.adaptive_lead_time import AdaptiveLeadTimeCalculator
from data.historical_data_pipeline import HistoricalDataPipeline


class HistoricalLeadAnalyzer:
    """Connect historical pipeline records to the adaptive lead-time model."""

    def __init__(self, pipeline: Optional[HistoricalDataPipeline] = None, calculator: Optional[AdaptiveLeadTimeCalculator] = None, min_samples: int = 5) -> None:
        self.pipeline = pipeline or HistoricalDataPipeline()
        self.calculator = calculator or AdaptiveLeadTimeCalculator(min_samples=min_samples)
        self.min_samples = max(1, int(min_samples))

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number == number and number not in (float("inf"), float("-inf")) else None

    @staticmethod
    def _timestamp(row: Mapping[str, Any]) -> Optional[datetime]:
        value = row.get("timestamp", row.get("datetime", row.get("time")))
        try:
            timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    @staticmethod
    def _event_id(row: Mapping[str, Any], index: int) -> Optional[str]:
        value = row.get("event_id", row.get("lead_event_id", row.get("event")))
        return str(value) if value not in (None, "") else None

    @classmethod
    def _gap(cls, row: Mapping[str, Any]) -> Optional[float]:
        for key in ("remaining_volume_gap", "remaining_gap", "volume_gap", "progress_gap", "remaining_progress"):
            value = cls._number(row.get(key))
            if value is not None:
                return max(0.0, value)
        return None

    @classmethod
    def _complete(cls, row: Mapping[str, Any]) -> bool:
        return row.get("completed", row.get("event_complete", row.get("is_complete", False))) is True or (cls._gap(row) == 0.0)

    @staticmethod
    def _group_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("market", "UNKNOWN")).upper(),
            str(row.get("symbol", "UNKNOWN")).upper(),
            str(row.get("segment", "OTHER")).upper(),
            str(row.get("timeframe", "UNKNOWN")).lower(),
        )

    @staticmethod
    def _period(timestamp: datetime) -> str:
        if timestamp.hour < 11:
            return "OPENING"
        if timestamp.hour < 14:
            return "MIDDAY"
        return "CLOSING"

    @staticmethod
    def _regime(row: Mapping[str, Any]) -> str:
        return str(row.get("regime", row.get("market_regime", "UNKNOWN"))).upper()

    @staticmethod
    def _metrics(observations: list[Dict[str, Any]], eta_key: str = "normal") -> Dict[str, Any]:
        if not observations:
            return {"samples": 0, "mae_minutes": None, "median_error_minutes": None, "p90_error_minutes": None, "coverage": None}
        errors = [abs(item["predicted_eta_minutes"][eta_key] - item["actual_elapsed_minutes"]) for item in observations if item["predicted_eta_minutes"].get(eta_key) is not None]
        covered = [item["actual_elapsed_minutes"] <= item["predicted_eta_minutes"][eta_key] for item in observations if item["predicted_eta_minutes"].get(eta_key) is not None]
        ordered = sorted(errors)
        percentile = lambda fraction: ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))] if ordered else None
        return {"samples": len(errors), "mae_minutes": sum(errors) / len(errors) if errors else None, "median_error_minutes": percentile(0.5), "p90_error_minutes": percentile(0.9), "coverage": sum(covered) / len(covered) if covered else None}

    def extract_events(self, frame: Any) -> Dict[str, Any]:
        """Extract explicit start/completion event pairs without fabricating IDs."""
        records = self.pipeline.records(frame) if hasattr(frame, "to_dict") else list(frame or [])
        grouped: Dict[tuple[str, str, str, str, str], list[Dict[str, Any]]] = {}
        unsupported = 0
        for index, row in enumerate(records):
            timestamp = self._timestamp(row)
            event_id = self._event_id(row, index)
            if timestamp is None or event_id is None:
                unsupported += 1
                continue
            key = (*self._group_key(row), event_id)
            grouped.setdefault(key, []).append({**row, "_parsed_timestamp": timestamp})
        events: list[Dict[str, Any]] = []
        for key, rows in grouped.items():
            rows.sort(key=lambda item: item["_parsed_timestamp"])
            start = rows[0]
            completion = next((row for row in rows[1:] if self._complete(row)), None)
            start_gap = self._gap(start)
            if completion is None or start_gap is None or start_gap <= 0:
                unsupported += 1
                continue
            final_gap = self._gap(completion) or 0.0
            elapsed = (completion["_parsed_timestamp"] - start["_parsed_timestamp"]).total_seconds() / 60.0
            progress = max(0.0, start_gap - final_gap)
            if elapsed <= 0 or progress <= 0:
                unsupported += 1
                continue
            events.append({"market": key[0], "symbol": key[1], "segment": key[2], "timeframe": key[3], "event_id": key[4], "start": start, "completion": completion, "initial_gap": start_gap, "final_gap": final_gap, "actual_progress": progress, "actual_elapsed_minutes": elapsed, "actual_speed": progress / elapsed, "period": str(start.get("period", self._period(start["_parsed_timestamp"]))).upper(), "regime": self._regime(start)})
        events.sort(key=lambda event: event["start"]["_parsed_timestamp"])
        return {"events": events, "unsupported_events": unsupported, "input_rows": len(records)}

    def analyze(self, source: Any, *, market: Optional[str] = None, segment: Optional[str] = None, symbol: Optional[str] = None, timeframe: Optional[str] = None, data_source: str = "historical") -> Dict[str, Any]:
        """Analyze all available metadata groups with strict walk-forward profiles."""
        frame = self.pipeline.normalize(source, market=market, segment=segment, symbol=symbol, timeframe=timeframe)
        extracted = self.extract_events(frame)
        prior: Dict[tuple[str, str, str, str], list[Dict[str, Any]]] = {}
        grouped: Dict[str, list[Dict[str, Any]]] = {}
        for event in extracted["events"]:
            group_tuple = (event["market"], event["symbol"], event["segment"], event["timeframe"])
            history = prior.setdefault(group_tuple, [])
            profile_rows = [{"market": event["market"], "segment": event["segment"], "timeframe": event["timeframe"], "timestamp": item["timestamp"], "speed": item["actual_speed"]} for item in history]
            estimate = self.calculator.estimate(event["initial_gap"], event["actual_speed"], profile_rows, event["market"], event["segment"], event["timeframe"], event["start"]["timestamp"])
            if estimate.get("sufficient_history"):
                state = estimate.get("speed_regime", "NORMAL")
                observation = {"market": event["market"], "symbol": event["symbol"], "segment": event["segment"], "timeframe": event["timeframe"], "event_id": event["event_id"], "period": event["period"], "regime": event["regime"], "initial_gap": event["initial_gap"], "final_gap": event["final_gap"], "actual_progress": event["actual_progress"], "actual_speed": event["actual_speed"], "actual_elapsed_minutes": event["actual_elapsed_minutes"], "speed_state": state, "predicted_eta_minutes": {"normal": estimate.get("normal_eta_minutes"), "fast": estimate.get("accelerated_eta_minutes"), "shock": estimate.get("shock_eta_minutes")}, "confidence": estimate.get("confidence", 0.0), "sample_count": estimate.get("sample_count", 0)}
                grouped.setdefault("/".join(group_tuple), []).append(observation)
            history.append({"timestamp": event["start"]["timestamp"], "actual_speed": event["actual_speed"]})
        reports: Dict[str, Any] = {}
        for group_key, observations in grouped.items():
            reports[group_key] = {"samples": len(observations), "reliable": len(observations) >= self.min_samples, "confidence": min(item["confidence"] for item in observations) if observations else 0.0, "normal": self._metrics(observations, "normal"), "fast": self._metrics(observations, "fast"), "shock": self._metrics(observations, "shock"), "by_period": {period: self._metrics([item for item in observations if item["period"] == period]) for period in ("OPENING", "MIDDAY", "CLOSING", "UNKNOWN")}, "by_regime": {regime: self._metrics([item for item in observations if item["regime"] == regime]) for regime in sorted({item["regime"] for item in observations})}, "speed_states": {state: self._metrics([item for item in observations if item["speed_state"] == state]) for state in ("NORMAL", "ACCELERATING", "SHOCK/EXPANSION")}, "eta_distributions": {"normal": [item["predicted_eta_minutes"]["normal"] for item in observations], "fast": [item["predicted_eta_minutes"]["fast"] for item in observations], "shock": [item["predicted_eta_minutes"]["shock"] for item in observations]}, "events": observations}
        report = {"data_source": str(data_source), "historical_only": True, "synthetic_data": str(data_source).lower() == "synthetic", "groups": reports, "input_rows": extracted["input_rows"], "candidate_events": len(extracted["events"]), "validated_events": sum(len(items) for items in grouped.values()), "unsupported_events": extracted["unsupported_events"], "minimum_samples": self.min_samples, "note": "Synthetic results are correctness checks, not market-performance claims." if str(data_source).lower() == "synthetic" else "Walk-forward historical analysis using only prior completed events."}
        json.dumps(report)
        return report

    analyze_historical_data = analyze