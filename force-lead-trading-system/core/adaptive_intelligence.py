"""Historical adaptive-intelligence orchestration for paper-safe predictions."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Mapping, Optional

from core.adaptive_lead_time import AdaptiveLeadTimeCalculator
from core.historical_lead_analyzer import HistoricalLeadAnalyzer
from core.prediction_engine import PredictionEngine
from data.historical_data_manager import HistoricalDataManager


class AdaptiveIntelligence:
    """Compose existing historical, lead-time, and prediction APIs safely."""

    def __init__(
        self,
        historical_manager: Optional[HistoricalDataManager] = None,
        prediction_engine: Optional[PredictionEngine] = None,
        lead_time_calculator: Optional[AdaptiveLeadTimeCalculator] = None,
        lead_analyzer: Optional[HistoricalLeadAnalyzer] = None,
        min_samples: int = 5,
        lookback_days: int = 36500,
    ) -> None:
        self.historical_manager = historical_manager
        self.prediction_engine = prediction_engine or PredictionEngine()
        self.lead_time_calculator = lead_time_calculator or AdaptiveLeadTimeCalculator(
            min_samples=min_samples,
            lookback_days=lookback_days,
        )
        self.lead_analyzer = lead_analyzer or HistoricalLeadAnalyzer(
            calculator=self.lead_time_calculator,
            min_samples=min_samples,
        )
        self.min_samples = max(1, int(min_samples))

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _timestamp(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            result = value
        else:
            try:
                result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return None
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.astimezone(timezone.utc)

    @classmethod
    def _json(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Mapping):
            return {str(key): cls._json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json(item) for item in value]
        if hasattr(value, "item"):
            try:
                return cls._json(value.item())
            except (TypeError, ValueError):
                return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _identity(row: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
        return (
            str(row.get("market", "UNKNOWN")).upper(),
            str(row.get("segment", "OTHER")).upper(),
            str(row.get("symbol", "UNKNOWN")).upper(),
            str(row.get("instrument", row.get("symbol", "UNKNOWN"))).upper(),
            str(row.get("timeframe", row.get("interval", "UNKNOWN"))).lower(),
            str(row.get("timezone", "UTC")).upper(),
        )

    @classmethod
    def _records(cls, source: Any, manager: Optional[HistoricalDataManager], dataset_id: Optional[str]) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
        metadata: Dict[str, Any] = {}
        if manager is not None and dataset_id is not None:
            records = manager.records(dataset_id)
            metadata = manager.quality(dataset_id)
        elif hasattr(source, "to_dict"):
            records = source.to_dict(orient="records")
        else:
            records = list(source or [])
        normalized = [dict(row) for row in records if isinstance(row, Mapping)]
        return normalized, metadata

    @classmethod
    def _prior_and_current(cls, rows: list[Dict[str, Any]], current_timestamp: Any = None) -> tuple[list[Dict[str, Any]], Optional[Dict[str, Any]], Optional[datetime]]:
        ordered = sorted(rows, key=lambda row: cls._timestamp(row.get("timestamp", row.get("datetime", row.get("time"))) or datetime.min.replace(tzinfo=timezone.utc)))
        cutoff = cls._timestamp(current_timestamp) if current_timestamp is not None else None
        if cutoff is not None:
            ordered = [row for row in ordered if cls._timestamp(row.get("timestamp", row.get("datetime", row.get("time")))) is not None and cls._timestamp(row.get("timestamp", row.get("datetime", row.get("time")))) <= cutoff]
        if not ordered:
            return [], None, cutoff
        current = ordered[-1]
        current_time = cls._timestamp(current.get("timestamp", current.get("datetime", current.get("time"))))
        return ordered[:-1], current, current_time

    @classmethod
    def _gap(cls, row: Mapping[str, Any]) -> float:
        for key in ("remaining_volume_gap", "remaining_gap", "volume_gap", "progress_gap", "remaining_progress"):
            value = cls._number(row.get(key))
            if value is not None:
                return max(0.0, value)
        return 0.0

    @classmethod
    def _current_speed(cls, row: Mapping[str, Any]) -> Optional[float]:
        for key in ("current_speed", "volume_per_minute", "speed"):
            value = cls._number(row.get(key))
            if value is not None:
                return max(0.0, value)
        return None

    @staticmethod
    def _analyzer_rows(rows: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
        # HistoricalLeadAnalyzer accepts symbol identity; remove only the field
        # that the legacy pipeline aliases to symbol during normalization.
        return [{key: value for key, value in row.items() if key != "instrument"} for row in rows]

    def _validation(self, rows: list[Dict[str, Any]], identity: tuple[str, str, str, str, str, str], data_source: str) -> Dict[str, Any]:
        market, segment, symbol, _instrument, timeframe, _timezone = identity
        try:
            lead_report = self.lead_time_calculator.validate_historical_events(
                rows, market=market, segment=segment, timeframe=timeframe, data_source=data_source,
            )
        except Exception as exc:
            lead_report = {"historical_only": True, "reports": {}, "validated_events": 0, "error": type(exc).__name__}
        try:
            analyzer_report = self.lead_analyzer.analyze(
                self._analyzer_rows(rows), market=market, segment=segment, symbol=symbol,
                timeframe=timeframe, data_source=data_source,
            )
        except Exception as exc:
            analyzer_report = {"historical_only": True, "groups": {}, "validated_events": 0, "error": type(exc).__name__}
        return {"adaptive_lead": lead_report, "historical_lead": analyzer_report}

    def analyze(
        self,
        source: Any = None,
        *,
        dataset_id: Optional[str] = None,
        current_timestamp: Any = None,
        data_source: str = "historical",
    ) -> Dict[str, Any]:
        """Analyze isolated normalized history without using rows after the cutoff."""
        rows, metadata = self._records(source, self.historical_manager, dataset_id)
        groups: Dict[tuple[str, str, str, str, str, str], list[Dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(self._identity(row), []).append(row)

        reports: Dict[str, Any] = {}
        for identity, group_rows in sorted(groups.items()):
            cutoff = self._timestamp(current_timestamp) if current_timestamp is not None else None
            visible_rows = [
                row for row in group_rows
                if cutoff is None or (timestamp := self._timestamp(row.get("timestamp", row.get("datetime", row.get("time"))))) is not None and timestamp <= cutoff
            ]
            prior, current, prediction_timestamp = self._prior_and_current(visible_rows, current_timestamp)
            market, segment, symbol, instrument, timeframe, timezone_name = identity
            profile = self.lead_time_calculator.build_profiles(prior, prediction_timestamp)
            estimate = self.lead_time_calculator.estimate(
                self._gap(current or {}),
                self._current_speed(current or {}),
                prior,
                market,
                segment,
                timeframe,
                prediction_timestamp,
            )
            validation = self._validation(visible_rows, identity, data_source) if visible_rows else {}
            group_key = "/".join(identity)
            reports[group_key] = {
                "identity": {"market": market, "segment": segment, "symbol": symbol, "instrument": instrument, "timeframe": timeframe, "timezone": timezone_name},
                "historical_rows": len(visible_rows),
                "training_rows": len(prior),
                "prediction_timestamp": self._json(prediction_timestamp),
                "profile": profile.get(market, {}).get(segment, {}).get(timeframe, {}),
                "adaptive_estimate": estimate,
                "validation": validation,
                "sample_quality": {
                    "sample_count": estimate.get("sample_count", 0),
                    "sufficient_history": bool(estimate.get("sufficient_history", False)),
                    "confidence": estimate.get("confidence", 0.0),
                    "stability": estimate.get("stability", 0.0),
                },
                "future_rows_excluded": max(0, len(group_rows) - len(visible_rows)),
            }

        result = {
            "historical_only": True,
            "paper_simulation_only": True,
            "data_source": str(data_source),
            "synthetic_data": str(data_source).lower() == "synthetic",
            "dataset_id": dataset_id,
            "metadata": metadata,
            "reports": reports,
            "groups": len(reports),
            "rows": len(rows),
            "note": "Synthetic results are code-validation only; estimates are not guaranteed." if str(data_source).lower() == "synthetic" else "Historical analysis uses only data available before each prediction timestamp.",
        }
        return self._json(result)

    def predict(
        self,
        current_data: Mapping[str, Any],
        historical_data: Any = None,
        *,
        dataset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the existing PredictionEngine with a strictly prior history prefix."""
        rows, _metadata = self._records(historical_data, self.historical_manager, dataset_id)
        current_time = self._timestamp(current_data.get("timestamp", current_data.get("datetime", current_data.get("time"))))
        prior = [row for row in rows if (timestamp := self._timestamp(row.get("timestamp", row.get("datetime", row.get("time"))))) is not None and (current_time is None or timestamp < current_time)]
        identity = self._identity(current_data)
        result = self.prediction_engine.predict(
            dict(current_data), prior,
            market=identity[0], segment=identity[1], timeframe=identity[4],
        )
        result = dict(result) if isinstance(result, Mapping) else {}
        result["historical_intelligence"] = {
            "training_samples": len(prior),
            "future_rows_excluded": len(rows) - len(prior),
            "prediction_timestamp": self._json(current_time),
            "confidence": result.get("confidence", 0.0),
            "paper_simulation_only": True,
        }
        return self._json(result)

    analyze_history = analyze
    build_intelligence = analyze
    calculate = predict


AdaptiveIntelligenceEngine = AdaptiveIntelligence
