"""Historical, segment-aware lead-time estimation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, Iterable, Mapping, Optional

from config.markets import MARKETS


TIMEFRAME_MINUTES = {
    "1m": 1.0, "3m": 3.0, "5m": 5.0, "15m": 15.0,
    "30m": 30.0, "1h": 60.0, "4h": 240.0, "1d": 1440.0,
}


class AdaptiveLeadTimeEngine:
    """Build historical speed profiles and estimate a live lead window."""

    def __init__(self, min_samples: int = 5, lookback_days: int = 180) -> None:
        self.min_samples = max(1, int(min_samples))
        self.lookback_days = max(1, int(lookback_days))
        self.profiles: Dict[str, Dict[str, Dict[str, Any]]] = {}

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _timestamp(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            result = value
        elif isinstance(value, (int, float)):
            try:
                result = datetime.fromtimestamp(value, timezone.utc)
            except (OverflowError, OSError, ValueError):
                return None
        elif isinstance(value, str):
            try:
                result = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        else:
            return None
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.astimezone(timezone.utc)

    @classmethod
    def _minutes(cls, timeframe: Any) -> Optional[float]:
        key = str(timeframe or "").strip().lower()
        if key in TIMEFRAME_MINUTES:
            return TIMEFRAME_MINUTES[key]
        if key.endswith("m"):
            return cls._number(key[:-1])
        if key.endswith("h"):
            hours = cls._number(key[:-1])
            return hours * 60.0 if hours is not None else None
        if key in {"d", "1day"}:
            return 1440.0
        return None

    @classmethod
    def _timeframe(cls, row: Mapping[str, Any], fallback: Any = None) -> Optional[str]:
        raw = row.get("timeframe", row.get("interval", row.get("tf", fallback)))
        if raw is None:
            return None
        text = str(raw).strip().lower().replace("day", "d")
        return text if cls._minutes(text) is not None else None

    @staticmethod
    def _segment_name(segment: Any) -> str:
        return str(segment or "OTHER").strip().upper() or "OTHER"

    @classmethod
    def _iter_rows(cls, data: Any) -> Iterable[tuple[Dict[str, Any], Optional[str]]]:
        if isinstance(data, Mapping):
            for key, value in data.items():
                if isinstance(value, list):
                    for row in value:
                        if isinstance(row, Mapping):
                            yield dict(row), str(key)
                elif isinstance(value, Mapping):
                    yield from cls._iter_rows(value)
        elif isinstance(data, list):
            for row in data:
                if isinstance(row, Mapping):
                    yield dict(row), None

    @classmethod
    def _speed(cls, row: Mapping[str, Any], previous: Optional[Mapping[str, Any]], minutes: float) -> Optional[float]:
        explicit_speed = cls._number(row.get("speed", row.get("volume_per_minute")))
        if explicit_speed is not None:
            return max(0.0, explicit_speed)
        current = cls._number(row.get("volume", row.get("progress_volume", row.get("underlying_volume", row.get("option_volume")))))
        if current is None:
            current = cls._number(row.get("progress", row.get("progress_delta", row.get("oi_change", row.get("open_interest_change")))))
        if current is None:
            return None
        cumulative = any(key in row for key in ("cumulative_volume", "total_volume", "cumulative_progress"))
        if cumulative and previous is not None:
            prior = cls._number(previous.get("cumulative_volume", previous.get("total_volume", previous.get("cumulative_progress"))))
            if prior is not None:
                current -= prior
        return max(0.0, current) / max(minutes, 1.0)

    @classmethod
    def _event_key(cls, row: Mapping[str, Any], fallback_timeframe: Optional[str]) -> tuple[str, str]:
        return cls._segment_name(row.get("segment", row.get("market_segment"))), cls._timeframe(row, fallback_timeframe) or "unknown"

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    @staticmethod
    def _stability(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        middle = len(ordered) // 2
        centre = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
        if len(values) < 2 or centre <= 0:
            return 0.0
        mean = sum(values) / len(values)
        deviation = math.sqrt(sum((item - mean) ** 2 for item in values) / len(values))
        return max(0.0, min(1.0, 1.0 / (1.0 + deviation / centre)))

    @staticmethod
    def _reliable_max(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        centre = AdaptiveLeadTimeEngine._percentile(ordered, 0.50)
        deviations = sorted(abs(item - centre) for item in ordered)
        mad = AdaptiveLeadTimeEngine._percentile(deviations, 0.50)
        robust_limit = centre + max(3.0 * mad, 3.0 * centre)
        robust = [item for item in ordered if item <= robust_limit]
        if robust and len(robust) < len(ordered):
            return max(robust)
        if len(ordered) >= 4:
            q1 = AdaptiveLeadTimeEngine._percentile(ordered, 0.25)
            q3 = AdaptiveLeadTimeEngine._percentile(ordered, 0.75)
            inner = [item for item in ordered if item <= q3 + 1.5 * (q3 - q1)]
            if inner and len(inner) < len(ordered):
                return max(inner)
        upper = AdaptiveLeadTimeEngine._percentile(ordered, 0.99)
        return max(item for item in values if item <= upper)

    @staticmethod
    def _outlier_count(values: list[float]) -> int:
        if len(values) < 3:
            return 0
        ordered = sorted(values)
        centre = AdaptiveLeadTimeEngine._percentile(ordered, 0.50)
        deviations = sorted(abs(item - centre) for item in ordered)
        mad = AdaptiveLeadTimeEngine._percentile(deviations, 0.50)
        robust_limit = centre + max(3.0 * mad, 3.0 * centre)
        if any(item > robust_limit for item in values):
            return sum(item > robust_limit for item in values)
        q1 = AdaptiveLeadTimeEngine._percentile(ordered, 0.25)
        q3 = AdaptiveLeadTimeEngine._percentile(ordered, 0.75)
        upper = q3 + 1.5 * (q3 - q1)
        lower = max(0.0, q1 - 1.5 * (q3 - q1))
        return sum(item < lower or item > upper for item in values)

    @staticmethod
    def _normal_values(values: list[float]) -> list[float]:
        if len(values) < 3:
            return list(values)
        ordered = sorted(values)
        centre = AdaptiveLeadTimeEngine._percentile(ordered, 0.50)
        deviations = sorted(abs(item - centre) for item in ordered)
        mad = AdaptiveLeadTimeEngine._percentile(deviations, 0.50)
        limit = centre + max(3.0 * mad, 3.0 * centre)
        normal = [item for item in ordered if item <= limit]
        return normal if normal else ordered

    @classmethod
    def _speed_profile(cls, speeds: list[float]) -> Dict[str, Any]:
        """Summarize speeds while retaining rare extremes as observed evidence."""
        if not speeds:
            return {"sample_count": 0, "median_speed": 0.0, "p75_speed": 0.0, "p90_speed": 0.0, "p95_speed": 0.0, "maximum_reliable_speed": 0.0, "outlier_count": 0, "stability": 0.0, "sufficient_history": False}
        return {
            "sample_count": len(speeds),
            "median_speed": cls._percentile(cls._normal_values(speeds), 0.50),
            "p75_speed": cls._percentile(speeds, 0.75),
            "p90_speed": cls._percentile(speeds, 0.90),
            "p95_speed": cls._percentile(speeds, 0.95),
            "maximum_reliable_speed": cls._reliable_max(speeds),
            "outlier_count": cls._outlier_count(speeds),
            "stability": cls._stability(speeds),
            "sufficient_history": len(speeds) >= 1,
        }

    @staticmethod
    def _period_bucket(timestamp: Optional[datetime], latest: Optional[datetime]) -> str:
        if timestamp is None or latest is None:
            return "unknown"
        age = (latest - timestamp).total_seconds() / 86400.0
        if age <= 90:
            return "recent"
        if age <= 365:
            return "medium_term"
        return "long_term"

    @classmethod
    def _volume_condition(cls, row: Mapping[str, Any], volume_median: Optional[float]) -> str:
        explicit = row.get("volume_condition", row.get("volume_regime", row.get("liquidity")))
        if explicit not in (None, ""):
            return str(explicit).upper()
        volume = cls._number(row.get("volume", row.get("underlying_volume", row.get("option_volume"))))
        if volume is None or volume_median is None or volume_median <= 0:
            return "UNKNOWN"
        if volume >= volume_median * 1.5:
            return "HIGH"
        if volume <= volume_median / 1.5:
            return "LOW"
        return "NORMAL"

    def build_profiles(self, historical_data: Any, current_timestamp: Any = None) -> Dict[str, Any]:
        """Build profiles independently for each market, segment and timeframe."""
        cutoff = self._timestamp(current_timestamp) if current_timestamp is not None else None
        lower_cutoff = cutoff - timedelta(days=self.lookback_days) if cutoff is not None else None
        grouped: Dict[tuple[str, str, str], list[float]] = {}
        observations: Dict[tuple[str, str, str], list[tuple[float, Optional[datetime], Dict[str, Any]]]] = {}
        previous: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        rows = list(self._iter_rows(historical_data))
        rows.sort(key=lambda item: self._timestamp(item[0].get("timestamp", item[0].get("datetime", item[0].get("time")))) or datetime.min.replace(tzinfo=timezone.utc))
        for row, grouped_timeframe in rows:
            timestamp = self._timestamp(row.get("timestamp", row.get("datetime", row.get("time"))))
            if cutoff is not None:
                if timestamp is None or timestamp > cutoff:
                    continue
                if lower_cutoff is not None and timestamp < lower_cutoff:
                    continue
            segment, timeframe = self._event_key(row, grouped_timeframe)
            if timeframe == "unknown":
                continue
            market = str(row.get("market", "UNKNOWN")).strip().upper() or "UNKNOWN"
            key = (market, segment, timeframe)
            speed = self._speed(row, previous.get(key), self._minutes(timeframe) or 1.0)
            previous[key] = row
            if speed is not None and speed > 0:
                grouped.setdefault(key, []).append(speed)
                observations.setdefault(key, []).append((speed, timestamp, row))

        result: Dict[str, Any] = {}
        for (market, segment, timeframe), speeds in grouped.items():
            grouped_observations = observations[(market, segment, timeframe)]
            latest = max((item[1] for item in grouped_observations if item[1] is not None), default=None)
            volume_values = [self._number(item[2].get("volume", item[2].get("underlying_volume", item[2].get("option_volume")))) for item in grouped_observations]
            volume_values = [value for value in volume_values if value is not None and value > 0]
            volume_median = self._percentile(volume_values, 0.50) if volume_values else None
            profile = {
                "market": market,
                "segment": segment,
                "timeframe": timeframe,
                "lookback_days": self.lookback_days,
                **self._speed_profile(speeds),
            }
            profile["sufficient_history"] = len(speeds) >= self.min_samples
            periods = {name: [] for name in ("recent", "medium_term", "long_term", "full_history")}
            conditions: Dict[str, list[float]] = {}
            for speed_value, timestamp, row in grouped_observations:
                periods["full_history"].append(speed_value)
                periods[self._period_bucket(timestamp, latest)].append(speed_value)
                if timestamp is not None:
                    conditions.setdefault(f"time_of_day:{self._period(timestamp, row)}", []).append(speed_value)
                regime = self._regime(row)
                if regime != "UNKNOWN":
                    conditions.setdefault(f"regime:{regime}", []).append(speed_value)
                volume_condition = self._volume_condition(row, volume_median)
                if volume_condition != "UNKNOWN":
                    conditions.setdefault(f"volume:{volume_condition}", []).append(speed_value)
            profile["period_profiles"] = {name: self._speed_profile(values) for name, values in periods.items()}
            profile["condition_profiles"] = {name: self._speed_profile(values) for name, values in sorted(conditions.items())}
            result.setdefault(market, {}).setdefault(segment, {})[timeframe] = profile
        self.profiles = result
        return result

    @classmethod
    def _event_id(cls, row: Mapping[str, Any], index: int) -> str:
        value = row.get("event_id", row.get("lead_event_id", row.get("event")))
        return str(value) if value not in (None, "") else f"row-{index}"

    @classmethod
    def _event_gap(cls, row: Mapping[str, Any]) -> Optional[float]:
        for key in ("remaining_volume_gap", "remaining_gap", "volume_gap", "progress_gap", "remaining_progress"):
            value = cls._number(row.get(key))
            if value is not None:
                return max(0.0, value)
        return None

    @classmethod
    def _is_complete(cls, row: Mapping[str, Any]) -> bool:
        if row.get("completed", row.get("event_complete", row.get("is_complete", False))) is True:
            return True
        gap = cls._event_gap(row)
        return gap is not None and gap <= 0.0

    @staticmethod
    def _period(timestamp: Optional[datetime], row: Mapping[str, Any]) -> str:
        explicit = row.get("period", row.get("session_period"))
        if explicit:
            return str(explicit).upper()
        if timestamp is None:
            return "UNKNOWN"
        hour = timestamp.hour + timestamp.minute / 60.0
        if hour < 11.0:
            return "OPENING"
        if hour < 14.0:
            return "MIDDAY"
        return "CLOSING"

    @staticmethod
    def _regime(row: Mapping[str, Any]) -> str:
        value = row.get("regime", row.get("market_regime"))
        return str(value).upper() if value else "UNKNOWN"

    def _validation_profile(self, speeds: list[float]) -> Dict[str, Any]:
        if not speeds:
            return {"sample_count": 0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "maximum_reliable": 0.0, "outlier_count": 0, "stability": 0.0}
        ordered = sorted(speeds)
        upper = self._percentile(ordered, 0.99)
        reliable = [value for value in ordered if value <= upper]
        normal = self._normal_values(speeds)
        return {
            "sample_count": len(speeds),
            "median": self._percentile(normal, 0.50),
            "p75": self._percentile(ordered, 0.75),
            "p90": self._percentile(ordered, 0.90),
            "p95": self._percentile(ordered, 0.95),
            "maximum_reliable": max(reliable) if reliable else 0.0,
            "outlier_count": self._outlier_count(speeds),
            "stability": self._stability(speeds),
        }

    @staticmethod
    def _error_metrics(errors: list[float], actuals: list[float], windows: list[Optional[float]]) -> Dict[str, Any]:
        if not errors:
            return {"mae_minutes": None, "median_error_minutes": None, "p90_error_minutes": None, "coverage": None, "samples": 0}
        covered = [actual <= window for actual, window in zip(actuals, windows) if window is not None]
        return {
            "mae_minutes": sum(errors) / len(errors),
            "median_error_minutes": AdaptiveLeadTimeEngine._percentile(errors, 0.50),
            "p90_error_minutes": AdaptiveLeadTimeEngine._percentile(errors, 0.90),
            "coverage": sum(covered) / len(covered) if covered else None,
            "samples": len(errors),
        }

    def validate_historical_events(self, historical_data: Any, market: Optional[str] = None, segment: Optional[str] = None, timeframe: Optional[str] = None, data_source: str = "historical") -> Dict[str, Any]:
        """Evaluate ETAs with walk-forward, pre-event historical profiles only.

        Rows must provide timestamps and an explicit event id plus an initial
        gap and completion marker (or zero final gap). Rows without those
        fields are reported as unsupported rather than being treated as a
        market-performance observation.
        """
        raw_rows = list(self._iter_rows(historical_data))
        invalid_timestamp_rows = sum(1 for row, _ in raw_rows if self._timestamp(row.get("timestamp", row.get("datetime", row.get("time")))) is None)
        rows = [(row, grouped_tf) for row, grouped_tf in raw_rows if self._timestamp(row.get("timestamp", row.get("datetime", row.get("time")))) is not None]
        rows.sort(key=lambda item: self._timestamp(item[0].get("timestamp", item[0].get("datetime", item[0].get("time")))))
        events: Dict[tuple[str, str, str, str], list[tuple[datetime, Dict[str, Any]]]] = {}
        unsupported = 0
        for index, (row, grouped_tf) in enumerate(rows):
            event_key = self._event_key(row, grouped_tf)
            event_market = str(row.get("market", "UNKNOWN")).upper()
            key = (event_market, event_key[0], event_key[1], self._event_id(row, index))
            events.setdefault(key, []).append((self._timestamp(row.get("timestamp", row.get("datetime", row.get("time")))), row))

        observations: list[Dict[str, Any]] = []
        prior_speeds: Dict[tuple[str, str, str], list[float]] = {}
        candidate_groups: set[str] = set()
        for (event_market, event_segment, event_timeframe, event_name), event_rows in events.items():
            if ((market and event_market != str(market).upper()) or (segment and event_segment != self._segment_name(segment)) or (timeframe and event_timeframe != self._timeframe({}, timeframe))):
                continue
            candidate_groups.add(f"{event_market}/{event_segment}/{event_timeframe}")
            event_rows.sort(key=lambda item: item[0])
            start_time, start_row = event_rows[0]
            start_gap = self._event_gap(start_row)
            completion = next(((timestamp, row) for timestamp, row in event_rows[1:] if self._is_complete(row)), None)
            if start_gap is None or completion is None or completion[0] <= start_time or start_gap <= 0.0:
                unsupported += 1
                continue
            elapsed = (completion[0] - start_time).total_seconds() / 60.0
            final_gap = self._event_gap(completion[1]) or 0.0
            actual_progress = max(0.0, start_gap - final_gap)
            actual_speed = actual_progress / elapsed if elapsed > 0 else 0.0
            key = (event_market, event_segment, event_timeframe)
            profile = self._validation_profile(prior_speeds.get(key, []))
            if profile["sample_count"] >= self.min_samples and profile["median"] > 0:
                etas = {name: start_gap / profile[name] if profile[name] > 0 else None for name in ("median", "p90", "maximum_reliable")}
                errors = {name: abs(etas[name] - elapsed) for name in etas if etas[name] is not None}
                speed_state = "SHOCK/EXPANSION" if actual_speed >= profile["p95"] else "ACCELERATING" if actual_speed >= profile["p75"] else "NORMAL"
                observations.append({"market": event_market, "segment": event_segment, "timeframe": event_timeframe, "event_id": event_name, "period": self._period(start_time, start_row), "regime": self._regime(start_row), "initial_gap": start_gap, "final_gap": final_gap, "actual_progress": actual_progress, "actual_elapsed_minutes": elapsed, "actual_speed": actual_speed, "predicted_eta_minutes": {"normal": etas["median"], "fast": etas["p90"], "shock": etas["maximum_reliable"]}, "errors_minutes": errors, "speed_state": speed_state, "profile_sample_count": profile["sample_count"]})
            prior_speeds.setdefault(key, []).append(actual_speed)

        groups: Dict[str, list[Dict[str, Any]]] = {}
        for observation in observations:
            group_key = f"{observation['market']}/{observation['segment']}/{observation['timeframe']}"
            groups.setdefault(group_key, []).append(observation)
        reports = {}
        for group_key in sorted(candidate_groups):
            group = groups.get(group_key, [])
            if not group:
                reports[group_key] = {"samples": 0, "reliable": False, "confidence": 0.0, "overall": self._error_metrics([], [], []), "by_speed_state": {state: self._error_metrics([], [], []) for state in ("NORMAL", "ACCELERATING", "SHOCK/EXPANSION")}, "by_period": {period: self._error_metrics([], [], []) for period in ("OPENING", "MIDDAY", "CLOSING", "UNKNOWN")}, "by_regime": {}, "speed_statistics": self._validation_profile([]), "events": [], "insufficient_history": True}
                continue
            errors_by_state: Dict[str, list[float]] = {"NORMAL": [], "ACCELERATING": [], "SHOCK/EXPANSION": []}
            actuals_by_state: Dict[str, list[float]] = {key: [] for key in errors_by_state}
            windows_by_state: Dict[str, list[Optional[float]]] = {key: [] for key in errors_by_state}
            for item in group:
                state = item["speed_state"]
                errors_by_state[state].append(item["errors_minutes"]["median"])
                actuals_by_state[state].append(item["actual_elapsed_minutes"])
                windows_by_state[state].append(item["predicted_eta_minutes"]["normal"])
            all_errors = errors_by_state["NORMAL"] + errors_by_state["ACCELERATING"] + errors_by_state["SHOCK/EXPANSION"]
            all_actuals = actuals_by_state["NORMAL"] + actuals_by_state["ACCELERATING"] + actuals_by_state["SHOCK/EXPANSION"]
            all_windows = windows_by_state["NORMAL"] + windows_by_state["ACCELERATING"] + windows_by_state["SHOCK/EXPANSION"]
            sample_count = len(group)
            reports[group_key] = {"samples": sample_count, "reliable": sample_count >= self.min_samples, "confidence": round(min(1.0, sample_count / self.min_samples) * self._stability([item["actual_speed"] for item in group]), 4), "overall": self._error_metrics(all_errors, all_actuals, all_windows), "by_speed_state": {state: self._error_metrics(errors_by_state[state], actuals_by_state[state], windows_by_state[state]) for state in errors_by_state}, "by_period": {period: self._error_metrics([item["errors_minutes"]["median"] for item in group if item["period"] == period], [item["actual_elapsed_minutes"] for item in group if item["period"] == period], [item["predicted_eta_minutes"]["normal"] for item in group if item["period"] == period]) for period in ("OPENING", "MIDDAY", "CLOSING", "UNKNOWN")}, "by_regime": {regime: self._error_metrics([item["errors_minutes"]["median"] for item in group if item["regime"] == regime], [item["actual_elapsed_minutes"] for item in group if item["regime"] == regime], [item["predicted_eta_minutes"]["normal"] for item in group if item["regime"] == regime]) for regime in sorted({item["regime"] for item in group})}, "speed_statistics": self._validation_profile([item["actual_speed"] for item in group]), "events": group}

        return {"data_source": str(data_source), "historical_only": True, "synthetic_data": str(data_source).lower() == "synthetic", "lookback_days": self.lookback_days, "minimum_samples": self.min_samples, "reports": reports, "unsupported_events": unsupported, "invalid_timestamp_rows": invalid_timestamp_rows, "total_events": len(events), "validated_events": len(observations), "note": "Synthetic results are code-correctness checks, not market-performance claims." if str(data_source).lower() == "synthetic" else "Walk-forward historical validation; each event uses only earlier completed events."}

    validate_historical_data = validate_historical_events
    validate = validate_historical_events

    @staticmethod
    def configured_timeframes(market: Optional[str] = None, segment: Optional[str] = None) -> Dict[str, Dict[str, list[str]]]:
        """Return configured market/segment timeframe support without mutating config."""
        result: Dict[str, Dict[str, list[str]]] = {}
        markets = [str(market).upper()] if market else list(MARKETS)
        for market_name in markets:
            market_data = MARKETS.get(market_name, {})
            for segment_name, segment_data in market_data.get("segments", {}).items():
                if segment and segment_name != str(segment).upper():
                    continue
                result.setdefault(market_name, {})[segment_name] = [
                    timeframe.lower().replace("1d", "1d")
                    for timeframe in segment_data.get("timeframes", [])
                    if AdaptiveLeadTimeEngine._minutes(timeframe) is not None
                ]
        return result

    def estimate(self, remaining_volume_gap: Any, current_speed: Any = None, historical_data: Any = None, market: str = "UNKNOWN", segment: str = "OTHER", timeframe: str = "5m", current_timestamp: Any = None, required_speed: Any = None) -> Dict[str, Any]:
        """Return a continuously recomputable, JSON-safe historical estimate."""
        timeframe = self._timeframe({}, timeframe) or "5m"
        market, segment = str(market or "UNKNOWN").upper(), self._segment_name(segment)
        if historical_data is not None:
            self.build_profiles(historical_data, current_timestamp)
        profile = self.profiles.get(market, {}).get(segment, {}).get(timeframe)
        gap = max(0.0, self._number(remaining_volume_gap) or 0.0)
        speed = max(0.0, self._number(current_speed) or 0.0)
        required = self._number(required_speed)
        if required is None:
            required = gap / (self._minutes(timeframe) or 1.0) if gap else 0.0
        profile = profile or {"market": market, "segment": segment, "timeframe": timeframe, "sample_count": 0, "lookback_days": self.lookback_days, "median_speed": 0.0, "p75_speed": 0.0, "p90_speed": 0.0, "p95_speed": 0.0, "maximum_reliable_speed": 0.0, "stability": 0.0, "sufficient_history": False}
        recent_profile = profile.get("period_profiles", {}).get("recent")
        if recent_profile and recent_profile.get("sample_count", 0) >= self.min_samples:
            profile = {**profile, **recent_profile}
            profile["sufficient_history"] = recent_profile["sample_count"] >= self.min_samples
        normal_speed = speed or profile["median_speed"]
        accelerated_speed, shock_speed = profile["p90_speed"], profile["maximum_reliable_speed"]
        eta = lambda rate: gap / rate if rate > 0 else None
        ratio = speed / normal_speed if normal_speed > 0 else 0.0
        shock_ratio = shock_speed / normal_speed if normal_speed > 0 else float("inf")
        regime = "SHOCK/EXPANSION" if shock_speed > accelerated_speed and speed >= shock_speed else "ACCELERATING" if speed >= accelerated_speed and accelerated_speed > normal_speed or ratio >= 1.25 else "NORMAL"
        sample_factor = min(1.0, profile["sample_count"] / max(self.min_samples, 1))
        similarity = max(0.0, min(1.0, ratio / 2.0)) if speed and normal_speed else 0.5
        confidence = round(sample_factor * (0.5 + 0.5 * similarity) * profile["stability"], 4)
        normal_eta, accelerated_eta, shock_eta = eta(normal_speed), eta(accelerated_speed), eta(shock_speed)
        minutes = self._minutes(timeframe) or 1.0
        return {
            "market": market, "segment": segment, "timeframe": timeframe,
            "status": regime, "speed_regime": regime, "historical_estimate": True,
            "model_note": "Historical/model-estimated lead window; shortest ETA is not guaranteed.",
            "remaining_volume_gap": gap, "current_speed": speed, "required_speed": max(0.0, required),
            "normal_eta_minutes": normal_eta, "accelerated_eta_minutes": accelerated_eta, "shock_eta_minutes": shock_eta,
            "estimated_lead_window_bars": {"normal": normal_eta / minutes if normal_eta is not None else None, "accelerated": accelerated_eta / minutes if accelerated_eta is not None else None, "shock": shock_eta / minutes if shock_eta is not None else None},
            "historical_speed_percentiles": {"median": profile["median_speed"], "p75": profile["p75_speed"], "p90": profile["p90_speed"], "p95": profile["p95_speed"], "maximum_reliable": profile["maximum_reliable_speed"]},
            "sample_count": profile["sample_count"], "lookback_days": profile["lookback_days"], "similarity": round(similarity, 4), "stability": profile["stability"], "confidence": confidence, "sufficient_history": profile["sufficient_history"],
        }

    calculate = estimate
    build_historical_profiles = build_profiles


AdaptiveLeadTimeCalculator = AdaptiveLeadTimeEngine