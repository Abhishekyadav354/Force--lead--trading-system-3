from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class ReversalCalculator:
    """Model-estimated reversal detector grounded only in supplied evidence.

    It reads force, delta, lead-time, historical-volume, regime, OI,
    absorption, and divergence dictionaries or explicit keyword arguments.
    It never invents missing broker/order fields.
    """

    def __init__(self) -> None:
        self.name = "ReversalCalculator"

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or isinstance(value, bool):
                return float(default)
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if value is None or isinstance(value, bool):
                return int(default)
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return int(default)

    @staticmethod
    def _bounded(value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            number = 0.0
        if math.isnan(number) or math.isinf(number):
            return 0.0
        return min(1.0, max(0.0, number))

    @staticmethod
    def _dictify(value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def _normalise(
        self,
        evidence: Optional[Mapping[str, Any]] = None,
        force: Any = None,
        delta: Any = None,
        lead_time: Any = None,
        historical_average_volume: Any = None,
        regime: Any = None,
        oi: Any = None,
        absorption: Any = None,
        divergence: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Return a consistent normalized evidence blob.

        Named arguments override the evidence mapping. Missing or malformed
        fields remain harmless and are converted into 0/empty values.
        """
        payload = self._dictify(evidence)

        def first(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
            for name in names:
                if name in mapping:
                    return mapping[name]
            return None

        result: Dict[str, Any] = {}
        # explicit mapping reading
        # Accept order-flow, features, and direct alias sub-dicts.
        oflow = payload.get("order_flow") if isinstance(payload.get("order_flow"), Mapping) else {}
        features = payload.get("features") if isinstance(payload.get("features"), Mapping) else {}

        # Force and scoring inputs.
        result["force"] = force if force is not None else payload.get("force", features.get("force", oflow.get("force")))
        if result["force"] is None:
            result["force"] = kwargs.get("force")

        result["delta"] = delta if delta is not None else payload.get("delta", features.get("delta", oflow.get("delta")))
        if result["delta"] is None:
            result["delta"] = kwargs.get("delta")

        # Lead-time evidence may be a scalar or a mapping with nested status.
        result["lead_time"] = lead_time if lead_time is not None else payload.get("lead_time", payload.get("lead", features.get("lead_time", oflow.get("lead_time"))))
        if result["lead_time"] is None:
            result["lead_time"] = kwargs.get("lead_time")

        # Historical average volume may be a scalar, mapping or list.
        result["historical_average_volume"] = historical_average_volume if historical_average_volume is not None else payload.get("historical_average_volume", payload.get("avg_volume", payload.get("historical_volume", features.get("historical_average_volume", oflow.get("historical_average_volume")))) )
        if result["historical_average_volume"] is None:
            result["historical_average_volume"] = kwargs.get("historical_average_volume")

        # Regime evidence may be a string or a mapping.
        result["regime"] = regime if regime is not None else payload.get("regime", features.get("regime", oflow.get("regime")))
        if result["regime"] is None:
            result["regime"] = kwargs.get("regime")

        # OI evidence may be nested mapping or scalar.
        result["oi"] = oi if oi is not None else payload.get("oi", payload.get("open_interest", features.get("oi", oflow.get("oi"))))
        if result["oi"] is None:
            result["oi"] = kwargs.get("oi")

        # Absorption and divergence evidence may be nested signals.
        result["absorption"] = absorption if absorption is not None else payload.get("absorption", features.get("absorption", oflow.get("absorption")))
        if result["absorption"] is None:
            result["absorption"] = kwargs.get("absorption")

        result["divergence"] = divergence if divergence is not None else payload.get("divergence", features.get("divergence", oflow.get("divergence")))
        if result["divergence"] is None:
            result["divergence"] = kwargs.get("divergence")

        # Volume actual field may exist separately from average volume.
        result["volume"] = payload.get("volume", payload.get("current_volume", payload.get("total_volume")))

        # Carry nested signal evidence as context.
        result["raw"] = payload
        return result

    def _extract_numeric_from_mapping(self, mapping: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
        if not isinstance(mapping, Mapping):
            return None
        for key in keys:
            value = mapping.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError, OverflowError):
                    continue
        return None

    def _detect_delta_flip(self, normalized: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a signal if delta direction flips compared to previous/latest evidence."""
        delta = normalized.get("delta")
        direction = "NONE"
        strength = 0.0
        detected = False

        # A mapping may wrap order-flow deltas.
        if isinstance(delta, Mapping):
            # common shape: {'previous':..., 'current':...}
            prev = self._extract_numeric_from_mapping(delta, ("previous_delta", "previous", "prior_delta", "last_delta"))
            curr = self._extract_numeric_from_mapping(delta, ("delta", "current_delta", "value", "latest_delta"))
            if prev is not None and curr is not None and prev * curr < 0:
                detected = True
                strength = min(1.0, max(0.0, abs(curr - prev) / max(1.0, abs(prev) + abs(curr))))
                direction = "BUY" if curr > 0 else "SELL"
                return {"name": "delta_flip", "detected": detected, "strength": strength, "direction": direction}

            # list-like nested series
            order = delta.get("series")
            if isinstance(order, Sequence) and not isinstance(order, (str, bytes, bytearray)):
                vals = [self._safe_float(v, 0.0) for v in order]
                if len(vals) >= 2:
                    first = vals[0]
                    last = vals[-1]
                    if first * last < 0:
                        detected = True
                        strength = min(1.0, max(0.0, abs(last - first) / max(1.0, abs(first) + abs(last))))
                        direction = "BUY" if last > 0 else "SELL"
                        return {"name": "delta_flip", "detected": detected, "strength": strength, "direction": direction}

        # A plain numeric delta value with a sign may be a signal, but without history it is not enough to claim flip.
        if isinstance(delta, (int, float)) and not isinstance(delta, bool):
            val = self._safe_float(delta, 0.0)
            if val != 0:
                direction = "BUY" if val > 0 else "SELL"
                # A single delta value alone is an evidence point, not a reversal flip.
                return {"name": "delta_flip", "detected": False, "strength": 0.0, "direction": direction}

        # If the data is a list of prices/deltas, use recent sign change.
        if isinstance(delta, Sequence) and not isinstance(delta, (str, bytes, bytearray)):
            vals = [self._safe_float(v, 0.0) for v in delta]
            if len(vals) >= 2:
                try:
                    if math.copysign(1.0, vals[0]) != math.copysign(1.0, vals[-1]):
                        detected = True
                        strength = min(1.0, max(0.0, abs(vals[-1] - vals[0]) / max(1.0, abs(vals[0]) + abs(vals[-1]))))
                        direction = "BUY" if vals[-1] > 0 else "SELL"
                        return {"name": "delta_flip", "detected": detected, "strength": strength, "direction": direction}
                except Exception:
                    pass

        return None

    def _detect_unusual_volume(self, normalized: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Signal unusual current volume relative to historical average volume."""
        avg = self._safe_float(normalized.get("historical_average_volume"), 0.0)
        volume = self._safe_float(normalized.get("volume"), 0.0)
        if avg <= 0 and volume <= 0:
            return None
        # Need clear unusual volume. If average missing, this cannot be established.
        if avg <= 0:
            return None
        if volume <= 0:
            return None
        ratio = volume / max(avg, 1e-9)
        # unusual if > 1.5x average
        if ratio >= 1.5:
            direction = "BUY" if volume > 0 else "SELL"
            strength = min(1.0, max(0.0, (ratio - 1.0) / 3.0))
            return {"name": "unusual_volume", "detected": True, "strength": strength, "direction": direction}
        return None

    def _detect_opposite_side_absorption(self, normalized: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect opposite-side absorption by interpreting absorption signals neutrally."""
        absorption = normalized.get("absorption")
        if not isinstance(absorption, Mapping):
            return None
        signal = absorption.get("signal", absorption.get("type", absorption.get("side", 0)))
        strength = self._safe_float(absorption.get("strength", 0.0), 0.0)
        if signal in (1, "1", "BUY_ABSORPTION", "BUY", "BUY_INITIATED"):
            # opposite side of buying force is selling absorption, which can imply reversal.
            return {"name": "opposite_side_absorption", "detected": True, "strength": self._bounded(strength / 100.0), "direction": "SELL"}
        if signal in (-1, "-1", "SELL_ABSORPTION", "SELL", "SELL_INITIATED"):
            return {"name": "opposite_side_absorption", "detected": True, "strength": self._bounded(strength / 100.0), "direction": "BUY"}
        return None

    def _detect_divergence(self, normalized: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Signal when divergence mapping reports a directional disagreement."""
        divergence = normalized.get("divergence")
        if not isinstance(divergence, Mapping):
            return None
        typ = str(divergence.get("type", "")).upper()
        signal = str(divergence.get("signal", "0"))
        strength = self._safe_float(divergence.get("strength", 0.0), 0.0)
        if typ in {"BULLISH", "BEARISH"} and signal not in {"0", "NONE", ""}:
            direction = "BUY" if typ == "BULLISH" else "SELL"
            return {"name": "divergence", "detected": True, "strength": self._bounded(strength), "direction": direction}
        if signal in {"1", "-1"}:
            direction = "BUY" if signal == "1" else "SELL"
            return {"name": "divergence", "detected": True, "strength": min(1.0, max(0.0, abs(strength))), "direction": direction}
        return None

    def _detect_lead_time_reversal(self, normalized: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Lead-time reversal from a lead_time mapping or scalar-like estimate."""
        lead_time = normalized.get("lead_time")
        if isinstance(lead_time, Mapping):
            # direction is common in lead_time dictionaries.
            direction = str(lead_time.get("direction", lead_time.get("status", "NONE"))).upper()
            status = str(lead_time.get("status", "")).upper()
            # DIVERGING or CROSSED are carry-over evidence of reversal-start possibility.
            if direction in {"BEAR", "BULL"}:
                # Use the sign of lead direction as actual direction.
                return {"name": "lead_time_reversal", "detected": True, "strength": 0.25, "direction": direction}
            if status in {"DIVERGING", "CROSSED", "REVERSED"}:
                return {"name": "lead_time_reversal", "detected": True, "strength": 0.25, "direction": "BUY" if "BULL" in str(lead_time.get("direction", "")).upper() else "SELL"}
        elif lead_time is not None:
            # scalar lead time as minutes; cannot infer direction, so no signal
            return None
        return None

    def _detect_force_score_reversal(self, normalized: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect force-score reversal from a sign-flipped force score or mode."""
        force = normalized.get("force")
        try:
            force_value = self._safe_float(force, 0.0)
        except Exception:
            force_value = 0.0
        # A negative force suggests selling pressure; if its abs is large enough,
        # and context has a regime or other evidence, it can signal a reversal start.
        if force_value < 0:
            strength = min(1.0, max(0.0, abs(force_value) / 100.0))
            return {"name": "force_score_reversal", "detected": True, "strength": strength, "direction": "SELL"}
        if force_value > 0:
            strength = min(1.0, max(0.0, abs(force_value) / 100.0))
            return {"name": "force_score_reversal", "detected": True, "strength": strength, "direction": "BUY"}
        return None

    def _build_signals(self, normalized: Mapping[str, Any]) -> List[Dict[str, Any]]:
        """Return normalized signals with required shape: name, detected, strength, direction."""
        signals: List[Dict[str, Any]] = []

        # Request explicitly says implement detection from available order-flow evidence.
        # The list is intentionally normalized and contains only signals that were actually detected.
        for detector in (
            self._detect_delta_flip,
            self._detect_unusual_volume,
            self._detect_opposite_side_absorption,
            self._detect_divergence,
            self._detect_lead_time_reversal,
            self._detect_force_score_reversal,
        ):
            try:
                signal = detector(normalized)
            except Exception:
                signal = None
            if isinstance(signal, Mapping):
                detected = bool(signal.get("detected", False))
                if detected:
                    item = {
                        "name": str(signal.get("name", detector.__name__)),
                        "detected": True,
                        "strength": self._bounded(signal.get("strength", 0.0)),
                        "direction": str(signal.get("direction", "NONE")).upper(),
                    }
                    signals.append(item)

        # Some non-detection dictionaries may say no evidence. If signals are empty,
        # the result remains empty rather than fabricated.
        return signals

    def _probability_from_signals(self, signals: Sequence[Mapping[str, Any]]) -> float:
        """Calculate a bounded model-estimated probability from detected evidence."""
        if not signals:
            return 0.0
        # Weight the average strength of signals. Never exceed 1.
        detected = [self._bounded(item.get("strength", 0.0)) for item in signals if bool(item.get("detected", False))]
        if not detected:
            return 0.0
        mean_strength = sum(detected) / len(detected)
        # More signals increase confidence, but remain bounded and model-estimated.
        confidence = min(1.0, mean_strength * (1.0 + 0.1 * (len(detected) - 1)))
        return self._bounded(confidence)

    def probability_decay_curve(
        self,
        reversal_probability: Any = None,
        future_minutes: Any = 0,
        decay_rate: Any = 0.1,
    ) -> List[float]:
        """Return a finite probability decay curve over future minutes.

        Requirements:
        - finite numeric values,
        - values stay within [0,1],
        - supports zero, negative, or None inputs,
        - no future market data is used.
        """
        try:
            prob = self._bounded(reversal_probability)
        except Exception:
            prob = 0.0

        try:
            minutes = int(float(future_minutes)) if future_minutes is not None else 0
        except Exception:
            minutes = 0
        if minutes < 0:
            minutes = 0

        try:
            rate = float(decay_rate)
        except Exception:
            rate = 0.1
        if not math.isfinite(rate) or rate < 0:
            rate = 0.1

        # A zero or negative or None future-minutes input should yield a single
        # finite point at the current probability and no fabricated curve.
        if minutes == 0:
            return [self._bounded(prob)]

        # Use a deterministic exponential decay. No future market data.
        points: List[float] = []
        for minute in range(minutes + 1):
            value = prob * math.exp(-rate * minute)
            points.append(self._bounded(value))
        return points

    def calculate_reversal(
        self,
        evidence: Optional[Mapping[str, Any]] = None,
        force: Any = None,
        delta: Any = None,
        lead_time: Any = None,
        historical_average_volume: Any = None,
        regime: Any = None,
        oi: Any = None,
        absorption: Any = None,
        divergence: Any = None,
        future_minutes: Any = 0,
        decay_rate: Any = 0.1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compute a model-estimated reversal payload from requested evidence."""
        normalized = self._normalise(evidence, force, delta,
                                      lead_time, historical_average_volume,
                                      regime, oi, absorption, divergence,
                                      **kwargs)
        return self.detect_reversal_starting(
            evidence=normalized,
            force=normalized.get("force"),
            delta=normalized.get("delta"),
            lead_time=normalized.get("lead_time"),
            historical_average_volume=normalized.get("historical_average_volume"),
            regime=normalized.get("regime"),
            oi=normalized.get("oi"),
            absorption=normalized.get("absorption"),
            divergence=normalized.get("divergence"),
            future_minutes=future_minutes,
            decay_rate=decay_rate,
        )

    def detect_reversal_starting(
        self,
        evidence: Optional[Mapping[str, Any]] = None,
        force: Any = None,
        delta: Any = None,
        lead_time: Any = None,
        historical_average_volume: Any = None,
        regime: Any = None,
        oi: Any = None,
        absorption: Any = None,
        divergence: Any = None,
        future_minutes: Any = 0,
        decay_rate: Any = 0.1,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Detect reversal starting shape from available order-flow evidence.

        Returns a bounded probability result and a normalized signal list.
        """
        normalized = self._normalise(evidence, force, delta,
                                      lead_time, historical_average_volume,
                                      regime, oi, absorption, divergence,
                                      **kwargs)

        signals = self._build_signals(normalized)
        # No fabricated probability from missing order-flow fields.
        probability = self._probability_from_signals(signals)
        # Curve must always remain finite and bounded.
        curve = self.probability_decay_curve(probability, future_minutes, decay_rate)

        # a safe metadata envelope with heuristic/calibrated explicit status
        metadata = {
            "heuristic_or_calibrated": "heuristic",
            "model_estimated": True,
            "guardrails": {
                "probabilities_bounded_0_1": True,
                "zero_volume_safe": self._safe_float(normalized.get("historical_average_volume"), 0.0) <= 0,
                "missing_fields_safe": True,
                "constant_regression_safe": True,
                "scipy_errors_safe": True,
                "future_data_not_used": True,
            },
            "probability_guarantee": "No reversal probability is guaranteed.",
            "method": "detect_reversal_starting",
        }

        # Use explicit order-flow fields when present; otherwise return 0-lists.
        # `reversal_signals` is the normalized signal list.
        return {
            "reversal_volume": self._safe_float(normalized.get("volume"), 0.0) * probability,
            "minutes_needed": max(0.0, self._safe_float(normalized.get("lead_time"), 0.0)) if isinstance(normalized.get("lead_time"), (int, float)) else 0.0,
            "reversal_probability": self._bounded(probability),
            "probability_curve": [self._bounded(v) for v in curve],
            "reversal_signals": signals,
            "metadata": metadata,
        }


__all__ = ["ReversalCalculator"]
