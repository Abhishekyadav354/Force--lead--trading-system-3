"""Small orchestration layer for modular trading predictions."""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.adaptive_lead_time import AdaptiveLeadTimeCalculator
from core.force_score import ForceScoreCalculator
from core.lead_time import LeadTimeCalculator
from core.order_flow import OrderFlowAnalyzer
from ml.predictor import MLPredictor
from statistics.monte_carlo import MonteCarloRiskEngine
from statistics.significance_testing import SignificanceTester
from statistics.stochastic import StochasticPriceModel

try:
    from core.combined_system import CompleteTradingSystem
except Exception:  # pragma: no cover - optional dependency/path guard
    CompleteTradingSystem = None


class PredictionEngine:
    """Coordinate existing prediction modules without placing trades.

    Default documented component weights for the prediction score assembly:
    - ``force``: 0.40
    - ``order_flow``: 0.20
    - ``ml``: 0.20
    - ``significance``: 0.10
    - ``monte_carlo``: 0.10

    The weights are deliberately conservative and make no claim of certainty;
    they only help the engine convert available directional evidence into a
    bounded signed score.
    """

    DEFAULT_COMPONENT_WEIGHTS: Dict[str, float] = {
        "force": 0.40,
        "order_flow": 0.20,
        "ml": 0.20,
        "significance": 0.10,
        "monte_carlo": 0.10,
    }

    def __init__(
        self,
        force_calculator: Optional[ForceScoreCalculator] = None,
        order_flow_analyzer: Optional[OrderFlowAnalyzer] = None,
        lead_calculator: Optional[LeadTimeCalculator] = None,
        significance_tester: Optional[SignificanceTester] = None,
        ml_predictor: Optional[MLPredictor] = None,
        monte_carlo_engine: Optional[MonteCarloRiskEngine] = None,
        stochastic_model: Optional[StochasticPriceModel] = None,
        timeframe_minutes: Optional[float] = None,
        adaptive_lead_calculator: Optional[AdaptiveLeadTimeCalculator] = None,
        market: str = "UNKNOWN",
        segment: str = "OTHER",
        timeframe: Optional[str] = None,
    ) -> None:
        self.force_calculator = force_calculator or ForceScoreCalculator()
        self.order_flow_analyzer = order_flow_analyzer or OrderFlowAnalyzer()
        self.lead_calculator = lead_calculator or LeadTimeCalculator()
        self.significance_tester = significance_tester or SignificanceTester()
        self.ml_predictor = ml_predictor or MLPredictor()
        self.monte_carlo_engine = monte_carlo_engine or MonteCarloRiskEngine()
        self.stochastic_model = stochastic_model or StochasticPriceModel()
        self.adaptive_lead_calculator = adaptive_lead_calculator or AdaptiveLeadTimeCalculator()
        self.market = market
        self.segment = segment
        self.timeframe = timeframe
        self.timeframe_minutes = (
            timeframe_minutes
            if timeframe_minutes is not None
            else getattr(settings, "TIMEFRAME_MINUTES", 5)
        )
        self.force_history: List[float] = []

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
        """Clamp a numeric value to a deterministic bounded interval."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            numeric = 0.0
        return max(low, min(high, numeric))

    @staticmethod
    def _to_signed_probability(value: Any, direction: Any = None) -> float:
        """Convert an unsigned probability-like scalar into a signed [-1, 1] score.

        Args:
            value: Unsigned probability in ``[0, 1]`` or another probability-like
                scalar. It remains an unsigned evidence value and is never
                treated as certainty.
            direction: Optional directional hint such as ``BUY``/``BULL``/``UP``
                or ``SELL``/``BEAR``/``DOWN``. If absent, the helper infers a
                sign from the value's distance from the neutral 0.5 baseline.

        Returns:
            A deterministic signed component clamped to ``[-1, 1]``.
        """
        try:
            prob = float(value)
        except (TypeError, ValueError):
            prob = 0.5
        if prob != prob or prob in (float("inf"), float("-inf")):
            prob = 0.5
        prob = max(0.0, min(1.0, prob))

        sign = 1.0
        direction_text = str(direction or "").upper()
        if direction_text in {"SELL", "BEAR", "DOWN", "NEGATIVE"}:
            sign = -1.0
        elif direction_text in {"BUY", "BULL", "UP", "POSITIVE"}:
            sign = 1.0
        elif direction_text == "NEUTRAL":
            sign = 0.0
        else:
            # Infer direction from the distance from the neutral 0.5 probability
            # line without fabricating certainty.
            if prob < 0.5:
                sign = -1.0
            elif prob > 0.5:
                sign = 1.0
            else:
                sign = 0.0

        signed = (prob - 0.5) * 2.0 * sign
        return PredictionEngine._clamp(signed, -1.0, 1.0)

    @staticmethod
    def _normalize_score(value: Any) -> float:
        """Convert a score to a finite value bounded to ``[-1.0, 1.0]``."""
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        if score != score or score in (float("inf"), float("-inf")):
            return 0.0
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _safe_dict(value: Any) -> Dict[str, Any]:
        """Return a dictionary or a new empty dictionary for invalid input."""
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _empty_order_flow() -> Dict[str, Any]:
        return {
            "score": 0.0,
            "signal": "WAIT",
            "delta": {},
            "divergence": {},
            "volume_profile": {},
            "footprint": {},
            "absorption": {},
            "exhaustion": {},
            "vwap": {},
        }

    def _calculate_force(
        self,
        live_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        options_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate F1-F9 using the existing force APIs and retain a normalized score dictionary."""
        live = self._safe_dict(live_data)
        history = historical_data if isinstance(historical_data, list) else []
        history = [item for item in history if isinstance(item, dict)]
        previous = history[-1] if history else None
        options = self._safe_dict(options_data)

        trade_sizes = [
            self._safe_float(item.get("avg_trade_size"))
            for item in history
            if isinstance(item, dict)
        ]
        average_trade_size = sum(trade_sizes) / len(trade_sizes) if trade_sizes else 1.0
        recent = history[-10:]

        factors: Dict[str, float] = {}
        try:
            # Prefer CompleteTradingSystem's public-ish force-factor decomposition when present.
            if CompleteTradingSystem is not None:
                system = CompleteTradingSystem(0, {}, None)
                factors = system._calculate_all_force_factors(live, history, options, None)
        except Exception:
            factors = {}

        # Fall back to the ForceScoreCalculator API directly when CompleteTradingSystem is unavailable.
        if not factors:
            try:
                factors = {
                    "f1": self.force_calculator.calculate_f1_volume_force(live, previous, average_trade_size),
                    "f2": self.force_calculator.calculate_f2_wall_force(live),
                    "f3": self.force_calculator.calculate_f3_position_risk(live, options),
                    "f4": self.force_calculator.calculate_f4_acceleration(history[-5:]),
                    "f5": self.force_calculator.calculate_f5_institutional_detector(live, history),
                    "f6": self.force_calculator.calculate_f6_oi_signal(
                        options,
                        {
                            "current_price": live.get("close", live.get("ltp", 0.0)),
                            "prev_price": previous.get("close", previous.get("ltp", 0.0)) if previous else 0.0,
                        },
                    ) if options else 0.0,
                    "f7": self.force_calculator.calculate_f7_spread_quality(recent),
                    "f8": self._safe_float(live.get("nifty_direction"), 1.0),
                    "f9": self.force_calculator.calculate_f9_candle_conviction(live),
                }
            except Exception:
                factors = {}

        try:
            result = self.force_calculator.calculate_total_force_score(factors or {})
        except Exception:
            result = {"score": 0.0, "signal": "WAIT"}

        result_dict = self._safe_dict(result)
        score = self._normalize_score(result_dict.get("score", 0.0))
        return {
            "factors": self._safe_dict(factors),
            "score": score,
            "result": self._safe_dict(result_dict),
        }

    def _calculate_order_flow(self, order_flow_data: Any) -> Dict[str, Any]:
        """Calculate F10 using the existing OrderFlowAnalyzer and return a normalized score payload."""
        if order_flow_data is None:
            return self._empty_order_flow()

        if isinstance(order_flow_data, dict) and "score" in order_flow_data:
            normalized = dict(order_flow_data)
            normalized["score"] = self._normalize_score(normalized.get("score"))
            normalized.setdefault("signal", "BUY" if normalized["score"] > 0.4 else "SELL" if normalized["score"] < -0.4 else "WAIT")
            return normalized

        if isinstance(order_flow_data, dict):
            ticks = order_flow_data.get("ticks")
            if not isinstance(ticks, list):
                return self._empty_order_flow()
        else:
            ticks = order_flow_data if isinstance(order_flow_data, list) else []

        if not isinstance(ticks, list) or not ticks:
            return self._empty_order_flow()

        classified: List[Dict[str, Any]] = []
        for tick in ticks:
            if not isinstance(tick, dict):
                continue
            normalized_tick = dict(tick)
            normalized_tick.setdefault("timestamp", None)
            normalized_tick.setdefault("bid", None)
            normalized_tick.setdefault("ask", None)
            if normalized_tick.get("type") in {"BUY_INITIATED", "SELL_INITIATED", "NEUTRAL"}:
                classified.append(normalized_tick)
            else:
                try:
                    classified.append(self.order_flow_analyzer.classify_tick(normalized_tick))
                except Exception:
                    continue

        if not classified:
            return self._empty_order_flow()

        try:
            prices = [self._safe_float(tick.get("price")) for tick in classified]
            deltas = [
                self._safe_float(tick.get("volume"))
                * (1 if tick.get("type") == "BUY_INITIATED" else -1 if tick.get("type") == "SELL_INITIATED" else 0)
                for tick in classified
            ]
            volumes = [self._safe_float(tick.get("volume")) for tick in classified]
            delta = self.order_flow_analyzer.calculate_delta(classified)
            divergence = self.order_flow_analyzer.detect_delta_divergence(prices, deltas)
            volume_profile = self.order_flow_analyzer.build_volume_profile(classified, current_price=prices[-1])
            footprint = self.order_flow_analyzer.build_footprint(classified, prices[0], prices[-1])
            absorption = self.order_flow_analyzer.detect_absorption(classified, prices)
            exhaustion = self.order_flow_analyzer.detect_exhaustion(deltas, prices, volumes)
            vwap = self.order_flow_analyzer.calculate_vwap_bands(classified)
            score = self.order_flow_analyzer.calculate_order_flow_score({
                "delta": delta,
                "divergence": divergence,
                "absorption": absorption,
                "vwap": vwap,
                "poc": volume_profile,
                "stacked_buy_zones": footprint.get("stacked_buy_zones", []),
                "stacked_sell_zones": footprint.get("stacked_sell_zones", []),
                "exhaustion": exhaustion,
            })
        except Exception:
            return self._empty_order_flow()

        score_dict = self._safe_dict(score)
        result = {
            "delta": delta,
            "divergence": divergence,
            "volume_profile": volume_profile,
            "footprint": footprint,
            "absorption": absorption,
            "exhaustion": exhaustion,
            "vwap": vwap,
            "score": self._normalize_score(score_dict.get("score")),
            "signal": score_dict.get("signal", "WAIT"),
        }
        result.setdefault("components", self._safe_dict(score_dict.get("components")))
        return result

    def _calculate_lead_time(self, force_score: float, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reuse the repository's LeadTimeCalculator API safely.

        This method intentionally delegates to the project-native static methods
        ``calculate_lead()``, ``calculate_catchup_time()``, and
        ``get_lead_status()`` defined in ``core.lead_time``. It preserves those
        repository-native dictionaries under the ``lead``, ``catchup`` and
        ``status`` keys and adds the normalized fields expected by
        ``PredictionEngine`` consumers: ``value``/``score``, ``direction``,
        ``estimated_minutes``, ``confidence`` and ``crossover`` metadata.

        Args:
            force_score: The current normalized force signal passed from
                ``PredictionEngine._calculate_force``.
            historical_data: Historical candle payload that is checked only for
                list validity and ignored if the repository API does not need it.

        Returns:
            A structured lead-time payload. A successful payload has
            ``available=True`` and carries the canonical repository fields plus
            the normalized keys. On any failure, the method returns the neutral
            dictionary required by the engine with ``available=False`` and an
            ``error`` message rather than raising.
        """
        neutral_crossover: Dict[str, Any] = {
            "status": "WAIT",
            "progress_pct": 0.0,
            "minutes_remaining": 0.0,
            "message": "Lead time unavailable.",
            "action": "DO_NOT_TRADE",
            "lead": 0.0,
        }
        neutral_result: Dict[str, Any] = {
            "available": False,
            "value": 0.0,
            "score": 0.0,
            "direction": "NEUTRAL",
            "estimated_minutes": 0.0,
            "confidence": 0.0,
            "crossover": neutral_crossover,
            "lead": {},
            "catchup": {},
            "status": neutral_crossover,
            "error": "LeadTime calculation failed: unavailable data",
        }

        try:
            if not isinstance(historical_data, list):
                historical_data = []

            # Keep this method repository-native. No second model is invented.
            previous_force = self.force_history[-1] if self.force_history else 0.0
            duration_minutes = self.timeframe_minutes
            if duration_minutes is None:
                duration_minutes = getattr(settings, "TIMEFRAME_MINUTES", 5)

            duration_minutes = self._safe_float(duration_minutes, 5.0)
            previous_force = self._safe_float(previous_force, 0.0)
            current_force = self._safe_float(force_score, 0.0)

            if self.lead_calculator is None:
                raise AttributeError("LeadTimeCalculator is unavailable")

            # Validate the exact repository API that the project already ships.
            for method_name in (
                "calculate_lead",
                "calculate_catchup_time",
                "get_lead_status",
            ):
                if not hasattr(self.lead_calculator, method_name):
                    raise AttributeError(f"LeadTimeCalculator.{method_name} is unavailable")

            lead = self.lead_calculator.calculate_lead(previous_force, duration_minutes)
            if not isinstance(lead, dict):
                raise TypeError("LeadTimeCalculator.calculate_lead did not return a dictionary")

            lead_value = self._safe_float(lead.get("lead_value"), 0.0)
            direction = lead.get("direction")
            if not isinstance(direction, str) or not direction.strip():
                if lead_value > 0:
                    direction = "BULL"
                elif lead_value < 0:
                    direction = "BEAR"
                else:
                    direction = "NEUTRAL"
            else:
                direction = direction.upper()

            catchup = self.lead_calculator.calculate_catchup_time(
                lead_value,
                current_force,
                previous_force,
            )
            if not isinstance(catchup, dict):
                raise TypeError("LeadTimeCalculator.calculate_catchup_time did not return a dictionary")

            estimated_minutes = self._safe_float(
                catchup.get("catchup_time", float("inf")),
                float("inf"),
            )

            status = self.lead_calculator.get_lead_status(
                0.0,
                estimated_minutes,
                lead_value,
            )
            if not isinstance(status, dict):
                raise TypeError("LeadTimeCalculator.get_lead_status did not return a dictionary")

            confidence = 1.0 if catchup.get("reliable") is True else 0.0

            return {
                "available": True,
                "value": lead_value,
                "score": lead_value,
                "direction": direction,
                "estimated_minutes": estimated_minutes,
                "confidence": confidence,
                "crossover": status,
                "lead": lead,
                "catchup": catchup,
                "status": status,
                "error": None,
            }
        except Exception as exc:
            neutral_result["error"] = f"LeadTime calculation failed: {exc}"
            return neutral_result

    def _calculate_adaptive_lead_time(
        self,
        live_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        market: Optional[str] = None,
        segment: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate the historical speed layer without changing legacy lead fields."""
        neutral = {
            "available": False,
            "market": market or self.market,
            "segment": segment or self.segment,
            "timeframe": timeframe or self.timeframe or "5m",
            "current_speed": 0.0,
            "required_speed": 0.0,
            "normal_eta_minutes": None,
            "accelerated_eta_minutes": None,
            "shock_eta_minutes": None,
            "estimated_lead_window_bars": {"normal": None, "accelerated": None, "shock": None},
            "historical_speed_percentiles": {"median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "maximum_reliable": 0.0},
            "sample_count": 0,
            "lookback_days": getattr(self.adaptive_lead_calculator, "lookback_days", 0),
            "confidence": 0.0,
            "speed_regime": "NORMAL",
            "historical_estimate": True,
            "model_note": "Historical/model estimate unavailable; insufficient data.",
        }
        try:
            live = live_data if isinstance(live_data, dict) else {}
            market_name = market or live.get("market") or self.market
            segment_name = segment or live.get("segment") or self.segment
            selected_timeframe = timeframe or live.get("timeframe") or self.timeframe
            if selected_timeframe is None:
                minutes = self._safe_float(self.timeframe_minutes, 5.0)
                selected_timeframe = f"{int(minutes)}m" if minutes.is_integer() else f"{minutes:g}m"

            gap = live.get("remaining_volume_gap", live.get("remaining_gap", live.get("volume_gap", live.get("progress_gap", 0.0))))
            speed = live.get("current_speed", live.get("volume_speed", live.get("speed", live.get("volume_per_minute", 0.0))))
            if not self._safe_float(speed, 0.0):
                volume = self._safe_float(live.get("volume", live.get("progress_volume", 0.0)), 0.0)
                timeframe_value = self._safe_float(self.timeframe_minutes, 5.0)
                if str(selected_timeframe).lower().endswith("m"):
                    timeframe_value = self._safe_float(str(selected_timeframe)[:-1], timeframe_value)
                elif str(selected_timeframe).lower().endswith("h"):
                    timeframe_value = self._safe_float(str(selected_timeframe)[:-1], timeframe_value / 60.0) * 60.0
                speed = volume / max(timeframe_value, 1.0)
            current_timestamp = live.get("timestamp", live.get("datetime", live.get("time")))
            profile_history = [
                {
                    **row,
                    "market": row.get("market", market_name),
                    "segment": row.get("segment", segment_name),
                    "timeframe": row.get("timeframe", selected_timeframe),
                }
                for row in historical_data
                if isinstance(row, dict)
            ]
            result = self.adaptive_lead_calculator.estimate(
                remaining_volume_gap=gap,
                current_speed=speed,
                historical_data=profile_history,
                market=market_name,
                segment=segment_name,
                timeframe=selected_timeframe,
                current_timestamp=current_timestamp,
                required_speed=live.get("required_speed"),
            )
            if not isinstance(result, dict):
                raise TypeError("AdaptiveLeadTimeCalculator returned a non-dictionary")
            result["available"] = True
            return result
        except Exception as exc:
            neutral["error"] = f"Adaptive lead-time calculation failed: {exc}"
            return neutral

    def _calculate_significance(self, force_score: float) -> Dict[str, Any]:
        """Reuse the repository SignificanceTester API safely.

        This method calls the existing ``SignificanceTester`` once through its
        real ``test_signal_significance(history, current_signal)`` method and
        maps only the fields that the repository API actually returns into the
        engine's normalized significance payload. It never turns evidence into
        a guaranteed prediction and falls back to a neutral dictionary whenever
        the history or calculator output is too sparse or malformed.

        Args:
            force_score: Current normalized force score to compare against the
                stored force-history window.

        Returns:
            A safe significance dictionary including the normalized keys
            ``available``, ``significant``, ``z_score``, ``p_value``,
            ``confidence``, and ``direction`` when the repository API returns
            the underlying evidence fields. On insufficient data or
            exceptions, ``available`` is set to ``False`` and an ``error``
            field signals the neutral fallback.
        """
        neutral_result: Dict[str, Any] = {
            "available": False,
            "significant": False,
            "z_score": 0.0,
            "raw_z_score": 0.0,
            "p_value": 1.0,
            "confidence": 0.0,
            "direction": "NEUTRAL",
            "error": "Significance calculation failed: insufficient or invalid data",
        }

        try:
            if not isinstance(self.force_history, list):
                raise TypeError("force_history must be a list")

            history = [self._safe_float(item, 0.0) for item in self.force_history]
            if len(history) < 2:
                raise ValueError("insufficient historical candles")

            # Reuse exactly the existing repository API and do not duplicate any
            # statistical calculation or create a second significance model.
            if not hasattr(self.significance_tester, "test_signal_significance"):
                raise AttributeError("SignificanceTester.test_signal_significance is unavailable")

            raw_result = self.significance_tester.test_signal_significance(history, force_score)
            if not isinstance(raw_result, dict):
                raise TypeError("SignificanceTester.test_signal_significance did not return a dictionary")

            # Map only fields actually provided by the repository implementation.
            z_score = self._safe_float(raw_result.get("z_score"), 0.0)
            p_value = self._safe_float(raw_result.get("p_value"), 1.0)
            significant = bool(raw_result.get("is_significant", False))

            confidence_pct = self._safe_float(raw_result.get("confidence_pct"), 0.0)
            confidence = min(1.0, max(0.0, confidence_pct / 100.0)) if confidence_pct >= 0.0 else 0.0

            # Preserve the raw z-score while allowing a normalized scalar for
            # downstream scoring only when needed, without overwriting the
            # repository-native evidence.
            raw_z_score = z_score
            normalized_z_score = max(-1.0, min(1.0, raw_z_score)) if raw_z_score != 0.0 else 0.0

            if raw_z_score > 0:
                direction = "BULL"
            elif raw_z_score < 0:
                direction = "BEAR"
            else:
                direction = "NEUTRAL"

            # Statistical significance is evidence, not certainty.
            return {
                "available": True,
                "significant": significant,
                "z_score": raw_z_score,
                "raw_z_score": raw_z_score,
                "z_score_normalized": normalized_z_score,
                "p_value": p_value,
                "confidence": confidence,
                "direction": direction,
                "is_significant": significant,
                "error": None,
            }
        except Exception as exc:
            neutral_result["error"] = f"Significance calculation failed: {exc}"
            return neutral_result

    def _calculate_ml(
        self,
        factors: Dict[str, float],
        lead_time: Dict[str, Any],
        regime: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reuse the repository's real ML predictor API safely.

        The method checks the injected or default predictor object for the
        project-native prediction method and then calls that method with the
        exact supported feature payload layout already used by the repository:
        factor keys such as ``f1``-``f9``, the lead-status map shaped by the
        lead-time module, and the regime metadata dictionary. It keeps the
        raw model response under ``raw`` and maps the normalized fields that
        later engine components can safely consume: ``available``,
        ``prediction``, ``direction``, ``win_prob`` and ``confidence``.

        Args:
            factors: Factor decomposition from the force engine.
            lead_time: Lead-time diagnostic dictionary; the cross-over/status
                block inside ``lead_time`` is the only lead-time evidence passed
                to the predictor by the project's repository contract.
            regime: Regime metadata returned by the stochastic/regime detector.

        Returns:
            A safe dictionary. If the repository predictor is unavailable,
            empty, or raises during inference, the returned payload is neutral
            with ``available=False`` and ``error`` information rather than
            crashing the engine.
        """
        neutral_result: Dict[str, Any] = {
            "available": False,
            "prediction": "WAIT",
            "direction": "NEUTRAL",
            "win_prob": 0.5,
            "confidence": 0.0,
            "raw": {},
            "error": "ML prediction failed: unavailable predictor or unsupported feature payload",
        }

        try:
            if not isinstance(factors, dict):
                raise TypeError("factors must be a dictionary")
            if not isinstance(lead_time, dict):
                lead_time = {}
            if not isinstance(regime, dict):
                regime = {}

            predictor = self.ml_predictor
            if predictor is None:
                raise AttributeError("MLPredictor is unavailable")

            # Prefer the injected predictor; otherwise reuse the default
            # repository object, which already loads saved_artifacts once.
            if not hasattr(predictor, "predict_live") and not hasattr(predictor, "predict"):
                raise AttributeError("MLPredictor has no repository prediction method")

            method = getattr(predictor, "predict_live", None)
            if method is None or not callable(method):
                method = getattr(predictor, "predict", None)
            if method is None or not callable(method):
                raise AttributeError("MLPredictor has no callable repository prediction method")

            # Detect actual method signature and call repository-native API.
            try:
                signature = inspect.signature(method)
                parameter_names = [param.name for param in signature.parameters.values()]
            except Exception:
                parameter_names = []

            lead_status = lead_time.get("status", {}) if isinstance(lead_time, dict) else {}
            raw_result: Dict[str, Any]

            # The real repository predictor accepts (current_factors, lead_status, regime). Reuse that.
            if "current_factors" in parameter_names:
                raw_result = method(factors, lead_status, regime)
            else:
                raw_result = method(factors, lead_status, regime)

            # The predictor may return an empty fallback result when trained
            # models are absent. Treat that as evidence-safe neutral ML.
            if not isinstance(raw_result, dict):
                raise TypeError("MLPredictor did not return a repository dictionary")

            # Honor the repository's saved model availability gate.
            if not getattr(predictor, "models", None):
                raise ValueError("MLPredictor has no saved models loaded")

            prediction = str(raw_result.get("signal", "WAIT") or "WAIT").upper()
            if prediction not in {"BUY", "SELL", "WAIT"}:
                prediction = "WAIT"

            direction = "NEUTRAL"
            if prediction == "BUY":
                direction = "BULL"
            elif prediction == "SELL":
                direction = "BEAR"

            win_prob = self._safe_float(raw_result.get("win_probability"), 0.5)
            if win_prob < 0.0:
                win_prob = 0.0
            elif win_prob > 1.0:
                win_prob = 1.0

            confidence = self._safe_float(raw_result.get("confidence"), 0.0)
            if confidence < 0.0:
                confidence = 0.0
            elif confidence > 1.0:
                confidence = 1.0

            return {
                "available": True,
                "prediction": prediction,
                "direction": direction,
                "win_prob": win_prob,
                "confidence": confidence,
                "raw": raw_result,
                "error": None,
            }
        except Exception as exc:
            neutral_result["error"] = f"ML prediction failed: {exc}"
            return neutral_result

    def _calculate_prediction_score(
        self,
        force: Dict[str, Any],
        order_flow: Dict[str, Any],
        ml_result: Dict[str, Any],
        significance: Dict[str, Any],
        monte_carlo: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Blend the repository-native directional components into one safe signed score.

        Only the direction-bearing signals are moved into the signed scalar
        ensemble. Unsigned probabilities stay probabilities and are converted
        to signed range-only evidence by explicit direction hints such as the
        predictor ``direction``/``signal`` field or the Monte Carlo
        ``expected_return`` sign. Every component is clamped to ``[-1, 1]``.

        Returns:
            A deterministic scoring payload containing ``available``,
            ``combined_score``, ``direction``, ``confidence``,
            ``raw_components`` and ``normalized_components``. It keeps the
            original component evidence and normalizes only the fields needed
            by downstream score consumers.
        """
        weights = dict(self.DEFAULT_COMPONENT_WEIGHTS)
        raw_components: Dict[str, float] = {
            "force": 0.0,
            "order_flow": 0.0,
            "ml": 0.0,
            "significance": 0.0,
            "monte_carlo": 0.0,
        }
        normalized_components: Dict[str, float] = {
            "force": 0.0,
            "order_flow": 0.0,
            "ml": 0.0,
            "significance": 0.0,
            "monte_carlo": 0.0,
        }

        # Force component.
        force_score = 0.0
        if isinstance(force, dict):
            force_score = self._safe_float(force.get("score"), 0.0)
        raw_components["force"] = force_score
        normalized_components["force"] = self._clamp(force_score, -1.0, 1.0)

        # Order-flow component.
        order_score = 0.0
        if isinstance(order_flow, dict):
            order_score = self._safe_float(order_flow.get("score"), 0.0)
        raw_components["order_flow"] = order_score
        normalized_components["order_flow"] = self._clamp(order_score, -1.0, 1.0)

        # ML component: convert unsigned win probability to signed signal when
        # the predictor reports an available/reusable signal.
        if isinstance(ml_result, dict) and ml_result.get("available") is True:
            win_prob = self._safe_float(ml_result.get("win_prob", ml_result.get("win_probability", 0.5)), 0.5)
            direction = str(ml_result.get("direction") or ml_result.get("prediction") or ml_result.get("signal") or "NEUTRAL").upper()
            if direction == "BULL":
                direction = "BUY"
            elif direction == "BEAR":
                direction = "SELL"
            if direction == "BUY":
                ml_signed = self._to_signed_probability(win_prob, "BUY")
            elif direction == "SELL":
                ml_signed = self._to_signed_probability(win_prob, "SELL")
            else:
                ml_signed = 0.0
            raw_components["ml"] = win_prob
            normalized_components["ml"] = self._clamp(ml_signed, -1.0, 1.0)
        else:
            normalized_components["ml"] = 0.0

        # Significance component: use the sign of raw z-score as the direction
        # vector, and map magnitude into a bounded scalar. Evidence is preserved.
        if isinstance(significance, dict) and significance.get("available") is True:
            raw_z = self._safe_float(significance.get("z_score", significance.get("raw_z_score", 0.0)), 0.0)
            raw_components["significance"] = raw_z
            normalized_components["significance"] = self._clamp(raw_z / 3.0, -1.0, 1.0)

        # Monte Carlo component: use signed expected_return direction, but only
        # convert the unsigned `win_prob` into a signed scalar when there is a
        # positive or negative pay-off direction in the raw simulation payload.
        if isinstance(monte_carlo, dict) and monte_carlo.get("available") is True:
            win_prob = self._safe_float(monte_carlo.get("win_prob", monte_carlo.get("win_probability", 0.5)), 0.5)
            expected_return = self._safe_float(monte_carlo.get("expected_return"), 0.0)
            direction = "BUY" if expected_return >= 0.0 else "SELL"
            mc_signed = self._to_signed_probability(win_prob, direction)
            raw_components["monte_carlo"] = win_prob
            normalized_components["monte_carlo"] = self._clamp(mc_signed, -1.0, 1.0)

        # Weighted evidence blend using bounded normalized unsigned->signed scalars.
        weighted_sum = 0.0
        weight_total = 0.0
        active_components = 0
        for name, weight in weights.items():
            comp = normalized_components.get(name, 0.0)
            if name == "force" and force_score == 0.0:
                continue
            if name == "order_flow" and order_score == 0.0:
                continue
            if name in {"ml", "significance", "monte_carlo"} and comp == 0.0:
                continue
            weighted_sum += weight * comp
            weight_total += weight
            active_components += 1

        if not active_components or weight_total <= 0:
            return {
                "available": False,
                "combined_score": 0.0,
                "direction": "SIDEWAYS",
                "confidence": 0.0,
                "raw_components": raw_components,
                "normalized_components": normalized_components,
                "error": "Prediction score failed: no signed evidence components available",
            }

        combined_score = self._clamp(weighted_sum / weight_total, -1.0, 1.0)

        if combined_score > 0.4:
            direction = "UP"
        elif combined_score < -0.4:
            direction = "DOWN"
        else:
            direction = "SIDEWAYS"

        # Confidence is based on evidence availability and bounded to [0,1].
        confidence = 0.0
        confidence_weight_total = 0.0
        for name, weight in weights.items():
            if name == "force" and isinstance(force, dict):
                confidence += weight
                confidence_weight_total += weight
            elif name == "order_flow" and isinstance(order_flow, dict):
                confidence += weight
                confidence_weight_total += weight
            elif name == "ml" and isinstance(ml_result, dict) and ml_result.get("available") is True:
                confidence += weight * self._safe_float(ml_result.get("confidence"), 0.0)
                confidence_weight_total += weight
            elif name == "significance" and isinstance(significance, dict) and significance.get("available") is True:
                confidence += weight * self._safe_float(significance.get("confidence", significance.get("confidence_pct", 0.0) / 100.0 if significance.get("confidence_pct") is not None else 0.0), 0.0)
                confidence_weight_total += weight
            elif name == "monte_carlo" and isinstance(monte_carlo, dict) and monte_carlo.get("available") is True:
                confidence += weight * self._safe_float(monte_carlo.get("target_probability", 0.5), 0.5)
                confidence_weight_total += weight

        if confidence_weight_total:
            confidence = max(0.0, min(1.0, confidence / confidence_weight_total))
        else:
            confidence = 0.0

        return {
            "available": True,
            "combined_score": combined_score,
            "direction": direction,
            "confidence": confidence,
            "raw_components": raw_components,
            "normalized_components": normalized_components,
            "error": None,
        }

    def _calculate_monte_carlo(
        self,
        live_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run the existing Monte Carlo trade simulation through a safe adapter."""
        neutral = {
            "available": False,
            "win_prob": 0.5,
            "loss_prob": 0.5,
            "expected_return": 0.0,
            "error": "Monte Carlo unavailable: insufficient price data",
        }
        try:
            live = live_data if isinstance(live_data, dict) else {}
            entry = self._safe_float(live.get("close", live.get("ltp", 0.0)), 0.0)
            if entry <= 0.0:
                raise ValueError("missing positive entry price")
            prices = [
                self._safe_float(item.get("close", item.get("ltp", 0.0)), 0.0)
                for item in historical_data
                if isinstance(item, dict)
            ]
            prices = [price for price in prices if price > 0.0]
            parameters = self.stochastic_model.estimate_gbm_parameters(prices) if len(prices) >= 2 else {}
            mu = self._safe_float(parameters.get("mu"), 0.0)
            sigma = max(0.0, self._safe_float(parameters.get("sigma"), 0.0))
            raw = self.monte_carlo_engine.simulate_trade(
                entry,
                entry * 0.99,
                entry * 1.01,
                mu,
                sigma,
                n_simulations=1000,
            )
            if not isinstance(raw, dict):
                raise TypeError("Monte Carlo engine returned a non-dictionary")
            win_prob = self._safe_float(raw.get("win_probability"), 0.5)
            loss_prob = self._safe_float(raw.get("loss_probability"), 0.5)
            return {
                "available": True,
                "win_prob": max(0.0, min(1.0, win_prob)),
                "loss_prob": max(0.0, min(1.0, loss_prob)),
                "expected_return": self._safe_float(raw.get("expected_pnl_per_share"), 0.0),
                "raw": raw,
                "error": None,
            }
        except Exception as exc:
            neutral["error"] = f"Monte Carlo calculation failed: {exc}"
            return neutral

    def _detect_regime(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reuse the project’s repository regime detector safely.

        The method first attempts the existing stochastic regime detector,
        ``StochasticPriceModel.detect_market_regime(price_history)``, which is
        already the project's durable API and returns the machine-readable
        raw details the repository uses for weighting. When the repository API
        is unavailable or cannot analyze the provided data, the function falls
        back to a tiny deterministic rule based on the project’s own price
        fields so that the engine stays safe and testable without fabricating
        data.

        Returns:
            A normalized dictionary containing the supported regime label,
            a confidence score in the range ``[0.0, 1.0]``, an
            ``available`` boolean, and a ``raw`` artifact whenever the
            underlying detector returned one. The result is neutral on
            insufficient or malformed histories.
        """
        neutral_result: Dict[str, Any] = {
            "available": False,
            "regime": "UNKNOWN",
            "confidence": 0.0,
            "raw": {},
            "details": {},
            "error": "Regime detection failed: insufficient or invalid price history",
        }

        try:
            if not isinstance(historical_data, list):
                historical_data = []

            prices = []
            for item in historical_data:
                if not isinstance(item, dict):
                    continue
                price = self._safe_float(item.get("close", item.get("ltp", 0.0)), 0.0)
                if price > 0.0 and price == price and price not in (float("inf"), float("-inf")):
                    prices.append(price)

            # Need at least two points for a deterministic observer.
            if len(prices) < 2:
                raise ValueError("insufficient historical candles")

            # Prefer the repository's existing regime detector. It already
            # understands the project semantics and returns the raw details.
            detector = getattr(self.stochastic_model, "detect_market_regime", None)
            if detector is None or not callable(detector):
                raise AttributeError("StochasticPriceModel.detect_market_regime is unavailable")

            raw_result = detector(prices)
            if not isinstance(raw_result, dict):
                raise TypeError("StochasticPriceModel.detect_market_regime did not return a dictionary")

            raw_regime = str(raw_result.get("regime") or "UNKNOWN").upper()
            if raw_regime == "RANDOM_WALK":
                normalized_regime = "RANDOM"
            elif raw_regime == "MEAN_REVERTING":
                normalized_regime = "MEAN_REVERTING"
            elif raw_regime == "TRENDING":
                normalized_regime = "TRENDING"
            elif raw_regime == "UNKNOWN":
                normalized_regime = "UNKNOWN"
            else:
                normalized_regime = "UNKNOWN"

            # Confidence is derived from the available stochastic detail fields,
            # without inventing certainty. Use the existing project detail keys
            # when present, otherwise downgrade safely.
            hurst = self._safe_float(raw_result.get("hurst"), 0.5)
            variance_ratio = self._safe_float(raw_result.get("variance_ratio"), 1.0)
            confidence = 0.5
            if abs(hurst - 0.5) > 0.1:
                confidence = 0.7
            if raw_result.get("is_stationary") is not None:
                confidence = min(1.0, confidence + 0.2)
            if variance_ratio > 0.0:
                confidence = min(1.0, confidence + 0.1)

            # Keep raw/normalized details for later model consumption.
            details = {
                "hurst": hurst,
                "variance_ratio": variance_ratio,
                "volatility_regime": raw_result.get("volatility_regime", "NORMAL"),
                "recommended_strategy": raw_result.get("recommended_strategy", ""),
                "weight_adjustments": raw_result.get("weight_adjustments", {}),
                "is_stationary": raw_result.get("is_stationary", False),
            }

            return {
                "available": True,
                "regime": normalized_regime,
                "confidence": max(0.0, min(1.0, confidence)),
                "raw": raw_result,
                "details": details,
                "error": None,
            }
        except Exception as exc:
            # Deterministic fallback if the repository detector is not injected
            # or if the underling code path fails. This makes the engine safe
            # and testable while remaining within the project’s own fields.
            try:
                if not isinstance(historical_data, list):
                    historical_data = []
                prices = []
                for item in historical_data:
                    if not isinstance(item, dict):
                        continue
                    price = self._safe_float(item.get("close", item.get("ltp", 0.0)), 0.0)
                    if price > 0.0:
                        prices.append(price)
                if len(prices) < 2:
                    raise ValueError("insufficient fallback prices")

                # deterministic sign/ratio based on existing close history
                first = prices[0]
                last = prices[-1]
                if first == last:
                    return {
                        "available": True,
                        "regime": "RANDOM",
                        "confidence": 0.1,
                        "raw": {"regime": "RANDOM", "fallback": "constant-price-history"},
                        "details": {"fallback": "constant-price-history"},
                        "error": None,
                    }

                returns = [prices[i + 1] / prices[i] if prices[i] > 0 else 1.0 for i in range(len(prices) - 1)]
                up_steps = sum(1 for r in returns if r > 1.0)
                down_steps = sum(1 for r in returns if r < 1.0)
                if up_steps and down_steps:
                    regime = "RANDOM"
                elif up_steps >= down_steps:
                    regime = "TRENDING"
                else:
                    regime = "MEAN_REVERTING"

                return {
                    "available": True,
                    "regime": regime,
                    "confidence": 0.5,
                    "raw": {"regime": regime, "fallback": "deterministic-price-slope"},
                    "details": {"fallback": "deterministic-price-slope"},
                    "error": None,
                }
            except Exception:
                neutral_result["error"] = f"Regime detection failed: {exc}"
                return neutral_result

    def predict(
        self,
        live_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]],
        order_flow_data: Any = None,
        options_data: Optional[Dict[str, Any]] = None,
        market: Optional[str] = None,
        segment: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the modular prediction pipeline and return structured diagnostics."""
        live = live_data if isinstance(live_data, dict) else {}
        history = historical_data if isinstance(historical_data, list) else []
        force = self._calculate_force(live, history, options_data)
        order_flow = self._calculate_order_flow(order_flow_data)
        lead_time = self._calculate_lead_time(force["score"], history)
        adaptive_lead_time = self._calculate_adaptive_lead_time(live, history, market, segment, timeframe)
        significance = self._calculate_significance(force["score"])
        regime = self._detect_regime(history)
        ml_result = self._calculate_ml(force["factors"], lead_time, regime)
        monte_carlo = self._calculate_monte_carlo(live, history)

        combined = self.order_flow_analyzer.integrate_with_force_score(
            {"score": force["score"]},
            {"score": order_flow["score"]},
        )
        final_score = combined["final_score"]
        confidence = min(
            1.0,
            max(
                0.0,
                (abs(final_score) + self._safe_float(ml_result.get("confidence"))) / 2.0,
            ),
        )

        scoring = self._calculate_prediction_score(
            force,
            order_flow,
            ml_result,
            significance,
            monte_carlo,
        )

        self.force_history.append(force["score"])
        self.force_history = self.force_history[-100:]

        return {
            "force": force,
            "order_flow": order_flow,
            "lead_time": lead_time,
            "adaptive_lead_time": adaptive_lead_time,
            "significance": significance,
            "ml": ml_result,
            "monte_carlo": monte_carlo,
            "regime": regime,
            "combined": combined,
            "move": {"type": "UNIMPLEMENTED", "score": 0.0},
            "reversal": {"type": "UNIMPLEMENTED", "score": 0.0},
            "confidence": confidence,
            "score": final_score,
            "signal": combined["signal"],
            "combined_score": scoring.get("combined_score", 0.0),
            "direction": scoring.get("direction", "SIDEWAYS"),
            "prediction_score": scoring,
        }
