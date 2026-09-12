"""Deterministic move classification helpers for prediction score evidence.

This module intentionally stays in the analysis layer only. It never places
orders, reads broker credentials, or mutates trade state. Its purpose is to
turn a signed combined prediction score and historical evidence into a safe,
structured, deterministic move expectation payload.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


class MoveClassifier:
    """Classify a directional move from repository-safe score and candle evidence.

    The classifier accepts the engine's normalized ``combined_score`` along with
    historical candles and a regime summary, then derives a deterministic
    `category`, `expected_move_pct`, `expected_move_points`, `duration_minutes`,
    `target_price`, `confidence`, and `available` reply. The implementation is
    intentionally helper-driven and avoids one oversized function.
    """

    MICRO = "MICRO"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    BIG = "BIG"
    EXPLOSIVE = "EXPLOSIVE"

    SUPPORTED_CATEGORIES = {
        MICRO: 0,
        SMALL: 1,
        MEDIUM: 2,
        BIG: 3,
        EXPLOSIVE: 4,
    }

    def classify_move(
        self,
        combined_score: Any,
        historical: Optional[Sequence[Mapping[str, Any]]],
        regime: Optional[Mapping[str, Any]],
        mc_paths: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """Classify the likely move from a combined signed score and evidence.

        Args:
            combined_score: Any finite numeric value in roughly ``[-1, 1]``.
            historical: Sequence of candle dictionaries containing a price field
                such as ``close`` or ``ltp``. Empty, malformed, or short histories
                produce a neutral result with ``available=False``.
            regime: Optional mapping containing the repository's detected regime,
                such as the normalized ``regime`` label and confidence fields.
            mc_paths: Optional iterable of simulated path outcomes or prices.
                It is accepted for shape compatibility but never treated as a
                source of unbounded certainty.

        Returns:
            A structured dictionary with keys:
            ``category``, ``expected_move_pct``, ``expected_move_points``,
            ``duration_minutes``, ``target_price``, ``confidence``, and
            ``available``.
        """
        available, normalized_score, current_price = self._validate_inputs(
            combined_score,
            historical,
        )
        if not available:
            return self._neutral_result(
                category=self.MICRO,
                expected_move_pct=0.0,
                expected_move_points=0.0,
                duration_minutes=0,
                target_price=0.0,
                confidence=0.0,
                available=False,
                reason="Move classification failed: invalid inputs",
            )

        prices = self._extract_prices(historical)
        if len(prices) < 2:
            return self._neutral_result(
                category=self.MICRO,
                expected_move_pct=0.0,
                expected_move_points=0.0,
                duration_minutes=0,
                target_price=current_price,
                confidence=0.0,
                available=False,
                reason="Move classification failed: insufficient candles",
            )

        volatility = self._volatility(prices)
        if volatility <= 0.0:
            return self._neutral_result(
                category=self.MICRO,
                expected_move_pct=0.0,
                expected_move_points=0.0,
                duration_minutes=0,
                target_price=current_price,
                confidence=0.0,
                available=False,
                reason="Move classification failed: zero volatility",
            )

        mc_path_count = self._count_mc_paths(mc_paths)
        if mc_path_count == 0:
            # Empty MC paths are acceptable evidence-safely. They suppress
            # the confidence boost but do not invalidate the classifier.
            mc_path_count = 0

        regime_label = self._regime_label(regime)
        regime_confidence = self._regime_confidence(regime)

        category = self._category_from_score(abs(normalized_score))
        move_pct = self._expected_move_pct(
            abs(normalized_score),
            volatility,
            current_price,
            regime_label,
            regime_confidence,
        )
        duration_minutes = self._duration_minutes(regime_label, len(prices), volatility)
        move_points = self._expected_move_points(move_pct, current_price)
        target_price = self._target_price(current_price, normalized_score, move_pct)
        confidence = self._confidence(
            abs(normalized_score),
            volatility,
            regime_confidence,
            mc_path_count,
        )

        return {
            "category": category,
            "expected_move_pct": self._clamp_pct(move_pct),
            "expected_move_points": self._clamp_points(move_points),
            "duration_minutes": int(duration_minutes),
            "target_price": self._round_price(target_price),
            "confidence": self._clamp_confidence(confidence),
            "available": True,
        }

    @staticmethod
    def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
        """Clamp a finite numeric value into a bounded interval deterministically."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return 0.0
        return max(low, min(high, numeric))

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Return a finite float or the safe default if the input is not numeric."""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
        if numeric != numeric or numeric in (float("inf"), float("-inf")):
            return default
        return numeric

    @staticmethod
    def _finite_positive_price(price: Any) -> Optional[float]:
        """Return the price only if it is cleanly positive and finite."""
        numeric = MoveClassifier._safe_float(price, 0.0)
        if numeric <= 0.0:
            return None
        return numeric

    def _validate_inputs(
        self,
        combined_score: Any,
        historical: Optional[Sequence[Mapping[str, Any]]],
    ) -> Tuple[bool, float, float]:
        """Validate the top-level inputs and return a numeric score plus current price."""
        try:
            normalized_score = self._safe_float(combined_score, 0.0)
        except Exception:
            normalized_score = 0.0
        if normalized_score != normalized_score or normalized_score in (float("inf"), float("-inf")):
            normalized_score = 0.0

        if not isinstance(historical, Sequence) or isinstance(historical, (str, bytes)):
            return False, normalized_score, 0.0
        if len(historical) < 2:
            return False, normalized_score, 0.0

        prices = self._extract_prices(historical)
        if len(prices) < 2:
            return False, normalized_score, 0.0

        current_price = prices[-1]
        if current_price <= 0.0:
            return False, normalized_score, 0.0

        return True, self._clamp(normalized_score, -1.0, 1.0), current_price

    def _extract_prices(self, historical: Optional[Sequence[Mapping[str, Any]]]) -> List[float]:
        """Extract valid candle-close or ltp prices from the repository's list shape."""
        if not isinstance(historical, Sequence) or isinstance(historical, (str, bytes)):
            return []

        prices: List[float] = []
        for candle in historical:
            if not isinstance(candle, Mapping):
                continue
            close_price = self._safe_float(candle.get("close", candle.get("ltp", candle.get("price", 0.0))), 0.0)
            if close_price <= 0.0:
                continue
            if close_price != close_price or close_price in (float("inf"), float("-inf")):
                continue
            prices.append(close_price)

        return prices

    def _current_price_from_historical(self, historical: Optional[Sequence[Mapping[str, Any]]]) -> float:
        """Return the most recent valid price from the untouched historical record."""
        prices = self._extract_prices(historical)
        if not prices:
            return 0.0
        return prices[-1]

    def _volatility(self, prices: Sequence[float]) -> float:
        """Estimate deterministic relative volatility from the last price series."""
        if len(prices) < 2:
            return 0.0
        series = [self._safe_float(price, 0.0) for price in prices]
        if not series:
            return 0.0
        mean = sum(series) / float(len(series))
        variance = sum((price - mean) ** 2 for price in series) / float(max(1, len(series) - 1))
        std = math.sqrt(max(variance, 0.0))
        if prices[-1] <= 0.0:
            return 0.0
        return std / prices[-1]

    def _count_mc_paths(self, mc_paths: Optional[Sequence[Any]]) -> int:
        """Count Monte Carlo paths safely while supporting empty or invalid lists."""
        if mc_paths is None:
            return 0
        if isinstance(mc_paths, (str, bytes)):
            return 0
        if not isinstance(mc_paths, Sequence):
            return 0
        # Accept either a raw numeric sequence or a mapping containing a list.
        if isinstance(mc_paths, Mapping):
            raw_paths = mc_paths.get("paths") or mc_paths.get("mc_paths") or mc_paths.get("data")
            if isinstance(raw_paths, Sequence) and not isinstance(raw_paths, (str, bytes)):
                return len(raw_paths)
            return 0
        try:
            return len(list(mc_paths))
        except Exception:
            return 0

    def _regime_label(self, regime: Optional[Mapping[str, Any]]) -> str:
        """Return the normalized regime label for deterministic duration scaling."""
        if not isinstance(regime, Mapping):
            return "UNKNOWN"
        raw_regime = str(regime.get("regime") or regime.get("name") or "UNKNOWN").upper()
        if raw_regime == "RANDOM_WALK":
            return "RANDOM"
        if raw_regime == "MEAN_REVERTING":
            return "MEAN_REVERTING"
        if raw_regime == "TRENDING":
            return "TRENDING"
        return raw_regime if raw_regime else "UNKNOWN"

    def _regime_confidence(self, regime: Optional[Mapping[str, Any]]) -> float:
        """Read regime confidence from the repository's normalized regime structure."""
        if not isinstance(regime, Mapping):
            return 0.0
        confidence = self._safe_float(regime.get("confidence"), 0.0)
        if confidence <= 0.0:
            confidence = self._safe_float(regime.get("conf"), 0.0)
        if confidence <= 0.0:
            confidence = 0.0
        return max(0.0, min(1.0, confidence))

    def _category_from_score(self, score_magnitude: float) -> str:
        """Map absolute combined-score magnitude to a supported move category."""
        if score_magnitude >= 0.75:
            return self.EXPLOSIVE
        if score_magnitude >= 0.55:
            return self.BIG
        if score_magnitude >= 0.35:
            return self.MEDIUM
        if score_magnitude >= 0.20:
            return self.SMALL
        return self.MICRO

    def _expected_move_pct(
        self,
        score_magnitude: float,
        volatility: float,
        current_price: float,
        regime_label: str,
        regime_confidence: float,
    ) -> float:
        """Estimate a deterministic percentage move from safe evidence sources."""
        # Deterministic baseline: score magnitude drives direction strength,
        # volatility adjusts the expected move size, and regime confidence
        # weakly modulates it. No fabricated certainty.
        base = 0.03 + score_magnitude * 0.35
        volatility_factor = max(volatility * 100.0, 0.0)
        regime_factor = 1.0 + (0.10 * regime_confidence)
        if regime_label == "TRENDING":
            regime_factor += 0.05
        elif regime_label == "MEAN_REVERTING":
            regime_factor -= 0.05
        elif regime_label == "RANDOM":
            regime_factor -= 0.025

        pct = base * regime_factor + volatility_factor * 0.015
        if pct <= 0.0:
            pct = 0.0
        if current_price <= 0.0:
            pct = 0.0
        return min(max(pct, 0.0), 10.0)

    def _duration_minutes(
        self,
        regime_label: str,
        candle_count: int,
        volatility: float,
    ) -> int:
        """Return the deterministic expected duration bucket in minutes."""
        # Use a conservative default; historical candle count and volatility
        # only influence the expected lifetime without inventing urgency.
        if regime_label == "TRENDING":
            base = 90
        elif regime_label == "MEAN_REVERTING":
            base = 60
        elif regime_label == "RANDOM":
            base = 45
        else:
            base = 60

        # Slightly scale with volatility but keep the result deterministic.
        if volatility > 0.0:
            base += int(round(volatility * 1000.0))
        base = max(5, min(base, 240))
        return base

    def _expected_move_points(self, expected_move_pct: float, current_price: float) -> float:
        """Convert the percentage move estimate into point units deterministically."""
        if current_price <= 0.0:
            return 0.0
        return (expected_move_pct / 100.0) * current_price

    def _target_price(self, current_price: float, combined_score: float, move_pct: float) -> float:
        """Compute an evidence-based target price using the direction from the score."""
        if current_price <= 0.0:
            return 0.0
        direction = 1.0 if combined_score >= 0.0 else -1.0
        return current_price * (1.0 + direction * (move_pct / 100.0))

    def _confidence(
        self,
        score_magnitude: float,
        volatility: float,
        regime_confidence: float,
        mc_path_count: int,
    ) -> float:
        """Return a deterministic confidence score in the repository-safe range [0,1]."""
        confidence = 0.35 * score_magnitude
        confidence += min(0.25, volatility * 50.0)
        confidence += 0.20 * regime_confidence
        if mc_path_count > 0:
            confidence += 0.20
        return max(0.0, min(1.0, confidence))

    def _round_price(self, price: float) -> float:
        """Round target price to a deterministic finite number without surprises."""
        if price != price or price in (float("inf"), float("-inf")):
            return 0.0
        return round(price, 4)

    def _clamp_pct(self, pct: float) -> float:
        """Clamp the expected percentage move to a stable, reportable range."""
        return max(0.0, min(10.0, self._safe_float(pct, 0.0)))

    def _clamp_points(self, points: float) -> float:
        """Clamp points to a non-negative finite range."""
        return max(0.0, self._safe_float(points, 0.0))

    def _clamp_confidence(self, confidence: float) -> float:
        """Clamp confidence deterministically to the repository-safe probability range."""
        return max(0.0, min(1.0, self._safe_float(confidence, 0.0)))

    def _neutral_result(
        self,
        category: str,
        expected_move_pct: float,
        expected_move_points: float,
        duration_minutes: int,
        target_price: float,
        confidence: float,
        available: bool,
        reason: str,
    ) -> Dict[str, Any]:
        """Build a neutral result dictionary that is safe on failure or malformed input."""
        return {
            "category": category,
            "expected_move_pct": max(0.0, expected_move_pct),
            "expected_move_points": max(0.0, expected_move_points),
            "duration_minutes": int(duration_minutes),
            "target_price": self._round_price(target_price),
            "confidence": self._clamp_confidence(confidence),
            "available": available,
            "reason": reason,
        }
