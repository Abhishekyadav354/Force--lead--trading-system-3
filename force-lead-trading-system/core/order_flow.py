"""Order-flow analysis helpers for market microstructure signals.

This module is intentionally kept as a lightweight, dependency-free skeleton for
future order-flow research and signal generation. The goal is to provide a
stable API surface for analyzing price/volume behavior without coupling the code
base to any broker, exchange, or live order-placement logic.

Expected input schema for future analysis methods:
- candle dictionaries containing price and volume fields such as open, high, low,
  close, volume, and timestamp values.
- tick-level price movement data represented as numeric deltas or sequences of
  candles that can be aggregated into cumulative movement statistics.
- derived trade/volume metrics used for divergence, absorption, exhaustion, and
  VWAP analysis.

Expected output schema for future analysis methods:
- classification strings such as "BUY", "SELL", "NEUTRAL", or directional
  labels for imbalance/delta conditions.
- numeric metrics including price deltas, cumulative delta, delta per minute,
  VWAP bands, imbalance scores, and summary statistics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional


class OrderFlowAnalyzer:
    """Analyze price and volume flow characteristics from candle or tick data.

    The initial implementation is a structural scaffold only. It defines the core
    state used by more advanced order-flow methods and exposes placeholder method
    signatures that raise NotImplementedError until the analysis logic is added.
    """

    def __init__(self) -> None:
        self.prev_price: Optional[float] = None
        self.prev_classification: str = "NEUTRAL"
        self.candle_deltas: List[float] = []
        self.vwap_ticks: List[float] = []

    def classify_tick(self, tick: Dict[str, Any]) -> Dict[str, Any]:
        """Classify one tick and return a normalized result payload."""
        if not isinstance(tick, dict):
            raise ValueError("Tick must be provided as a dictionary.")

        required_fields = ("price", "volume", "timestamp", "bid", "ask")
        missing_fields = [field for field in required_fields if field not in tick]
        if missing_fields:
            raise ValueError(f"Tick is missing required fields: {', '.join(missing_fields)}")

        try:
            price = float(tick["price"])
        except (TypeError, ValueError):
            raise ValueError("Tick price must be a valid number.")

        try:
            volume = float(tick["volume"])
        except (TypeError, ValueError):
            raise ValueError("Tick volume must be a valid number.")

        if not (price > 0):
            raise ValueError("Tick price must be greater than zero.")
        if volume < 0:
            raise ValueError("Tick volume cannot be negative.")

        bid = tick.get("bid")
        ask = tick.get("ask")
        try:
            bid_value = float(bid) if bid is not None else None
        except (TypeError, ValueError):
            bid_value = None

        try:
            ask_value = float(ask) if ask is not None else None
        except (TypeError, ValueError):
            ask_value = None

        previous_price = self.prev_price
        previous_classification = self.prev_classification

        if price >= ask_value if ask_value is not None else False:
            classification = "BUY_INITIATED"
        elif price <= bid_value if bid_value is not None else False:
            classification = "SELL_INITIATED"
        else:
            if previous_price is not None:
                if price > previous_price:
                    classification = "BUY_INITIATED"
                elif price < previous_price:
                    classification = "SELL_INITIATED"
                elif previous_classification in {"BUY_INITIATED", "SELL_INITIATED"}:
                    classification = previous_classification
                else:
                    classification = "NEUTRAL"
            else:
                classification = "NEUTRAL"

        self.prev_price = price
        self.prev_classification = classification

        result = {
            "type": classification,
            "volume": volume,
            "price": price,
            "timestamp": tick["timestamp"],
        }

        return result

    def calculate_delta(self, classified_ticks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate directional volume from classified ticks into a delta summary.

        Args:
            classified_ticks: Sequence of tick dictionaries produced by classify_tick().

        Returns:
            Dictionary with directional volume totals, delta, total volume, and delta ratio.
        """
        if classified_ticks is None:
            return {
                "delta": 0.0,
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "delta_ratio": 0.5,
                "total_volume": 0.0,
            }

        buy_volume = 0.0
        sell_volume = 0.0

        for tick in classified_ticks:
            if not isinstance(tick, dict):
                continue

            tick_type = str(tick.get("type", "")).upper()
            volume = tick.get("volume", 0)

            try:
                volume_value = float(volume)
            except (TypeError, ValueError):
                continue

            # Validation: directional volume should never be negative.
            if volume_value < 0:
                raise ValueError("Tick volume cannot be negative.")

            if tick_type == "BUY_INITIATED":
                buy_volume += volume_value
            elif tick_type == "SELL_INITIATED":
                sell_volume += volume_value

        # Directional delta is net buying pressure: buy volume minus sell volume.
        delta = buy_volume - sell_volume
        total_directional_volume = buy_volume + sell_volume

        # For a zero-directional-volume sample, use a neutral ratio to avoid division-by-zero.
        if total_directional_volume == 0:
            delta_ratio = 0.5
        else:
            delta_ratio = buy_volume / total_directional_volume

        return {
            "delta": delta,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "delta_ratio": delta_ratio,
            "total_volume": total_directional_volume,
        }

    def calculate_cumulative_delta(
        self,
        candle_deltas: List[Dict[str, Any]],
    ) -> List[Dict[str, float]]:
        """Build cumulative delta series from candle-level delta records.

        Each entry in the input list is expected to contain a timestamp and a numeric
        "delta" field. The method returns a list of dictionaries with the original
        timestamp, the individual delta, and the running cumulative total.
        """
        cumulative: List[Dict[str, float]] = []
        running_total = 0.0

        for record in candle_deltas or []:
            if not isinstance(record, dict):
                continue

            try:
                value = float(record.get("delta", 0.0))
            except (TypeError, ValueError):
                value = 0.0

            running_total += value
            cumulative.append({
                "timestamp": record.get("timestamp"),
                "delta": value,
                "cumulative_delta": running_total,
            })

        if len(cumulative) < 2:
            for item in cumulative:
                item["slope"] = 0.0
            return cumulative

        recent_values = [float(item["cumulative_delta"]) for item in cumulative[-5:]]
        x_values = list(range(len(recent_values)))
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(recent_values) / len(recent_values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, recent_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        slope = numerator / denominator if denominator != 0 else 0.0

        for item in cumulative:
            item["slope"] = slope

        return cumulative

    def calculate_delta_per_minute(
        self,
        delta: float,
        elapsed_seconds: float,
        historical_avg_delta_pm: Optional[float] = None,
    ) -> Dict[str, float]:
        """Convert a raw delta into a per-minute value and optional normalized reading.

        Args:
            delta: Directional delta magnitude for the interval.
            elapsed_seconds: Length of the interval in seconds.
            historical_avg_delta_pm: Baseline average delta-per-minute for normalization.

        Returns:
            A dictionary containing delta_per_minute, normalized_delta, and historical average.
        """
        try:
            delta_value = float(delta)
        except (TypeError, ValueError):
            delta_value = 0.0

        try:
            elapsed = float(elapsed_seconds)
        except (TypeError, ValueError):
            elapsed = 0.0

        if elapsed <= 0:
            delta_per_minute = 0.0
            normalized_delta = 0.0
            historical_baseline = 0.0 if historical_avg_delta_pm is None else float(historical_avg_delta_pm)
            return {
                "delta_per_minute": delta_per_minute,
                "normalized_delta": normalized_delta,
                "historical_avg_delta_pm": historical_baseline,
            }

        delta_per_minute = delta_value / (elapsed / 60.0)

        if historical_avg_delta_pm is None:
            historical_baseline = 0.0
            normalized_delta = 0.0
        else:
            try:
                historical_baseline = float(historical_avg_delta_pm)
            except (TypeError, ValueError):
                historical_baseline = 0.0

            if historical_baseline == 0:
                normalized_delta = 0.0
            else:
                normalized_delta = delta_per_minute / historical_baseline

        return {
            "delta_per_minute": delta_per_minute,
            "normalized_delta": normalized_delta,
            "historical_avg_delta_pm": historical_baseline,
        }

    @staticmethod
    def _linear_regression_slope(values: List[float]) -> float:
        """Return the slope from a simple least-squares fit over equally spaced points."""
        if len(values) < 2:
            return 0.0

        x_values = list(range(len(values)))
        x_mean = sum(x_values) / len(x_values)
        y_mean = sum(values) / len(values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def detect_delta_divergence(
        self,
        price_series: List[float],
        delta_series: List[float],
        window: int = 5,
    ) -> Dict[str, Any]:
        """Detect directional divergence between price trend and delta trend.

        This is a statistical diagnostic only. It does not guarantee a future move.
        """
        if not isinstance(price_series, list) or not isinstance(delta_series, list):
            raise ValueError("price_series and delta_series must be lists.")

        if len(price_series) != len(delta_series):
            raise ValueError("price_series and delta_series must have equal length.")

        if window <= 0:
            return {
                "type": "NONE",
                "strength": 0.0,
                "price_slope": 0.0,
                "delta_slope": 0.0,
                "signal": "0",
            }

        if len(price_series) < 2 or len(delta_series) < 2:
            return {
                "type": "NONE",
                "strength": 0.0,
                "price_slope": 0.0,
                "delta_slope": 0.0,
                "signal": "0",
            }

        latest_prices = [float(value) for value in price_series[-window:]]
        latest_deltas = [float(value) for value in delta_series[-window:]]

        if len(latest_prices) < 2 or len(latest_deltas) < 2:
            return {
                "type": "NONE",
                "strength": 0.0,
                "price_slope": 0.0,
                "delta_slope": 0.0,
                "signal": "0",
            }

        price_slope = self._linear_regression_slope(latest_prices)
        delta_slope = self._linear_regression_slope(latest_deltas)

        # Bullish divergence: falling prices with rising delta pressure.
        # Bearish divergence: rising prices with falling delta pressure.
        if price_slope < 0 and delta_slope > 0:
            divergence_type = "BULLISH"
            signal = "1"
        elif price_slope > 0 and delta_slope < 0:
            divergence_type = "BEARISH"
            signal = "-1"
        else:
            return {
                "type": "NONE",
                "strength": 0.0,
                "price_slope": price_slope,
                "delta_slope": delta_slope,
                "signal": "0",
            }

        denominator = abs(price_slope) + abs(delta_slope)
        if denominator == 0:
            strength = 0.0
        else:
            strength = min(1.0, max(0.0, abs(delta_slope) / (denominator + 1e-12)))

        return {
            "type": divergence_type,
            "strength": strength,
            "price_slope": price_slope,
            "delta_slope": delta_slope,
            "signal": signal,
        }

    def build_volume_profile(
        self,
        ticks: List[Dict[str, Any]],
        price_levels: int = 20,
        current_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build a simple volume-profile summary from tick data.

        Args:
            ticks: Sequence of tick dictionaries with price, volume, bid, ask, and type.
            price_levels: Number of price bins to use for the profile.
            current_price: Optional price used for proximity-based profile metadata.

        Returns:
            Dictionary containing per-level volume statistics, POC, VAH, VAL, and signal.
        """
        if price_levels < 1:
            raise ValueError("price_levels must be at least 1.")

        if not ticks:
            return {
                "profile": {},
                "poc": 0.0,
                "vah": 0.0,
                "val": 0.0,
                "poc_signal": "0",
                "total_volume": 0.0,
            }

        valid_prices: List[float] = []
        for tick in ticks:
            if not isinstance(tick, dict):
                continue
            try:
                value = float(tick.get("price", 0.0))
            except (TypeError, ValueError):
                continue
            if value > 0:
                valid_prices.append(value)

        if not valid_prices:
            return {
                "profile": {},
                "poc": 0.0,
                "vah": 0.0,
                "val": 0.0,
                "poc_signal": "0",
                "total_volume": 0.0,
            }

        low = min(valid_prices)
        high = max(valid_prices)

        if high == low:
            price_step = 1.0
            profile_levels = {0.0: {"buy_volume": 0.0, "sell_volume": 0.0, "total_volume": 0.0}}
        else:
            price_step = (high - low) / max(1, price_levels)
            profile_levels: Dict[float, Dict[str, float]] = {}
            for index in range(price_levels):
                level_start = low + (index * price_step)
                level_end = low + ((index + 1) * price_step)
                level_mid = (level_start + level_end) / 2.0
                profile_levels[round(level_mid, 6)] = {
                    "buy_volume": 0.0,
                    "sell_volume": 0.0,
                    "total_volume": 0.0,
                }

        for tick in ticks:
            if not isinstance(tick, dict):
                continue

            try:
                price = float(tick.get("price", 0.0))
                volume = float(tick.get("volume", 0.0))
            except (TypeError, ValueError):
                continue

            if price <= 0 or volume < 0:
                continue

            if high == low:
                level_key = 0.0
            else:
                level_index = min(int((price - low) / max(price_step, 1e-12)), price_levels - 1)
                level_key = round(low + (level_index * price_step) + (price_step / 2.0), 6)

            bucket = profile_levels.setdefault(level_key, {
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "total_volume": 0.0,
            })

            tick_type = str(tick.get("type", "")).upper()
            if tick_type == "BUY_INITIATED":
                bucket["buy_volume"] += volume
            elif tick_type == "SELL_INITIATED":
                bucket["sell_volume"] += volume

            bucket["total_volume"] += volume

        level_totals = [entry["total_volume"] for entry in profile_levels.values()]
        if not level_totals or max(level_totals) == 0:
            poc = float(round((low + high) / 2.0, 6))
            vah = float(round(high, 6))
            val = float(round(low, 6))
            poc_signal = "0"
            profile_summary = {"levels": profile_levels}
            return {
                "profile": profile_summary,
                "poc": poc,
                "vah": vah,
                "val": val,
                "poc_signal": poc_signal,
                "total_volume": 0.0,
            }

        ranked_levels = sorted(
            profile_levels.items(),
            key=lambda item: item[1].get("total_volume", 0.0),
            reverse=True,
        )
        poc_key, poc_bucket = ranked_levels[0]
        poc = float(round(poc_key, 6))

        total_profile_volume = sum(bucket["total_volume"] for _, bucket in ranked_levels)
        target_volume = total_profile_volume * 0.70
        cumulative = 0.0
        value_area_levels: List[float] = []
        for level_key, bucket in ranked_levels:
            cumulative += bucket["total_volume"]
            value_area_levels.append(float(level_key))
            if cumulative >= target_volume:
                break

        if value_area_levels:
            vah = float(round(max(value_area_levels), 6))
            val = float(round(min(value_area_levels), 6))
        else:
            vah = float(round(poc, 6))
            val = float(round(poc, 6))

        if current_price is None:
            current_price = max((float(tick.get("price", 0.0)) for tick in ticks if isinstance(tick, dict) and tick.get("price") is not None), default=(low + high) / 2.0)

        current_price_value = float(current_price)
        poc_signal = self.get_poc_signal(current_price_value, poc, vah, val)

        return {
            "profile": {"levels": profile_levels},
            "poc": poc,
            "vah": vah,
            "val": val,
            "poc_signal": poc_signal,
            "total_volume": float(round(total_profile_volume, 6)),
        }

    def get_poc_signal(
        self,
        current_price: Optional[float],
        poc: Optional[float],
        vah: Optional[float],
        val: Optional[float],
        tolerance: float = 0.001,
    ) -> str:
        """Classify the current price relative to the volume profile state.

        This helper is descriptive only; it does not produce orders or trading signals.
        """
        try:
            current_value = float(current_price) if current_price is not None else None
        except (TypeError, ValueError):
            current_value = None

        try:
            poc_value = float(poc) if poc is not None else None
        except (TypeError, ValueError):
            poc_value = None

        try:
            vah_value = float(vah) if vah is not None else None
        except (TypeError, ValueError):
            vah_value = None

        try:
            val_value = float(val) if val is not None else None
        except (TypeError, ValueError):
            val_value = None

        if current_value is None or poc_value is None or vah_value is None or val_value is None:
            return "NEAR_POC"

        # Boundary checks are intentionally conservative to avoid noisy transitions.
        if current_value > vah_value:
            return "ABOVE_VALUE"
        if current_value < val_value:
            return "BELOW_VALUE"

        if abs(current_value - poc_value) <= max(0.0, float(tolerance)):
            return "NEAR_POC"

        if current_value > poc_value:
            return "ABOVE_POC"
        if current_value < poc_value:
            return "BELOW_POC"

        return "NEAR_POC"

    def build_footprint(
        self,
        ticks: List[Dict[str, Any]],
        candle_open: Optional[float] = None,
        candle_close: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Aggregate tick-level volume by rounded price and summarize buy/sell imbalance.

        Args:
            ticks: Sequence of tick dictionaries containing price, volume, and type.
            candle_open: Optional candle open price used only for context.
            candle_close: Optional candle close price used only for context.

        Returns:
            Dictionary with the main footprint summary and stacked imbalance zones.
        """
        if not isinstance(ticks, list):
            raise ValueError("ticks must be provided as a list.")

        if not ticks:
            return {
                "footprint": [],
                "stacked_buy_zones": [],
                "stacked_sell_zones": [],
                "max_buy_level": 0.0,
                "max_sell_level": 0.0,
                "candle_open": candle_open,
                "candle_close": candle_close,
            }

        grouped: Dict[float, Dict[str, Any]] = {}
        for tick in ticks:
            if not isinstance(tick, dict):
                continue

            try:
                price = float(tick.get("price", 0.0))
                volume = float(tick.get("volume", 0.0))
            except (TypeError, ValueError):
                continue

            if price <= 0 or volume < 0:
                continue

            rounded_price = round(price, 2)
            bucket = grouped.setdefault(
                rounded_price,
                {"price": rounded_price, "buy_volume": 0.0, "sell_volume": 0.0, "total_volume": 0.0},
            )

            tick_type = str(tick.get("type", "")).upper()
            if tick_type == "BUY_INITIATED":
                bucket["buy_volume"] += volume
            elif tick_type == "SELL_INITIATED":
                bucket["sell_volume"] += volume
            bucket["total_volume"] += volume

        footprint_levels: List[Dict[str, Any]] = []
        buy_levels: List[Dict[str, Any]] = []
        sell_levels: List[Dict[str, Any]] = []

        for price in sorted(grouped.keys()):
            bucket = grouped[price]
            buy_volume = float(bucket["buy_volume"])
            sell_volume = float(bucket["sell_volume"])
            total_volume = float(bucket["total_volume"])

            if buy_volume > 0 and sell_volume > 0:
                ratio = buy_volume / sell_volume if sell_volume else float("inf")
                if ratio > 3:
                    imbalance = "BUY_IMBALANCE"
                elif (sell_volume / buy_volume) > 3:
                    imbalance = "SELL_IMBALANCE"
                else:
                    imbalance = "BALANCED"
            elif buy_volume > 0:
                imbalance = "BUY_IMBALANCE"
            elif sell_volume > 0:
                imbalance = "SELL_IMBALANCE"
            else:
                imbalance = "BALANCED"

            level_entry = {
                "price": float(price),
                "buy_volume": buy_volume,
                "sell_volume": sell_volume,
                "total_volume": total_volume,
                "imbalance": imbalance,
            }
            footprint_levels.append(level_entry)

            if imbalance == "BUY_IMBALANCE":
                buy_levels.append({"price": float(price), "imbalance_type": imbalance})
            elif imbalance == "SELL_IMBALANCE":
                sell_levels.append({"price": float(price), "imbalance_type": imbalance})

        buy_prices = [float(level["price"]) for level in buy_levels]
        sell_prices = [float(level["price"]) for level in sell_levels]
        stacked_buy = self.detect_stacked_imbalance(
            {"levels": buy_levels},
            buy_prices,
            "BUY_IMBALANCE",
            min_stack=3,
        )
        stacked_sell = self.detect_stacked_imbalance(
            {"levels": sell_levels},
            sell_prices,
            "SELL_IMBALANCE",
            min_stack=3,
        )

        max_buy_level = max((level["price"] for level in buy_levels), default=0.0)
        max_sell_level = max((level["price"] for level in sell_levels), default=0.0)

        return {
            "footprint": footprint_levels,
            "stacked_buy_zones": stacked_buy,
            "stacked_sell_zones": stacked_sell,
            "max_buy_level": max_buy_level,
            "max_sell_level": max_sell_level,
            "candle_open": candle_open,
            "candle_close": candle_close,
        }

    def detect_stacked_imbalance(
        self,
        footprint: Optional[Dict[str, Any]],
        prices: Optional[List[float]],
        imbalance_type: str,
        min_stack: int = 3,
    ) -> List[Dict[str, Any]]:
        """Identify consecutive price levels that match a repeated imbalance type.

        The detector walks a price series in ascending order and groups consecutive
        levels whose imbalance matches the requested type. A valid stack must meet
        or exceed ``min_stack`` consecutive levels. This helper intentionally only
        reports contiguous price zones and does not claim support/resistance.

        Args:
            footprint: Optional footprint payload containing level entries.
            prices: Ascending price levels to scan.
            imbalance_type: Imbalance label to match, e.g. "BUY_IMBALANCE".
            min_stack: Minimum number of consecutive matching levels required.

        Returns:
            List of zone dictionaries with ``price_low``, ``price_high``, ``levels``,
            and ``strength``. Each zone is independent and reusable.

        Raises:
            ValueError: If ``min_stack`` is less than 1.
        """
        if min_stack < 1:
            raise ValueError("min_stack must be >= 1")

        if not isinstance(footprint, dict):
            footprint = {}

        levels = footprint.get("levels") or []
        if not isinstance(levels, list):
            levels = []

        if not prices:
            return []

        normalized_prices = []
        for price in prices:
            try:
                value = float(price)
            except (TypeError, ValueError):
                continue
            if value > 0:
                normalized_prices.append(value)

        if not normalized_prices:
            return []

        ordered_prices = sorted(set(normalized_prices))
        desired_type = str(imbalance_type).upper()

        level_lookup = {
            float(item.get("price", 0.0)): str(item.get("imbalance", "")).upper()
            for item in levels
            if isinstance(item, dict)
        }

        zones: List[Dict[str, Any]] = []
        current_start: Optional[float] = None
        current_end: Optional[float] = None
        current_levels: List[float] = []

        for price in ordered_prices:
            match = level_lookup.get(price, "") == desired_type
            if match:
                if current_start is None:
                    current_start = price
                    current_end = price
                    current_levels = [price]
                else:
                    current_end = price
                    current_levels.append(price)
                continue

            if current_start is not None and len(current_levels) >= min_stack:
                zones.append({
                    "price_low": current_start,
                    "price_high": current_end,
                    "levels": [
                        {"price": level, "imbalance": desired_type, "strength": 1}
                        for level in current_levels
                    ],
                    "strength": len(current_levels),
                })

            current_start = None
            current_end = None
            current_levels = []

        if current_start is not None and len(current_levels) >= min_stack:
            zones.append({
                "price_low": current_start,
                "price_high": current_end,
                "levels": [
                    {"price": level, "imbalance": desired_type, "strength": 1}
                    for level in current_levels
                ],
                "strength": len(current_levels),
            })

        return zones

    def detect_absorption(
        self,
        ticks: List[Dict[str, Any]],
        price_series: List[float],
        window_seconds: int = 60,
    ) -> Dict[str, Any]:
        """Detect possible absorption from directional tick volume and price stability.

        Recent ticks are selected from their timestamps when timestamps are
        available and parseable. If timestamps are unavailable, the supplied
        sequence is used as-is without assuming a time scale. This identifies a
        possible price/volume condition only; it is not evidence of institutional
        activity or a guarantee of support or resistance.

        Args:
            ticks: Tick dictionaries with ``type`` (or ``classification``),
                ``volume``, and optional ``timestamp`` fields.
            price_series: Ordered prices used to measure recent price movement.
            window_seconds: Lookback window when usable timestamps are present.

        Returns:
            A dictionary containing ``type``, bounded ``strength`` (0-100), and
            ``signal``: 1 for possible buy absorption, -1 for possible sell
            absorption, and 0 for no absorption.
        """
        if window_seconds < 0:
            raise ValueError("window_seconds must be >= 0")

        if not isinstance(ticks, list) or not isinstance(price_series, list):
            return {"type": "NONE", "strength": 0.0, "signal": 0}

        def parse_timestamp(value: Any) -> Optional[float]:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, datetime):
                timestamp = value
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                return timestamp.timestamp()
            if isinstance(value, str):
                try:
                    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=timezone.utc)
                    return timestamp.timestamp()
                except ValueError:
                    return None
            return None

        timestamped_ticks = []
        for tick in ticks:
            if not isinstance(tick, dict):
                continue
            timestamp = parse_timestamp(tick.get("timestamp"))
            if timestamp is not None:
                timestamped_ticks.append((timestamp, tick))

        if timestamped_ticks:
            latest_timestamp = max(timestamp for timestamp, _ in timestamped_ticks)
            recent_ticks = [
                tick
                for timestamp, tick in timestamped_ticks
                if latest_timestamp - timestamp <= window_seconds
            ]
        else:
            recent_ticks = [tick for tick in ticks if isinstance(tick, dict)]

        buy_volume = 0.0
        sell_volume = 0.0
        for tick in recent_ticks:
            try:
                volume = float(tick.get("volume", 0.0))
            except (TypeError, ValueError):
                continue
            if volume <= 0:
                continue

            classification = str(
                tick.get("type", tick.get("classification", ""))
            ).upper()
            if classification in {"BUY_INITIATED", "BUY"}:
                buy_volume += volume
            elif classification in {"SELL_INITIATED", "SELL"}:
                sell_volume += volume

        try:
            numeric_prices = [float(price) for price in price_series]
        except (TypeError, ValueError):
            numeric_prices = []
        numeric_prices = [price for price in numeric_prices if price > 0]

        total_volume = buy_volume + sell_volume
        if total_volume <= 0 or len(numeric_prices) < 2:
            return {"type": "NONE", "strength": 0.0, "signal": 0}

        price_start = numeric_prices[0]
        price_change = numeric_prices[-1] - price_start
        stable_threshold = max(abs(price_start) * 0.001, 1e-9)
        sell_pressure = sell_volume / total_volume
        buy_pressure = buy_volume / total_volume
        pressure_threshold = 0.6

        if sell_pressure >= pressure_threshold and price_change >= -stable_threshold:
            return {
                "type": "BUY_ABSORPTION",
                "strength": min(100.0, round(sell_pressure * 100.0, 2)),
                "signal": 1,
            }

        if buy_pressure >= pressure_threshold and price_change <= stable_threshold:
            return {
                "type": "SELL_ABSORPTION",
                "strength": min(100.0, round(buy_pressure * 100.0, 2)),
                "signal": -1,
            }

        return {"type": "NONE", "strength": 0.0, "signal": 0}

    def detect_exhaustion(
        self,
        delta_series: List[float],
        price_series: List[float],
        volume_series: List[float],
    ) -> Dict[str, Any]:
        """Detect declining delta-to-price efficiency in recent observations.

        Efficiency is measured as volume-normalized absolute delta divided by
        absolute price movement. Near-zero price moves are skipped rather than
        allowed to create unstable ratios. A negative least-squares slope across
        the recent efficiency values is treated as a possible exhaustion signal.
        The returned reversal value is a bounded heuristic score, not a calibrated
        statistical probability.

        Args:
            delta_series: Delta values aligned with the price and volume series.
            price_series: Price observations aligned with the other series.
            volume_series: Volume observations used to normalize delta pressure.

        Returns:
            Dictionary containing ``is_exhausted``, bounded ``strength``, the
            regression ``efficiency_trend``, and a bounded ``reversal_probability``.
            The latter is a heuristic ranking value for backtesting and paper
            simulation, not a calibrated probability.
        """
        empty_result = {
            "is_exhausted": False,
            "strength": 0.0,
            "efficiency_trend": 0.0,
            "reversal_probability": 0.0,
        }

        if not all(isinstance(series, list) for series in (delta_series, price_series, volume_series)):
            return empty_result

        observation_count = min(len(delta_series), len(price_series), len(volume_series), 5)
        if observation_count < 3:
            return empty_result

        deltas = delta_series[-observation_count:]
        prices = price_series[-observation_count:]
        volumes = volume_series[-observation_count:]
        efficiencies: List[float] = []
        near_zero = 1e-12

        for previous_price, current_price, delta, volume in zip(
            prices[:-1], prices[1:], deltas[1:], volumes[1:]
        ):
            try:
                price_change = abs(float(current_price) - float(previous_price))
                delta_value = abs(float(delta))
                volume_value = abs(float(volume))
            except (TypeError, ValueError):
                continue

            if price_change <= near_zero or volume_value <= near_zero:
                continue

            normalized_delta = delta_value / volume_value
            efficiencies.append(normalized_delta / price_change)

        if len(efficiencies) < 3:
            return empty_result

        efficiency_trend = self._linear_regression_slope(efficiencies)
        average_efficiency = sum(efficiencies) / len(efficiencies)
        if average_efficiency <= near_zero or efficiency_trend >= 0:
            return {
                "is_exhausted": False,
                "strength": 0.0,
                "efficiency_trend": efficiency_trend,
                "reversal_probability": 0.0,
            }

        decline_rate = (-efficiency_trend * len(efficiencies)) / average_efficiency
        strength = min(1.0, max(0.0, decline_rate))
        reversal_probability = min(0.9, max(0.0, strength * 0.9))

        return {
            "is_exhausted": strength > 0.0,
            "strength": strength,
            "efficiency_trend": efficiency_trend,
            "reversal_probability": reversal_probability,
        }

    def calculate_vwap_bands(
        self,
        ticks: List[Dict[str, Any]],
        std_multiplier: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Calculate current VWAP, volume-weighted deviation, and descriptive bands.

        VWAP distance describes where the current weighted price sits relative to
        the supplied ticks. It is not a guaranteed mean-reversion signal.

        Args:
            ticks: Tick dictionaries containing numeric ``price`` and ``volume``.
            std_multiplier: Optional positive band multipliers. Defaults to
                ``[1, 2, 3]``.

        Returns:
            Dictionary containing ``vwap``, a list of ``bands`` with upper and
            lower levels, signed ``distance_std``, ``signal``, and ``std_dev``.
        """
        if std_multiplier is None:
            multipliers = [1.0, 2.0, 3.0]
        elif isinstance(std_multiplier, (int, float)):
            multipliers = [float(std_multiplier)]
        else:
            multipliers = list(std_multiplier)
        valid_multipliers: List[float] = []
        for multiplier in multipliers:
            try:
                numeric_multiplier = float(multiplier)
            except (TypeError, ValueError):
                continue
            if numeric_multiplier > 0:
                valid_multipliers.append(numeric_multiplier)
        if not valid_multipliers:
            valid_multipliers = [1.0, 2.0, 3.0]

        cumulative_price_volume = 0.0
        cumulative_volume = 0.0
        cumulative_squared_price_volume = 0.0
        current_price = 0.0

        if isinstance(ticks, list):
            for tick in ticks:
                if not isinstance(tick, dict):
                    continue
                try:
                    price = float(tick.get("price", 0.0))
                    volume = float(tick.get("volume", 0.0))
                except (TypeError, ValueError):
                    continue
                if price <= 0 or volume <= 0:
                    continue

                cumulative_price_volume += price * volume
                cumulative_volume += volume
                cumulative_squared_price_volume += price * price * volume
                current_price = price

        if cumulative_volume <= 0:
            return {
                "vwap": 0.0,
                "bands": [
                    {"multiplier": multiplier, "upper": 0.0, "lower": 0.0}
                    for multiplier in valid_multipliers
                ],
                "distance_std": 0.0,
                "signal": "NEAR_VWAP",
                "std_dev": 0.0,
            }

        vwap = cumulative_price_volume / cumulative_volume
        variance = (cumulative_squared_price_volume / cumulative_volume) - (vwap * vwap)
        std_dev = math.sqrt(max(0.0, variance))
        bands = [
            {
                "multiplier": multiplier,
                "upper": vwap + (multiplier * std_dev),
                "lower": vwap - (multiplier * std_dev),
            }
            for multiplier in valid_multipliers
        ]

        if std_dev <= 1e-12:
            distance_std = 0.0
            signal = "NEAR_VWAP"
        else:
            distance_std = (current_price - vwap) / std_dev
            inner_threshold = min(valid_multipliers)
            outer_threshold = max(valid_multipliers)
            if distance_std >= outer_threshold:
                signal = "EXTREME_ABOVE"
            elif distance_std >= inner_threshold:
                signal = "ABOVE"
            elif distance_std <= -outer_threshold:
                signal = "EXTREME_BELOW"
            elif distance_std <= -inner_threshold:
                signal = "BELOW"
            else:
                signal = "NEAR_VWAP"

        return {
            "vwap": vwap,
            "bands": bands,
            "distance_std": distance_std,
            "signal": signal,
            "std_dev": std_dev,
        }

    def calculate_order_flow_score(
        self,
        all_order_flow_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine order-flow components into a bounded directional score.

        Component scores are normalized to approximately ``[-1, 1]`` and combined
        with fixed weights. Exhaustion is deliberately applied as a conviction
        multiplier rather than as another weighted directional component.

        Args:
            all_order_flow_data: Mapping containing any of ``delta``, ``divergence``,
                ``absorption``, ``vwap``, ``poc``, ``stacked``, and ``exhaustion``.

        Returns:
            Dictionary containing the clamped ``score``, normalized ``components``,
            and ``signal`` (``BUY``, ``SELL``, or ``WAIT``).
        """
        if not isinstance(all_order_flow_data, dict):
            all_order_flow_data = {}

        def clamp(value: float) -> float:
            return min(1.0, max(-1.0, value))

        def strength_value(value: Any, default: float = 1.0) -> float:
            try:
                strength = abs(float(value))
            except (TypeError, ValueError):
                strength = default
            if strength > 1.0:
                strength /= 100.0
            return min(1.0, max(0.0, strength))

        def directional_value(value: Any) -> float:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return clamp(float(value))
            text = str(value).upper()
            if text in {"BUY", "BULLISH", "BUY_ABSORPTION", "ABOVE", "ABOVE_VALUE", "ABOVE_POC"}:
                return 1.0
            if text in {"SELL", "BEARISH", "SELL_ABSORPTION", "BELOW", "BELOW_VALUE", "BELOW_POC"}:
                return -1.0
            return 0.0

        delta_data = all_order_flow_data.get("delta")
        if isinstance(delta_data, dict):
            raw_delta = delta_data.get("delta", delta_data.get("value", 0.0))
            total_volume = delta_data.get("total_volume")
            try:
                delta_value = float(raw_delta)
                volume_value = float(total_volume)
                normalized_delta = delta_value / volume_value if volume_value > 0 else delta_value
            except (TypeError, ValueError):
                normalized_delta = 0.0
        else:
            try:
                normalized_delta = float(delta_data)
            except (TypeError, ValueError):
                normalized_delta = 0.0
        delta_score = clamp(math.tanh(normalized_delta))

        divergence_data = all_order_flow_data.get("divergence")
        if isinstance(divergence_data, dict):
            divergence_direction = directional_value(divergence_data.get("signal", ""))
            if divergence_direction == 0.0:
                divergence_direction = directional_value(divergence_data.get("type", ""))
            divergence_score = clamp(
                divergence_direction * strength_value(divergence_data.get("strength"))
            )
        else:
            divergence_score = directional_value(divergence_data)

        absorption_data = all_order_flow_data.get("absorption")
        if isinstance(absorption_data, dict):
            absorption_direction = directional_value(
                absorption_data.get("signal", absorption_data.get("type", ""))
            )
            absorption_score = clamp(
                absorption_direction * strength_value(absorption_data.get("strength"))
            )
        else:
            absorption_score = directional_value(absorption_data)

        vwap_data = all_order_flow_data.get("vwap")
        if isinstance(vwap_data, dict):
            vwap_score = directional_value(vwap_data.get("signal", ""))
            if vwap_score == 0.0:
                try:
                    vwap_score = clamp(math.tanh(float(vwap_data.get("distance_std", 0.0)) / 3.0))
                except (TypeError, ValueError):
                    vwap_score = 0.0
        else:
            vwap_score = directional_value(vwap_data)

        poc_data = all_order_flow_data.get("poc")
        if isinstance(poc_data, dict):
            poc_data = poc_data.get("poc_signal", poc_data.get("signal", ""))
        poc_score = directional_value(poc_data)
        if poc_score != 0.0 and isinstance(all_order_flow_data.get("poc"), dict):
            poc_score *= 0.5 if str(poc_data).upper() in {"ABOVE_POC", "BELOW_POC"} else 1.0

        stacked_data = all_order_flow_data.get("stacked")
        if stacked_data is None and (
            "stacked_buy_zones" in all_order_flow_data or "stacked_sell_zones" in all_order_flow_data
        ):
            stacked_data = {
                "stacked_buy_zones": all_order_flow_data.get("stacked_buy_zones", []),
                "stacked_sell_zones": all_order_flow_data.get("stacked_sell_zones", []),
            }
        if isinstance(stacked_data, dict):
            stacked_score = directional_value(stacked_data.get("signal", stacked_data.get("type", "")))
            if stacked_score == 0.0:
                buy_count = len(stacked_data.get("stacked_buy_zones", []))
                sell_count = len(stacked_data.get("stacked_sell_zones", []))
                total_count = buy_count + sell_count
                stacked_score = clamp((buy_count - sell_count) / total_count) if total_count else 0.0
        elif isinstance(stacked_data, list):
            buy_count = sum(
                1 for zone in stacked_data
                if isinstance(zone, dict) and directional_value(zone.get("type", zone.get("imbalance", ""))) > 0
            )
            sell_count = sum(
                1 for zone in stacked_data
                if isinstance(zone, dict) and directional_value(zone.get("type", zone.get("imbalance", ""))) < 0
            )
            total_count = buy_count + sell_count
            stacked_score = clamp((buy_count - sell_count) / total_count) if total_count else 0.0
        else:
            stacked_score = directional_value(stacked_data)

        component_scores = {
            "delta": delta_score,
            "divergence": clamp(divergence_score),
            "absorption": clamp(absorption_score),
            "vwap": clamp(vwap_score),
            "poc": clamp(poc_score),
            "stacked": clamp(stacked_score),
        }
        weights = {
            "delta": 0.30,
            "divergence": 0.20,
            "absorption": 0.20,
            "vwap": 0.15,
            "poc": 0.10,
            "stacked": 0.05,
        }
        weighted_score = sum(weights[name] * value for name, value in component_scores.items())

        exhaustion_data = all_order_flow_data.get("exhaustion")
        exhaustion_modifier = 1.0
        if isinstance(exhaustion_data, dict) and exhaustion_data.get("is_exhausted"):
            exhaustion_modifier = 1.0 - strength_value(exhaustion_data.get("strength"), default=1.0)
        exhaustion_modifier = min(1.0, max(0.0, exhaustion_modifier))
        score = clamp(weighted_score * exhaustion_modifier)
        if score > 0.40:
            signal = "BUY"
        elif score < -0.40:
            signal = "SELL"
        else:
            signal = "WAIT"

        return {
            "score": score,
            "components": {
                **component_scores,
                "exhaustion_modifier": exhaustion_modifier,
            },
            "signal": signal,
        }

    def integrate_with_force_score(
        self,
        force_score_result: Any,
        order_flow_score: Any,
    ) -> Dict[str, Any]:
        """Blend force and order-flow scores without placing orders.

        Force contributes 75% and order flow contributes 25%. Strong opposing
        signals receive a substantial conviction reduction, while strong aligned
        signals receive a confirmation boost. These adjustments are descriptive
        score integration only and do not trigger execution.

        Args:
            force_score_result: Force score number or result dictionary containing
                a ``score`` field.
            order_flow_score: Order-flow score number or result dictionary
                containing a ``score`` field.

        Returns:
            Dictionary containing final score, normalized components, override
            metadata, and a ``BUY``, ``SELL``, or ``WAIT`` signal.
        """
        def extract_score(value: Any) -> float:
            if isinstance(value, dict):
                value = value.get("score", 0.0)
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = 0.0
            return max(-1.0, min(1.0, numeric_value))

        force_component = extract_score(force_score_result)
        order_flow_component = extract_score(order_flow_score)
        combined = (force_component * 0.75) + (order_flow_component * 0.25)
        override: Dict[str, Any] = {
            "applied": False,
            "type": "NONE",
            "reason": "No contradiction or confirmation override applied.",
        }

        if force_component > 0.5 and order_flow_component < -0.6:
            combined *= 0.25
            override = {
                "applied": True,
                "type": "CONTRADICTION",
                "reason": "Strong bullish force conflicts with strong bearish order flow.",
            }
        elif force_component < -0.5 and order_flow_component > 0.6:
            combined *= 0.25
            override = {
                "applied": True,
                "type": "CONTRADICTION",
                "reason": "Strong bearish force conflicts with strong bullish order flow.",
            }
        elif force_component > 0.5 and order_flow_component > 0.5:
            combined *= 1.10
            override = {
                "applied": True,
                "type": "CONFIRMATION",
                "reason": "Strong bullish force is confirmed by strong bullish order flow.",
            }
        elif force_component < -0.5 and order_flow_component < -0.5:
            combined *= 1.10
            override = {
                "applied": True,
                "type": "CONFIRMATION",
                "reason": "Strong bearish force is confirmed by strong bearish order flow.",
            }

        final_score = max(-1.0, min(1.0, combined))
        if final_score > 0.40:
            signal = "BUY"
        elif final_score < -0.40:
            signal = "SELL"
        else:
            signal = "WAIT"

        return {
            "final_score": final_score,
            "force_component": force_component,
            "order_flow_component": order_flow_component,
            "override": override,
            "signal": signal,
        }

    def advanced_delta_statistics(
        self,
        delta_history: List[float],
    ) -> Dict[str, Any]:
        """Compute robust descriptive statistics for a delta history.

        The Hurst value is a lightweight estimate of persistence, not a certainty
        or calibrated regime classifier. Entropy measures directional uncertainty
        and signal clarity measures the dominance of the mean signed delta.

        Args:
            delta_history: Ordered historical delta observations.

        Returns:
            A stable dictionary containing z-score, extreme-delta, momentum,
            entropy, clarity, autocorrelation, Hurst-style, and mean-reversion
            diagnostics.
        """
        empty_result: Dict[str, Any] = {
            "delta_zscore": 0.0,
            "is_extreme_delta": False,
            "delta_momentum": 0.0,
            "momentum_signal": 0,
            "entropy": 0.0,
            "signal_clarity": 0.0,
            "hurst": 0.5,
            "hurst_estimate": 0.5,
            "autocorrelation": 0.0,
            "mean_reversion": False,
            "mean_reversion_signal": 0,
        }
        if not isinstance(delta_history, list):
            return empty_result

        values: List[float] = []
        for value in delta_history:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric_value):
                values.append(numeric_value)

        if not values:
            return empty_result

        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        standard_deviation = math.sqrt(max(0.0, variance))
        if standard_deviation <= 1e-12:
            z_score = 0.0
            is_extreme = False
        else:
            z_score = (values[-1] - mean_value) / standard_deviation
            is_extreme = abs(z_score) >= 2.0

        if len(values) >= 2:
            delta_momentum = values[-1] - values[-2]
        else:
            delta_momentum = 0.0

        momentum_signal = 0
        if len(values) >= 3:
            acceleration = values[-1] - (2.0 * values[-2]) + values[-3]
            acceleration_threshold = max(1e-12, standard_deviation * 0.05)
            if acceleration > acceleration_threshold:
                momentum_signal = 1
            elif acceleration < -acceleration_threshold:
                momentum_signal = -1

        positive_volume = sum(value for value in values if value > 0)
        negative_volume = sum(abs(value) for value in values if value < 0)
        total_directional_volume = positive_volume + negative_volume
        if total_directional_volume <= 1e-12:
            entropy = 0.0
        else:
            probabilities = [
                volume / total_directional_volume
                for volume in (positive_volume, negative_volume)
                if volume > 0
            ]
            entropy = -sum(probability * math.log(probability, 2) for probability in probabilities)
            entropy = min(1.0, max(0.0, entropy))

        absolute_mean = sum(abs(value) for value in values) / len(values)
        if absolute_mean <= 1e-12:
            signal_clarity = 0.0
        else:
            signal_clarity = min(1.0, max(0.0, abs(mean_value) / absolute_mean))

        autocorrelation = self._lag_autocorrelation(values, lag=1)
        hurst = self._estimate_hurst(values)
        mean_reversion_signal = -1 if autocorrelation < -0.1 else 0
        mean_reversion = mean_reversion_signal == -1 or hurst < 0.45

        return {
            "delta_zscore": z_score,
            "is_extreme_delta": is_extreme,
            "delta_momentum": delta_momentum,
            "momentum_signal": momentum_signal,
            "entropy": entropy,
            "signal_clarity": signal_clarity,
            "hurst": hurst,
            "hurst_estimate": hurst,
            "autocorrelation": autocorrelation,
            "mean_reversion": mean_reversion,
            "mean_reversion_signal": mean_reversion_signal,
        }

    @staticmethod
    def _lag_autocorrelation(values: List[float], lag: int = 1) -> float:
        """Return a bounded Pearson-style autocorrelation without statsmodels."""
        if lag < 1 or len(values) <= lag:
            return 0.0
        mean_value = sum(values) / len(values)
        numerator = sum(
            (values[index] - mean_value) * (values[index - lag] - mean_value)
            for index in range(lag, len(values))
        )
        denominator = sum((value - mean_value) ** 2 for value in values)
        if denominator <= 1e-12:
            return 0.0
        return min(1.0, max(-1.0, numerator / denominator))

    @classmethod
    def _estimate_hurst(cls, values: List[float]) -> float:
        """Estimate Hurst persistence from log lag/log increment dispersion."""
        if len(values) < 8 or max(values) == min(values):
            return 0.5

        points: List[tuple[float, float]] = []
        max_lag = min(10, len(values) // 2)
        for lag in range(1, max_lag + 1):
            differences = [
                values[index] - values[index - lag]
                for index in range(lag, len(values))
            ]
            if len(differences) < 2:
                continue
            mean_difference = sum(differences) / len(differences)
            dispersion = math.sqrt(
                sum((difference - mean_difference) ** 2 for difference in differences)
                / len(differences)
            )
            if dispersion > 1e-12:
                points.append((math.log(float(lag)), math.log(dispersion)))

        if len(points) < 2:
            return 0.5
        x_mean = sum(point[0] for point in points) / len(points)
        y_mean = sum(point[1] for point in points) / len(points)
        denominator = sum((point[0] - x_mean) ** 2 for point in points)
        if denominator <= 1e-12:
            return 0.5
        slope = sum(
            (x - x_mean) * (y - y_mean)
            for x, y in points
        ) / denominator
        return min(1.0, max(0.0, slope))


class TickProcessor:
    """Process ticks through an :class:`OrderFlowAnalyzer` composition boundary.

    This class only manages in-memory simulation and analysis state. It has no
    broker, network, or Angel One integration. Malformed ticks are ignored so a
    single bad market-data record cannot stop a paper-trading loop.
    """

    def __init__(
        self,
        analyzer: Optional[OrderFlowAnalyzer] = None,
        update_interval: int = 10,
        max_tick_buffer: int = 1000,
        max_candle_history: int = 100,
        max_vwap_history: int = 1000,
    ) -> None:
        if update_interval < 1:
            raise ValueError("update_interval must be at least 1")
        if max_tick_buffer < 1 or max_candle_history < 1 or max_vwap_history < 1:
            raise ValueError("history limits must be at least 1")

        self.analyzer = analyzer or OrderFlowAnalyzer()
        self.update_interval = update_interval
        self.max_tick_buffer = max_tick_buffer
        self.max_candle_history = max_candle_history
        self.max_vwap_history = max_vwap_history
        self.tick_buffer: List[Dict[str, Any]] = []
        self.candle_deltas: List[Dict[str, Any]] = []
        self.vwap_ticks: List[Dict[str, Any]] = []
        self.prev_bid: Optional[float] = None
        self.prev_ask: Optional[float] = None
        self._ticks_since_update = 0

    @staticmethod
    def _normalise_tick(tick: Any) -> Optional[Dict[str, Any]]:
        """Return a numeric, analyzer-compatible copy of a tick or ``None``."""
        if not isinstance(tick, dict):
            return None
        try:
            price = float(tick.get("price"))
            volume = float(tick.get("volume", 0.0))
        except (TypeError, ValueError):
            return None
        if price <= 0 or volume < 0:
            return None

        normalized = dict(tick)
        normalized["price"] = price
        normalized["volume"] = volume
        normalized.setdefault("timestamp", None)
        normalized.setdefault("bid", None)
        normalized.setdefault("ask", None)
        return normalized

    @staticmethod
    def _numeric_or_none(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def process_tick(self, tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Classify and buffer a tick, returning metrics at the update interval.

        Invalid ticks are ignored and return ``None``. Valid ticks are classified
        by the composed analyzer, retain their bid/ask context, and contribute to
        the VWAP history. No external market-data or order API is contacted.
        """
        normalized = self._normalise_tick(tick)
        if normalized is None:
            return None

        try:
            classified = self.analyzer.classify_tick(normalized)
        except (TypeError, ValueError):
            return None

        classified_tick = {**normalized, **classified}
        self.tick_buffer.append(classified_tick)
        self.tick_buffer = self.tick_buffer[-self.max_tick_buffer:]
        self.vwap_ticks.append({
            "price": classified_tick["price"],
            "volume": classified_tick["volume"],
        })
        self.vwap_ticks = self.vwap_ticks[-self.max_vwap_history:]
        self.prev_bid = self._numeric_or_none(normalized.get("bid"))
        self.prev_ask = self._numeric_or_none(normalized.get("ask"))
        self._ticks_since_update += 1

        if self._ticks_since_update < self.update_interval:
            return None
        self._ticks_since_update = 0

        try:
            live_delta = self.analyzer.calculate_delta(self.tick_buffer)
            prices = [item["price"] for item in self.tick_buffer]
            absorption = self.analyzer.detect_absorption(
                self.tick_buffer,
                prices,
            )
            vwap = self.analyzer.calculate_vwap_bands(self.vwap_ticks)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

        return {
            "live_delta": live_delta,
            "vwap": vwap,
            "absorption": absorption,
        }

    def on_candle_complete(self) -> Optional[Dict[str, Any]]:
        """Finalize the active candle and return its order-flow analytics."""
        if not self.tick_buffer:
            return None

        candle_ticks = list(self.tick_buffer)
        prices = [tick["price"] for tick in candle_ticks]
        volumes = [tick["volume"] for tick in candle_ticks]
        try:
            delta = self.analyzer.calculate_delta(candle_ticks)
            candle_delta = {
                "delta": delta["delta"],
                "price": prices[-1],
                "volume": sum(volumes),
                "timestamp": candle_ticks[-1].get("timestamp"),
            }
            self.candle_deltas.append(candle_delta)
            self.candle_deltas = self.candle_deltas[-self.max_candle_history:]

            footprint = self.analyzer.build_footprint(
                candle_ticks,
                candle_open=prices[0],
                candle_close=prices[-1],
            )
            volume_profile = self.analyzer.build_volume_profile(
                candle_ticks,
                current_price=prices[-1],
            )
            absorption = self.analyzer.detect_absorption(candle_ticks, prices)
            exhaustion = self.analyzer.detect_exhaustion(
                [item["delta"] for item in self.candle_deltas],
                [item["price"] for item in self.candle_deltas],
                [item["volume"] for item in self.candle_deltas],
            )
        except (TypeError, ValueError, ZeroDivisionError):
            self.tick_buffer.clear()
            self._ticks_since_update = 0
            return None

        self.tick_buffer.clear()
        self.vwap_ticks.clear()
        self._ticks_since_update = 0
        return {
            "candle_delta": candle_delta,
            "footprint": footprint,
            "volume_profile": volume_profile,
            "absorption": absorption,
            "exhaustion": exhaustion,
        }

    def reset(self) -> None:
        """Clear processor buffers, counters, bid/ask state, and analyzer state."""
        self.tick_buffer.clear()
        self.candle_deltas.clear()
        self.vwap_ticks.clear()
        self.prev_bid = None
        self.prev_ask = None
        self._ticks_since_update = 0
        self.analyzer.prev_price = None
        self.analyzer.prev_classification = "NEUTRAL"
        self.analyzer.candle_deltas.clear()
        self.analyzer.vwap_ticks.clear()
