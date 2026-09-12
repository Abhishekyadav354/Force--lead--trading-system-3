from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


class ForceScoreCalculator:
    """Implements the F1 to F9 trading signal engine and final force score."""

    @staticmethod
    def _normalize(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
        """Normalize a raw value to the range [-1, 1]."""
        if not math.isfinite(value):
            return 0.0
        try:
            return max(lower, min(upper, math.tanh(value)))
        except Exception:
            return 0.0

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_dict(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        values = list(values)
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def calculate_f1_volume_force(current_candle: Dict[str, Any], previous_candle: Optional[Dict[str, Any]], historical_avg_trade_size: float = 1.0) -> float:
        """F1 = Volume Force."""
        if not current_candle:
            return 0.0
        prev = previous_candle or current_candle

        volume_ratio = (current_candle.get("volume_per_minute", 0.0) or 0.0) / max((prev.get("volume_per_minute", 0.0) or 0.0), 1e-6)
        buyers_pct = current_candle.get("buyers_pct", 0.0) or 0.0
        sellers_pct = current_candle.get("sellers_pct", 0.0) or 0.0
        net_pressure = (buyers_pct - sellers_pct) / 100.0
        institutional_score = (current_candle.get("avg_trade_size", 0.0) or 0.0) / max(historical_avg_trade_size, 1e-6)
        f1 = volume_ratio * net_pressure * (1 + institutional_score * 0.3)
        return ForceScoreCalculator._normalize(f1)

    @staticmethod
    def calculate_f2_wall_force(current_candle: Dict[str, Any]) -> float:
        """F2 = Wall Force."""
        if not current_candle:
            return 0.0

        bid_total = float(current_candle.get("bid_total", 0.0) or 0.0)
        ask_total = float(current_candle.get("ask_total", 0.0) or 0.0)
        bid_levels = current_candle.get("bid_levels") or []
        ask_levels = current_candle.get("ask_levels") or []

        total = bid_total + ask_total
        if total <= 0:
            return 0.0

        wall_ratio = (bid_total - ask_total) / total
        top_bid_qty = float((bid_levels[0][1] if len(bid_levels) > 0 and isinstance(bid_levels[0], (list, tuple)) else 0.0) or 0.0)
        top_ask_qty = float((ask_levels[0][1] if len(ask_levels) > 0 and isinstance(ask_levels[0], (list, tuple)) else 0.0) or 0.0)

        top_bid_concentration = top_bid_qty / max(bid_total, 1e-6)
        top_ask_concentration = top_ask_qty / max(ask_total, 1e-6)
        concentration_factor = (top_bid_concentration + top_ask_concentration) / 2.0

        if top_bid_concentration > 0.30:
            wall_ratio *= 0.8

        f2 = wall_ratio * concentration_factor
        return ForceScoreCalculator._normalize(f2)

    @staticmethod
    def calculate_f3_position_risk(candle: Dict[str, Any], day_data: Dict[str, Any]) -> float:
        """F3 = Price Position Risk."""
        if not candle:
            return 0.0

        current_price = float(candle.get("close", candle.get("ltp", 0.0)) or 0.0)
        day_high = float(day_data.get("day_high", 0.0) or 0.0)
        day_low = float(day_data.get("day_low", 0.0) or 0.0)

        if day_high <= day_low:
            return 0.0

        position = (current_price - day_low) / (day_high - day_low)
        return max(0.0, min(1.0, position))

    @staticmethod
    def calculate_f4_acceleration(candles_list: List[Dict[str, Any]]) -> float:
        """F4 = Momentum Acceleration."""
        if len(candles_list) < 2:
            return 0.0

        current = candles_list[-1]
        prev = candles_list[-2]

        current_open = float(current.get("open", 0.0) or 0.0)
        current_close = float(current.get("close", current.get("ltp", 0.0)) or 0.0)
        prev_open = float(prev.get("open", 0.0) or 0.0)
        prev_close = float(prev.get("close", prev.get("ltp", 0.0)) or 0.0)

        if current_open == 0 or prev_open == 0:
            return 0.0

        current_pct_change = (current_close - current_open) / current_open * 100.0
        prev_pct_change = (prev_close - prev_open) / prev_open * 100.0
        acceleration = current_pct_change - prev_pct_change

        roc_values = []
        for i in range(1, min(len(candles_list), 4)):
            current_item = candles_list[-i]
            prev_item = candles_list[-(i + 1)] if -(i + 1) >= -len(candles_list) else candles_list[0]
            c_close = float(current_item.get("close", current_item.get("ltp", 0.0)) or 0.0)
            p_close = float(prev_item.get("close", prev_item.get("ltp", 0.0)) or 0.0)
            if p_close != 0:
                roc_values.append(((c_close - p_close) / p_close) * 100.0)

        mom_values = []
        for i in range(1, min(len(candles_list), 6)):
            current_item = candles_list[-1]
            past_item = candles_list[-(i + 1)] if -(i + 1) >= -len(candles_list) else candles_list[0]
            c_close = float(current_item.get("close", current_item.get("ltp", 0.0)) or 0.0)
            p_close = float(past_item.get("close", past_item.get("ltp", 0.0)) or 0.0)
            if p_close != 0:
                mom_values.append(((c_close - p_close) / p_close) * 100.0)

        roc = ForceScoreCalculator._mean(roc_values) if roc_values else 0.0
        mom = ForceScoreCalculator._mean(mom_values) if mom_values else 0.0
        total = acceleration * 0.5 + roc * 0.3 + mom * 0.2
        return ForceScoreCalculator._normalize(total)

    @staticmethod
    def calculate_f5_institutional_detector(current_candle: Dict[str, Any], historical: List[Dict[str, Any]]) -> float:
        """F5 = Institutional vs Retail."""
        if not current_candle:
            return 0.0

        volume = float(current_candle.get("volume", 0.0) or 0.0)
        trade_count = float(current_candle.get("trade_count", 0.0) or 0.0)
        avg_trade_size = volume / max(trade_count, 1.0)
        historical_avg = ForceScoreCalculator._mean(
            [float(item.get("avg_trade_size", 0.0) or 0.0) for item in historical[-20:]]
        ) if historical else avg_trade_size

        institutional_ratio = avg_trade_size / max(historical_avg, 1e-6)
        if institutional_ratio > 3.0:
            f5 = 1.0
        elif institutional_ratio > 1.5:
            f5 = 0.5
        elif institutional_ratio < 0.5:
            f5 = -0.3
        else:
            f5 = 0.0

        volume_direction = 1.0 if (current_candle.get("close", 0.0) or 0.0) >= (current_candle.get("open", 0.0) or 0.0) else -1.0
        return f5 * volume_direction

    @staticmethod
    def calculate_f6_oi_signal(oi_data: Dict[str, Any], price_data: Dict[str, Any]) -> float:
        """F6 = Options Open Interest Signal."""
        if not oi_data:
            return 0.0

        current_call_oi = float(oi_data.get("current_call_oi", 0.0) or 0.0)
        prev_call_oi = float(oi_data.get("prev_call_oi", 0.0) or 0.0)
        current_put_oi = float(oi_data.get("current_put_oi", 0.0) or 0.0)
        prev_put_oi = float(oi_data.get("prev_put_oi", 0.0) or 0.0)

        call_oi_change = current_call_oi - prev_call_oi
        put_oi_change = current_put_oi - prev_put_oi
        raw_signal = (call_oi_change - put_oi_change) / (abs(call_oi_change) + abs(put_oi_change) + 1.0)

        current_price = float(price_data.get("current_price", 0.0) or 0.0)
        prev_price = float(price_data.get("prev_price", 0.0) or 0.0)
        price_direction = 1.0 if current_price > prev_price else -1.0 if current_price < prev_price else 0.0

        if price_direction > 0 and call_oi_change >= 0:
            confirmation_multiplier = 1.0
        elif price_direction > 0 and put_oi_change >= 0:
            confirmation_multiplier = -0.5
        elif price_direction < 0 and put_oi_change >= 0:
            confirmation_multiplier = -1.0
        elif price_direction < 0 and call_oi_change >= 0:
            confirmation_multiplier = 0.5
        else:
            confirmation_multiplier = 1.0

        return ForceScoreCalculator._normalize(raw_signal * confirmation_multiplier)

    @staticmethod
    def calculate_f7_spread_quality(candles_list: List[Dict[str, Any]]) -> float:
        """F7 = Signal Quality Filter."""
        if not candles_list:
            return 1.0

        current = candles_list[-1]
        current_spread = float(current.get("spread", 0.0) or 0.0)
        recent_spreads = [float(item.get("spread", 0.0) or 0.0) for item in candles_list[-10:]]
        avg_spread = ForceScoreCalculator._mean(recent_spreads) if recent_spreads else max(current_spread, 1e-6)
        if avg_spread <= 0:
            return 1.0

        ratio = current_spread / avg_spread
        if ratio < 0.8:
            quality = 1.0
        elif ratio <= 1.2:
            quality = 0.7
        elif ratio <= 1.5:
            quality = 0.4
        else:
            quality = 0.0
        return max(0.0, min(1.0, quality))

    @staticmethod
    def calculate_f8_nifty_alignment(nifty_data: Dict[str, Any], stock_data: Dict[str, Any]) -> float:
        """F8 = Market Alignment Multiplier."""
        nifty_direction = float(nifty_data.get("direction", 0.0) or 0.0)
        stock_direction = float(stock_data.get("direction", 0.0) or 0.0)

        if nifty_direction == 0:
            return 1.0
        if stock_direction == nifty_direction:
            return 1.5
        return 0.5

    @staticmethod
    def calculate_f9_candle_conviction(current_candle: Dict[str, Any]) -> float:
        """F9 = Candle Body Conviction."""
        if not current_candle:
            return 0.0

        open_price = float(current_candle.get("open", 0.0) or 0.0)
        close_price = float(current_candle.get("close", current_candle.get("ltp", 0.0)) or 0.0)
        high_price = float(current_candle.get("high", max(open_price, close_price)) or 0.0)
        low_price = float(current_candle.get("low", min(open_price, close_price)) or 0.0)

        total_size = high_price - low_price
        if total_size == 0:
            return 0.0

        body_size = abs(close_price - open_price)
        body_ratio = body_size / total_size
        if body_ratio > 0.7:
            conviction = 1.0
        elif body_ratio >= 0.4:
            conviction = 0.6
        else:
            conviction = 0.2

        direction = 1 if close_price > open_price else -1
        return conviction * direction

    @staticmethod
    def calculate_total_force_score(all_factors: Dict[str, float]) -> Dict[str, Any]:
        """FINAL WEIGHTED FORCE SCORE."""
        f1 = float(all_factors.get("f1", 0.0) or 0.0)
        f2 = float(all_factors.get("f2", 0.0) or 0.0)
        f3 = float(all_factors.get("f3", 0.0) or 0.0)
        f4 = float(all_factors.get("f4", 0.0) or 0.0)
        f5 = float(all_factors.get("f5", 0.0) or 0.0)
        f6 = float(all_factors.get("f6", 0.0) or 0.0)
        f7 = float(all_factors.get("f7", 1.0) or 1.0)
        f8 = float(all_factors.get("f8", 1.0) or 1.0)
        f9 = float(all_factors.get("f9", 0.0) or 0.0)

        weights = {
            "f1": 0.25,
            "f2": 0.20,
            "f4": 0.20,
            "f5": 0.15,
            "f6": 0.10,
            "f9": 0.10,
        }

        position_risk_deduction = f3 * 0.10
        raw_score = (
            f1 * weights["f1"]
            + f2 * weights["f2"]
            + f4 * weights["f4"]
            + f5 * weights["f5"]
            + f6 * weights["f6"]
            + f9 * weights["f9"]
            - position_risk_deduction
        )

        quality_adjusted = raw_score * f7
        final_score = quality_adjusted * f8
        final_score = max(-1.0, min(1.0, final_score))

        signal = "BUY" if final_score > 0.4 else "SELL" if final_score < -0.4 else "WAIT"
        return {
            "score": final_score,
            "signal": signal,
            "strength": abs(final_score),
            "components": {
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "f4": f4,
                "f5": f5,
                "f6": f6,
                "f7": f7,
                "f8": f8,
                "f9": f9,
            },
        }


def calculate_force_score(price_series=None, momentum=0.0, volume=0.0, trend=0.0):
    """Backward-compatible wrapper for the earlier simple raw force score."""
    if not price_series:
        return 0.0

    recent_close = float(price_series[-1])
    baseline = float(price_series[0]) if len(price_series) > 0 else recent_close
    trend_component = ((recent_close - baseline) / baseline) * 100 if baseline else 0.0
    score = 50 + trend_component * 0.8 + momentum * 0.5 + volume * 0.2 + trend * 0.6
    return float(max(0, min(100, score)))
