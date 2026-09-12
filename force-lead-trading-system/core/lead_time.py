from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


class LeadTimeCalculator:
    """Pursuit-problem lead time engine for current-vs-previous force dynamics."""

    FACTOR_INERTIA = {
        "f1_volume": 1.0,
        "f2_wall": 0.8,
        "f4_acceleration": 1.2,
        "f5_institutional": 0.5,
        "f6_oi": 0.3,
        "f9_candle": 1.0,
    }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def calculate_lead(previous_force_score: float, duration_minutes: float) -> Dict[str, Any]:
        """Lead = Previous force ka accumulated advantage."""
        lead_value = float(previous_force_score) * float(duration_minutes)
        direction = "BULL" if lead_value > 0 else "BEAR"
        return {
            "lead_value": lead_value,
            "direction": direction,
            "calculated_at": LeadTimeCalculator._safe_now().isoformat(),
        }

    @staticmethod
    def calculate_catchup_time(lead: float, current_force: float, previous_force: float) -> Dict[str, Any]:
        """Pursuit-problem catch-up formula."""
        relative_speed = float(current_force) - float(previous_force)

        if relative_speed == 0:
            return {"catchup_time": float("inf"), "reliable": False}
        if math.copysign(1.0, relative_speed) == math.copysign(1.0, lead):
            return {"catchup_time": float("inf"), "reliable": False}

        catchup_minutes = abs(float(lead)) / abs(relative_speed)
        return {
            "catchup_time": catchup_minutes,
            "reliable": catchup_minutes < 30,
        }

    @staticmethod
    def calculate_per_factor_lead_times(prev_factors: Dict[str, float], curr_factors: Dict[str, float]) -> Dict[str, Any]:
        """Overall weighted catch-up across factor-specific reaction speeds."""
        factor_catchups: Dict[str, float] = {}
        weighted_total = 0.0
        total_inertia = 0.0

        for factor_name, inertia in LeadTimeCalculator.FACTOR_INERTIA.items():
            prev_value = float(prev_factors.get(factor_name, 0.0) or 0.0)
            curr_value = float(curr_factors.get(factor_name, 0.0) or 0.0)
            factor_lead = prev_value * 1.0
            factor_speed = abs(curr_value - prev_value) * inertia
            factor_catchup = (abs(factor_lead) / abs(factor_speed)) if factor_speed > 0 else float("inf")
            factor_catchups[factor_name] = factor_catchup
            weighted_total += factor_catchup * inertia
            total_inertia += inertia

        weighted_catchup = weighted_total / total_inertia if total_inertia else float("inf")
        values = [v for v in factor_catchups.values() if math.isfinite(v)]
        if not values:
            std_dev = 0.0
        else:
            mean = sum(values) / len(values)
            std_dev = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

        most_reliable_factor = min(
            factor_catchups,
            key=lambda k: (abs(factor_catchups[k] - weighted_catchup), factor_catchups[k]) if math.isfinite(factor_catchups[k]) else float("inf"),
            default="f1_volume",
        )

        return {
            "overall_catchup": weighted_catchup,
            "per_factor": factor_catchups,
            "confidence": 1 / (1 + std_dev),
            "most_reliable_factor": most_reliable_factor,
        }

    @staticmethod
    def get_lead_status(elapsed_minutes: float, catchup_time: float, lead: float) -> Dict[str, Any]:
        """Assess where we are in the pursuit problem."""
        if catchup_time == float("inf"):
            return {
                "status": "DIVERGING",
                "progress_pct": 0.0,
                "minutes_remaining": 0.0,
                "message": "Old momentum still dominating; gap is increasing.",
                "action": "DO_NOT_TRADE",
            }

        progress = elapsed_minutes / catchup_time if catchup_time > 0 else 0.0
        if progress < 0.3:
            status = "EARLY"
            message = "Old momentum very strong, wait."
            action = "DO_NOT_TRADE"
        elif progress < 0.6:
            status = "APPROACHING"
            message = "Old momentum weakening, prepare."
            action = "WATCH_CLOSELY"
        elif progress < 0.85:
            status = "NEAR_CROSSOVER"
            message = "Crossover imminent, get ready."
            action = "PREPARE_ENTRY"
        elif progress <= 1.0:
            status = "CROSSOVER_ZONE"
            message = "Entering now, high probability."
            action = "ENTER_TRADE"
        else:
            status = "CROSSED"
            message = "New force dominant."
            action = "IN_TRADE_OR_MISSED"

        return {
            "status": status,
            "progress_pct": max(0.0, min(100.0, progress * 100.0)),
            "minutes_remaining": max(0.0, catchup_time - elapsed_minutes),
            "message": message,
            "action": action,
            "lead": lead,
        }

    @staticmethod
    def dynamic_update(new_data: Dict[str, Any], current_lead_state: Dict[str, Any]) -> Dict[str, Any]:
        """Recalculate lead status every 30 seconds using the current live force snapshot."""
        previous_force = float(current_lead_state.get("previous_force", 0.0) or 0.0)
        current_force = float(new_data.get("current_force", 0.0) or 0.0)
        lead = float(current_lead_state.get("lead", 0.0) or 0.0)

        catchup = LeadTimeCalculator.calculate_catchup_time(lead, current_force, previous_force)
        elapsed_minutes = float(new_data.get("elapsed_minutes", 0.0) or 0.0)
        status = LeadTimeCalculator.get_lead_status(elapsed_minutes, catchup.get("catchup_time", float("inf")), lead)

        updated = dict(current_lead_state)
        updated["current_force"] = current_force
        updated["previous_force"] = previous_force
        updated["lead"] = lead
        updated["catchup_time"] = catchup.get("catchup_time", float("inf"))
        updated["reliable"] = catchup.get("reliable", False)
        updated["status"] = status

        if current_lead_state.get("catchup_time") is not None:
            prev_time = float(current_lead_state.get("catchup_time", 0.0) or 0.0)
            if math.isfinite(prev_time) and math.isfinite(catchup.get("catchup_time", prev_time)):
                delta = abs(catchup["catchup_time"] - prev_time)
                if delta > (prev_time * 0.5):
                    updated["alert"] = "SIGNIFICANT_CATCHUP_CHANGE"

        return updated


def estimate_lead_time(signal_strength: float, volatility: float = 1.0):
    """Backward-compatible simple estimate of lead time in bars."""
    if volatility <= 0:
        volatility = 1.0
    return max(1, int(round((100 - signal_strength) / volatility)))
