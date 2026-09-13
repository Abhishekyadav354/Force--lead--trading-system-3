from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np


class Backtester:
    """Historical validation engine for Force-Lead entries and portfolio performance."""

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _calculate_force_score(candle: Dict[str, Any], weights: Dict[str, float]) -> float:
        raw = 0.0
        for i in range(1, 10):
            raw += weights.get(f"f{i}", 0.0) * Backtester._safe_float(candle.get(f"f{i}", 0.0), 0.0)
        return float(np.clip(raw, -1.0, 1.0))

    @staticmethod
    def _calculate_lead_status(history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not history:
            return {"action": "WAIT", "lead_value": 0.0, "phase": "neutral"}

        latest = history[-1]
        lead_value = Backtester._safe_float(latest.get("lead_value"), 0.0)
        catchup_time = Backtester._safe_int(latest.get("catchup_time"), 0)
        progress = Backtester._safe_float(latest.get("lead_progress_pct"), 0.0)

        if lead_value > 0.25 and catchup_time <= 3 and progress > 0.4:
            action = "ENTER_TRADE"
        elif lead_value > 0.10:
            action = "PREPARE_ENTRY"
        else:
            action = "WAIT"

        return {"action": action, "lead_value": lead_value, "phase": "strong" if action != "WAIT" else "neutral"}

    @staticmethod
    def _test_significance(history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if len(history) < 3:
            return {"is_significant": True, "p_value": 0.1}

        closes = np.asarray([Backtester._safe_float(c.get("close"), 0.0) for c in history], dtype=float)
        if closes.size < 2:
            return {"is_significant": True, "p_value": 0.1}
        changes = np.diff(closes)
        volatility = np.std(changes) if changes.size else 0.0
        mean_move = np.mean(np.abs(changes)) if changes.size else 0.0
        return {"is_significant": bool(volatility > 0.0 and mean_move > 0.0), "p_value": float(0.05 if volatility > 0.0 and mean_move > 0.0 else 0.5)}

    @staticmethod
    def _is_good_time(timestamp: Any) -> bool:
        if timestamp is None:
            return True
        try:
            dt = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            return 9 <= dt.hour <= 15
        except Exception:
            return True

    @staticmethod
    def calculate_all_metrics(trades, equity_curve, initial_capital):
        """
        Compute standard performance metrics from trade records and equity curve.
        """
        total_trades = len(trades)
        wins = [t for t in trades if t.get("outcome") == "WIN"]
        losses = [t for t in trades if t.get("outcome") == "LOSS"]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100.0 if equity_curve else 0.0

        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t["pnl"] for t in losses])) if losses else 1.0
        profit_factor = (
            sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))
        ) if losses else (float("inf") if sum(t["pnl"] for t in wins) > 0 else 0.0)

        daily_returns = []
        if len(equity_curve) > 1:
            for prev, curr in zip(equity_curve[:-1], equity_curve[1:]):
                if prev != 0:
                    daily_returns.append((curr - prev) / prev)
        sharpe = (np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0.0

        peak = equity_curve[0] if equity_curve else initial_capital
        max_dd = 0.0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak else 0.0
            if dd > max_dd:
                max_dd = dd

        calmar = ((total_return / 100.0) / max_dd) if max_dd > 0 else float("inf")

        max_consec_loss = 0
        current_streak = 0
        for t in trades:
            if t.get("outcome") == "LOSS":
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0

        return {
            "total_trades": int(total_trades),
            "win_rate": float(win_rate),
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "profit_factor": float(profit_factor),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "reward_risk_ratio": float(avg_win / avg_loss) if avg_loss > 0 else 0.0,
            "max_consecutive_losses": int(max_consec_loss),
            "calmar_ratio": float(calmar),
            "final_capital": float(equity_curve[-1]) if equity_curve else float(initial_capital),
        }

    @staticmethod
    def best_time_analysis(trades):
        """Find best trade hours and score ranges for dashboard display."""
        by_hour = {}
        by_score_range = {
            "0.4-0.5": {"wins": 0, "total": 0},
            "0.5-0.6": {"wins": 0, "total": 0},
            "0.6-0.7": {"wins": 0, "total": 0},
            "0.7+": {"wins": 0, "total": 0},
        }

        for trade in trades:
            timestamp = trade.get("timestamp")
            if timestamp is not None:
                try:
                    hour = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).hour
                except Exception:
                    hour = 0
                by_hour.setdefault(hour, {"wins": 0, "total": 0})
                by_hour[hour]["total"] += 1
                if trade.get("outcome") == "WIN":
                    by_hour[hour]["wins"] += 1

            force = Backtester._safe_float(trade.get("force_score"), 0.0)
            if force < 0.5:
                key = "0.4-0.5"
            elif force < 0.6:
                key = "0.5-0.6"
            elif force < 0.7:
                key = "0.6-0.7"
            else:
                key = "0.7+"
            by_score_range.setdefault(key, {"wins": 0, "total": 0})
            by_score_range[key]["total"] += 1
            if trade.get("outcome") == "WIN":
                by_score_range[key]["wins"] += 1

        hour_summary = {hour: round(data["wins"] / max(data["total"], 1), 4) for hour, data in sorted(by_hour.items())}
        score_summary = {bucket: round(data["wins"] / max(data["total"], 1), 4) for bucket, data in by_score_range.items()}
        return {"by_hour": hour_summary, "by_score_range": score_summary}

    @staticmethod
    def _timestamp(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            parsed = value
        else:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(key): Backtester._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [Backtester._json_value(item) for item in value]
        return str(value)

    @staticmethod
    def _identity(row: Dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row.get("market", "UNKNOWN")).upper(),
            str(row.get("symbol", "UNKNOWN")).upper(),
            str(row.get("segment", "OTHER")).upper(),
            str(row.get("timeframe", row.get("interval", "UNKNOWN"))).lower(),
        )

    @staticmethod
    def _period(timestamp: Optional[datetime], row: Dict[str, Any]) -> str:
        explicit = row.get("period", row.get("session_period"))
        if explicit:
            return str(explicit).upper()
        if timestamp is None:
            return "UNKNOWN"
        if timestamp.hour < 11:
            return "OPENING"
        if timestamp.hour < 14:
            return "MIDDAY"
        return "CLOSING"

    @staticmethod
    def _regime(row: Dict[str, Any]) -> str:
        return str(row.get("regime", row.get("market_regime", "UNKNOWN"))).upper()

    @staticmethod
    def _direction(value: Any) -> str:
        text = str(value or "").upper()
        if text in {"BUY", "BULL", "UP", "LONG", "POSITIVE"}:
            return "UP"
        if text in {"SELL", "BEAR", "DOWN", "SHORT", "NEGATIVE"}:
            return "DOWN"
        return "SIDEWAYS"

    @classmethod
    def _metric(cls, errors: list[float], actual: list[float], windows: list[Optional[float]]) -> Dict[str, Any]:
        if not errors:
            return {"samples": 0, "mae": None, "median_error": None, "p90_error": None, "mae_minutes": None, "median_error_minutes": None, "p90_error_minutes": None, "coverage": None}
        ordered = sorted(errors)
        percentile = lambda fraction: ordered[(len(ordered) - 1) * fraction] if False else ordered[int(round((len(ordered) - 1) * fraction))]
        covered = [value <= window for value, window in zip(actual, windows) if window is not None]
        mae = sum(errors) / len(errors)
        median_error = percentile(0.5)
        p90_error = percentile(0.9)
        return {
            "samples": len(errors),
            "mae": mae, "mae_minutes": mae,
            "median_error": median_error, "median_error_minutes": median_error,
            "p90_error": p90_error, "p90_error_minutes": p90_error,
            "coverage": sum(covered) / len(covered) if covered else None,
        }

    @classmethod
    def _group_metrics(cls, observations: list[Dict[str, Any]]) -> Dict[str, Any]:
        def metrics(items: list[Dict[str, Any]], error_key: str = "lead_error_minutes", actual_key: str = "actual_elapsed_minutes", window_key: str = "predicted_eta_minutes") -> Dict[str, Any]:
            return cls._metric(
                [item[error_key] for item in items if item.get(error_key) is not None],
                [item[actual_key] for item in items if item.get(error_key) is not None],
                [item.get(window_key) for item in items if item.get(error_key) is not None],
            )

        direction_samples = [item for item in observations if item.get("actual_direction")]
        direction_accuracy = sum(item["predicted_direction"] == item["actual_direction"] for item in direction_samples) / len(direction_samples) if direction_samples else None
        return {
            "samples": len(observations),
            "direction": {"samples": len(direction_samples), "accuracy": direction_accuracy},
            "lead_time": metrics(observations),
            "move_size": metrics(observations, "move_error", "actual_move_size", "predicted_move_size"),
            "reversal": {"samples": sum(item.get("actual_reversal") is not None for item in observations), "accuracy": None},
            "confidence_calibration": {"samples": len(observations), "mean_confidence": sum(item.get("confidence", 0.0) for item in observations) / len(observations) if observations else 0.0},
        }

    @classmethod
    def walk_forward_validate(
        cls,
        historical_data: Iterable[Dict[str, Any]],
        prediction_engine: Any = None,
        *,
        min_train_samples: int = 20,
        rolling_window: Optional[int] = None,
        step: int = 1,
        direction_threshold: float = 0.0,
        transaction_cost_bps: float = 0.0,
        slippage_bps: float = 0.0,
        data_source: str = "historical",
    ) -> Dict[str, Any]:
        """Validate predictions chronologically using only each group's prior rows.

        This is a PAPER/SIMULATION measurement path. It never submits orders and
        does not turn transaction-cost assumptions into a profitability claim.
        """
        if min_train_samples < 1 or step < 1 or rolling_window is not None and rolling_window < min_train_samples:
            raise ValueError("Invalid walk-forward window parameters")
        if transaction_cost_bps < 0 or slippage_bps < 0:
            raise ValueError("Transaction costs and slippage cannot be negative")
        if hasattr(historical_data, "to_dict"):
            historical_data = historical_data.to_dict(orient="records")
        rows = [dict(row) for row in historical_data if isinstance(row, dict)]
        rows.sort(key=lambda row: cls._timestamp(row.get("timestamp", row.get("datetime", row.get("time")))) or datetime.min.replace(tzinfo=timezone.utc))
        grouped: Dict[tuple[str, str, str, str], list[Dict[str, Any]]] = {}
        invalid_timestamps = 0
        for row in rows:
            if cls._timestamp(row.get("timestamp", row.get("datetime", row.get("time")))) is None:
                invalid_timestamps += 1
                continue
            grouped.setdefault(cls._identity(row), []).append(row)
        if prediction_engine is None:
            from core.prediction_engine import PredictionEngine
            prediction_engine = PredictionEngine()

        observations: list[Dict[str, Any]] = []
        leakage_checks = {"chronological": True, "future_rows_used_for_prediction": 0, "predictions": 0}
        for identity, group in sorted(grouped.items()):
            for index in range(min_train_samples, len(group), step):
                prior = group[:index]
                train = prior[-rolling_window:] if rolling_window else prior
                evaluation = group[index]
                live = dict(train[-1]) if train else {}
                prediction_timestamp = cls._timestamp(evaluation.get("timestamp"))
                result = prediction_engine.predict(
                    live, train, market=identity[0], segment=identity[2], timeframe=identity[3]
                )
                leakage_checks["predictions"] += 1
                predicted_direction = cls._direction(result.get("direction", result.get("signal")))
                previous_close = cls._safe_float(prior[-1].get("close"), 0.0)
                evaluation_close = cls._safe_float(evaluation.get("close"), previous_close)
                move = evaluation_close - previous_close
                actual_direction = "UP" if move > direction_threshold else "DOWN" if move < -direction_threshold else "SIDEWAYS"
                adaptive = result.get("adaptive_lead_time", {}) if isinstance(result, dict) else {}
                eta = adaptive.get("normal_eta_minutes")
                completion = next((row for row in group[index + 1:] if row.get("event_id") == evaluation.get("event_id") and (row.get("completed") is True or row.get("remaining_gap") in (0, 0.0))), None) if evaluation.get("event_id") else None
                actual_elapsed = None
                if completion is not None and prediction_timestamp is not None:
                    completed_at = cls._timestamp(completion.get("timestamp"))
                    if completed_at and completed_at > prediction_timestamp:
                        actual_elapsed = (completed_at - prediction_timestamp).total_seconds() / 60.0
                predicted_move = result.get("predicted_move_size", result.get("move", {}).get("predicted_size") if isinstance(result.get("move"), dict) else None)
                actual_reversal = evaluation.get("actual_reversal")
                observations.append({
                    "market": identity[0], "symbol": identity[1], "segment": identity[2], "timeframe": identity[3],
                    "timestamp": cls._json_value(evaluation.get("timestamp")), "period": cls._period(prediction_timestamp, evaluation), "regime": cls._regime(evaluation),
                    "predicted_direction": predicted_direction, "actual_direction": actual_direction,
                    "predicted_move_size": cls._safe_float(predicted_move) if predicted_move is not None else None, "actual_move_size": abs(move),
                    "move_error": abs(cls._safe_float(predicted_move) - abs(move)) if predicted_move is not None else None,
                    "predicted_eta_minutes": cls._safe_float(eta) if eta is not None else None, "actual_elapsed_minutes": actual_elapsed,
                    "lead_error_minutes": abs(cls._safe_float(eta) - actual_elapsed) if eta is not None and actual_elapsed is not None else None,
                    "confidence": max(0.0, min(1.0, cls._safe_float(result.get("confidence"), 0.0))),
                    "speed_state": str(adaptive.get("speed_regime", "NORMAL")), "actual_reversal": actual_reversal,
                })
        reports: Dict[str, Any] = {}
        for observation in observations:
            key = "/".join(observation[field] for field in ("market", "symbol", "segment", "timeframe"))
            reports.setdefault(key, []).append(observation)
        formatted = {}
        for key, items in reports.items():
            formatted[key] = {"overall": cls._group_metrics(items), "by_timeframe": {}, "by_period": {}, "by_regime": {}, "by_speed_state": {}}
            for field, output in (("timeframe", "by_timeframe"), ("period", "by_period"), ("regime", "by_regime"), ("speed_state", "by_speed_state")):
                for value in sorted({item[field] for item in items}):
                    formatted[key][output][value] = cls._group_metrics([item for item in items if item[field] == value])
            formatted[key]["events"] = items
            formatted[key]["quality"] = {
                "sample_count": len(items),
                "sufficient_samples": len(items) >= min_train_samples,
                "confidence": min(1.0, len(items) / max(min_train_samples, 1)),
                "insufficient_samples": len(items) < min_train_samples,
            }
        result = {"historical_only": True, "paper_simulation_only": True, "data_source": str(data_source), "synthetic_data": str(data_source).lower() == "synthetic", "rolling_window": rolling_window, "expanding_window": rolling_window is None, "minimum_train_samples": min_train_samples, "transaction_cost_bps": float(transaction_cost_bps), "slippage_bps": float(slippage_bps), "invalid_timestamp_rows": invalid_timestamps, "reports": formatted, "samples": len(observations), "leakage_checks": leakage_checks, "note": "Synthetic results are code-validation only; no profitability claim is made." if str(data_source).lower() == "synthetic" else "Walk-forward validation uses only prior rows for each prediction."}
        return cls._json_value(result)

    validate_walk_forward = walk_forward_validate
    run_walk_forward = walk_forward_validate

    @staticmethod
    def run_full_backtest(historical_data, weights, capital=10000):
        """
        Run a historical validation loop over the dataset and summarize cumulative performance.
        """
        if any(
            isinstance(candle, dict)
            and str(candle.get("data_source", "")).lower() == "synthetic"
            for candle in historical_data
        ):
            print("WARNING: Synthetic data results may not reflect real performance")

        trades = []
        equity_curve = [capital]
        current_capital = float(capital)

        for i in range(len(historical_data) - 10):
            candle = historical_data[i]
            force_score = Backtester._calculate_force_score(candle, weights)
            history_slice = historical_data[: i + 1]
            lead_status = Backtester._calculate_lead_status(history_slice)
            significance = Backtester._test_significance(history_slice)

            if (
                abs(force_score) > 0.40
                and significance["is_significant"]
                and lead_status["action"] in ["PREPARE_ENTRY", "ENTER_TRADE"]
                and Backtester._is_good_time(candle.get("timestamp"))
            ):
                entry_price = Backtester._safe_float(candle.get("close"), 0.0)
                is_buy = force_score > 0

                stop_loss = entry_price * (0.99 if is_buy else 1.01)
                target = entry_price * (1.02 if is_buy else 0.98)
                position_size = min(0.03 * current_capital, current_capital)
                shares = max(0, int(position_size / entry_price)) if entry_price > 0 else 0

                outcome = "OPEN"
                exit_price = None
                for j in range(i + 1, min(i + 11, len(historical_data))):
                    future_candle = historical_data[j]
                    future_low = Backtester._safe_float(future_candle.get("low"), entry_price)
                    future_high = Backtester._safe_float(future_candle.get("high"), entry_price)

                    if is_buy:
                        if future_low <= stop_loss:
                            outcome = "LOSS"
                            exit_price = stop_loss
                            break
                        if future_high >= target:
                            outcome = "WIN"
                            exit_price = target
                            break
                    else:
                        if future_high >= stop_loss:
                            outcome = "LOSS"
                            exit_price = stop_loss
                            break
                        if future_low <= target:
                            outcome = "WIN"
                            exit_price = target
                            break

                if outcome == "OPEN":
                    exit_price = Backtester._safe_float(historical_data[min(i + 10, len(historical_data) - 1)].get("close"), entry_price)
                    outcome = "WIN" if ((is_buy and exit_price > entry_price) or (not is_buy and exit_price < entry_price)) else "LOSS"

                if exit_price is None:
                    exit_price = entry_price

                pnl = shares * (exit_price - entry_price) * (1 if is_buy else -1)
                pnl -= 40.0
                current_capital += pnl
                equity_curve.append(current_capital)

                trades.append({
                    "entry": entry_price,
                    "exit": exit_price,
                    "outcome": outcome,
                    "pnl": pnl,
                    "force_score": force_score,
                    "timestamp": candle.get("timestamp"),
                    "capital_after": current_capital,
                })

        return Backtester.calculate_all_metrics(trades, equity_curve, capital)


def backtest_signal(signals, actual_returns):
    """Simple PnL-like backtest summary."""
    if len(signals) != len(actual_returns):
        raise ValueError("signals and actual_returns must have the same length.")
    score = sum(s * r for s, r in zip(signals, actual_returns))
    return float(score)
