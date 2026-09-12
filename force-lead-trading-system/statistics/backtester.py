from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Sequence

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
    def run_full_backtest(historical_data, weights, capital=10000):
        """
        Run a historical validation loop over the dataset and summarize cumulative performance.
        """
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
