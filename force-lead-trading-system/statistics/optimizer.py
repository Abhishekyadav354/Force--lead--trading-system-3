from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from scipy.optimize import minimize


class SystemOptimizer:
    """Optimization engine for factor weights and walk-forward validation."""

    FACTOR_NAMES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]

    @staticmethod
    def _default_start_weights():
        return np.array([0.25, 0.20, 0.10, 0.20, 0.10, 0.05, 0.05, 0.03, 0.02], dtype=float)

    @staticmethod
    def _as_float_list(values):
        return np.asarray(values, dtype=float)

    @staticmethod
    def _normalize_weights(weights):
        arr = np.asarray(weights, dtype=float)
        if arr.size == 0:
            return np.array([], dtype=float)
        arr = np.clip(arr, 0.02, 0.50)
        total = arr.sum()
        if total <= 0:
            arr = np.full(arr.shape, 1.0 / arr.size)
            total = arr.sum()
        return arr / total

    @staticmethod
    def _calculate_metrics(trades):
        if not trades:
            return {"sharpe_ratio": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

        pnl_values = np.asarray([float(t.get("pnl", 0.0)) for t in trades], dtype=float)
        wins = pnl_values[pnl_values > 0]
        losses = pnl_values[pnl_values < 0]

        gross_profit = float(wins.sum()) if wins.size else 0.0
        gross_loss = abs(float(losses.sum())) if losses.size else 0.0
        win_rate = float(np.mean(pnl_values > 0)) if pnl_values.size else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        returns = pnl_values / max(abs(pnl_values).sum(), 1.0)
        mean_return = float(np.mean(returns)) if returns.size else 0.0
        std_return = float(np.std(returns)) if returns.size else 0.0
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0

        cumulative = np.cumsum(pnl_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / np.maximum(running_max, 1e-9)
        max_drawdown = float(np.max(drawdown)) if drawdown.size else 0.0

        return {
            "sharpe_ratio": float(sharpe_ratio),
            "win_rate": float(win_rate),
            "profit_factor": float(profit_factor),
            "max_drawdown": float(max_drawdown),
        }

    @staticmethod
    def run_backtest_with_weights(weights, data):
        """Compute basic trade metrics from a weight vector and market data."""
        weights = SystemOptimizer._normalize_weights(weights)
        trades = []

        for i in range(len(data) - 1):
            row = data[i]
            close = float(row.get("close", 0.0) or 0.0)
            if close <= 0:
                continue
            score = float(np.dot(weights[: min(len(weights), 9)], np.asarray([row.get(f"f{idx}", 0.0) for idx in range(1, 10)], dtype=float)[: len(weights)]))
            if abs(score) <= 0.4:
                continue
            entry = close
            sl = entry * (0.99 if score > 0 else 1.01)
            target = entry * (1.02 if score > 0 else 0.98)
            next_value = float(data[i + 1].get("close", close) or close)
            pnl = (target - entry) if score > 0 else (entry - sl)
            if next_value <= sl and score > 0:
                pnl = -abs(entry - sl)
            if next_value >= target and score < 0:
                pnl = -abs(target - entry)
            if score > 0:
                pnl = max((next_value - entry), -abs(entry - sl))
            else:
                pnl = max((entry - next_value), -abs(entry - sl))
            trades.append({"pnl": float(pnl), "score": float(score)})

        return SystemOptimizer._calculate_metrics(trades)

    @staticmethod
    def optimize_weights_scipy(historical_data, target="sharpe"):
        """Optimizes the 9 factor weights via constrained minimization."""
        x0 = SystemOptimizer._default_start_weights()
        bounds = [(0.05, 0.50)] * 9

        def objective(weights):
            metrics = SystemOptimizer.run_backtest_with_weights(weights, historical_data)
            if target == "sharpe":
                return -metrics["sharpe_ratio"]
            if target == "winrate":
                return -metrics["win_rate"]
            if target == "profit_factor":
                return -metrics["profit_factor"]
            return -metrics["sharpe_ratio"]

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000},
        )

        optimal_weights = SystemOptimizer._normalize_weights(result.x if result.success else x0)
        config_path = Path(__file__).resolve().parents[1] / "config" / "optimal_weights.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({factor: float(weight) for factor, weight in zip(SystemOptimizer.FACTOR_NAMES, optimal_weights)}, indent=2), encoding="utf-8")

        print("Optimal weights found:")
        for factor, weight in zip(SystemOptimizer.FACTOR_NAMES, optimal_weights):
            print(f"  {factor}: {weight:.3f}")
        return optimal_weights

    @staticmethod
    def walk_forward_optimization(historical_data, train_days=60, test_days=10):
        """Roll-forward evaluation using training and testing segments."""
        results = []
        start = 0

        while start + train_days + test_days < len(historical_data):
            train = historical_data[start : start + train_days]
            test = historical_data[start + train_days : start + train_days + test_days]
            optimal_w = SystemOptimizer.optimize_weights_scipy(train)
            test_metrics = SystemOptimizer.run_backtest_with_weights(optimal_w, test)
            results.append({
                "period": f"Day {start} to {start + train_days + test_days}",
                "optimal_weights": optimal_w,
                "test_sharpe": test_metrics["sharpe_ratio"],
                "test_winrate": test_metrics["win_rate"],
                "test_profit_factor": test_metrics["profit_factor"],
            })
            start += test_days

        if not results:
            return [], np.array([], dtype=float)

        avg_out_of_sample_sharpe = float(np.mean([r["test_sharpe"] for r in results]))
        avg_win_rate = float(np.mean([r["test_winrate"] for r in results]))
        best_idx = int(np.argmax([r["test_sharpe"] for r in results]))
        best_period_weights = np.asarray(results[best_idx]["optimal_weights"], dtype=float)

        print(f"Walk-Forward Win Rate: {avg_win_rate:.1%}")
        print(f"Walk-Forward Sharpe: {avg_out_of_sample_sharpe:.2f}")
        return results, best_period_weights

    @staticmethod
    def adaptive_weight_update(recent_10_trades, current_weights, lr=0.05):
        """Online learning update based on recent trade factor contributions."""
        current_weights = np.asarray(current_weights, dtype=float)
        if not recent_10_trades or current_weights.size == 0:
            return current_weights, {}

        weight_changes = {}
        updated_weights = current_weights.copy()

        for idx in range(len(current_weights)):
            factor_name = f"f{idx + 1}"
            wins_with_high_factor = sum(1 for trade in recent_10_trades if trade.get("outcome") == "WIN" and float(trade.get(factor_name, 0.0)) > 0.5)
            losses_with_high_factor = sum(1 for trade in recent_10_trades if trade.get("outcome") == "LOSS" and float(trade.get(factor_name, 0.0)) > 0.5)
            total_wins = max(sum(1 for trade in recent_10_trades if trade.get("outcome") == "WIN"), 1)
            total_losses = max(sum(1 for trade in recent_10_trades if trade.get("outcome") == "LOSS"), 1)

            factor_win_contribution = wins_with_high_factor / total_wins
            factor_loss_contribution = losses_with_high_factor / total_losses
            gradient = factor_win_contribution - factor_loss_contribution
            new_weight = float(current_weights[idx] + lr * gradient)
            updated_weights[idx] = new_weight
            weight_changes[factor_name] = float(new_weight - current_weights[idx])

        updated_weights = np.clip(updated_weights, 0.02, 0.50)
        updated_weights = updated_weights / updated_weights.sum() if updated_weights.sum() > 0 else updated_weights
        return updated_weights, weight_changes

    @staticmethod
    def multi_objective_pareto(historical_data):
        """Search random weight combinations and return Pareto-optimal strategies."""
        pareto_results = []
        rng = np.random.default_rng()

        for _ in range(1000):
            weights = rng.dirichlet(np.ones(9))
            metrics = SystemOptimizer.run_backtest_with_weights(weights, historical_data)
            pareto_results.append({
                "weights": weights,
                "sharpe": metrics["sharpe_ratio"],
                "winrate": metrics["win_rate"],
                "max_drawdown": metrics["max_drawdown"],
            })

        pareto_front = []
        for i, candidate in enumerate(pareto_results):
            dominated = False
            for j, other in enumerate(pareto_results):
                if i == j:
                    continue
                if (
                    other["sharpe"] >= candidate["sharpe"]
                    and other["winrate"] >= candidate["winrate"]
                    and other["max_drawdown"] <= candidate["max_drawdown"]
                    and (
                        other["sharpe"] > candidate["sharpe"]
                        or other["winrate"] > candidate["winrate"]
                        or other["max_drawdown"] < candidate["max_drawdown"]
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(candidate)

        if not pareto_front:
            return {"conservative": None, "balanced": None, "aggressive": None}

        conservative = max(pareto_front, key=lambda x: -x["max_drawdown"])
        balanced = max(pareto_front, key=lambda x: x["sharpe"])
        aggressive = max(pareto_front, key=lambda x: x["winrate"])

        return {
            "conservative": conservative,
            "balanced": balanced,
            "aggressive": aggressive,
        }


def optimize_threshold(scores, target=None):
    """Choose a threshold that best aligns with the desired target level.

    If a target is omitted, we fall back to the median score.
    """
    if not scores:
        return 0.0

    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return 0.0

    if target is None:
        return float(np.median(arr))

    target_value = float(target)
    idx = int(np.argmin(np.abs(arr - target_value)))
    return float(arr[idx])


def threshold_from_scores(scores, target=None):
    """Convenience alias for threshold optimization."""
    return optimize_threshold(scores, target)
