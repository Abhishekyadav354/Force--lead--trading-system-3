from __future__ import annotations

import numpy as np


class MonteCarloRiskEngine:
    """Monte Carlo engine for trade risk, portfolio risk, and risk-adjusted sizing."""

    @staticmethod
    def simulate_trade(entry_price, stop_loss, target, mu, sigma, n_simulations=10000):
        """Run GBM-based Monte Carlo simulations to assess trade quality."""
        entry = float(entry_price)
        stop = float(stop_loss)
        target_price = float(target)
        risk_amount = max(entry - stop, 0.0)
        reward_amount = max(target_price - entry, 0.0)

        outcomes = []
        rng = np.random.default_rng()

        for _ in range(int(n_simulations)):
            price = entry
            result = "OPEN"

            for _ in range(30):
                dW = rng.normal(0.0, 1.0)
                price = price * np.exp((mu - 0.5 * sigma ** 2) + sigma * dW)

                if price <= stop:
                    result = "LOSS"
                    break
                if price >= target_price:
                    result = "WIN"
                    break

            if result == "OPEN":
                result = "LOSS" if price < entry else "WIN"

            outcomes.append(result)

        outcomes_arr = np.asarray(outcomes)
        win_rate = float(np.mean(outcomes_arr == "WIN"))
        loss_rate = float(np.mean(outcomes_arr == "LOSS"))
        expected_pnl = (win_rate * reward_amount) - (loss_rate * risk_amount)

        return {
            "win_probability": win_rate,
            "loss_probability": loss_rate,
            "expected_pnl_per_share": expected_pnl,
            "trade_worthwhile": expected_pnl > 0,
            "simulations_run": int(n_simulations),
        }

    @staticmethod
    def calculate_var_cvar(entry_price, mu, sigma, horizon=10, n_sim=10000):
        """Estimate VaR and CVaR over the next horizon using GBM price paths."""
        entry = float(entry_price)
        rng = np.random.default_rng()
        final_prices = []

        for _ in range(int(n_sim)):
            price = entry
            for _ in range(int(horizon)):
                price *= np.exp((mu - 0.5 * sigma ** 2) + sigma * rng.normal(0.0, 1.0))
            final_prices.append(price)

        final_prices = np.asarray(final_prices, dtype=float)
        returns = [(p - entry) / entry for p in final_prices]
        returns = np.sort(np.asarray(returns, dtype=float))

        var_95 = float(np.percentile(returns, 5) * entry)
        worst_5pct = returns[: int(0.05 * n_sim)]
        cvar_95 = float(np.mean(worst_5pct) * entry)

        return {
            "var_95_rupees": abs(var_95),
            "cvar_95_rupees": abs(cvar_95),
            "interpretation": f"95% confident max loss: {abs(var_95):.2f}",
        }

    @staticmethod
    def optimal_position_sizer(capital, win_prob, risk_per_share, reward_per_share, mc_results, current_volatility):
        """Size a position using Kelly, volatility, and Monte Carlo confidence adjustments."""
        capital_value = float(capital)
        win_probability = float(win_prob)
        risk = float(risk_per_share)
        reward = float(reward_per_share)
        current_vol = float(current_volatility)

        if reward <= 0 or risk <= 0:
            return {
                "position_rupees": 0.0,
                "shares": 0,
                "pct_of_capital": 0.0,
                "kelly_raw": 0.0,
                "half_kelly": 0.0,
                "final_adjusted": 0.0,
            }

        kelly = win_probability - ((1.0 - win_probability) / (reward / risk))
        half_kelly = kelly / 2.0

        historical_avg_vol = 0.01
        vol_ratio = current_vol / historical_avg_vol if historical_avg_vol > 0 else 1.0
        vol_adjustment = 1.0 / vol_ratio if vol_ratio > 0 else 1.0

        mc_confidence = float(mc_results.get("win_probability", win_probability)) if isinstance(mc_results, dict) else win_probability
        mc_adjustment = mc_confidence / 0.5 if mc_confidence > 0 else 1.0

        raw_fraction = half_kelly * vol_adjustment * mc_adjustment
        capped_fraction = min(max(raw_fraction, 0.0), 0.05)
        position_rupees = capped_fraction * capital_value
        shares = int(position_rupees / max(risk, 1e-6))

        return {
            "position_rupees": float(position_rupees),
            "shares": shares,
            "pct_of_capital": float(capped_fraction * 100.0),
            "kelly_raw": float(kelly),
            "half_kelly": float(half_kelly),
            "final_adjusted": float(capped_fraction),
        }

    @staticmethod
    def stress_test(current_state, scenarios):
        """Evaluate portfolio survival under extreme market scenarios."""
        if not scenarios:
            return {}

        current_sigma = float(current_state.get("sigma", 0.01) or 0.01)
        entry_price = float(current_state.get("entry_price", 100.0) or 100.0)
        stop_loss = float(current_state.get("stop_loss", 95.0) or 95.0)
        target = float(current_state.get("target", 110.0) or 110.0)

        stress_results = {}
        for name, scenario in scenarios.items():
            scenario_mu = float(scenario.get("mu", 0.0) or 0.0)
            scenario_sigma = float(scenario.get("sigma", current_sigma) or current_sigma)
            result = MonteCarloRiskEngine.simulate_trade(entry_price, stop_loss, target, scenario_mu, scenario_sigma, n_simulations=1000)
            expected_loss = max((1.0 - result["win_probability"]) * max(entry_price - stop_loss, 0.0), 0.0)
            stop_hit_probability = result["loss_probability"]
            survival_probability = 1.0 - stop_hit_probability
            stress_results[name] = {
                "expected_loss": float(expected_loss),
                "stop_hit_probability": float(stop_hit_probability),
                "survival_probability": float(survival_probability),
            }

        return stress_results


def monte_carlo_expectation(values, simulations: int = 1000, rng=None):
    """Estimate the expected value via repeated random sampling from the input distribution."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if simulations <= 0:
        return float(arr.mean())

    generator = np.random.default_rng() if rng is None else rng
    sample = generator.choice(arr, size=int(simulations))
    return float(np.mean(sample))


def monte_carlo_projection(start_value: float, drift: float, volatility: float, steps: int, simulations: int = 1000, dt: float = 1.0, rng=None):
    """Simulate several price paths using geometric Brownian motion."""
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    generator = np.random.default_rng() if rng is None else rng
    paths = np.empty((int(simulations), int(steps) + 1), dtype=float)
    paths[:, 0] = start_value

    for step in range(1, int(steps) + 1):
        shock = generator.normal(0.0, 1.0, size=int(simulations))
        paths[:, step] = paths[:, step - 1] * np.exp((drift - 0.5 * volatility ** 2) * dt + volatility * np.sqrt(dt) * shock)

    return paths
