from __future__ import annotations

from statistics.monte_carlo import MonteCarloRiskEngine


def test_simulate_trade():
    result = MonteCarloRiskEngine.simulate_trade(
        entry_price=100,
        stop_loss=96,
        target=110,
        mu=0.0002,
        sigma=0.01,
        n_simulations=500,
    )
    assert result["win_probability"] >= 0.0
    assert result["loss_probability"] >= 0.0
    assert result["trade_worthwhile"] in {True, False}
    assert result["simulations_run"] == 500


def test_calculate_var_cvar():
    result = MonteCarloRiskEngine.calculate_var_cvar(100.0, mu=0.0001, sigma=0.01, horizon=5, n_sim=500)
    assert "var_95_rupees" in result
    assert "cvar_95_rupees" in result
    assert result["var_95_rupees"] >= 0.0
    assert result["cvar_95_rupees"] >= 0.0


def test_optimal_position_sizer():
    result = MonteCarloRiskEngine.optimal_position_sizer(
        capital=10000,
        win_prob=0.55,
        risk_per_share=4,
        reward_per_share=8,
        mc_results={"win_probability": 0.55},
        current_volatility=0.008,
    )
    assert result["position_rupees"] >= 0.0
    assert result["pct_of_capital"] >= 0.0
    assert result["shares"] >= 0


def test_stress_test():
    result = MonteCarloRiskEngine.stress_test(
        {"entry_price": 100, "stop_loss": 96, "target": 110, "sigma": 0.01},
        {
            "crash_5pct": {"mu": -0.05, "sigma": 0.03},
            "pump_5pct": {"mu": 0.05, "sigma": 0.03},
        },
    )
    assert "crash_5pct" in result
    assert "pump_5pct" in result
    assert result["crash_5pct"]["stop_hit_probability"] >= 0.0
