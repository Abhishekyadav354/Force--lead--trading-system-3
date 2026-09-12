from __future__ import annotations

import numpy as np

from statistics.stochastic import StochasticPriceModel, geometric_brownian_motion


def test_estimate_gbm_parameters():
    params = StochasticPriceModel.estimate_gbm_parameters([100, 101, 102, 101.5, 103])
    assert "mu" in params
    assert "sigma" in params
    assert params["annualized_volatility"] >= 0


def test_simulate_gbm_paths():
    result = StochasticPriceModel.simulate_gbm_paths(100.0, mu=0.0005, sigma=0.01, horizon_minutes=5, n_paths=50)
    assert result["all_paths"].shape == (50, 6)
    assert result["prob_up"] >= 0.0 and result["prob_up"] <= 1.0
    assert result["prob_down"] >= 0.0 and result["prob_down"] <= 1.0


def test_detect_market_regime():
    prices = np.linspace(100, 110, 50)
    regime = StochasticPriceModel.detect_market_regime(prices)
    assert regime["regime"] in {"TRENDING", "MEAN_REVERTING", "RANDOM_WALK"}
    assert "weight_adjustments" in regime


def test_ornstein_uhlenbeck_mean_reversion():
    theta, mu, sigma_ou, r_squared = StochasticPriceModel.ornstein_uhlenbeck_mean_reversion([100, 101, 100.5, 101.2, 100.8, 101.1])
    assert theta >= 0.0
    assert sigma_ou >= 0.0
    assert r_squared >= 0.0


def test_geometric_brownian_motion_shape():
    path = geometric_brownian_motion(100.0, 0.0005, 0.01, steps=5)
    assert len(path) == 6
    assert path[0] == 100.0
