from __future__ import annotations

import numpy as np

from statistics.optimizer import SystemOptimizer


def test_optimize_weights_scipy():
    data = [
        {"close": 100.0, "f1": 0.8, "f2": 0.5, "f3": 0.6, "f4": 0.9, "f5": 0.7, "f6": 0.3, "f7": 0.8, "f8": 1.2, "f9": 0.6},
        {"close": 101.0, "f1": 0.9, "f2": 0.7, "f3": 0.5, "f4": 0.8, "f5": 0.8, "f6": 0.4, "f7": 0.7, "f8": 1.1, "f9": 0.5},
        {"close": 102.0, "f1": 0.6, "f2": 0.4, "f3": 0.7, "f4": 0.7, "f5": 0.6, "f6": 0.2, "f7": 0.6, "f8": 1.0, "f9": 0.4},
        {"close": 103.0, "f1": 0.5, "f2": 0.3, "f3": 0.8, "f4": 0.6, "f5": 0.5, "f6": 0.5, "f7": 0.9, "f8": 0.9, "f9": 0.7},
    ]
    weights = SystemOptimizer.optimize_weights_scipy(data)
    assert len(weights) == 9
    assert np.isclose(weights.sum(), 1.0)


def test_walk_forward_optimization():
    data = [
        {"close": 100.0 + i * 0.5, "f1": 0.7, "f2": 0.5, "f3": 0.6, "f4": 0.8, "f5": 0.7, "f6": 0.4, "f7": 0.8, "f8": 1.1, "f9": 0.5}
        for i in range(100)
    ]
    results, best_weights = SystemOptimizer.walk_forward_optimization(data, train_days=30, test_days=10)
    assert isinstance(results, list)
    assert len(best_weights) == 9


def test_adaptive_weight_update():
    current = np.array([0.25, 0.20, 0.10, 0.20, 0.10, 0.05, 0.05, 0.03, 0.02])
    trades = [
        {"outcome": "WIN", "f1": 0.8, "f2": 0.4, "f3": 0.5, "f4": 0.7, "f5": 0.6, "f6": 0.3, "f7": 0.8, "f8": 0.9, "f9": 0.6},
        {"outcome": "LOSS", "f1": 0.2, "f2": 0.7, "f3": 0.8, "f4": 0.5, "f5": 0.4, "f6": 0.6, "f7": 0.2, "f8": 0.6, "f9": 0.5},
        {"outcome": "WIN", "f1": 0.9, "f2": 0.3, "f3": 0.7, "f4": 0.8, "f5": 0.7, "f6": 0.5, "f7": 0.9, "f8": 1.0, "f9": 0.8},
    ]
    updated, deltas = SystemOptimizer.adaptive_weight_update(trades, current, lr=0.05)
    assert len(updated) == len(current)
    assert sum(updated) > 0.0
    assert isinstance(deltas, dict)


def test_multi_objective_pareto():
    data = [
        {"close": 100.0 + i * 0.3, "f1": 0.8, "f2": 0.5, "f3": 0.6, "f4": 0.7, "f5": 0.7, "f6": 0.4, "f7": 0.8, "f8": 1.0, "f9": 0.5}
        for i in range(40)
    ]
    solutions = SystemOptimizer.multi_objective_pareto(data)
    assert set(solutions.keys()) == {"conservative", "balanced", "aggressive"}
    assert solutions["balanced"] is not None
