from __future__ import annotations

import numpy as np


def probability_of_event(successes: int, trials: int) -> float:
    if trials <= 0:
        return 0.0
    return float(successes / trials)


def normal_probability(x: float, mean: float = 0.0, std: float = 1.0) -> float:
    coeff = 1 / (np.sqrt(2 * np.pi) * std)
    exponent = -((x - mean) ** 2) / (2 * std ** 2)
    return float(coeff * np.exp(exponent))
