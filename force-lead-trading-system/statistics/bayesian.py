from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_FACTOR_NAMES = [f"f{i}" for i in range(1, 10)]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_probability(value: Any, default: float = 0.5) -> float:
    numeric = _as_float(value, default)
    if numeric <= 0:
        return 0.01
    if numeric >= 1:
        return 0.99
    return numeric


def bayes_update(prior: float, likelihood: float, evidence: float) -> float:
    """Compute the posterior probability using Bayes' theorem.

    P(H|E) = P(E|H) * P(H) / P(E)
    """
    prior_value = _as_float(prior, 0.5)
    likelihood_value = _as_float(likelihood, 0.5)
    evidence_value = _as_float(evidence, 0.5)

    if evidence_value == 0:
        return float(_clamp_probability(prior_value))

    posterior = (prior_value * likelihood_value) / evidence_value
    return float(_clamp_probability(posterior))


def posterior_probability(prior: float, likelihood: float, evidence: float) -> float:
    """Backward-compatible alias for the Bayesian update formula."""
    return bayes_update(prior, likelihood, evidence)


class BayesianUpdater:
    """Bayesian probability engine for trading signals and persistent learning."""

    def __init__(self, prior_win_rate: float = 0.5, factor_likelihoods: Optional[Mapping[str, Mapping[str, float]]] = None):
        self.prior_win_rate = _clamp_probability(_as_float(prior_win_rate, 0.5))
        self.factor_likelihoods = dict(factor_likelihoods or {})

    @staticmethod
    def _get_factor_value(factor_scores: Mapping[str, Any], factor_name: str) -> float:
        if not isinstance(factor_scores, Mapping):
            return 0.0
        return _as_float(factor_scores.get(factor_name, 0.0), 0.0)

    @staticmethod
    def _safe_factor_likelihoods(factor_likelihoods: Optional[Mapping[str, Mapping[str, float]]]) -> Dict[str, Dict[str, float]]:
        if not factor_likelihoods:
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for name, probs in factor_likelihoods.items():
            if not isinstance(probs, Mapping):
                continue
            result[str(name)] = {
                "given_win": _normalize_probability(probs.get("given_win", 0.5)),
                "given_loss": _normalize_probability(probs.get("given_loss", 0.5)),
            }
        return result

    def bayesian_update(self, prior_win_rate: float, new_evidence_factors: Mapping[str, float], factor_likelihoods: Optional[Mapping[str, Mapping[str, float]]] = None) -> float:
        """Apply chained Bayes updates using each factor signal as evidence.

        Formula used:
        posterior = (likelihood_win * prior) /
                    ((likelihood_win * prior) + (likelihood_loss * (1 - prior)))
        """
        prior = _clamp_probability(_as_float(prior_win_rate, 0.5))
        if not new_evidence_factors:
            return prior

        likelihood_map = self._safe_factor_likelihoods(self.factor_likelihoods if factor_likelihoods is None else factor_likelihoods)
        for factor_name, factor_value in new_evidence_factors.items():
            normalized_name = str(factor_name).lower()
            factor_value = _as_float(factor_value, 0.0)

            if normalized_name in likelihood_map:
                win_likelihood = _normalize_probability(likelihood_map[normalized_name].get("given_win", 0.5))
                loss_likelihood = _normalize_probability(likelihood_map[normalized_name].get("given_loss", 0.5))
            else:
                threshold = 0.5
                win_likelihood = 0.6 if factor_value > threshold else 0.4
                loss_likelihood = 0.45 if factor_value > threshold else 0.55
                win_likelihood = _normalize_probability(win_likelihood)
                loss_likelihood = _normalize_probability(loss_likelihood)

            evidence_term = (win_likelihood * prior) + (loss_likelihood * (1.0 - prior))
            if evidence_term == 0:
                continue
            prior = (win_likelihood * prior) / evidence_term
            prior = _clamp_probability(prior)

        return float(prior)

    def update_factor_likelihoods(self, completed_trades_csv: str | Path) -> Dict[str, Dict[str, float]]:
        """Learn per-factor win/loss likelihoods from a trade log CSV."""
        csv_path = Path(completed_trades_csv)
        if not csv_path.exists():
            return {}

        trade_rows: List[Dict[str, Any]] = []
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                trade_rows.append(row)

        if not trade_rows:
            return {}

        factor_likelihoods: Dict[str, Dict[str, float]] = {}
        for factor_name in DEFAULT_FACTOR_NAMES:
            win_count = 0
            loss_count = 0
            total_win = 0
            total_loss = 0

            for row in trade_rows:
                outcome = str(row.get("outcome", "")).strip().lower()
                is_win = outcome in {"win", "winner", "profit", "1", "true", "successful"}
                factor_value = _as_float(row.get(factor_name, 0.0), 0.0)

                if is_win:
                    total_win += 1
                    if factor_value > 0.5:
                        win_count += 1
                else:
                    total_loss += 1
                    if factor_value > 0.5:
                        loss_count += 1

            factor_likelihoods[factor_name] = {
                "given_win": _normalize_probability(win_count / total_win) if total_win else 0.5,
                "given_loss": _normalize_probability(loss_count / total_loss) if total_loss else 0.5,
            }

        config_dir = csv_path.resolve().parents[1] / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        json_path = config_dir / "factor_likelihoods.json"
        json_path.write_text(json.dumps(factor_likelihoods, indent=2), encoding="utf-8")
        self.factor_likelihoods = factor_likelihoods
        return factor_likelihoods

    @staticmethod
    def naive_bayes_classifier(factor_scores: Mapping[str, float] | Sequence[float], historical_data: Optional[Sequence[Mapping[str, float]]] = None, outcomes: Optional[Sequence[int]] = None):
        """Predict the win probability using a Gaussian Naive Bayes model."""
        try:
            from sklearn.naive_bayes import GaussianNB
            import numpy as np
        except ImportError:
            return {"win_probability": 0.5, "model_confidence": 0.5}

        if historical_data is None or outcomes is None:
            if isinstance(factor_scores, Mapping):
                current = np.array([[float(factor_scores.get(f"f{i}", 0.0)) for i in range(1, 10)]])
                return {"win_probability": 0.5, "model_confidence": 0.5, "current": current.tolist()}
            return {"win_probability": 0.5, "model_confidence": 0.5}

        rows = []
        for row in historical_data:
            if isinstance(row, Mapping):
                rows.append([float(row.get(f"f{i}", 0.0)) for i in range(1, 10)])
            else:
                rows.append([float(item) for item in row])

        if len(rows) < 2:
            return {"win_probability": 0.5, "model_confidence": 0.5}

        X = np.asarray(rows, dtype=float)
        y = np.asarray(outcomes, dtype=int)
        model = GaussianNB()
        model.fit(X, y)

        if isinstance(factor_scores, Mapping):
            current = np.asarray([[float(factor_scores.get(f"f{i}", 0.0)) for i in range(1, 10)]], dtype=float)
        else:
            current = np.asarray([list(factor_scores)], dtype=float)

        probabilities = model.predict_proba(current)[0]
        win_probability = float(probabilities[1] if len(probabilities) > 1 else probabilities[0])
        return {
            "win_probability": _clamp_probability(win_probability),
            "model_confidence": float(max(probabilities)),
        }

    @staticmethod
    def conditional_probability_table(factor_data: Sequence[Mapping[str, float]], outcomes: Sequence[int]):
        """Compute win rates for important factor pairs that may drive the trade decision."""
        pair_list = [("f1", "f8"), ("f4", "f7"), ("f2", "f5"), ("f6", "f4")]
        results: Dict[str, Dict[str, Dict[str, float]]] = {}

        for left_name, right_name in pair_list:
            buckets = {
                "both_high": {"wins": 0, "total": 0},
                "both_low": {"wins": 0, "total": 0},
                "left_high_right_low": {"wins": 0, "total": 0},
                "left_low_right_high": {"wins": 0, "total": 0},
            }

            for row, outcome in zip(factor_data, outcomes):
                left_value = _as_float(row.get(left_name, 0.0), 0.0)
                right_value = _as_float(row.get(right_name, 0.0), 0.0)
                outcome_flag = int(bool(outcome))

                if left_value > 0.5 and right_value > 0.5:
                    bucket = "both_high"
                elif left_value <= 0.5 and right_value <= 0.5:
                    bucket = "both_low"
                elif left_value > 0.5 and right_value <= 0.5:
                    bucket = "left_high_right_low"
                else:
                    bucket = "left_low_right_high"

                buckets[bucket]["total"] += 1
                buckets[bucket]["wins"] += outcome_flag

            results[f"{left_name}_{right_name}"] = {
                key: {
                    "win_rate": (value["wins"] / value["total"]) if value["total"] else 0.0,
                    "wins": value["wins"],
                    "total": value["total"],
                }
                for key, value in buckets.items()
            }

        return results

    @staticmethod
    def expected_value_calculator(probability: float, risk_amount: float, reward_amount: float, available_capital: float = 0.0, max_position_fraction: float = 0.05):
        """Compute expected value, Kelly fraction, and safe position sizing."""
        p = _clamp_probability(_as_float(probability, 0.5))
        risk = max(0.0, _as_float(risk_amount, 0.0))
        reward = max(0.0, _as_float(reward_amount, 0.0))

        ev = (p * reward) - ((1.0 - p) * risk)

        if reward > 0 and risk > 0:
            odds = reward / risk
            kelly_fraction = p - ((1.0 - p) / odds)
        else:
            kelly_fraction = 0.0

        half_kelly = max(0.0, kelly_fraction / 2.0)
        capital = max(0.0, _as_float(available_capital, 0.0))
        max_size = capital * max_position_fraction if capital > 0 else 0.0
        recommended_size = min(half_kelly * capital, max_size) if capital > 0 else 0.0

        return {
            "expected_value": float(ev),
            "kelly_fraction": float(_clamp_probability(kelly_fraction + 0.5) - 0.5) if kelly_fraction > 0 else 0.0,
            "recommended_size": float(recommended_size),
            "trade_worthwhile": bool(ev > 0),
        }

    def update(self, prior: float, likelihood: float, evidence: float) -> float:
        return bayes_update(prior, likelihood, evidence)

    def posterior(self, prior: float, likelihood: float, evidence: float) -> float:
        return posterior_probability(prior, likelihood, evidence)


# Backwards-compatible function names
bayes_update = bayes_update
posterior_probability = posterior_probability
