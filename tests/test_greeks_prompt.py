from __future__ import annotations

from core.greeks_calculator import GreeksCalculator


def _chain():
    return [
        {"strike": 99.0, "type": "CALL", "last_price": 3.1, "oi": 2000},
        {"strike": 100.0, "type": "CALL", "last_price": 2.8, "oi": 2200},
        {"strike": 101.0, "type": "CALL", "last_price": 2.4, "oi": 2100},
        {"strike": 99.0, "type": "PUT", "last_price": 1.8, "oi": 1800},
        {"strike": 100.0, "type": "PUT", "last_price": 2.1, "oi": 1900},
        {"strike": 101.0, "type": "PUT", "last_price": 2.6, "oi": 1700},
    ]


def test_calculate_delta_from_chain():
    calc = GreeksCalculator()
    res = calc.calculate_delta_from_chain(_chain(), 100.0)
    assert set(res.keys()) >= {"atm_call_delta", "atm_put_delta", "sentiment", "force_score_addition"}
    assert -1.0 <= res["atm_call_delta"] <= 1.0
    assert -1.0 <= res["atm_put_delta"] <= 1.0
    assert res["sentiment"] in {"STRONGLY_BULLISH", "NEUTRAL", "BEARISH"}


def test_calculate_gamma_exposure():
    calc = GreeksCalculator()
    res = calc.calculate_gamma_exposure(_chain(), 100.0)
    assert set(res.keys()) >= {"net_gamma", "position_multiplier", "warning"}
    assert res["position_multiplier"] in {0.7, 1.0}


def test_implied_volatility_analysis():
    calc = GreeksCalculator()
    chain = _chain()
    chain[0]["last_price"] = 1.7
    chain[1]["last_price"] = 1.3
    chain[2]["last_price"] = 1.1
    chain[3]["last_price"] = 1.9
    chain[4]["last_price"] = 2.2
    chain[5]["last_price"] = 2.7
    res = calc.implied_volatility_analysis(chain)
    assert set(res.keys()) >= {"iv_skew", "iv_percentile", "implied_move", "skew_signal"}
    assert res["skew_signal"] in {"BULLISH", "BEARISH", "NEUTRAL"}


def test_max_pain_analysis():
    calc = GreeksCalculator()
    res = calc.max_pain_analysis(_chain(), "2026-09-02")
    assert set(res.keys()) >= {"max_pain", "distance_pct", "days_to_expiry", "pinning_likely"}
    assert isinstance(res["days_to_expiry"], int)
