from __future__ import annotations

from core.reversal_calculator import ReversalCalculator


def test_no_reversal_signals_returns_empty_list_and_zero_probability():
    calc = ReversalCalculator()
    result = calc.detect_reversal_starting({})
    assert result["reversal_signals"] == []
    assert result["reversal_probability"] == 0.0
    assert result["probability_curve"] == [0.0]


def test_one_signal_is_normalized_and_detected():
    calc = ReversalCalculator()
    result = calc.detect_reversal_starting(
        evidence={
            "divergence": {"type": "BULLISH", "signal": "1", "strength": 0.8},
        }
    )
    assert len(result["reversal_signals"]) == 1
    signal = result["reversal_signals"][0]
    assert signal["name"] == "divergence"
    assert signal["detected"] is True
    assert 0.0 <= signal["strength"] <= 1.0
    assert signal["direction"] == "BUY"
    assert 0.0 <= result["reversal_probability"] <= 1.0


def test_multiple_signals_are_normalized_and_return_probabilities_bounded():
    calc = ReversalCalculator()
    result = calc.detect_reversal_starting(
        evidence={
            "delta": {"delta": -5.0},
            "volume": 3000,
            "historical_average_volume": 1000,
            "divergence": {"type": "BEARISH", "signal": "-1", "strength": 0.9},
            "absorption": {"signal": -1, "strength": 70.0},
        }
    )
    assert len(result["reversal_signals"]) >= 2
    assert all(0.0 <= signal["strength"] <= 1.0 for signal in result["reversal_signals"])
    assert 0.0 <= result["reversal_probability"] <= 1.0


def test_missing_order_flow_fields_are_safe_and_empty():
    calc = ReversalCalculator()
    result = calc.detect_reversal_starting({
        "force": None,
        "delta": None,
        "lead_time": None,
        "historical_average_volume": None,
        "regime": None,
        "oi": None,
        "absorption": None,
        "divergence": None,
    })
    assert result["reversal_signals"] == []
    assert result["reversal_probability"] == 0.0
    assert result["metadata"]["guardrails"]["missing_fields_safe"] is True


def test_zero_historical_volume_stays_safe_and_bounded():
    calc = ReversalCalculator()
    result = calc.detect_reversal_starting({
        "historical_average_volume": 0,
        "volume": 0,
        "divergence": {"type": "BULLISH", "signal": "1", "strength": 0.1},
    })
    assert result["probability_curve"]
    assert all(0.0 <= value <= 1.0 for value in result["probability_curve"])
    assert result["reversal_probability"] <= 1.0
    assert result["reversal_probability"] >= 0.0
