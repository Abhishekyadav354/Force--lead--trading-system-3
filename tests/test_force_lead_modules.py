from __future__ import annotations

from math import isclose

from core.factor_model import FactorModel
from core.order_flow import OrderFlowAnalyzer, TickProcessor
from statistics.bayesian import BayesianUpdater, bayes_update, posterior_probability
from statistics.monte_carlo import monte_carlo_expectation
from statistics.optimizer import optimize_threshold
from statistics.stochastic import geometric_brownian_motion


def test_bayes_update_returns_probability_bound():
    result = bayes_update(0.5, 0.8, 0.4)
    assert isclose(result, 1.0, rel_tol=1e-9)
    assert 0.0 <= result <= 1.0


def test_posterior_alias_matches_bayes_rule():
    result = posterior_probability(0.2, 0.9, 0.3)
    assert isclose(result, 0.6, rel_tol=1e-9)


def test_bayesian_updater_chains_factor_evidence():
    updater = BayesianUpdater(prior_win_rate=0.5)
    result = updater.bayesian_update(
        0.5,
        {"f1": 0.8, "f4": 0.7},
        {"f1": {"given_win": 0.8, "given_loss": 0.3}, "f4": {"given_win": 0.7, "given_loss": 0.4}},
    )
    assert 0.0 <= result <= 1.0
    assert result > 0.5


def test_expected_value_calculator_returns_trade_signal():
    result = BayesianUpdater.expected_value_calculator(0.6, 1000, 2000, available_capital=10000)
    assert result["trade_worthwhile"] is True
    assert result["recommended_size"] >= 0.0


def test_factor_model_aggregates_scores():
    model = FactorModel()
    score = model.aggregate_score({
        "momentum": 0.8,
        "volume": 0.6,
        "trend": 0.7,
        "volatility": 0.2,
    })
    assert 0.0 <= score <= 100.0
    assert score > 0


def test_gbm_path_has_expected_length_and_starts_at_s0():
    path = geometric_brownian_motion(100.0, 0.05, 0.2, steps=10)
    assert len(path) == 11
    assert path[0] == 100.0


def test_factor_model_linear_algebra_methods():
    model = FactorModel()
    candles = [
        {"close": 100.0, "open": 99.0, "high": 101.0, "low": 98.5, "volume": 1200, "nifty_direction": 1.0, "spread": 0.05},
        {"close": 101.5, "open": 100.0, "high": 102.0, "low": 99.2, "volume": 1500, "nifty_direction": 1.0, "spread": 0.04},
        {"close": 102.7, "open": 101.5, "high": 103.5, "low": 101.0, "volume": 1700, "nifty_direction": 1.0, "spread": 0.03},
    ]
    matrix, returns = model.build_factor_matrix(candles)
    assert matrix.shape[1] == 9
    assert len(returns) == len(candles) - 1

    corr, condition_number = model.calculate_correlation_matrix(matrix)
    assert corr.shape == (9, 9)
    assert condition_number >= 0

    reduced, n_components, explained = model.pca_analysis(matrix, variance_threshold=0.90)
    assert reduced.shape[1] >= 1
    assert n_components >= 1
    assert len(explained) == min(matrix.shape[0], matrix.shape[1])

    weights = model.derive_optimal_weights_regression(matrix, returns)
    assert set(weights.keys()) == {
        "F1_Volume",
        "F2_Wall",
        "F3_Position",
        "F4_Accel",
        "F5_Institutional",
        "F6_OI",
        "F7_Spread",
        "F8_Nifty",
        "F9_Candle",
    }
    assert abs(sum(weights.values()) - 1.0) < 1e-6


def test_monte_carlo_and_optimizer_basic_behavior():
    expectation = monte_carlo_expectation([1.0, 2.0, 3.0], simulations=100)
    assert 1.0 <= expectation <= 3.0
    threshold = optimize_threshold([10, 20, 30], target=25)
    assert threshold > 0


def test_detect_stacked_imbalance_thresholds_and_interruption():
    analyzer = __import__("core.order_flow", fromlist=["OrderFlowAnalyzer"]).OrderFlowAnalyzer()

    exact_three = analyzer.detect_stacked_imbalance(
        footprint={"levels": [{"price": 100.0, "imbalance": "BUY_IMBALANCE"}, {"price": 101.0, "imbalance": "BUY_IMBALANCE"}, {"price": 102.0, "imbalance": "BUY_IMBALANCE"}]},
        prices=[100.0, 101.0, 102.0],
        imbalance_type="BUY_IMBALANCE",
        min_stack=3,
    )
    assert len(exact_three) == 1
    assert exact_three[0]["price_low"] == 100.0
    assert exact_three[0]["price_high"] == 102.0
    assert exact_three[0]["strength"] == 3

    four_plus = analyzer.detect_stacked_imbalance(
        footprint={"levels": [{"price": 200.0, "imbalance": "BUY_IMBALANCE"}, {"price": 201.0, "imbalance": "BUY_IMBALANCE"}, {"price": 202.0, "imbalance": "BUY_IMBALANCE"}, {"price": 203.0, "imbalance": "BUY_IMBALANCE"}]},
        prices=[200.0, 201.0, 202.0, 203.0],
        imbalance_type="BUY_IMBALANCE",
        min_stack=3,
    )
    assert len(four_plus) == 1
    assert four_plus[0]["strength"] == 4

    interrupted = analyzer.detect_stacked_imbalance(
        footprint={"levels": [{"price": 300.0, "imbalance": "BUY_IMBALANCE"}, {"price": 301.0, "imbalance": "BUY_IMBALANCE"}, {"price": 302.0, "imbalance": "SELL_IMBALANCE"}, {"price": 303.0, "imbalance": "BUY_IMBALANCE"}]},
        prices=[300.0, 301.0, 302.0, 303.0],
        imbalance_type="BUY_IMBALANCE",
        min_stack=3,
    )
    assert interrupted == []

    empty_result = analyzer.detect_stacked_imbalance(
        footprint={"levels": []},
        prices=[],
        imbalance_type="BUY_IMBALANCE",
        min_stack=3,
    )
    assert empty_result == []

    try:
        analyzer.detect_stacked_imbalance(
            footprint={"levels": [{"price": 400.0, "imbalance": "BUY_IMBALANCE"}]},
            prices=[400.0],
            imbalance_type="BUY_IMBALANCE",
            min_stack=0,
        )
        raise AssertionError("min_stack=0 should raise ValueError")
    except ValueError:
        pass


def test_detect_absorption_direction_timestamp_window_and_zero_volume():
    analyzer = OrderFlowAnalyzer()

    buy_absorption = analyzer.detect_absorption(
        [
            {"type": "SELL_INITIATED", "volume": 90, "timestamp": 100},
            {"type": "BUY_INITIATED", "volume": 10, "timestamp": 100},
        ],
        [100.0, 100.1],
    )
    assert buy_absorption == {"type": "BUY_ABSORPTION", "strength": 90.0, "signal": 1}

    sell_absorption = analyzer.detect_absorption(
        [
            {"type": "BUY_INITIATED", "volume": 95, "timestamp": "2026-09-08T12:00:00Z"},
            {"type": "SELL_INITIATED", "volume": 5, "timestamp": "2026-09-08T11:58:00Z"},
            {"type": "SELL_INITIATED", "volume": 5, "timestamp": "2026-09-08T12:00:30Z"},
        ],
        [100.0, 99.95],
        window_seconds=60,
    )
    assert sell_absorption == {"type": "SELL_ABSORPTION", "strength": 95.0, "signal": -1}

    no_volume = analyzer.detect_absorption(
        [{"type": "SELL_INITIATED", "volume": 0}],
        [100.0, 100.0],
    )
    assert no_volume == {"type": "NONE", "strength": 0.0, "signal": 0}

    sequence_fallback = analyzer.detect_absorption(
        [{"type": "SELL_INITIATED", "volume": 10}],
        [100.0, 100.0],
    )
    assert sequence_fallback["type"] == "BUY_ABSORPTION"
    assert 0.0 <= sequence_fallback["strength"] <= 100.0


def test_detect_exhaustion_uses_recent_efficiency_trend():
    analyzer = OrderFlowAnalyzer()

    result = analyzer.detect_exhaustion(
        delta_series=[100, 80, 60, 40, 20, 10],
        price_series=[100, 101, 102, 103, 104, 105],
        volume_series=[100, 100, 100, 100, 100, 100],
    )
    assert result["is_exhausted"] is True
    assert result["efficiency_trend"] < 0.0
    assert 0.0 <= result["strength"] <= 1.0
    assert 0.0 <= result["reversal_probability"] <= 0.9

    insufficient = analyzer.detect_exhaustion(
        delta_series=[100, 50, 25],
        price_series=[100, 101, 102],
        volume_series=[100, 100, 100],
    )
    assert insufficient["is_exhausted"] is False

    constant_prices = analyzer.detect_exhaustion(
        delta_series=[100, 80, 60, 40],
        price_series=[100, 100, 100, 100],
        volume_series=[100, 100, 100, 100],
    )
    assert constant_prices == {
        "is_exhausted": False,
        "strength": 0.0,
        "efficiency_trend": 0.0,
        "reversal_probability": 0.0,
    }

    near_zero_move = analyzer.detect_exhaustion(
        delta_series=[100, 80, 60, 40],
        price_series=[100, 100 + 1e-13, 101, 102],
        volume_series=[100, 100, 100, 100],
    )
    assert near_zero_move["is_exhausted"] is False


def test_calculate_vwap_bands_is_weighted_and_zero_safe():
    analyzer = OrderFlowAnalyzer()
    ticks = [
        {"price": 100.0, "volume": 1.0},
        {"price": 101.0, "volume": 1.0},
        {"price": 102.0, "volume": 8.0},
    ]
    original_ticks = [tick.copy() for tick in ticks]

    result = analyzer.calculate_vwap_bands(ticks, std_multiplier=[1, 2])
    assert result["vwap"] == 101.7
    assert result["std_dev"] > 0.0
    assert len(result["bands"]) == 2
    assert result["bands"][0]["multiplier"] == 1.0
    assert result["bands"][1]["upper"] > result["bands"][0]["upper"]
    assert result["signal"] == "NEAR_VWAP"
    assert ticks == original_ticks

    extreme_above = analyzer.calculate_vwap_bands(
        [{"price": 100.0, "volume": 100.0}, {"price": 110.0, "volume": 1.0}]
    )
    assert extreme_above["signal"] == "EXTREME_ABOVE"

    empty = analyzer.calculate_vwap_bands([])
    zero_volume = analyzer.calculate_vwap_bands([{"price": 100.0, "volume": 0.0}])
    assert empty == zero_volume
    assert empty["vwap"] == 0.0
    assert empty["std_dev"] == 0.0
    assert empty["signal"] == "NEAR_VWAP"


def test_calculate_order_flow_score_weights_components_and_modifies_conviction():
    analyzer = OrderFlowAnalyzer()
    bullish_data = {
        "delta": {"delta": 100.0, "total_volume": 100.0},
        "divergence": {"type": "BULLISH", "signal": "1", "strength": 100.0},
        "absorption": {"type": "BUY_ABSORPTION", "signal": 1, "strength": 100.0},
        "vwap": {"signal": "ABOVE"},
        "poc": {"poc_signal": "ABOVE_VALUE"},
        "stacked_buy_zones": [{"price_low": 100.0, "price_high": 102.0}],
    }
    bullish = analyzer.calculate_order_flow_score(bullish_data)
    assert bullish["signal"] == "BUY"
    assert 0.40 < bullish["score"] <= 1.0
    assert bullish["components"]["divergence"] == 1.0
    assert bullish["components"]["stacked"] == 1.0

    exhausted = analyzer.calculate_order_flow_score({
        **bullish_data,
        "exhaustion": {"is_exhausted": True, "strength": 0.5},
    })
    assert exhausted["score"] == bullish["score"] * 0.5
    assert exhausted["components"]["exhaustion_modifier"] == 0.5

    neutral = analyzer.calculate_order_flow_score({})
    assert neutral["signal"] == "WAIT"
    assert neutral["score"] == 0.0
    assert all(
        -1.0 <= value <= 1.0
        for name, value in neutral["components"].items()
        if name != "exhaustion_modifier"
    )


def test_integrate_with_force_score_applies_symmetric_overrides():
    analyzer = OrderFlowAnalyzer()

    confirmed_buy = analyzer.integrate_with_force_score({"score": 0.8}, {"score": 0.7})
    assert confirmed_buy["signal"] == "BUY"
    assert confirmed_buy["override"]["type"] == "CONFIRMATION"
    assert isclose(confirmed_buy["final_score"], 0.8525)

    contradiction_buy = analyzer.integrate_with_force_score(0.8, -0.8)
    assert contradiction_buy["override"]["type"] == "CONTRADICTION"
    assert isclose(contradiction_buy["final_score"], 0.1)
    assert contradiction_buy["signal"] == "WAIT"

    confirmed_sell = analyzer.integrate_with_force_score(-0.8, -0.8)
    assert confirmed_sell["signal"] == "SELL"
    assert confirmed_sell["override"]["type"] == "CONFIRMATION"

    contradiction_sell = analyzer.integrate_with_force_score(-0.8, 0.8)
    assert contradiction_sell["override"]["type"] == "CONTRADICTION"
    assert isclose(contradiction_sell["final_score"], -0.1)

    neutral = analyzer.integrate_with_force_score({}, {})
    assert neutral["final_score"] == 0.0
    assert neutral["signal"] == "WAIT"
    assert neutral["override"]["applied"] is False

    clamped = analyzer.integrate_with_force_score(5.0, -5.0)
    assert -1.0 <= clamped["final_score"] <= 1.0
    assert clamped["force_component"] == 1.0
    assert clamped["order_flow_component"] == -1.0


def test_tick_processor_updates_completes_candle_and_resets_safely():
    analyzer = OrderFlowAnalyzer()
    processor = TickProcessor(
        analyzer=analyzer,
        max_tick_buffer=3,
        max_candle_history=2,
        max_vwap_history=3,
    )

    assert processor.process_tick({"price": "bad", "volume": 1}) is None
    assert processor.tick_buffer == []

    update = None
    for index in range(10):
        update = processor.process_tick({
            "price": 100.0 + index * 0.1,
            "volume": 10.0,
            "timestamp": index,
            "bid": 99.9 + index * 0.1,
            "ask": 100.1 + index * 0.1,
        })
        if index < 9:
            assert update is None

    assert update is not None
    assert set(update) == {"live_delta", "vwap", "absorption"}
    assert "delta" in update["live_delta"]
    assert "vwap" in update["vwap"]
    assert len(processor.tick_buffer) == 3
    assert len(processor.vwap_ticks) == 3
    assert processor.prev_bid is not None
    assert processor.prev_ask is not None

    completed = processor.on_candle_complete()
    assert completed is not None
    assert set(completed) == {
        "candle_delta",
        "footprint",
        "volume_profile",
        "absorption",
        "exhaustion",
    }
    assert len(processor.tick_buffer) == 0
    assert len(processor.vwap_ticks) == 0
    assert len(processor.candle_deltas) == 1

    processor.process_tick({
        "price": 101.0,
        "volume": 5.0,
        "timestamp": 11,
        "bid": 100.9,
        "ask": 101.1,
    })
    processor.reset()
    assert processor.tick_buffer == []
    assert processor.candle_deltas == []
    assert processor.vwap_ticks == []
    assert processor.prev_bid is None
    assert processor.prev_ask is None
    assert analyzer.prev_price is None


def test_advanced_delta_statistics_handles_empty_and_constant_series():
    analyzer = OrderFlowAnalyzer()

    empty = analyzer.advanced_delta_statistics([])
    assert empty["delta_zscore"] == 0.0
    assert empty["is_extreme_delta"] is False
    assert empty["momentum_signal"] == 0
    assert empty["entropy"] == 0.0
    assert empty["signal_clarity"] == 0.0
    assert empty["hurst"] == 0.5

    constant = analyzer.advanced_delta_statistics([5.0] * 8)
    assert constant["delta_zscore"] == 0.0
    assert constant["is_extreme_delta"] is False
    assert constant["delta_momentum"] == 0.0
    assert constant["momentum_signal"] == 0
    assert constant["entropy"] == 0.0
    assert constant["autocorrelation"] == 0.0
    assert constant["mean_reversion"] is False


def test_advanced_delta_statistics_reports_acceleration_and_bounds_metrics():
    analyzer = OrderFlowAnalyzer()
    result = analyzer.advanced_delta_statistics(
        [1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0, 37.0, 46.0]
    )

    assert result["momentum_signal"] == 1
    assert result["delta_momentum"] > 0.0
    assert 0.0 <= result["entropy"] <= 1.0
    assert 0.0 <= result["signal_clarity"] <= 1.0
    assert 0.0 <= result["hurst"] <= 1.0
    assert 0.0 <= result["hurst_estimate"] <= 1.0
    assert -1.0 <= result["autocorrelation"] <= 1.0
