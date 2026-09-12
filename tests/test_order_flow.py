from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "force-lead-trading-system"
for path in (ROOT, PACKAGE_ROOT):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from core.order_flow import OrderFlowAnalyzer, TickProcessor


@contextmanager
def raises(expected_exception: type[BaseException]):
    """Small pytest.raises-compatible context for standalone execution."""
    try:
        yield
    except expected_exception:
        return
    raise AssertionError(f"Expected {expected_exception.__name__} to be raised")


def make_tick(
    price: float,
    volume: float = 10.0,
    bid: float | None = 99.0,
    ask: float | None = 101.0,
    timestamp: int = 1,
) -> dict:
    return {
        "price": price,
        "volume": volume,
        "bid": bid,
        "ask": ask,
        "timestamp": timestamp,
    }


def classified_tick(price: float, volume: float, tick_type: str) -> dict:
    return {
        "price": price,
        "volume": volume,
        "type": tick_type,
        "timestamp": 1,
    }


def test_classify_buy_at_ask_and_sell_at_bid():
    analyzer = OrderFlowAnalyzer()

    buy = analyzer.classify_tick(make_tick(101.0, bid=100.0, ask=101.0))
    assert set(buy) == {"type", "volume", "price", "timestamp"}
    assert buy["type"] == "BUY_INITIATED"

    sell = OrderFlowAnalyzer().classify_tick(make_tick(99.0, bid=99.0, ask=100.0))
    assert sell["type"] == "SELL_INITIATED"


def test_classify_tick_rule_and_unchanged_price_fallback():
    analyzer = OrderFlowAnalyzer()
    analyzer.classify_tick(make_tick(100.0, bid=None, ask=None))
    rising = analyzer.classify_tick(make_tick(101.0, bid=None, ask=None))
    unchanged = analyzer.classify_tick(make_tick(101.0, bid=None, ask=None))
    assert rising["type"] == "BUY_INITIATED"
    assert unchanged["type"] == "BUY_INITIATED"

    neutral = OrderFlowAnalyzer().classify_tick(make_tick(100.0, bid=None, ask=None))
    assert neutral["type"] == "NEUTRAL"


def test_invalid_tick_and_zero_volume():
    analyzer = OrderFlowAnalyzer()
    with raises(ValueError):
        analyzer.classify_tick({"price": "invalid", "volume": 1, "timestamp": 1, "bid": 99, "ask": 101})

    zero = analyzer.classify_tick(make_tick(101.0, volume=0.0))
    assert zero["type"] == "BUY_INITIATED"
    assert zero["volume"] == 0.0


def test_delta_cumulative_delta_and_delta_per_minute():
    analyzer = OrderFlowAnalyzer()
    ticks = [
        classified_tick(100.0, 8.0, "BUY_INITIATED"),
        classified_tick(100.0, 3.0, "SELL_INITIATED"),
    ]
    delta = analyzer.calculate_delta(ticks)
    assert set(delta) == {"delta", "buy_volume", "sell_volume", "delta_ratio", "total_volume"}
    assert delta["delta"] == 5.0
    assert delta["total_volume"] == 11.0

    cumulative = analyzer.calculate_cumulative_delta([
        {"timestamp": 1, "delta": 2.0},
        {"timestamp": 2, "delta": -1.0},
        {"timestamp": 3, "delta": 4.0},
    ])
    assert [item["cumulative_delta"] for item in cumulative] == [2.0, 1.0, 5.0]
    assert all("slope" in item for item in cumulative)

    per_minute = analyzer.calculate_delta_per_minute(30.0, 30.0, historical_avg_delta_pm=60.0)
    assert set(per_minute) == {"delta_per_minute", "normalized_delta", "historical_avg_delta_pm"}
    assert per_minute["delta_per_minute"] == 60.0
    assert per_minute["normalized_delta"] == 1.0


def test_divergence_insufficient_bullish_bearish_and_flat():
    analyzer = OrderFlowAnalyzer()
    insufficient = analyzer.detect_delta_divergence([100.0], [1.0])
    assert insufficient["type"] == "NONE"
    assert insufficient["signal"] == "0"

    bullish = analyzer.detect_delta_divergence([103.0, 102.0, 101.0, 100.0], [1.0, 2.0, 3.0, 4.0])
    assert bullish["type"] == "BULLISH"
    assert bullish["signal"] == "1"

    bearish = analyzer.detect_delta_divergence([100.0, 101.0, 102.0, 103.0], [4.0, 3.0, 2.0, 1.0])
    assert bearish["type"] == "BEARISH"
    assert bearish["signal"] == "-1"

    flat = analyzer.detect_delta_divergence([100.0, 100.0, 100.0], [1.0, 1.0, 1.0])
    assert flat["type"] == "NONE"
    assert flat["strength"] == 0.0


def test_volume_profiles_empty_single_price_and_value_area():
    analyzer = OrderFlowAnalyzer()
    empty = analyzer.build_volume_profile([])
    assert set(empty) == {"profile", "poc", "vah", "val", "poc_signal", "total_volume"}
    assert empty["total_volume"] == 0.0

    single = analyzer.build_volume_profile([
        classified_tick(100.0, 5.0, "BUY_INITIATED"),
    ], current_price=100.0)
    assert single["poc"] == 0.0
    assert single["total_volume"] == 5.0
    assert single["vah"] >= single["val"]

    profile = analyzer.build_volume_profile([
        classified_tick(100.0, 10.0, "BUY_INITIATED"),
        classified_tick(101.0, 30.0, "SELL_INITIATED"),
        classified_tick(102.0, 5.0, "BUY_INITIATED"),
    ], price_levels=3, current_price=102.0)
    assert profile["total_volume"] == 45.0
    assert profile["vah"] >= profile["val"]
    assert profile["poc"] >= profile["val"]
    assert profile["poc"] <= profile["vah"]
    assert profile["poc_signal"] in {"ABOVE_VALUE", "ABOVE_POC", "NEAR_POC", "BELOW_POC", "BELOW_VALUE"}


def test_footprint_imbalance_and_stacked_zone_final_flush():
    analyzer = OrderFlowAnalyzer()
    ticks = [
        classified_tick(100.0, 10.0, "BUY_INITIATED"),
        classified_tick(101.0, 10.0, "BUY_INITIATED"),
        classified_tick(102.0, 10.0, "BUY_INITIATED"),
        classified_tick(103.0, 10.0, "SELL_INITIATED"),
    ]
    footprint = analyzer.build_footprint(ticks, candle_open=100.0, candle_close=103.0)
    assert set(footprint) == {
        "footprint", "stacked_buy_zones", "stacked_sell_zones",
        "max_buy_level", "max_sell_level", "candle_open", "candle_close",
    }
    assert footprint["footprint"][0]["imbalance"] == "BUY_IMBALANCE"

    levels = [
        {"price": 100.0, "imbalance": "BUY_IMBALANCE"},
        {"price": 101.0, "imbalance": "BUY_IMBALANCE"},
        {"price": 102.0, "imbalance": "BUY_IMBALANCE"},
        {"price": 103.0, "imbalance": "SELL_IMBALANCE"},
    ]
    zones = analyzer.detect_stacked_imbalance(
        {"levels": levels}, [103.0, 102.0, 101.0, 100.0], "BUY_IMBALANCE", min_stack=3
    )
    assert len(zones) == 1
    assert set(zones[0]) == {"price_low", "price_high", "levels", "strength"}
    assert zones[0]["price_low"] == 100.0
    assert zones[0]["price_high"] == 102.0
    assert zones[0]["strength"] == 3


def test_absorption_exhaustion_and_vwap_statistics():
    analyzer = OrderFlowAnalyzer()
    absorption = analyzer.detect_absorption([
        classified_tick(100.0, 90.0, "SELL_INITIATED"),
        classified_tick(100.1, 10.0, "BUY_INITIATED"),
    ], [100.0, 100.1])
    assert set(absorption) == {"type", "strength", "signal"}
    assert absorption["type"] == "BUY_ABSORPTION"
    assert absorption["signal"] == 1

    exhaustion = analyzer.detect_exhaustion(
        [100.0, 80.0, 60.0, 40.0, 20.0],
        [100.0, 101.0, 102.0, 103.0, 104.0],
        [100.0] * 5,
    )
    assert set(exhaustion) == {"is_exhausted", "strength", "efficiency_trend", "reversal_probability"}
    assert exhaustion["is_exhausted"] is True
    assert 0.0 <= exhaustion["reversal_probability"] <= 0.9

    vwap = analyzer.calculate_vwap_bands([
        {"price": 100.0, "volume": 1.0},
        {"price": 101.0, "volume": 1.0},
        {"price": 102.0, "volume": 2.0},
    ])
    assert set(vwap) == {"vwap", "bands", "distance_std", "signal", "std_dev"}
    assert math.isclose(vwap["vwap"], 101.25)
    assert len(vwap["bands"]) == 3

    zero_std = analyzer.calculate_vwap_bands([
        {"price": 100.0, "volume": 10.0},
        {"price": 100.0, "volume": 20.0},
    ])
    assert zero_std["std_dev"] == 0.0
    assert zero_std["distance_std"] == 0.0
    assert zero_std["signal"] == "NEAR_VWAP"


def test_f10_score_bounds_and_force_order_flow_combination():
    analyzer = OrderFlowAnalyzer()
    score = analyzer.calculate_order_flow_score({
        "delta": {"delta": 100.0, "total_volume": 100.0},
        "divergence": {"type": "BULLISH", "strength": 1.0},
        "absorption": {"type": "BUY_ABSORPTION", "strength": 100.0},
        "vwap": {"signal": "ABOVE"},
        "poc": {"poc_signal": "ABOVE_VALUE"},
        "stacked_buy_zones": [{"price_low": 100.0, "price_high": 102.0}],
    })
    assert set(score) == {"score", "components", "signal"}
    assert -1.0 <= score["score"] <= 1.0
    assert all(-1.0 <= value <= 1.0 for value in score["components"].values())

    combined = analyzer.integrate_with_force_score({"score": 0.8}, score)
    assert set(combined) == {
        "final_score", "force_component", "order_flow_component", "override", "signal",
    }
    assert -1.0 <= combined["final_score"] <= 1.0
    assert combined["signal"] in {"BUY", "SELL", "WAIT"}


def test_tick_processor_reset_and_candle_completion():
    processor = TickProcessor(max_tick_buffer=20, max_candle_history=3, max_vwap_history=20)
    for index in range(3):
        result = processor.process_tick(make_tick(
            100.0 + index * 0.1,
            volume=10.0,
            bid=99.9 + index * 0.1,
            ask=100.1 + index * 0.1,
            timestamp=index,
        ))
        assert result is None

    completed = processor.on_candle_complete()
    assert completed is not None
    assert set(completed) == {
        "candle_delta", "footprint", "volume_profile", "absorption", "exhaustion",
    }
    assert processor.tick_buffer == []
    assert processor.vwap_ticks == []
    assert len(processor.candle_deltas) == 1

    processor.process_tick(make_tick(101.0, timestamp=10))
    processor.reset()
    assert processor.tick_buffer == []
    assert processor.candle_deltas == []
    assert processor.vwap_ticks == []
    assert processor.prev_bid is None
    assert processor.prev_ask is None
    assert processor.analyzer.prev_price is None


if __name__ == "__main__":
    failures = 0
    test_functions = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]
    for name, function in test_functions:
        try:
            function()
        except Exception as error:  # pragma: no cover - standalone failure path
            failures += 1
            print(f"FAIL: {name}: {error}")
    print(f"{len(test_functions) - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)
