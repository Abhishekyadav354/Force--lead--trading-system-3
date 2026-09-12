import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_TRADER_PATH = ROOT / "paper_trading" / "paper_trader.py"

spec = importlib.util.spec_from_file_location("paper_trader", PAPER_TRADER_PATH)
paper_trader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper_trader)
PaperTrader = paper_trader.PaperTrader


def test_virtual_paper_trade_lifecycle():
    trader = PaperTrader(virtual_capital=10000)

    buy_signal = {
        "signal": "BUY",
        "position_size": {"pct_of_capital": 10},
        "stop_loss": 99.0,
        "target": 105.0,
        "force_score": 0.8,
        "symbol": "TEST",
    }

    assert trader.check_entry(buy_signal, 100.0) is not None
    assert trader.open_position is not None
    assert trader.open_position["direction"] == "BUY"
    assert trader.open_position["virtual"] is True
    assert trader.open_position["broker_connected"] is False

    wait_signal = dict(buy_signal)
    wait_signal["signal"] = "WAIT"
    assert trader.check_entry(wait_signal, 101.0) is None
    assert trader.open_position is not None

    second_buy = dict(buy_signal)
    second_buy["signal"] = "BUY"
    assert trader.check_entry(second_buy, 102.0) is None
    assert trader.open_position is not None

    pre_exit_capital = trader.virtual_capital
    target_exit = trader.check_exit(105.0, "14:30:00")

    assert target_exit is not None
    assert trader.open_position is None
    assert target_exit["reason"] == "TARGET"
    assert target_exit["outcome"] == "WIN"
    assert trader.virtual_capital > pre_exit_capital
    assert trader.trades[-1]["reason"] == "TARGET"
    assert trader.trades[-1]["outcome"] == "WIN"
    assert trader.trades[-1]["capital"] == trader.virtual_capital
    assert trader.trades[-1]["force_score"] == 0.8
    assert all(not trade.get("broker_connected", False) for trade in trader.trades if isinstance(trade, dict))

    trader = PaperTrader(virtual_capital=10000)
    stop_signal = {
        "signal": "BUY",
        "position_size": {"pct_of_capital": 10},
        "stop_loss": 99.0,
        "target": 105.0,
        "force_score": 0.7,
        "symbol": "TEST",
    }

    assert trader.check_entry(stop_signal, 100.0) is not None
    stop_loss_exit = trader.check_exit(98.5, "14:30:00")

    assert stop_loss_exit is not None
    assert stop_loss_exit["reason"] == "SL"
    assert stop_loss_exit["outcome"] == "LOSS"
    assert trader.virtual_capital < 10000
    assert trader.trades[-1]["reason"] == "SL"
    assert trader.trades[-1]["outcome"] == "LOSS"
    assert all(not trade.get("broker_connected", False) for trade in trader.trades if isinstance(trade, dict))
