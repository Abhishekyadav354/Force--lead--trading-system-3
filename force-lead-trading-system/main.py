import argparse
import csv
import json
import os
import sys
import subprocess
import threading

from dotenv import load_dotenv


load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("paper", "live", "backtest", "test"),
        default=os.getenv("TRADING_MODE", "paper"),
    )
    parser.add_argument(
        "--symbol",
        default=os.getenv("STOCK_SYMBOL", "NIFTY"),
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=float(os.getenv("CAPITAL", "10000")),
    )
    parser.add_argument("--port", type=int, default=5000)
    return parser.parse_args()


def run_backtest_mode():
    print("Running backtest mode.")

    from data.historical import HistoricalData
    from statistics.backtester import Backtester

    fallback_weights = [0.25, 0.20, 0.10, 0.20, 0.10, 0.05, 0.05, 0.03, 0.02]
    weights_path = os.path.join(os.path.dirname(__file__), "config", "optimal_weights.json")
    try:
        with open(weights_path, "r", encoding="utf-8") as weights_file:
            weights = json.load(weights_file)
    except (OSError, json.JSONDecodeError):
        weights = fallback_weights

    historical_data = HistoricalData(None)
    cache_path = os.path.join(os.path.dirname(__file__), "logs", "historical_cache.csv")
    try:
        with open(cache_path, "r", encoding="utf-8", newline="") as cache_file:
            candles = list(csv.DictReader(cache_file))
    except OSError:
        candles = historical_data.fetch_last_200_candles()

    backtester = Backtester()
    results = backtester.run_full_backtest(candles, weights)
    for key, value in results.items():
        print(f"{key}: {value}")

    return results


def run_test_mode():
    print("Running integration tests...")
    completed = subprocess.run(["python", "tests/test_system.py"])
    return completed.returncode


def main():
    args = parse_args()
    os.environ["TRADING_MODE"] = args.mode
    os.environ["STOCK_SYMBOL"] = args.symbol
    os.environ["CAPITAL"] = str(args.capital)

    if args.mode == "backtest":
        run_backtest_mode()
        return 0

    if args.mode == "test":
        return run_test_mode()

    from startup import initialize_system
    from main_loop import run_main_loop
    from utils.trade_logger import TradeLogger
    from dashboard.prediction_server import broadcast_prediction, start_dashboard

    if args.mode == "live":
        print("WARNING: LIVE TRADING MODE is selected.")
        confirmation = input("Type 'yes' to continue: ").strip().lower()
        if confirmation != "yes":
            print("Live mode cancelled. Use --mode paper to run without live trading.")
            return 0

    try:
        feed, system, historical, nifty = initialize_system()
    except Exception as exc:
        print(f"Startup failed: {str(exc)}")
        print("Check your .env file and API credentials.")
        return 1

    logger = TradeLogger()
    main_loop_thread = threading.Thread(
        target=run_main_loop,
        args=(feed, system, historical, nifty, broadcast_prediction, logger),
        name="MainLoop",
        daemon=True,
    )
    main_loop_thread.start()
    print(f"Analysis loop running in thread: {main_loop_thread.name}")

    print(f"Dashboard available at http://localhost:{args.port}")
    try:
        start_dashboard(host="0.0.0.0", port=args.port)
    except KeyboardInterrupt:
        print("Shutting down dashboard and trading system...")
        feed.disconnect()
        print("Goodbye.")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
