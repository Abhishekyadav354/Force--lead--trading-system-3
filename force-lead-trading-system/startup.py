import json
import math
import time
from pathlib import Path

import joblib

from config.settings import settings
from data.historical import HistoricalData
from data.live_feed import LiveDataFeed
from ml.model_trainer import ModelTrainer
from statistics.optimizer import SystemOptimizer

DEFAULT_WEIGHTS = [0.25, 0.20, 0.10, 0.20, 0.10, 0.05, 0.05, 0.03, 0.02]


def _valid_weights(value):
    if isinstance(value, list):
        return bool(value) and all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in value
        )
    if isinstance(value, dict):
        return bool(value) and all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in value.values()
        )
    return False


print(
    """
=============================================
  FORCE-LEAD TRADING SYSTEM STARTING
  =============================================

  Do not initialize any components yet.
"""
)


def initialize_system():
    """Initialize all system components and return: feed, system, historical_candles, nifty_candles."""
    feed = LiveDataFeed(
        api_key=settings.angel_api_key,
        client_id=settings.angel_client_id,
        password=settings.angel_password,
        totp_secret=settings.angel_totp_secret,
    )

    try:
        feed.connect()
    except Exception as exc:
        raise ConnectionError(
            "Angel One API connection failed. Check credentials in .env"
        ) from exc

    if not getattr(feed, "connected", False):
        raise ConnectionError(
            "Angel One API connection failed. Check credentials in .env"
        )

    print("API connection successful: Angel One live feed connected.")

    stock_symbol = settings.stock_symbol
    timeframe = settings.timeframe

    if not hasattr(feed, "api"):
        feed.api = getattr(feed, "api_client", None)

    hist_loader = HistoricalData(feed.api)
    hist_loader.symbol = stock_symbol
    hist_loader.timeframe = timeframe
    hist_loader.manager.symbol = stock_symbol
    hist_loader.manager.interval = f"{timeframe}minute"

    historical_candles = hist_loader.fetch_last_200_candles()

    if len(historical_candles) < 50:
        print(f"Warning: only {len(historical_candles)} candles loaded for {stock_symbol}.")
    else:
        print(f"{len(historical_candles)} candles loaded successfully.")

    nifty_loader = HistoricalData(feed.api, symbol="NIFTY", timeframe=timeframe, limit=200)
    nifty_candles = nifty_loader.fetch_last_200_candles()
    print("NIFTY data loaded successfully.")

    trainer = ModelTrainer()
    base_dir = Path(__file__).resolve().parent
    models_path = base_dir / "ml" / "saved_models.pkl"
    scaler_path = base_dir / "ml" / "scaler.pkl"

    try:
        if models_path.exists():
            trainer.models_ = joblib.load(models_path)
            print("Saved models were loaded.")

        if scaler_path.exists():
            trainer.scaler = joblib.load(scaler_path)
    except Exception:
        print("No saved models found. Training fresh...")
        X, y = trainer.prepare_features(historical_candles)
        if len(X) < 30:
            print("Warning: not enough training samples to train the ML ensemble.")
            models = None
        else:
            models = trainer.train_ensemble(X, y)
            print("ML ensemble trained successfully.")

    optimizer = SystemOptimizer()
    weights_path = base_dir / "config" / "optimal_weights.json"
    optimal_weights = DEFAULT_WEIGHTS.copy()
    try:
        if weights_path.exists():
            with open(weights_path, "r", encoding="utf-8") as f:
                loaded_weights = json.load(f)
            if not _valid_weights(loaded_weights):
                raise ValueError("optimal weights must contain finite numeric values")
            optimal_weights = loaded_weights
            print("Saved weights were loaded.")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Invalid optimal weights; using defaults: {exc}")

    if not weights_path.exists():
        print("Running optimization (takes ~30 seconds)...")
        weights_array = optimizer.optimize_weights_scipy(
            historical_candles,
            target='sharpe'
        )
        if hasattr(weights_array, "tolist"):
            optimal_weights = weights_array.tolist()
        else:
            optimal_weights = list(weights_array)

        weights_path.parent.mkdir(parents=True, exist_ok=True)
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(optimal_weights, f, indent=2)

        print(f"Optimized weights: {optimal_weights}")

    from core.combined_system import CompleteTradingSystem

    capital = settings.capital
    system = CompleteTradingSystem(
        capital=capital,
        optimal_weights=optimal_weights,
        trained_models=models if "models" in locals() else None,
    )

    trading_mode = settings.trading_mode.upper()
    print(f"Symbol: {stock_symbol}")
    print(f"Capital: {capital}")
    print(f"Trading mode: {trading_mode}")
    print("SYSTEM READY")

    return (
        feed,
        system,
        historical_candles,
        nifty_candles,
    )
