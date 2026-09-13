from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import settings

try:
    import pyotp
    from smartapi import SmartConnect
except Exception:  # pragma: no cover - optional dependency handling
    pyotp = None
    SmartConnect = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "errors.log")

logger = logging.getLogger("angel_historical")
logger.setLevel(logging.ERROR)
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)


def log_error(message: str, exc: Optional[BaseException] = None) -> None:
    if exc is not None:
        logger.exception(message)
    else:
        logger.error(message)


class HistoricalDataManager:
    """Fetch and refresh historical candles for backtesting and ML workflows."""

    CANDLE_COLUMNS = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "buy_volume",
        "sell_volume",
        "trade_count",
        "avg_trade_size",
        "bid_total",
        "ask_total",
        "bid_levels",
        "ask_levels",
        "buyers_pct",
        "sellers_pct",
        "spread",
        "volume_per_minute",
        "institutional_score",
        "spread_ratio",
        "data_source",
    ]

    def __init__(self, symbol: Optional[str] = None, interval: str = "5minute", limit: int = 200):
        self.symbol = symbol or settings.stock_symbol
        self.interval = interval
        self.limit = limit
        self.df = pd.DataFrame(columns=self.CANDLE_COLUMNS)
        self.last_refresh = None

    def _build_session(self):
        if SmartConnect is None:
            raise RuntimeError("smartapi-python is not installed.")
        if not all([settings.angel_api_key, settings.angel_client_id, settings.angel_password, settings.angel_totp_secret]):
            raise ValueError("Angel One credentials are missing from environment variables.")

        client = SmartConnect(api_key=settings.angel_api_key)
        otp = pyotp.TOTP(settings.angel_totp_secret).now()
        client.generateSession(settings.angel_client_id, settings.angel_password, otp)
        return client

    def _synthetic_candles(self) -> pd.DataFrame:
        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(minutes=self.limit * 5)
        timestamps = pd.date_range(start=start_ts, end=end_ts, freq="5min")
        base = 100.0
        price_series = [base + i * 1.25 for i in range(len(timestamps))]

        frame = pd.DataFrame({
            "symbol": self.symbol,
            "timestamp": timestamps,
            "open": price_series,
            "high": [p + 0.8 for p in price_series],
            "low": [p - 0.8 for p in price_series],
            "close": [p + 0.2 for p in price_series],
            "volume": [1200 + (i * 30) for i in range(len(timestamps))],
            "buy_volume": [int(v * 0.55) for v in range(len(timestamps))],
            "sell_volume": [int(v * 0.45) for v in range(len(timestamps))],
            "trade_count": [25 + i for i in range(len(timestamps))],
            "avg_trade_size": [100.0 + i * 0.75 for i in range(len(timestamps))],
            "bid_total": [5000 + i * 10 for i in range(len(timestamps))],
            "ask_total": [5100 + i * 12 for i in range(len(timestamps))],
            "bid_levels": [[ [p, 100], [p - 0.2, 80], [p - 0.4, 60], [p - 0.6, 50], [p - 0.8, 40] ] for p in price_series],
            "ask_levels": [[ [p + 0.2, 90], [p + 0.4, 70], [p + 0.6, 60], [p + 0.8, 50], [p + 1.0, 40] ] for p in price_series],
            "buyers_pct": [55.0 + (i % 10) for i in range(len(timestamps))],
            "sellers_pct": [45.0 - (i % 10) for i in range(len(timestamps))],
            "spread": [0.15 + (i * 0.01) for i in range(len(timestamps))],
            "volume_per_minute": [float(v / 5) for v in [1200 + (i * 30) for i in range(len(timestamps))]],
            "institutional_score": [1.0 + (i * 0.05) for i in range(len(timestamps))],
            "spread_ratio": [1.0 + (i * 0.02) for i in range(len(timestamps))],
            "data_source": "synthetic",
        })
        return frame

    def _fetch_from_api(self) -> pd.DataFrame:
        """Fetch candle data from Angel One when credentials are available."""
        try:
            client = self._build_session()
            if hasattr(client, "getCandleData"):
                response = client.getCandleData(
                    symbol=self.symbol,
                    interval=self.interval,
                    fromdate=(datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d %H:%M:%S"),
                    todate=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                )
                if isinstance(response, list):
                    return pd.DataFrame(response)
        except Exception as exc:
            log_error(f"Historical API fetch failed for {self.symbol}", exc)
        return self._synthetic_candles()

    def refresh(self) -> pd.DataFrame:
        """Refresh the historical DataFrame and return the latest dataset."""
        try:
            data = self._fetch_from_api()
            if data.empty:
                data = self._synthetic_candles()
            if "timestamp" in data.columns:
                data["timestamp"] = pd.to_datetime(data["timestamp"])
            for column in self.CANDLE_COLUMNS:
                if column not in data.columns:
                    data[column] = "live" if column == "data_source" else 0
            self.df = data[self.CANDLE_COLUMNS].copy()
            self.last_refresh = datetime.now(timezone.utc)
            return self.df
        except Exception as exc:
            log_error(f"Failed to refresh historical data for {self.symbol}", exc)
            self.df = self._synthetic_candles()[self.CANDLE_COLUMNS].copy()
            return self.df

    def start_market_refresh_loop(self, sleep_seconds: int = 60, market_open_hour: int = 9, market_close_hour: int = 16) -> None:
        """Auto-refresh candles during active market hours. Safe against runtime errors."""
        while True:
            try:
                now = datetime.now(timezone.utc)
                current_hour = now.hour
                if market_open_hour <= current_hour < market_close_hour:
                    self.refresh()
                time.sleep(sleep_seconds)
            except Exception as exc:
                log_error("Historical refresh loop crashed unexpectedly", exc)
                time.sleep(sleep_seconds)


class HistoricalData:
    """Compatibility wrapper used by startup initialization."""

    def __init__(self, api=None, symbol: Optional[str] = None, timeframe: Optional[int] = None, limit: int = 200):
        self.api = api
        self.symbol = symbol or os.getenv("STOCK_SYMBOL", getattr(settings, "stock_symbol", "NIFTY"))
        self.timeframe = int(timeframe if timeframe is not None else os.getenv("TIMEFRAME", getattr(settings, "timeframe", 5)))
        self.limit = limit
        self.manager = HistoricalDataManager(symbol=self.symbol, interval=f"{self.timeframe}minute", limit=self.limit)

    def fetch_last_200_candles(self):
        df = self.manager.refresh()
        if isinstance(df, pd.DataFrame):
            return df.tail(self.limit).to_dict(orient="records")
        return []


def fetch_historical_data(symbol: str, timeframe: str = "5m", limit: int = 200) -> pd.DataFrame:
    manager = HistoricalDataManager(symbol=symbol, interval=timeframe, limit=limit)
    return manager.refresh()


if __name__ == "__main__":
    df = fetch_historical_data(symbol=settings.stock_symbol, timeframe="5minute", limit=200)
    print(df.head())
