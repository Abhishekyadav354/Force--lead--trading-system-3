from __future__ import annotations

import json
import logging
import os
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from threading import Event, Lock
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from config.settings import settings

try:
    import pyotp
    from smartapi import SmartConnect
    from smartapi.smartWebSocketV2 import SmartWebSocketV2
except Exception:  # pragma: no cover - optional dependency handling
    pyotp = None
    SmartConnect = None
    SmartWebSocketV2 = None


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "errors.log")

logger = logging.getLogger("angel_live_feed")
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


class AngelLiveFeed:
    """Live market-data collector for Angel One SmartAPI with candle aggregation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        client_id: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        self.api_key = api_key or settings.angel_api_key
        self.client_id = client_id or settings.angel_client_id
        self.password = password or settings.angel_password
        self.totp_secret = totp_secret or settings.angel_totp_secret
        self.symbol = symbol or settings.stock_symbol

        self.api_client = None
        self.ws = None
        self.connected = False
        self.lock = Lock()
        self.last_tick_time: Optional[datetime] = None
        self.last_disconnect_alert = None
        self.reconnect_attempts = 0
        self.stop_event = Event()

        self.tick_history: deque = deque(maxlen=50)
        self.candles: Dict[str, deque] = {
            "1min": deque(maxlen=200),
            "5min": deque(maxlen=200),
            "10min": deque(maxlen=200),
        }

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _utc_now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _parse_timestamp(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc)
        if isinstance(value, str):
            try:
                candidate = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if candidate.tzinfo is None:
                    return candidate.replace(tzinfo=timezone.utc)
                return candidate.astimezone(timezone.utc)
            except ValueError:
                return self._utc_now()
        return self._utc_now()

    def _extract_levels(self, depth_data: Any) -> List[List[float]]:
        result: List[List[float]] = []
        try:
            if isinstance(depth_data, dict):
                depth_data = depth_data.get("bids", []) if "bids" in depth_data else depth_data.get("asks", [])
            for item in list(depth_data)[:5]:
                if isinstance(item, dict):
                    price = self._safe_float(item.get("price", item.get("p", 0.0)))
                    qty = self._safe_float(item.get("quantity", item.get("q", 0.0)))
                    result.append([price, qty])
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    result.append([self._safe_float(item[0]), self._safe_float(item[1])])
        except Exception:
            pass
        return result

    def _ensure_session(self):
        if SmartConnect is None:
            raise RuntimeError("smartapi-python is not installed.")
        if not all([self.api_key, self.client_id, self.password, self.totp_secret]):
            raise ValueError("Angel One credentials are missing from environment variables.")

        if self.api_client is None:
            self.api_client = SmartConnect(api_key=self.api_key)
            otp = pyotp.TOTP(self.totp_secret).now()
            self.api_client.generateSession(self.client_id, self.password, otp)
        return self.api_client

    def connect(self) -> Dict[str, Any]:
        """Create a SmartAPI session and initialize the websocket client."""
        try:
            self._ensure_session()
            if SmartWebSocketV2 is not None:
                try:
                    self.ws = SmartWebSocketV2(self.api_client.token, self.api_key, self.client_id)
                except TypeError:
                    self.ws = SmartWebSocketV2(self.api_key, self.client_id)
            self.connected = True
            self.reconnect_attempts = 0
            return {"status": "connected", "symbol": self.symbol, "client_id": self.client_id}
        except Exception as exc:
            self.connected = False
            log_error(f"Angel One connect failed for {self.symbol}", exc)
            return {"status": "failed", "symbol": self.symbol, "error": str(exc)}

    def subscribe(self) -> Dict[str, Any]:
        """Subscribe to the selected symbol stream. Safe fallback if websocket is unavailable."""
        try:
            if self.ws is not None and hasattr(self.ws, "subscribe"):
                self.ws.subscribe(self.symbol)
            return {"status": "subscribed", "symbol": self.symbol}
        except Exception as exc:
            log_error(f"Subscription failed for {self.symbol}", exc)
            return {"status": "subscribed_fallback", "symbol": self.symbol, "error": str(exc)}

    def _normalize_tick(self, raw_tick: Dict[str, Any]) -> Dict[str, Any]:
        raw_tick = raw_tick or {}
        depth = raw_tick.get("depth", {})
        bid_levels = self._extract_levels(depth.get("bids") or raw_tick.get("bids") or [])
        ask_levels = self._extract_levels(depth.get("asks") or raw_tick.get("asks") or [])

        ltp = self._safe_float(raw_tick.get("ltp", raw_tick.get("last_price", raw_tick.get("lastTradedPrice", 0.0))) )
        volume = self._safe_int(raw_tick.get("volume", raw_tick.get("total_volume", raw_tick.get("totalVolume", 0))))
        trade_count = self._safe_int(raw_tick.get("trade_count", raw_tick.get("trades", raw_tick.get("tradeCount", 0))))
        open_price = self._safe_float(raw_tick.get("open", raw_tick.get("open_price", 0.0)))
        high_price = self._safe_float(raw_tick.get("high", 0.0))
        low_price = self._safe_float(raw_tick.get("low", 0.0))
        close_price = self._safe_float(raw_tick.get("close", raw_tick.get("prev_close", ltp)))
        buy_volume = self._safe_int(raw_tick.get("buy_volume", raw_tick.get("buyer_initiated_volume", 0)))
        sell_volume = self._safe_int(raw_tick.get("sell_volume", raw_tick.get("seller_initiated_volume", 0)))
        bid_total = sum(qty for _, qty in bid_levels)
        ask_total = sum(qty for _, qty in ask_levels)

        best_bid = bid_levels[0][0] if bid_levels else 0.0
        best_ask = ask_levels[0][0] if ask_levels else 0.0
        spread = best_ask - best_bid if best_ask and best_bid else 0.0

        buyers_pct = (buy_volume / (buy_volume + sell_volume)) * 100 if (buy_volume + sell_volume) else 0.0
        sellers_pct = (sell_volume / (buy_volume + sell_volume)) * 100 if (buy_volume + sell_volume) else 0.0

        timestamp = self._parse_timestamp(raw_tick.get("timestamp", raw_tick.get("last_updated_time", self._utc_now())))
        avg_trade_size = (volume / trade_count) if trade_count else 0.0
        historical_avg = max(self._rolling_avg_trade_size(), 1.0)
        avg_spread_20 = max(self._avg_spread_20(), 1.0)
        elapsed_minutes = self._elapsed_minutes_since_last_tick(timestamp)
        volume_per_minute = (volume / elapsed_minutes) if elapsed_minutes > 0 else float(volume)
        institutional_score = (avg_trade_size / historical_avg) if historical_avg > 0 else 0.0
        spread_ratio = (spread / avg_spread_20) if avg_spread_20 > 0 else 0.0

        tick = {
            "symbol": self.symbol,
            "timestamp": timestamp,
            "ltp": ltp,
            "volume": volume,
            "open": open_price or ltp,
            "high": max(high_price, ltp),
            "low": min(low_price, ltp) if low_price else ltp,
            "close": close_price or ltp,
            "trade_count": trade_count,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "avg_trade_size": avg_trade_size,
            "bid_total": bid_total,
            "ask_total": ask_total,
            "bid_levels": bid_levels[:5],
            "ask_levels": ask_levels[:5],
            "buyers_pct": buyers_pct,
            "sellers_pct": sellers_pct,
            "spread": spread,
            "volume_per_minute": volume_per_minute,
            "institutional_score": institutional_score,
            "spread_ratio": spread_ratio,
        }
        return tick

    def _elapsed_minutes_since_last_tick(self, current_ts: datetime) -> float:
        if self.last_tick_time is None:
            return 1.0
        delta = current_ts - self.last_tick_time
        return max(delta.total_seconds() / 60.0, 1.0)

    def _rolling_avg_trade_size(self) -> float:
        if not self.tick_history:
            return 1.0
        values = [tick.get("avg_trade_size", 0.0) for tick in self.tick_history]
        return float(sum(values) / len(values))

    def _avg_spread_20(self) -> float:
        spreads = [tick.get("spread", 0.0) for tick in self.tick_history]
        if not spreads:
            return 1.0
        return float(sum(spreads[-20:]) / min(len(spreads[-20:]), 20))

    def _bucket_timestamp(self, ts: datetime, timeframe: str) -> datetime:
        if timeframe == "1min":
            return ts.replace(second=0, microsecond=0)
        if timeframe == "5min":
            minute = (ts.minute // 5) * 5
            return ts.replace(minute=minute, second=0, microsecond=0)
        if timeframe == "10min":
            minute = (ts.minute // 10) * 10
            return ts.replace(minute=minute, second=0, microsecond=0)
        return ts.replace(second=0, microsecond=0)

    def _update_candles(self, tick: Dict[str, Any]) -> None:
        timestamp = tick["timestamp"]
        for timeframe in ("1min", "5min", "10min"):
            bucket = self._bucket_timestamp(timestamp, timeframe)
            candle = None
            for existing in self.candles[timeframe]:
                if existing["timestamp"] == bucket:
                    candle = existing
                    break
            if candle is None:
                candle = {
                    "symbol": self.symbol,
                    "timestamp": bucket,
                    "open": tick["ltp"],
                    "high": tick["ltp"],
                    "low": tick["ltp"],
                    "close": tick["ltp"],
                    "volume": tick["volume"],
                    "buy_volume": tick["buy_volume"],
                    "sell_volume": tick["sell_volume"],
                    "trade_count": tick["trade_count"],
                    "avg_trade_size": tick["avg_trade_size"],
                    "bid_total": tick["bid_total"],
                    "ask_total": tick["ask_total"],
                    "bid_levels": tick["bid_levels"],
                    "ask_levels": tick["ask_levels"],
                    "buyers_pct": tick["buyers_pct"],
                    "sellers_pct": tick["sellers_pct"],
                    "spread": tick["spread"],
                    "volume_per_minute": tick["volume_per_minute"],
                    "institutional_score": tick["institutional_score"],
                    "spread_ratio": tick["spread_ratio"],
                }
                self.candles[timeframe].append(candle)
                continue

            candle["high"] = max(candle["high"], tick["ltp"])
            candle["low"] = min(candle["low"], tick["ltp"])
            candle["close"] = tick["ltp"]
            candle["volume"] += tick["volume"]
            candle["buy_volume"] += tick["buy_volume"]
            candle["sell_volume"] += tick["sell_volume"]
            candle["trade_count"] += tick["trade_count"]
            candle["avg_trade_size"] = candle["volume"] / max(candle["trade_count"], 1)
            candle["bid_total"] = tick["bid_total"]
            candle["ask_total"] = tick["ask_total"]
            candle["bid_levels"] = tick["bid_levels"]
            candle["ask_levels"] = tick["ask_levels"]
            candle["buyers_pct"] = tick["buyers_pct"]
            candle["sellers_pct"] = tick["sellers_pct"]
            candle["spread"] = tick["spread"]
            candle["volume_per_minute"] = candle["volume"] / max((self._elapsed_minutes_since_last_tick(timestamp)), 1.0)
            candle["institutional_score"] = tick["institutional_score"]
            candle["spread_ratio"] = tick["spread_ratio"]

    def _check_disconnect(self) -> None:
        if self.last_tick_time is None:
            return
        disconnected_for = (self._utc_now() - self.last_tick_time).total_seconds()
        if disconnected_for > 30:
            if self.last_disconnect_alert is None or (self._utc_now() - self.last_disconnect_alert).total_seconds() > 30:
                self.last_disconnect_alert = self._utc_now()
                log_error(f"Live feed disconnected for more than 30 seconds for {self.symbol}.")

    def process_tick(self, raw_tick: Dict[str, Any]) -> Dict[str, Any]:
        """Accept a raw tick payload and store the normalized tick + candle aggregates."""
        try:
            tick = self._normalize_tick(raw_tick)
            with self.lock:
                self.tick_history.append(tick)
                self._update_candles(tick)
            self.last_tick_time = tick["timestamp"]
            self._check_disconnect()
            return tick
        except Exception as exc:
            log_error("Error while processing live tick", exc)
            return {"status": "error", "symbol": self.symbol, "message": str(exc)}

    def get_recent_ticks(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.tick_history)[-limit:]

    def get_candles(self, timeframe: str = "1min", limit: int = 20) -> List[Dict[str, Any]]:
        data = list(self.candles.get(timeframe, deque()))
        return data[-limit:]

    def reconnect(self, max_attempts: int = 5) -> bool:
        """Reconnect with exponential backoff and graceful error handling."""
        attempt = 0
        while attempt < max_attempts:
            try:
                time.sleep(2 ** attempt)
                result = self.connect()
                if result.get("status") == "connected":
                    self.subscribe()
                    return True
            except Exception as exc:
                log_error(f"Reconnect attempt {attempt + 1} failed for {self.symbol}", exc)
            attempt += 1
        log_error(f"Max reconnection attempts reached for {self.symbol}.")
        return False

    def start_stream(self, callback=None) -> bool:
        """Run the stream loop without crashing on data errors."""
        self.connected = False
        try:
            self.connect()
            self.subscribe()
        except Exception as exc:
            log_error("Streaming startup failed", exc)
            return False

        while not self.stop_event.is_set():
            try:
                if callback is not None:
                    callback(self.get_recent_ticks())
                time.sleep(1)
            except Exception as exc:
                log_error("Stream loop failed", exc)
                if not self.reconnect(max_attempts=5):
                    break
        return True

    def stop(self):
        self.stop_event.set()
        self.connected = False


LiveDataFeed = AngelLiveFeed


if __name__ == "__main__":
    feed = AngelLiveFeed(symbol=settings.stock_symbol)
    sample_tick = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ltp": 24650.0,
        "volume": 1200,
        "trade_count": 30,
        "open": 24610.0,
        "high": 24680.0,
        "low": 24590.0,
        "close": 24650.0,
        "depth": {
            "bids": [{"price": 24650.0, "quantity": 100}, {"price": 24648.0, "quantity": 80}, {"price": 24645.0, "quantity": 70}],
            "asks": [{"price": 24651.0, "quantity": 90}, {"price": 24654.0, "quantity": 75}, {"price": 24657.0, "quantity": 60}],
        },
        "buy_volume": 700,
        "sell_volume": 500,
    }
    result = feed.process_tick(sample_tick)
    print(json.dumps(result, default=str, indent=2))
    print(feed.get_candles("1min"))
