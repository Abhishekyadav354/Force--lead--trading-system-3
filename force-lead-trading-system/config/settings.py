"""Centralized, import-safe application configuration."""

import os
import re
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class Settings:
    MIN_LEAD_PROGRESS = 0.70  # Minimum progress toward a lead crossover.
    MIN_SPREAD_QUALITY = 0.40  # Minimum acceptable market spread quality.
    MIN_VOLUME_SPEED_RATIO = 1.5  # Minimum volume speed relative to its baseline.
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    NO_TRADE_BEFORE = "09:30"
    NO_TRADE_AFTER = "15:00"
    BEST_TRADING_START = "09:30"
    BEST_TRADING_END = "11:30"
    AFTERNOON_START = "13:00"
    AFTERNOON_END = "15:00"
    ML_RETRAIN_DAYS = 7
    ML_MIN_TRAINING_SAMPLES = 100
    UPDATE_INTERVAL_SECONDS = 5
    RECONNECT_MAX_ATTEMPTS = 5
    LOG_LEVEL = "INFO"
    DASHBOARD_PORT = 5000
    DASHBOARD_HOST = "0.0.0.0"
    BACKTEST_TRAIN_DAYS = 60
    BACKTEST_TEST_DAYS = 10
    REGIME_WEIGHTS = {
        "TRENDING": {
            "f1": 0.30,
            "f2": 0.15,
            "f3": 0.08,
            "f4": 0.25,
            "f5": 0.10,
            "f6": 0.05,
            "f7": 0.04,
            "f8": 0.02,
            "f9": 0.01,
        },
        "MEAN_REVERTING": {
            "f1": 0.15,
            "f2": 0.30,
            "f3": 0.15,
            "f4": 0.10,
            "f5": 0.10,
            "f6": 0.08,
            "f7": 0.06,
            "f8": 0.04,
            "f9": 0.02,
        },
        "RANDOM_WALK": {
            "f1": 0.20,
            "f2": 0.20,
            "f3": 0.10,
            "f4": 0.20,
            "f5": 0.10,
            "f6": 0.08,
            "f7": 0.06,
            "f8": 0.04,
            "f9": 0.02,
        },
    }

    def __init__(self):
        self.ANGEL_API_KEY = os.getenv("ANGEL_API_KEY", "")
        self.ANGEL_CLIENT_ID = os.getenv("ANGEL_CLIENT_ID", "")
        self.ANGEL_PASSWORD = os.getenv("ANGEL_PASSWORD", "")
        self.ANGEL_TOTP_SECRET = os.getenv("ANGEL_TOTP_SECRET", "")
        self.TRADING_MODE = os.getenv("TRADING_MODE", "paper").strip().lower() or "paper"
        self.STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "NIFTY")
        self.TIMEFRAME_MINUTES = _env_int("TIMEFRAME", 5)
        self.CAPITAL = _env_float("CAPITAL", 10000.0)

        self.angel_api_key = self.ANGEL_API_KEY
        self.angel_client_id = self.ANGEL_CLIENT_ID
        self.angel_password = self.ANGEL_PASSWORD
        self.angel_totp_secret = self.ANGEL_TOTP_SECRET
        self.trading_mode = self.TRADING_MODE
        self.stock_symbol = self.STOCK_SYMBOL
        self.timeframe = self.TIMEFRAME_MINUTES
        self.capital = self.CAPITAL

        self.FORCE_SCORE_THRESHOLD = 0.40  # Minimum force score for a signal.
        self.SIGNIFICANCE_THRESHOLD = 0.05  # Maximum p-value for statistical significance.
        self.MIN_WIN_PROBABILITY = 0.55  # Minimum estimated probability of a winning trade.
        self.MIN_ML_CONFIDENCE = 0.58  # Minimum confidence required from the ML model.
        self.MIN_MONTE_CARLO_WIN = 0.55  # Minimum Monte Carlo win probability.
        self.MAX_POSITION_PCT = 0.05  # Maximum portfolio percentage for one position.
        self.MIN_POSITION_PCT = 0.01  # Minimum portfolio percentage for one position.
        self.STOP_LOSS_PCT = 0.01  # Maximum loss percentage before exiting a position.
        self.TARGET_PCT = 0.02  # Target profit percentage for a position.
        self.MAX_DAILY_LOSS_PCT = 0.05  # Maximum allowed daily portfolio loss percentage.
        self.MAX_TRADES_PER_DAY = 5  # Maximum number of trades allowed per day.

    @property
    def angel_api_missing_fields(self):
        """Return missing credential field names without exposing secret values."""
        return [
            name
            for name, value in (
                ("ANGEL_API_KEY", self.ANGEL_API_KEY),
                ("ANGEL_CLIENT_ID", self.ANGEL_CLIENT_ID),
                ("ANGEL_PASSWORD", self.ANGEL_PASSWORD),
                ("ANGEL_TOTP_SECRET", self.ANGEL_TOTP_SECRET),
            )
            if not str(value or "").strip()
        ]

    @property
    def angel_api_configured(self):
        """True only when every Angel One credential is present."""
        return not self.angel_api_missing_fields

    @property
    def angel_api_status(self):
        """Safe status suitable for diagnostics; never includes credential values."""
        missing = self.angel_api_missing_fields
        return {
            "configured": not missing,
            "missing_fields": missing,
            "message": "Angel One API configured."
            if not missing
            else f"Angel One API credentials missing: {', '.join(missing)}.",
            "trading_mode": self.TRADING_MODE,
            "paper_mode": self.TRADING_MODE == "paper",
        }

    def angel_api_config_message(self):
        """Return a concise setup message without exposing secrets."""
        return self.angel_api_status["message"]

    def validate(self):
        if self.TRADING_MODE not in {"paper", "live", "backtest", "test"}:
            raise ValueError("TRADING_MODE must be one of: paper, live, backtest, test")

        credential_patterns = {
            "ANGEL_API_KEY": r"[A-Za-z0-9]{8,128}",
            "ANGEL_CLIENT_ID": r"[A-Za-z0-9]{8,32}",
            "ANGEL_PASSWORD": r"\S+",
            "ANGEL_TOTP_SECRET": r"[A-Z2-7]{16,64}",
        }
        for name, pattern in credential_patterns.items():
            value = getattr(self, name)
            if not re.fullmatch(pattern, str(value or "").strip()):
                raise ValueError(f"{name} is missing or has an invalid format")

        if self.CAPITAL <= 0:
            raise ValueError("CAPITAL must be greater than 0")
        if self.TIMEFRAME_MINUTES <= 0:
            raise ValueError("TIMEFRAME_MINUTES must be greater than 0")

        risk_percentages = {
            "MAX_POSITION_PCT": self.MAX_POSITION_PCT,
            "MIN_POSITION_PCT": self.MIN_POSITION_PCT,
            "STOP_LOSS_PCT": self.STOP_LOSS_PCT,
            "TARGET_PCT": self.TARGET_PCT,
            "MAX_DAILY_LOSS_PCT": self.MAX_DAILY_LOSS_PCT,
        }
        for name, value in risk_percentages.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        signal_probabilities = {
            "MIN_WIN_PROBABILITY": self.MIN_WIN_PROBABILITY,
            "MIN_ML_CONFIDENCE": self.MIN_ML_CONFIDENCE,
            "MIN_MONTE_CARLO_WIN": self.MIN_MONTE_CARLO_WIN,
        }
        for name, value in signal_probabilities.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")

        if self.MAX_TRADES_PER_DAY < 1:
            raise ValueError("MAX_TRADES_PER_DAY must be at least 1")
        if not 1 <= self.DASHBOARD_PORT <= 65535:
            raise ValueError("DASHBOARD_PORT must be between 1 and 65535")


settings = Settings()
settings.validate()
