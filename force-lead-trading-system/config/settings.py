"""Centralized, import-safe application configuration."""

import os

from dotenv import load_dotenv


load_dotenv()


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
        self.STOCK_SYMBOL = os.getenv("STOCK_SYMBOL", "NIFTY")
        self.TIMEFRAME_MINUTES = _env_int("TIMEFRAME", 5)
        self.CAPITAL = _env_float("CAPITAL", 10000.0)

        self.angel_api_key = self.ANGEL_API_KEY
        self.angel_client_id = self.ANGEL_CLIENT_ID
        self.angel_password = self.ANGEL_PASSWORD
        self.angel_totp_secret = self.ANGEL_TOTP_SECRET
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

    def validate(self):
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
