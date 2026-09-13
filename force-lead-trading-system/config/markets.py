"""Centralized market and trading-segment configuration.

This module is configuration-only. It does not load broker credentials, contact
external services, or place orders. Consumers receive copies of configurations
so a dashboard, backtest, or prediction engine cannot mutate shared defaults.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


_COMMON_TIMEFRAMES = ["1m", "5m", "15m", "1h", "1D"]


def _segment(
    name: str,
    description: str,
    exchanges: list[str],
    instruments: list[str],
    best_timeframe: str,
    stop_loss_pct: float | None,
    target_pct: float | None,
    leverage: str,
    rules: list[str],
    auto_square_off: str | None = None,
    timeframes: list[str] | None = None,
    adjustments: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build one consistently shaped market segment definition."""
    sl_target = {
        "stop_loss_pct": stop_loss_pct,
        "target_pct": target_pct,
    }
    return {
        "name": name,
        "description": description,
        "exchanges": list(exchanges),
        "instruments": list(instruments),
        "timeframes": list(timeframes or _COMMON_TIMEFRAMES),
        "best_timeframe": best_timeframe,
        "sl_target": sl_target,
        "stop_loss_pct": stop_loss_pct,
        "target_pct": target_pct,
        "leverage": leverage,
        "leverage_margin": leverage,
        "margin": leverage,
        "auto_square_off": auto_square_off,
        "rules": list(rules),
        "adjustments": dict(adjustments or {}),
    }


MARKETS: dict[str, dict[str, Any]] = {
    "INDIA": {
        "name": "India",
        "market_name": "India",
        "flag": "IN",
        "timezone": "Asia/Kolkata",
        "currency": "INR",
        "symbol": "₹",
        "market_open": "09:15",
        "market_close": "15:30",
        "india_time": {"market_open": "09:15", "market_close": "15:30", "timezone": "Asia/Kolkata"},
        "india_time_conversion": {"market_open": "09:15", "market_close": "15:30", "timezone": "Asia/Kolkata"},
        "segments": {
            "INTRADAY": _segment(
                "Intraday",
                "Same-session equity trading in the Indian cash market.",
                ["NSE", "BSE"],
                ["equities", "ETFs", "indices"],
                "5m",
                0.01,
                0.02,
                "Broker-defined intraday margin; verify before use",
                ["No overnight carry.", "Respect exchange and broker risk limits.", "Paper/simulation mode does not place orders."],
                "15:15 IST (broker/exchange policy applies)",
                ["1m", "5m", "15m", "30m", "1h"],
                {"force_threshold": 0.40, "lead_time_max": 30.0, "volume_importance": 1.0, "oi_importance": 0.20, "greeks_importance": 0.10, "volatility_adjustment": 1.0},
            ),
            "DELIVERY": _segment(
                "Delivery",
                "Indian cash-equity positions intended to be carried beyond one session.",
                ["NSE", "BSE"],
                ["equities", "ETFs"],
                "1D",
                0.03,
                0.06,
                "No intraday leverage; broker margin and settlement rules apply",
                ["Overnight and settlement risk applies.", "Check corporate actions and circuit limits.", "Broker holdings and margin rules apply."],
                None,
                ["15m", "1h", "1D"],
                {"force_threshold": 0.45, "lead_time_max": 240.0, "volume_importance": 0.90, "oi_importance": 0.10, "greeks_importance": 0.05, "volatility_adjustment": 1.10},
            ),
            "FUTURES": _segment(
                "Futures",
                "Exchange-traded Indian index or stock futures.",
                ["NSE", "BSE"],
                ["index futures", "stock futures"],
                "15m",
                0.015,
                0.03,
                "Contract and broker margin; leverage is variable",
                ["Contract expiry and rollover matter.", "Initial and maintenance margin are exchange/broker defined.", "Do not assume leverage is constant."],
                "15:15 IST for intraday positions; contract policy applies",
                ["5m", "15m", "1h", "1D"],
                {"force_threshold": 0.45, "lead_time_max": 60.0, "volume_importance": 0.90, "oi_importance": 1.0, "greeks_importance": 0.10, "volatility_adjustment": 1.15},
            ),
            "OPTIONS_BUY": _segment(
                "Options Buy",
                "Long calls or puts where the premium is paid upfront.",
                ["NSE", "BSE"],
                ["index call/put", "stock call/put"],
                "5m",
                0.30,
                0.60,
                "Premium paid; position risk is limited to premium before costs",
                ["Premium decay and implied volatility affect value.", "Expiry, strike, liquidity and spreads must be checked.", "Loss can reach the premium paid."],
                "15:15 IST for intraday positions; expiry rules apply",
                ["1m", "5m", "15m", "1h"],
                {"force_threshold": 0.50, "lead_time_max": 45.0, "volume_importance": 0.80, "oi_importance": 0.80, "greeks_importance": 1.0, "volatility_adjustment": 1.30},
            ),
            "OPTIONS_SELL": _segment(
                "Options Sell",
                "Short calls or puts subject to margin and assignment/expiry risk.",
                ["NSE", "BSE"],
                ["index call/put", "stock call/put"],
                "15m",
                0.10,
                0.20,
                "Exchange and broker SPAN/exposure margin required",
                ["Short options can carry substantial or theoretically unlimited risk.", "Maintain required margin and monitor volatility.", "Expiry and settlement rules apply."],
                "15:15 IST for intraday positions; expiry rules apply",
                ["5m", "15m", "1h", "1D"],
                {"force_threshold": 0.55, "lead_time_max": 60.0, "volume_importance": 0.80, "oi_importance": 1.0, "greeks_importance": 1.20, "volatility_adjustment": 1.40},
            ),
            "COMMODITY": _segment(
                "Commodity",
                "Indian exchange-traded commodity futures and options.",
                ["MCX", "NCDEX"],
                ["metals", "energy", "agricultural commodities"],
                "15m",
                0.02,
                0.04,
                "Contract-specific exchange and broker margin",
                ["Trading hours vary by commodity and exchange.", "Contract expiry, delivery and tick size rules apply.", "Check the contract specification before analysis."],
                "Contract/session specific; broker policy applies",
                ["5m", "15m", "1h", "1D"],
                {"force_threshold": 0.50, "lead_time_max": 90.0, "volume_importance": 1.0, "oi_importance": 0.90, "greeks_importance": 0.30, "volatility_adjustment": 1.25},
            ),
            "CURRENCY": _segment(
                "Currency",
                "Indian exchange-traded currency derivatives.",
                ["NSE", "BSE", "MSEI"],
                ["USDINR", "EURINR", "GBPINR", "JPYINR"],
                "15m",
                0.01,
                0.02,
                "Contract-specific exchange and broker margin",
                ["Contract size, expiry and quote convention matter.", "Currency market hours and holidays differ from equities.", "Check exchange contract specifications."],
                "Session/contract specific; broker policy applies",
                ["5m", "15m", "1h", "1D"],
                {"force_threshold": 0.45, "lead_time_max": 90.0, "volume_importance": 0.80, "oi_importance": 0.90, "greeks_importance": 0.20, "volatility_adjustment": 1.10},
            ),
        },
    },
    "US": {
        "name": "United States",
        "market_name": "United States",
        "flag": "US",
        "timezone": "America/New_York",
        "currency": "USD",
        "symbol": "$",
        "market_open": "09:30",
        "market_close": "16:00",
        "india_time": {"market_open": "19:00 DST / 20:00 standard", "market_close": "01:30 DST / 02:30 standard", "timezone": "Asia/Kolkata"},
        "india_time_conversion": {"market_open": "19:00 DST / 20:00 standard", "market_close": "01:30 DST / 02:30 standard", "timezone": "Asia/Kolkata"},
        "segments": {
            "STOCKS": _segment(
                "Stocks",
                "US-listed equities and exchange-traded funds.",
                ["NYSE", "NASDAQ", "AMEX"],
                ["equities", "ETFs", "ADR"],
                "5m",
                0.01,
                0.02,
                "Regulation-T or broker-defined margin",
                ["Regular session is 09:30–16:00 Eastern Time.", "Pre-market and after-hours liquidity differs.", "Short-sale and locate rules may apply."],
                None,
                ["1m", "5m", "15m", "1h", "1D"],
                {"force_threshold": 0.45, "lead_time_max": 60.0, "volume_importance": 1.0, "oi_importance": 0.10, "greeks_importance": 0.10, "volatility_adjustment": 1.10},
            ),
            "OPTIONS": _segment(
                "Options",
                "US-listed calls and puts with standardized contracts.",
                ["CBOE", "NYSE", "NASDAQ"],
                ["equity options", "index options", "ETF options"],
                "15m",
                0.20,
                0.40,
                "Option and account approval margin; broker-defined",
                ["Expiration, multiplier, implied volatility and Greeks matter.", "Assignment and exercise rules apply.", "Liquidity and spread must be checked."],
                None,
                ["5m", "15m", "1h", "1D"],
                {"force_threshold": 0.50, "lead_time_max": 90.0, "volume_importance": 0.80, "oi_importance": 0.90, "greeks_importance": 1.0, "volatility_adjustment": 1.30},
            ),
            "CRYPTO": _segment(
                "Crypto",
                "Continuous crypto spot or derivative market monitoring.",
                ["Exchange-specific"],
                ["BTC", "ETH", "crypto spot", "crypto derivatives"],
                "15m",
                0.03,
                0.06,
                "Venue and product specific; high leverage risk",
                ["Crypto trades continuously, including weekends.", "Venue, custody, funding and liquidation rules vary.", "Volatility and liquidity can change abruptly."],
                None,
                ["1m", "5m", "15m", "1h", "1D"],
                {"force_threshold": 0.60, "lead_time_max": 120.0, "volume_importance": 1.10, "oi_importance": 0.80, "greeks_importance": 0.20, "volatility_adjustment": 1.50},
            ),
        },
    },
    "EUROPE": {
        "name": "Europe",
        "market_name": "Europe",
        "flag": "EU",
        "timezone": "Europe/London",
        "currency": "EUR",
        "symbol": "€",
        "market_open": "08:00",
        "market_close": "16:30",
        "india_time": {"market_open": "12:30 DST / 13:30 standard", "market_close": "21:00 DST / 22:00 standard", "timezone": "Asia/Kolkata"},
        "india_time_conversion": {"market_open": "12:30 DST / 13:30 standard", "market_close": "21:00 DST / 22:00 standard", "timezone": "Asia/Kolkata"},
        "segments": {
            "STOCKS": _segment(
                "Stocks",
                "European listed equities and exchange-traded funds.",
                ["Euronext", "LSE", "Xetra", "SIX"],
                ["equities", "ETFs", "indices"],
                "15m",
                0.015,
                0.03,
                "Broker-defined margin; market-specific",
                ["Trading hours and holidays vary by exchange.", "Currency, listing venue and local settlement rules matter.", "Check the local exchange session before use."],
                None,
                ["5m", "15m", "1h", "1D"],
                {"force_threshold": 0.50, "lead_time_max": 90.0, "volume_importance": 0.95, "oi_importance": 0.10, "greeks_importance": 0.10, "volatility_adjustment": 1.15},
            ),
        },
    },
}


def get_market_config(market: str, segment: str) -> dict[str, Any]:
    """Return a normalized, independent configuration for ``market``/``segment``.

    Args:
        market: One of ``INDIA``, ``US`` or ``EUROPE`` (case-insensitive).
        segment: A segment supported by that market (case-insensitive).

    Raises:
        ValueError: If either name is missing or unsupported.
    """
    if not isinstance(market, str) or not market.strip():
        raise ValueError("market must be a non-empty string")
    if not isinstance(segment, str) or not segment.strip():
        raise ValueError("segment must be a non-empty string")
    market_key = market.strip().upper()
    segment_key = segment.strip().upper()
    if market_key not in MARKETS:
        supported = ", ".join(sorted(MARKETS))
        raise ValueError(f"Unsupported market '{market}'; expected one of: {supported}")
    market_data = MARKETS[market_key]
    if segment_key not in market_data["segments"]:
        supported = ", ".join(sorted(market_data["segments"]))
        raise ValueError(f"Unsupported segment '{segment}' for {market_key}; expected one of: {supported}")

    segment_data = deepcopy(market_data["segments"][segment_key])
    config = {
        "market": market_key,
        "market_name": market_data["market_name"],
        "flag": market_data["flag"],
        "segment": segment_key,
        "segment_name": segment_data.pop("name"),
        "timezone": market_data["timezone"],
        "currency": market_data["currency"],
        "symbol": market_data["symbol"],
        "market_open": market_data["market_open"],
        "market_close": market_data["market_close"],
        "india_time": deepcopy(market_data["india_time"]),
        "india_time_conversion": deepcopy(market_data["india_time_conversion"]),
    }
    config.update(segment_data)
    config.setdefault("adjustments", {})
    config["adjustments"] = {
        "force_threshold": 0.40,
        "lead_time_max": 60.0,
        "volume_importance": 1.0,
        "oi_importance": 0.0,
        "greeks_importance": 0.0,
        "volatility_adjustment": 1.0,
        **config["adjustments"],
    }
    config["system_adjustments"] = deepcopy(config["adjustments"])
    config["sl_target"] = deepcopy(config.get("sl_target", {}))
    return config


def apply_market_adjustments(base_settings: Mapping[str, Any], market_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return settings adjusted for a market without mutating either input.

    Both uppercase application-style names and lowercase normalized names are
    supported. Missing adjustment values retain safe neutral defaults.
    """
    if not isinstance(base_settings, Mapping):
        raise ValueError("base_settings must be a mapping")
    if not isinstance(market_config, Mapping):
        raise ValueError("market_config must be a mapping")
    result = deepcopy(dict(base_settings))
    raw_adjustments = market_config.get("system_adjustments", market_config.get("adjustments", {}))
    adjustments = raw_adjustments if isinstance(raw_adjustments, Mapping) else {}
    defaults = {"force_threshold": 0.40, "lead_time_max": 60.0, "volume_importance": 1.0, "oi_importance": 0.0, "greeks_importance": 0.0, "volatility_adjustment": 1.0}
    merged = {**defaults, **adjustments}
    aliases = {
        "force_threshold": ("FORCE_SCORE_THRESHOLD", "force_threshold"),
        "lead_time_max": ("LEAD_TIME_MAX", "lead_time_max"),
        "volume_importance": ("VOLUME_IMPORTANCE", "volume_importance"),
        "oi_importance": ("OI_IMPORTANCE", "oi_importance"),
        "greeks_importance": ("GREEKS_IMPORTANCE", "greeks_importance"),
        "volatility_adjustment": ("VOLATILITY_ADJUSTMENT", "volatility_adjustment"),
    }
    for key, names in aliases.items():
        value = merged[key]
        for name in names:
            if name in result or name == names[0]:
                result[name] = value
    return result


def validate_markets() -> None:
    """Perform a lightweight import-time-independent configuration self-check."""
    for market, market_data in MARKETS.items():
        if not market_data.get("segments"):
            raise ValueError(f"Market {market} has no segments")
        for segment in market_data["segments"]:
            normalized = get_market_config(market, segment)
            required = {"market", "market_name", "flag", "segment", "segment_name", "timezone", "currency", "symbol", "market_open", "market_close", "best_timeframe", "instruments", "sl_target", "leverage", "rules", "adjustments", "auto_square_off"}
            missing = required - set(normalized)
            if missing:
                raise ValueError(f"{market}/{segment} is missing fields: {sorted(missing)}")


if __name__ == "__main__":
    validate_markets()
