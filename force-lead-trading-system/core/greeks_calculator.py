from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


class GreeksCalculator:
    """Options sentiment and risk calculator for Force-Lead trading signals."""

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _option_type(option: Dict[str, Any]) -> str:
        return str(option.get("type", "")).upper()

    @staticmethod
    def _strike_distance(current_price: float, strike: float) -> float:
        return abs(current_price - strike)

    @staticmethod
    def _calculate_delta_for_price(option: Dict[str, Any], current_price: float) -> float:
        """Approximate delta via finite-difference around the current price."""
        if not option:
            return 0.0

        option_type = GreeksCalculator._option_type(option)
        dS = max(current_price * 0.01, 1e-6)
        low_price = max(current_price - dS, 1e-6)
        high_price = current_price + dS

        def price_value(price: float, typ: str) -> float:
            if typ == "CALL":
                return max(price - option.get("strike", price), 0.0)
            if typ == "PUT":
                return max(option.get("strike", price) - price, 0.0)
            return 0.0

        if option.get("last_price") is None:
            return 0.0
        low_value = price_value(low_price, option_type)
        high_value = price_value(high_price, option_type)
        return float((high_value - low_value) / (2.0 * dS))

    @staticmethod
    def _atm_option(options_chain: Sequence[Dict[str, Any]], current_price: float, option_type: str):
        matching = [opt for opt in options_chain if GreeksCalculator._option_type(opt) == option_type.upper()]
        if not matching:
            return None
        return min(matching, key=lambda opt: abs(GreeksCalculator._safe_float(opt.get("strike"), current_price) - current_price))

    @staticmethod
    def calculate_delta_from_chain(options_chain, current_price):
        """
        For each option in chain:
            delta = (V(S+dS) - V(S-dS)) / (2 * dS)
        """
        chain = list(options_chain or [])
        if not chain or current_price <= 0:
            return {
                "atm_call_delta": 0.0,
                "atm_put_delta": 0.0,
                "sentiment": "NEUTRAL",
                "force_score_addition": 0.0,
            }

        atm_call = GreeksCalculator._atm_option(chain, current_price, "CALL")
        atm_put = GreeksCalculator._atm_option(chain, current_price, "PUT")

        atm_call_delta = GreeksCalculator._calculate_delta_for_price(atm_call, current_price) if atm_call else 0.0
        atm_put_delta = GreeksCalculator._calculate_delta_for_price(atm_put, current_price) if atm_put else 0.0

        if atm_call_delta > 0.55:
            sentiment = "STRONGLY_BULLISH"
            force_addition = 0.3
        elif atm_call_delta > 0.45:
            sentiment = "NEUTRAL"
            force_addition = 0.0
        else:
            sentiment = "BEARISH"
            force_addition = -0.3

        return {
            "atm_call_delta": float(atm_call_delta),
            "atm_put_delta": float(atm_put_delta),
            "sentiment": sentiment,
            "force_score_addition": float(force_addition),
        }

    @staticmethod
    def calculate_gamma_exposure(options_chain, current_price):
        """
        Gamma = Rate of change of delta = ∂²V/∂S²
        """
        chain = list(options_chain or [])
        if not chain or current_price <= 0:
            return {"net_gamma": 0.0, "position_multiplier": 1.0, "warning": "No market data."}

        threshold = 0.5
        net_gamma = 0.0
        for option in chain:
            opt_type = GreeksCalculator._option_type(option)
            strike = GreeksCalculator._safe_float(option.get("strike"), current_price)
            oi = GreeksCalculator._safe_float(option.get("oi"), 0.0)
            if opt_type == "CALL":
                gamma = max(0.0, 1.0 / (1.0 + abs(strike - current_price)))
                net_gamma += gamma * oi
            elif opt_type == "PUT":
                gamma = max(0.0, 1.0 / (1.0 + abs(strike - current_price)))
                net_gamma -= gamma * oi

        if net_gamma > threshold:
            position_size_multiplier = 0.7
            warning = "High gamma, explosive moves possible"
        else:
            position_size_multiplier = 1.0
            warning = "Gamma within normal bounds"

        return {
            "net_gamma": float(net_gamma),
            "position_multiplier": float(position_size_multiplier),
            "warning": warning,
        }

    @staticmethod
    def _black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0.0) if option_type.upper() == "CALL" else max(K - S, 0.0)
        d1 = (math.log(S / K) + (r + sigma ** 2 / 2.0) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type.upper() == "CALL":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @staticmethod
    def _implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: str) -> float:
        def target(sigma: float) -> float:
            return GreeksCalculator._black_scholes_price(S, K, T, r, sigma, option_type) - market_price

        try:
            return float(brentq(target, 1e-6, 10.0))
        except ValueError:
            return 0.0

    @staticmethod
    def implied_volatility_analysis(options_chain):
        """
        IV Analysis - market's expectation of future volatility.
        """
        chain = list(options_chain or [])
        if not chain:
            return {"iv_skew": 0.0, "iv_percentile": 0.0, "implied_move": 0.0, "skew_signal": "NEUTRAL"}

        valid = [opt for opt in chain if GreeksCalculator._safe_float(opt.get("last_price"), 0.0) > 0.0]
        if not valid:
            return {"iv_skew": 0.0, "iv_percentile": 0.0, "implied_move": 0.0, "skew_signal": "NEUTRAL"}

        current_price = float(np.median([GreeksCalculator._safe_float(opt.get("strike"), 0.0) for opt in valid]))
        atm_call = GreeksCalculator._atm_option(valid, current_price, "CALL") or valid[0]
        atm_put = GreeksCalculator._atm_option(valid, current_price, "PUT") or valid[0]

        S = max(current_price, 1e-6)
        K_call = GreeksCalculator._safe_float(atm_call.get("strike"), S)
        K_put = GreeksCalculator._safe_float(atm_put.get("strike"), S)
        T = 0.05
        r = 0.0
        call_iv = GreeksCalculator._implied_volatility(GreeksCalculator._safe_float(atm_call.get("last_price"), 0.0), S, K_call, T, r, "CALL")
        put_iv = GreeksCalculator._implied_volatility(GreeksCalculator._safe_float(atm_put.get("last_price"), 0.0), S, K_put, T, r, "PUT")
        iv_skew = put_iv - call_iv

        if iv_skew > 0.1:
            skew_signal = "BEARISH"
        elif iv_skew < -0.1:
            skew_signal = "BULLISH"
        else:
            skew_signal = "NEUTRAL"

        iv_percentile = 50.0
        atm_straddle = ((GreeksCalculator._safe_float(atm_call.get("last_price"), 0.0) + GreeksCalculator._safe_float(atm_put.get("last_price"), 0.0)) / 2.0)

        return {
            "iv_skew": float(iv_skew),
            "iv_percentile": float(iv_percentile),
            "implied_move": float(atm_straddle),
            "skew_signal": skew_signal,
        }

    @staticmethod
    def max_pain_analysis(options_chain, expiry_date):
        """
        Max Pain = Price where maximum options expire worthless.
        """
        chain = list(options_chain or [])
        if not chain:
            return {"max_pain": 0.0, "distance_pct": 0.0, "days_to_expiry": 0, "pinning_likely": False}

        prices = sorted({GreeksCalculator._safe_float(opt.get("strike"), 0.0) for opt in chain})
        if not prices:
            return {"max_pain": 0.0, "distance_pct": 0.0, "days_to_expiry": 0, "pinning_likely": False}

        current_price = float(np.median(prices))
        try:
            expiry_dt = datetime.fromisoformat(str(expiry_date))
            days_to_expiry = max((expiry_dt - datetime.now()).days, 0)
        except Exception:
            days_to_expiry = 0

        pain_by_price = []
        for price in prices:
            total_pain = 0.0
            for option in chain:
                strike = GreeksCalculator._safe_float(option.get("strike"), 0.0)
                oi = GreeksCalculator._safe_float(option.get("oi"), 0.0)
                option_type = GreeksCalculator._option_type(option)
                if option_type == "CALL":
                    total_pain += max(0.0, strike - price) * oi
                elif option_type == "PUT":
                    total_pain += max(0.0, price - strike) * oi
            pain_by_price.append((price, total_pain))

        max_pain_price = min(pain_by_price, key=lambda item: item[1])[0]
        distance = (max_pain_price - current_price) / current_price * 100.0 if current_price else 0.0
        pinning_likely = abs(distance) < 1.0 and days_to_expiry < 3

        return {
            "max_pain": float(max_pain_price),
            "distance_pct": float(distance),
            "days_to_expiry": int(days_to_expiry),
            "pinning_likely": bool(pinning_likely),
        }
