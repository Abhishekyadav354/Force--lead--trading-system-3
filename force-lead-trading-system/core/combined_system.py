from __future__ import annotations

from config.settings import settings
from core.force_score import ForceScoreCalculator
from core.lead_time import LeadTimeCalculator
from statistics.significance_testing import SignificanceTester
from statistics.bayesian import BayesianUpdater
from statistics.stochastic import StochasticPriceModel
from statistics.monte_carlo import MonteCarloRiskEngine
from core.greeks_calculator import GreeksCalculator
from core.order_flow import OrderFlowAnalyzer
from ml.predictor import MLPredictor
from utils.logger import TradeLogger


class CompleteTradingSystem:
    def __init__(self, capital, optimal_weights, trained_models):
        self.capital = capital
        self.optimal_weights = optimal_weights
        self.force_history = []
        self.elapsed_minutes = 0

        self.force_calc = ForceScoreCalculator()
        self.lead_calc = LeadTimeCalculator()
        self.significance = SignificanceTester()
        self.bayesian = BayesianUpdater()
        self.stochastic = StochasticPriceModel()
        self.monte_carlo = MonteCarloRiskEngine()
        self.greeks_calc = GreeksCalculator()
        self.order_flow_analyzer = OrderFlowAnalyzer()
        self.ml = MLPredictor()
        self.trade_logger = TradeLogger()

    def _calculate_force_factors_part1(
        self,
        live_candle,
        historical_candles,
        options_data,
    ):
        historical_avg_trade_size = 1.0
        if historical_candles:
            trade_sizes = [
                float(candle.get("avg_trade_size", 0.0) or 0.0)
                for candle in historical_candles
                if isinstance(candle, dict)
            ]
            if trade_sizes:
                historical_avg_trade_size = sum(trade_sizes) / len(trade_sizes)

        previous_candle = historical_candles[-1] if historical_candles else None

        f1 = self.force_calc.calculate_f1_volume_force(
            live_candle,
            previous_candle,
            historical_avg_trade_size=historical_avg_trade_size,
        )
        f2 = self.force_calc.calculate_f2_wall_force(live_candle)
        f3 = self.force_calc.calculate_f3_position_risk(live_candle, options_data or {})

        return {"f1": f1, "f2": f2, "f3": f3}

    def _calculate_force_factors_part2(
        self,
        live_candle,
        historical_candles,
        options_data,
    ):
        recent_history = historical_candles[-5:] if historical_candles else []

        f4 = self.force_calc.calculate_f4_acceleration(recent_history)
        f5 = self.force_calc.calculate_f5_institutional_detector(live_candle, historical_candles or [])

        if not options_data:
            f6 = 0.0
        else:
            f6 = self.force_calc.calculate_f6_oi_signal(options_data, {"current_price": live_candle.get("close", live_candle.get("ltp", 0.0)), "prev_price": (recent_history[-1].get("close", recent_history[-1].get("ltp", 0.0)) if recent_history else 0.0)})

        return {"f4": f4, "f5": f5, "f6": f6}

    def _calculate_force_factors_part3(
        self,
        live_candle,
        historical_candles,
        nifty_candle,
    ):
        recent_10 = historical_candles[-10:] if historical_candles else []

        f7 = self.force_calc.calculate_f7_spread_quality(recent_10)

        if nifty_candle is None:
            f8 = 1.0
        else:
            f8 = self.force_calc.calculate_f8_nifty_alignment(nifty_candle, live_candle)

        f9 = self.force_calc.calculate_f9_candle_conviction(live_candle)

        return {"f7": f7, "f8": f8, "f9": f9}

    def _calculate_all_force_factors(
        self,
        live_candle,
        historical_candles,
        options_data,
        nifty_candle,
    ):
        part1 = self._calculate_force_factors_part1(live_candle, historical_candles, options_data)
        part2 = self._calculate_force_factors_part2(live_candle, historical_candles, options_data)
        part3 = self._calculate_force_factors_part3(live_candle, historical_candles, nifty_candle)
        merged = {}
        merged.update(part1)
        merged.update(part2)
        merged.update(part3)
        return merged

    def _check_force_significance(self, factors):
        force_score_result = self.force_calc.calculate_total_force_score(factors)
        force_score = force_score_result.get("score", 0.0)
        sig = self.significance.test_signal_significance(self.force_history, force_score)
        return {"force_score": force_score, "sig": sig}

    def _handle_insignificant_signal(self, force_score, factors, sig):
        if sig.get("is_significant"):
            return None

        p_value = sig.get("p_value", 1.0)
        return {
            "signal": "WAIT",
            "reason": "Not statistically significant",
            "p_value": p_value,
            "combined_score": 0,
            "force_score": force_score,
            "factors_detail": factors,
        }

    def _calculate_regime_adjusted_score(
        self,
        factors,
        historical_candles,
    ):
        regime_data = self.stochastic.detect_market_regime(historical_candles or [])
        regime = regime_data.get("regime", "RANDOM_WALK")

        regime_weights = getattr(settings, "REGIME_WEIGHTS", None)
        if isinstance(regime_weights, dict):
            regime_weights = regime_weights.get(regime, regime_weights.get("default", self.optimal_weights or {}))
        else:
            regime_weights = regime_data.get("weight_adjustments") or self.optimal_weights or {}

        if not isinstance(regime_weights, dict) or not regime_weights:
            regime_weights = self.optimal_weights or {}

        try:
            force_score_result = self.force_calc.calculate_total_force_score(factors, regime_weights)
        except TypeError:
            force_score_result = self.force_calc.calculate_total_force_score(factors)

        force_score_adjusted = force_score_result.get("score", 0.0) if isinstance(force_score_result, dict) else float(force_score_result)
        return {
            "regime": regime,
            "force_score_adjusted": force_score_adjusted,
        }

    def _calculate_lead_data(
        self,
        force_score_adjusted,
    ):
        prev_score = self.force_history[-1] if self.force_history else 0.0
        timeframe_minutes = getattr(settings, "TIMEFRAME_MINUTES", getattr(settings, "timeframe", 5))

        lead = self.lead_calc.calculate_lead(
            prev_score,
            timeframe_minutes,
        )
        catchup = self.lead_calc.calculate_catchup_time(
            lead["lead_value"],
            force_score_adjusted,
            prev_score,
        )
        lead_status = self.lead_calc.get_lead_status(
            self.elapsed_minutes,
            catchup["catchup_time"],
            lead["lead_value"],
        )

        return {
            "lead": lead,
            "catchup": catchup,
            "lead_status": lead_status,
        }

    def _calculate_probability_data(
        self,
        historical_candles,
        factors,
    ):
        closes = []
        for candle in (historical_candles or [])[-50:]:
            if isinstance(candle, dict):
                value = candle.get("close", candle.get("ltp", 0.0))
                try:
                    closes.append(float(value))
                except (TypeError, ValueError):
                    continue

        gbm_params = self.stochastic.estimate_gbm_parameters(closes)
        mu = gbm_params.get("mu", 0.0)
        sigma = gbm_params.get("sigma", 0.0)

        prior = 0.5
        prior_getter = getattr(self.bayesian, "get_prior", None)
        if callable(prior_getter):
            prior = prior_getter()
        elif hasattr(self.bayesian, "prior_win_rate"):
            prior = self.bayesian.prior_win_rate

        bayesian_prob = self.bayesian.bayesian_update(prior, factors)

        return {
            "mu": mu,
            "sigma": sigma,
            "bayesian_prob": bayesian_prob,
        }

    def _update_force_history(self, force_score):
        self.force_history.append(float(force_score))
        self.force_history = self.force_history[-100:]
        return self.force_history

    def _calculate_greeks_and_ml(
        self,
        options_data,
        live_candle,
        factors,
        lead_status,
        regime,
    ):
        greeks = {}
        if options_data:
            try:
                close_price = float((live_candle or {}).get("close", (live_candle or {}).get("ltp", 0.0)))
                greeks = self.greeks_calc.calculate_all(options_data, close_price)
            except Exception:
                greeks = {}

        ml_result = self.ml.predict_live(factors, lead_status or {}, regime or {})
        return {
            "greeks": greeks,
            "ml_result": ml_result,
        }

    def _calculate_trade_risk(
        self,
        live_candle,
        force_score_adjusted,
        mu,
        sigma,
    ):
        candle = live_candle or {}
        entry_price = float(candle.get("close", candle.get("ltp", 0.0)) or 0.0)
        bullish = force_score_adjusted > 0

        stop_loss_pct = getattr(settings, "STOP_LOSS_PCT", 0.01)
        target_pct = getattr(settings, "TARGET_PCT", 0.02)
        mc_simulations = getattr(settings, "MC_SIMULATIONS", 1000)

        if bullish:
            stop_loss = entry_price * (1.0 - float(stop_loss_pct))
            target = entry_price * (1.0 + float(target_pct))
        else:
            stop_loss = entry_price * (1.0 + float(stop_loss_pct))
            target = entry_price * (1.0 - float(target_pct))

        mc_result = self.monte_carlo.simulate_trade(
            entry_price,
            stop_loss,
            target,
            float(mu),
            float(sigma),
            n_simulations=int(mc_simulations),
        )

        return {
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "mc_result": mc_result,
        }

    def _calculate_final_score(
        self,
        force_score_adjusted,
        bayesian_prob,
        ml_result,
        sig,
        mc_result,
        greeks,
    ):
        if isinstance(ml_result, dict):
            ml_score = float(ml_result.get("win_probability", 0.5))
        else:
            ml_score = 0.5

        sig_confidence = 0.0
        if isinstance(sig, dict):
            sig_confidence = float(sig.get("confidence_pct", 0.0)) / 100.0

        mc_score = 0.0
        if isinstance(mc_result, dict):
            mc_score = 1.0 if mc_result.get("trade_worthwhile") else 0.0

        greeks_force = 0.0
        if isinstance(greeks, dict):
            greeks_force = float(greeks.get("force_score_addition", 0.0))

        force_component = float(force_score_adjusted)
        bayesian_component = float(bayesian_prob)
        ml_component = float(ml_score)
        stat_component = float(sig_confidence)
        mc_component = float(mc_score)

        combined_score = (
            force_component * 0.30
            + bayesian_component * 0.25
            + ml_component * 0.20
            + stat_component * 0.10
            + mc_component * 0.15
            + greeks_force * 0.10
        )

        combined_score = max(-1.0, min(1.0, combined_score))
        return combined_score

    def _calculate_order_flow_data(self, order_flow_input):
        """Calculate F10 only from supplied tick data, returning neutral when absent."""
        neutral = {
            "delta": {"delta": 0.0, "buy_volume": 0.0, "sell_volume": 0.0, "delta_ratio": 0.5, "total_volume": 0.0},
            "divergence": {"type": "NONE", "strength": 0.0, "price_slope": 0.0, "delta_slope": 0.0, "signal": "0"},
            "volume_profile": {"profile": {}, "poc": 0.0, "vah": 0.0, "val": 0.0, "poc_signal": "0", "total_volume": 0.0},
            "footprint": {"footprint": [], "stacked_buy_zones": [], "stacked_sell_zones": [], "max_buy_level": 0.0, "max_sell_level": 0.0, "candle_open": None, "candle_close": None},
            "absorption": {"type": "NONE", "strength": 0.0, "signal": 0},
            "exhaustion": {"is_exhausted": False, "strength": 0.0, "efficiency_trend": 0.0, "reversal_probability": 0.0},
            "vwap": {"vwap": 0.0, "bands": [], "distance_std": 0.0, "signal": "NEAR_VWAP", "std_dev": 0.0},
            "score": 0.0,
            "signal": "WAIT",
        }
        if order_flow_input is None:
            return neutral

        if isinstance(order_flow_input, dict):
            ticks = order_flow_input.get("ticks")
            if ticks is None:
                ticks = order_flow_input.get("classified_ticks")
        elif isinstance(order_flow_input, list):
            ticks = order_flow_input
        else:
            return neutral

        if not isinstance(ticks, list) or not ticks:
            return neutral

        classified_ticks = []
        for tick in ticks:
            if not isinstance(tick, dict):
                continue
            if tick.get("type") in {"BUY_INITIATED", "SELL_INITIATED", "NEUTRAL"}:
                classified_ticks.append(dict(tick))
                continue
            try:
                classified_ticks.append(self.order_flow_analyzer.classify_tick(tick))
            except (TypeError, ValueError):
                continue

        if not classified_ticks:
            return neutral

        prices = [float(tick["price"]) for tick in classified_ticks if tick.get("price") is not None]
        deltas = []
        for tick in classified_ticks:
            volume = float(tick.get("volume", 0.0) or 0.0)
            tick_type = str(tick.get("type", "")).upper()
            deltas.append(volume if tick_type == "BUY_INITIATED" else -volume if tick_type == "SELL_INITIATED" else 0.0)
        volumes = [float(tick.get("volume", 0.0) or 0.0) for tick in classified_ticks]
        current_price = prices[-1] if prices else None

        try:
            delta = self.order_flow_analyzer.calculate_delta(classified_ticks)
            divergence = self.order_flow_analyzer.detect_delta_divergence(prices, deltas)
            volume_profile = self.order_flow_analyzer.build_volume_profile(classified_ticks, current_price=current_price)
            footprint = self.order_flow_analyzer.build_footprint(classified_ticks, candle_open=prices[0] if prices else None, candle_close=current_price)
            absorption = self.order_flow_analyzer.detect_absorption(classified_ticks, prices)
            exhaustion = self.order_flow_analyzer.detect_exhaustion(deltas, prices, volumes)
            vwap = self.order_flow_analyzer.calculate_vwap_bands(classified_ticks)
            score_result = self.order_flow_analyzer.calculate_order_flow_score({
                "delta": delta,
                "divergence": divergence,
                "absorption": absorption,
                "vwap": vwap,
                "poc": volume_profile,
                "stacked_buy_zones": footprint.get("stacked_buy_zones", []),
                "stacked_sell_zones": footprint.get("stacked_sell_zones", []),
                "exhaustion": exhaustion,
            })
        except (TypeError, ValueError, ZeroDivisionError):
            return neutral

        return {
            "delta": delta,
            "divergence": divergence,
            "volume_profile": volume_profile,
            "footprint": footprint,
            "absorption": absorption,
            "exhaustion": exhaustion,
            "vwap": vwap,
            "score": score_result.get("score", 0.0),
            "signal": score_result.get("signal", "WAIT"),
        }

    def _calculate_position(
        self,
        entry_price,
        stop_loss,
        target,
        mc_result,
        sigma,
    ):
        risk_per_share = max(float(entry_price) - float(stop_loss), 0.0)
        reward_per_share = max(float(target) - float(entry_price), 0.0)
        win_prob = float((mc_result or {}).get("win_probability", 0.5)) if isinstance(mc_result, dict) else 0.5
        return self.monte_carlo.optimal_position_sizer(
            self.capital,
            win_prob,
            risk_per_share,
            reward_per_share,
            mc_result or {},
            float(sigma),
        )

    def analyze(
        self,
        live_candle,
        historical_candles,
        nifty_candle,
        options_data,
        order_flow_data=None,
    ):
        factors = self._calculate_all_force_factors(live_candle, historical_candles, options_data, nifty_candle)
        order_flow_result = self._calculate_order_flow_data(order_flow_data)
        score_info = self._check_force_significance(factors)
        force_score = score_info["force_score"]
        sig = score_info["sig"]
        if not sig.get("is_significant"):
            result = {
                "signal": "WAIT",
                "reason": "Not statistically significant",
                "p_value": sig.get("p_value", 1.0),
                "combined_score": 0,
                "force_score": force_score,
                "factors_detail": factors,
                "order_flow": order_flow_result,
                "final_combined_score": force_score,
                "final_signal": "WAIT",
            }
            self.trade_logger.log_signal(result)
            return result

        regime_data = self._calculate_regime_adjusted_score(factors, historical_candles)
        force_score_adjusted = regime_data["force_score_adjusted"]
        lead_data = self._calculate_lead_data(force_score_adjusted)
        self._update_force_history(force_score_adjusted)
        probability_data = self._calculate_probability_data(historical_candles, factors)
        ml_data = self._calculate_greeks_and_ml(options_data, live_candle, factors, lead_data["lead_status"], regime_data)
        risk_data = self._calculate_trade_risk(live_candle, force_score_adjusted, probability_data["mu"], probability_data["sigma"])
        combined_score = self._calculate_final_score(
            force_score_adjusted,
            probability_data["bayesian_prob"],
            ml_data["ml_result"],
            sig,
            risk_data["mc_result"],
            ml_data["greeks"],
        )
        integration = self.order_flow_analyzer.integrate_with_force_score(
            {"score": force_score_adjusted},
            {"score": order_flow_result["score"]},
        )
        position = self._calculate_position(
            risk_data["entry_price"],
            risk_data["stop_loss"],
            risk_data["target"],
            risk_data["mc_result"],
            probability_data["sigma"],
        )

        threshold = getattr(settings, "FORCE_SCORE_THRESHOLD", 0.5)
        lead_action = lead_data["lead_status"].get("action", "WATCH_CLOSELY")
        if combined_score > threshold and lead_action != "DO_NOT_TRADE":
            signal = "BUY"
            reason = "Score above threshold and lead not blocked"
        elif combined_score < -threshold and lead_action != "DO_NOT_TRADE":
            signal = "SELL"
            reason = "Score below threshold and lead not blocked"
        else:
            signal = "WAIT"
            reason = "Threshold not met or lead action blocked"

        result = {
            "signal": signal,
            "reason": reason,
            "combined_score": combined_score,
            "force_score": force_score,
            "force_score_adjusted": force_score_adjusted,
            "factors": factors,
            "regime": regime_data["regime"],
            "lead_status": lead_data["lead_status"],
            "lead": lead_data["lead"],
            "catchup": lead_data["catchup"],
            "sig": sig,
            "mu": probability_data["mu"],
            "sigma": probability_data["sigma"],
            "bayesian_prob": probability_data["bayesian_prob"],
            "greeks": ml_data["greeks"],
            "ml_result": ml_data["ml_result"],
            "entry_price": risk_data["entry_price"],
            "stop_loss": risk_data["stop_loss"],
            "target": risk_data["target"],
            "mc_result": risk_data["mc_result"],
            "position": position,
            "order_flow": order_flow_result,
            "final_combined_score": integration["final_score"],
            "final_signal": integration["signal"],
        }
        self.trade_logger.log_signal(result)
        return result
