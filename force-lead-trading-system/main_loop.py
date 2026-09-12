import os
import time
from datetime import datetime

from config.settings import settings


def is_trading_time():
    current_datetime = datetime.now()

    if current_datetime.weekday() >= 5:
        return False

    current_time = current_datetime.time()
    if current_time < settings.NO_TRADE_BEFORE:
        return False
    if current_time >= settings.NO_TRADE_AFTER:
        return False

    return True


def generate_alerts(result, prev_result=None):
    alerts = []
    combined_score = result["combined_score"]
    signal = result["signal"]
    confidence = result["confidence"]
    order_flow = (result.get("components") or {}).get("order_flow") or {}

    if signal in ("BUY", "SELL") and abs(combined_score) > 0.65:
        alerts.append({
            "level": "HIGH",
            "message": f"{signal} signal with score {combined_score} and confidence {confidence}",
        })

    divergence = order_flow.get("divergence", "NONE")
    divergence_strength = order_flow.get("divergence_strength", 0)
    if divergence in ("BULLISH", "BEARISH"):
        alerts.append({
            "level": "HIGH",
            "message": f"{divergence} divergence with strength {divergence_strength}",
        })

    absorption_type = order_flow.get("absorption_type", "NONE")
    absorption_strength = order_flow.get("absorption_strength", 0)
    if absorption_type != "NONE" and absorption_strength > 2:
        alerts.append({
            "level": "MEDIUM",
            "message": f"{absorption_type} absorption with strength {absorption_strength}",
        })

    lead_status = result.get("lead_status") or {}
    if lead_status.get("status") == "NEAR_CROSSOVER":
        alerts.append({
            "level": "MEDIUM",
            "message": f"Near crossover with {lead_status.get('minutes_remaining', 0)} minutes remaining",
        })

    reversal = result.get("reversal") or {}
    reversal_signal = reversal.get("reversal_signal") or {}
    if reversal_signal.get("reversal_starting") is True:
        detected_signals = reversal.get("detected_signals") or []
        alerts.append({
            "level": "HIGH",
            "message": f"Reversal starting with {len(detected_signals)} detected reversal signals",
        })

    move = result.get("move") or {}
    if move.get("size") == "EXPLOSIVE":
        alerts.append({
            "level": "HIGH",
            "message": f"Explosive move with expected move of {move.get('expected_move_pct', 0)}%",
        })

    significance = result.get("significance") or {}
    p_value = significance.get("p_value") or 0
    if p_value > 0.04:
        alerts.append({
            "level": "LOW",
            "message": f"Low significance with p-value {p_value:.3f}",
        })

    return alerts


def _daily_limit_reached(trades_today, logger):
    if trades_today >= settings.MAX_TRADES_PER_DAY:
        logger.log_error(f"Daily trade limit reached: {trades_today} trades today.")
        return True

    return False


def _get_live_market_data(feed):
    live_candle = feed.get_live_candle()
    nifty_candle = feed.get_nifty_candle()
    options_data = feed.get_options_data()
    order_flow = feed.get_order_flow()

    return live_candle, nifty_candle, options_data, order_flow


def _analyze_market(system, live_candle, historical_candles, nifty_candle, options_data, order_flow=None):
    return system.analyze(
        live_candle,
        historical_candles[-100:],
        nifty_candle,
        options_data,
        order_flow,
    )


def _enrich_result(result, feed, system, live_candle, trades_today, daily_pnl):
    get_order_flow = getattr(feed, "get_order_flow", None)
    if callable(get_order_flow):
        order_flow = get_order_flow()
        if order_flow is not None and "order_flow" not in result:
            result["order_flow"] = order_flow

    ml = getattr(system, "ml", None)
    result["system"] = {
        "connected": bool(getattr(feed, "connected", False)),
        "apiOk": bool(getattr(feed, "api_client", None)),
        "mlLoaded": bool(getattr(ml, "models", None)),
        "tradesTotal": trades_today,
        "dailyPnl": daily_pnl,
        "mode": os.getenv("TRADING_MODE", "paper"),
        "symbol": os.getenv("STOCK_SYMBOL", "NIFTY"),
        "time": datetime.now().isoformat(),
    }

    return result


def _add_price_info(result, live_candle):
    candle = live_candle or {}
    close_price = float(candle.get("close", 0.0) or 0.0)
    open_price = float(candle.get("open", 0.0) or 0.0)
    change = close_price - open_price

    result["price"] = close_price
    result["change"] = change
    result["changePct"] = (change / open_price * 100) if open_price else 0.0
    result["symbol"] = os.getenv("STOCK_SYMBOL", "NIFTY")

    return result


def _print_market_status(result, live_candle):
    candle = live_candle or {}
    price = float(candle.get("close", 0.0) or 0.0)
    signal = result.get("signal", "UNKNOWN")
    combined_score = float(result.get("combined_score", 0.0) or 0.0)
    win_probability = float(result.get("win_probability", 0.0) or 0.0)
    symbol = result.get("symbol", os.getenv("STOCK_SYMBOL", "NIFTY"))
    current_time = datetime.now().strftime("%H:%M:%S")

    print(
        f"[{current_time}] {symbol} | Price: {price:.2f} | "
        f"Signal: {signal} | Score: {combined_score:.2f} | "
        f"Win Probability: {win_probability:.1%}"
    )


def _update_historical_candles(feed, live_candle, historical_candles):
    if feed.is_candle_complete() is True:
        historical_candles.append(live_candle)
        historical_candles = historical_candles[-200:]
        print(f"Historical candle count: {len(historical_candles)}")

    return historical_candles


def run_main_loop(feed, system, historical_candles, nifty_candles, broadcast_fn, logger):
    prev_result = None
    trades_today = 0
    daily_pnl = 0.0

    print("Main loop started. Watching market...")

    while True:
        try:
            if not is_trading_time():
                time.sleep(30)
                continue

            if _daily_limit_reached(trades_today, logger):
                time.sleep(60)
                continue

            live_candle, nifty_candle, options_data, order_flow = _get_live_market_data(feed)
            if not live_candle:
                time.sleep(5)
                continue

            result = _analyze_market(
                system,
                live_candle,
                historical_candles,
                nifty_candle,
                options_data,
                order_flow,
            )
            if order_flow is not None and "order_flow" not in result:
                result["order_flow"] = order_flow

            result["alerts"] = generate_alerts(result, prev_result)
            result = _enrich_result(
                result,
                feed,
                system,
                live_candle,
                trades_today,
                daily_pnl,
            )
            result = _add_price_info(result, live_candle)
            broadcast_fn(result)
            _print_market_status(result, live_candle)
            historical_candles = _update_historical_candles(
                feed,
                live_candle,
                historical_candles,
            )
            prev_result = result
            time.sleep(5)
        except KeyboardInterrupt:
            print("System stopped by user.")
            break
        except Exception as exc:
            logger.log_error(f"Main loop error: {str(exc)}")
            print(f"Main loop error: {str(exc)}")
            time.sleep(10)