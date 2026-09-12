"""Utilities for generating mock market data in system tests."""

from datetime import datetime, timedelta

import numpy as np


def generate_mock_candles(n_candles=200, trend="mixed"):
	candles = []
	price = 500.0
	base_time = datetime(2024, 1, 1, 9, 15)
	base_volume = 50000
	current_day = None
	day_high = None
	day_low = None

	for index in range(n_candles):
		if trend == "bullish":
			drift = 0.002
		elif trend == "bearish":
			drift = -0.002
		else:
			drift = np.random.choice([-0.001, 0.001, 0])

		open_price = price
		price_change = open_price * (drift + np.random.normal(0, 0.001))
		close_price = open_price + price_change
		high_price = max(open_price, close_price) + abs(np.random.normal(0, open_price * 0.0005))
		low_price = min(open_price, close_price) - abs(np.random.normal(0, open_price * 0.0005))

		timestamp = base_time + timedelta(minutes=5 * index)
		if timestamp.date() != current_day:
			current_day = timestamp.date()
			day_high = high_price
			day_low = low_price
		else:
			day_high = max(day_high, high_price)
			day_low = min(day_low, low_price)
		if 9 <= timestamp.hour < 10:
			volume_multiplier = 1.5
		elif 14 <= timestamp.hour < 15:
			volume_multiplier = 1.25
		else:
			volume_multiplier = 0.8
		volume = max(1, int(base_volume * volume_multiplier + np.random.normal(0, base_volume * 0.05)))
		volume_per_minute = round(volume / 5, 2)
		direction = 1 if close_price > open_price else -1 if close_price < open_price else 0
		if close_price > open_price:
			buyers_pct = np.random.randint(55, 76)
		else:
			buyers_pct = np.random.randint(25, 46)
		sellers_pct = 100 - buyers_pct
		if np.random.random() < 0.8:
			avg_trade_size = np.random.randint(100, 501)
		else:
			avg_trade_size = np.random.randint(2000, 10001)
		trade_count = int(volume / avg_trade_size)
		spread = round(np.random.uniform(0.05, 0.20), 4)
		bid_price = round(close_price - spread / 2, 2)
		ask_price = round(close_price + spread / 2, 2)
		bid_total = int(volume * buyers_pct / 100)
		ask_total = int(volume * sellers_pct / 100)
		quantity_proportions = [0.30, 0.25, 0.20, 0.15, 0.10]
		bid_levels = [
			[round(bid_price - spread * (level + 1), 2), int(bid_total * proportion)]
			for level, proportion in enumerate(quantity_proportions)
		]
		ask_levels = [
			[round(ask_price + spread * (level + 1), 2), int(ask_total * proportion)]
			for level, proportion in enumerate(quantity_proportions)
		]

		candles.append({
			"timestamp": timestamp,
			"open": round(open_price, 2),
			"high": round(high_price, 2),
			"low": round(low_price, 2),
			"close": round(close_price, 2),
			"volume": volume,
			"volume_per_minute": volume_per_minute,
			"direction": direction,
			"buyers_pct": buyers_pct,
			"sellers_pct": sellers_pct,
			"avg_trade_size": avg_trade_size,
			"trade_count": trade_count,
			"spread": spread,
			"bid_price": bid_price,
			"ask_price": ask_price,
			"bid_total": bid_total,
			"ask_total": ask_total,
			"bid_levels": bid_levels,
			"ask_levels": ask_levels,
			"day_high": round(day_high, 2),
			"day_low": round(day_low, 2),
		})
		price = close_price

	return candles


def generate_mock_options():
	call_oi = int(np.random.randint(10000, 100001))
	put_oi = int(np.random.randint(10000, 100001))
	return {
		"call_oi": call_oi,
		"put_oi": put_oi,
		"current_call_oi": call_oi,
		"current_put_oi": put_oi,
		"prev_call_oi": int(np.random.randint(10000, 100001)),
		"prev_put_oi": int(np.random.randint(10000, 100001)),
		"atm_call_iv": round(np.random.uniform(0.15, 0.35), 4),
		"atm_put_iv": round(np.random.uniform(0.15, 0.35), 4),
		"max_pain": round(np.random.normal(500, 10), 2),
	}
