import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "force-lead-trading-system"

for path in (str(ROOT), str(PACKAGE_ROOT)):
	if path not in sys.path:
		sys.path.insert(0, path)

from tests.mock_data_generator import generate_mock_candles, generate_mock_options
from core.force_score import ForceScoreCalculator
from core.lead_time import LeadTimeCalculator
from core.combined_system import CompleteTradingSystem
from statistics.significance_testing import SignificanceTester
from statistics.bayesian import BayesianUpdater
from statistics.stochastic import StochasticPriceModel
from statistics.monte_carlo import MonteCarloRiskEngine
from statistics.backtester import Backtester


def test_force_score():
	np.random.seed(42)
	candles = generate_mock_candles(50)
	calculator = ForceScoreCalculator()
	current_candle = candles[-1]
	previous_candle = candles[-2]

	f1 = calculator.calculate_f1_volume_force(current_candle, previous_candle)
	f2 = calculator.calculate_f2_wall_force(current_candle)
	f3 = calculator.calculate_f3_position_risk(current_candle, current_candle)
	f4 = calculator.calculate_f4_acceleration(candles[-5:])

	assert -1 <= f1 <= 1
	assert -1 <= f2 <= 1
	assert 0 <= f3 <= 1
	assert -1 <= f4 <= 1

	print(f"F1 volume force: {f1}")
	print(f"F2 wall force: {f2}")
	print(f"F3 position risk: {f3}")
	print(f"F4 acceleration: {f4}")

	result = calculator.calculate_total_force_score({
		"f1": f1,
		"f2": f2,
		"f3": f3,
		"f4": f4,
		"f5": 0.2,
		"f6": 0.1,
		"f7": 0.8,
		"f8": 1.0,
		"f9": 0.5,
	})
	score = result["score"]
	signal = result["signal"]

	assert -1 <= score <= 1
	assert signal in {"BUY", "SELL", "WAIT"}

	print(f"Total score: {score}")
	print(f"Signal: {signal}")


def test_lead_time():
	calculator = LeadTimeCalculator()
	previous_score = -0.7
	elapsed_minutes = 10

	lead_result = calculator.calculate_lead(previous_score, elapsed_minutes)
	lead = lead_result["lead_value"]

	current_score = 0.5
	catchup_result = calculator.calculate_catchup_time(lead, current_score, previous_score)
	catchup_time = catchup_result["catchup_time"]
	assert catchup_time > 0

	status_result = calculator.get_lead_status(elapsed_minutes, catchup_time, lead)
	status = status_result["status"]
	assert status in {
		"EARLY",
		"APPROACHING",
		"NEAR_CROSSOVER",
		"CROSSOVER_ZONE",
		"CROSSED",
		"DIVERGING",
	}

	print(f"Lead: {lead}")
	print(f"Catch-up time: {catchup_time}")
	print(f"Lead status: {status}")


def test_statistics():
	np.random.seed(42)
	tester = SignificanceTester()
	force_score_history = np.random.normal(0, 0.3, 50).tolist()
	current_score = 0.75

	result = tester.test_signal_significance(force_score_history, current_score)
	p_value = result["p_value"]

	assert 0 <= p_value <= 1
	assert "is_significant" in result

	print(f"P-value: {p_value}")
	print(f"Significant: {result['is_significant']}")


def test_monte_carlo():
	np.random.seed(42)
	engine = MonteCarloRiskEngine()
	result = engine.simulate_trade(
		entry_price=500,
		stop_loss=495,
		target=510,
		mu=0.0002,
		sigma=0.005,
		n_simulations=1000,
	)
	win_probability = result["win_probability"]
	loss_probability = result["loss_probability"]

	assert 0 <= win_probability <= 1
	assert 0 <= loss_probability <= 1
	assert abs((win_probability + loss_probability) - 1) <= 0.01

	print(f"Win probability: {win_probability}")
	print(f"Loss probability: {loss_probability}")


def test_full_pipeline():
	np.random.seed(42)
	normal_candles = generate_mock_candles(100, trend="mixed")
	bullish_candles = generate_mock_candles(100, trend="bullish")
	default_weights = {
		"f1": 0.25,
		"f2": 0.20,
		"f3": 0.10,
		"f4": 0.20,
		"f5": 0.10,
		"f6": 0.05,
		"f7": 0.05,
		"f8": 0.03,
		"f9": 0.02,
	}

	system = CompleteTradingSystem(
		capital=10000,
		optimal_weights=default_weights,
		trained_models={},
	)
	signals = []
	valid_signals = {"BUY", "SELL", "WAIT"}

	for index in range(10, len(normal_candles)):
		result = system.analyze(
			normal_candles[index],
			normal_candles[max(0, index - 50):index],
			bullish_candles[index],
			generate_mock_options(),
		)
		signal = result["signal"]
		combined_score = result["combined_score"]

		assert signal
		assert signal in valid_signals
		assert "combined_score" in result
		assert -1 <= combined_score <= 1
		signals.append(signal)

	print(f"BUY count: {signals.count('BUY')}")
	print(f"SELL count: {signals.count('SELL')}")
	print(f"WAIT count: {signals.count('WAIT')}")
	print("Full pipeline test passed")


def test_backtester():
	np.random.seed(42)
	candles = generate_mock_candles(150)
	backtester = Backtester()
	default_weights = {
		"f1": 0.25,
		"f2": 0.20,
		"f3": 0.10,
		"f4": 0.20,
		"f5": 0.10,
		"f6": 0.05,
		"f7": 0.05,
		"f8": 0.03,
		"f9": 0.02,
	}

	metrics = backtester.run_full_backtest(candles, default_weights, capital=10000)
	required_metrics = {
		"total_trades",
		"win_rate",
		"total_return",
		"sharpe_ratio",
		"max_drawdown",
	}

	assert isinstance(metrics, dict)
	assert required_metrics <= metrics.keys()

	for metric in required_metrics:
		print(f"{metric}: {metrics[metric]}")
	print("Backtest passed")


if __name__ == '__main__':
	failed = False
	print("=" * 60)
	print("Force Lead Trading System Test Suite")
	print("=" * 60)

	try:
		test_force_score()
		test_lead_time()
		test_statistics()
		test_monte_carlo()
		test_full_pipeline()
		test_backtester()
		print("TEST SUITE PASSED")
	except AssertionError as error:
		failed = True
		print(f"TEST FAILED: {error}")
	except Exception as error:
		failed = True
		print(f"ERROR: {error}")
		traceback.print_exc()

	raise SystemExit(1 if failed else 0)
