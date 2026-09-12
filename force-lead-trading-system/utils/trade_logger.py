import csv
import os
from datetime import datetime


class TradeLogger:
	def __init__(self, log_dir="logs"):
		self.log_dir = log_dir
		os.makedirs(self.log_dir, exist_ok=True)

		self.trade_log_path = os.path.join(self.log_dir, "trade_log.csv")
		self.signal_log_path = os.path.join(self.log_dir, "signal_log.csv")
		self.error_log_path = os.path.join(self.log_dir, "errors.log")

		self.trades_today = 0
		self.daily_pnl = 0.0
		self._init_trade_log()
		self._init_signal_log()
		open(self.error_log_path, "a", encoding="utf-8").close()

	def _init_trade_log(self):
		if not os.path.exists(self.trade_log_path):
			with open(self.trade_log_path, "w", newline="", encoding="utf-8") as log_file:
				csv.writer(log_file).writerow([
					"timestamp",
					"symbol",
					"signal",
					"entry_price",
					"stop_loss",
					"target",
					"exit_price",
					"outcome",
					"pnl",
					"shares",
					"force_score",
					"combined_score",
					"f1",
					"f2",
					"f3",
					"f4",
					"f5",
					"f6",
					"f7",
					"f8",
					"f9",
					"lead_status",
					"lead_progress",
					"bayesian_prob",
					"ml_prob",
					"mc_prob",
					"significance_p",
					"regime",
					"capital_after",
					"notes",
				])

	def _init_signal_log(self):
		if not os.path.exists(self.signal_log_path):
			with open(self.signal_log_path, "w", newline="", encoding="utf-8") as log_file:
				csv.writer(log_file).writerow([
					"timestamp",
					"signal",
					"combined_score",
					"force_score",
					"win_probability",
					"ml_confidence",
					"p_value",
					"regime",
				])

	def log_signal(self, signal_data):
		significance = signal_data.get("significance") or signal_data.get("sig") or {}
		regime_data = signal_data.get("regime") or {}
		regime = regime_data.get("regime", "") if isinstance(regime_data, dict) else regime_data
		timestamp = signal_data.get("timestamp") or datetime.now().isoformat()

		with open(self.signal_log_path, "a", newline="", encoding="utf-8") as log_file:
			csv.writer(log_file).writerow([
				timestamp,
				signal_data["signal"],
				signal_data.get("combined_score", ""),
				signal_data.get("force_score", ""),
				signal_data.get("win_probability", ""),
				signal_data.get("ml_confidence", ""),
				significance.get("p_value", "") if isinstance(significance, dict) else "",
				regime,
			])

		return signal_data

	def log_trade_entry(self, trade_data):
		self.trades_today += 1
		factors = trade_data.get("factors_detail", {}) or {}
		factors = factors if isinstance(factors, dict) else {}
		lead_status_data = trade_data.get("lead_status", {}) or {}
		position_data = trade_data.get("position", {}) or {}
		position_data = position_data if isinstance(position_data, dict) else {}
		ml_data = trade_data.get("ml_result", {}) or {}
		mc_data = trade_data.get("mc_result", {}) or {}
		significance = trade_data.get("significance") or trade_data.get("sig") or {}
		regime_data = trade_data.get("regime") or {}
		regime = regime_data.get("regime", "") if isinstance(regime_data, dict) else regime_data
		timestamp = trade_data.get("timestamp") or trade_data.get("entry_time") or datetime.now().isoformat()

		with open(self.trade_log_path, "a", newline="", encoding="utf-8") as log_file:
			csv.writer(log_file).writerow([
				timestamp,
				trade_data.get("symbol", ""),
				trade_data.get("signal", ""),
				trade_data.get("entry_price", ""),
				trade_data.get("stop_loss", ""),
				trade_data.get("target", ""),
				"",
				"",
				"",
				trade_data.get("shares", position_data.get("shares", "")),
				trade_data.get("force_score", ""),
				trade_data.get("combined_score", ""),
				*[factors.get(f"f{index}", "") for index in range(1, 10)],
				lead_status_data.get("status", "") if isinstance(lead_status_data, dict) else "",
				lead_status_data.get("progress_pct", lead_status_data.get("progress", "")) if isinstance(lead_status_data, dict) else "",
				trade_data.get("bayesian_prob", ""),
				ml_data.get("win_probability", trade_data.get("ml_prob", "")) if isinstance(ml_data, dict) else "",
				mc_data.get("win_probability", trade_data.get("mc_prob", "")) if isinstance(mc_data, dict) else "",
				significance.get("p_value", "") if isinstance(significance, dict) else "",
				regime,
				"",
				trade_data.get("notes", ""),
			])

		return trade_data

	def log_trade_exit(self, entry_timestamp, exit_price, outcome, pnl, capital_after):
		import pandas as pd

		try:
			trade_log = pd.read_csv(self.trade_log_path)
		except pd.errors.EmptyDataError as error:
			raise ValueError("Trade log is empty and cannot be updated") from error
		except (pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
			raise ValueError(f"Unable to read trade log: {error}") from error

		required_columns = {
			"timestamp",
			"exit_price",
			"outcome",
			"pnl",
			"capital_after",
		}
		missing_columns = required_columns - set(trade_log.columns)
		if missing_columns:
			raise ValueError(
				f"Malformed trade log; missing columns: {', '.join(sorted(missing_columns))}"
			)
		if trade_log.empty:
			raise ValueError("Trade log contains no trade rows")

		matching_rows = trade_log["timestamp"].astype(str) == str(entry_timestamp)
		if not matching_rows.any():
			raise ValueError(f"No trade found with entry timestamp: {entry_timestamp}")

		trade_log["outcome"] = trade_log["outcome"].astype(object)
		trade_log.loc[matching_rows, "exit_price"] = exit_price
		trade_log.loc[matching_rows, "outcome"] = outcome
		trade_log.loc[matching_rows, "pnl"] = pnl
		trade_log.loc[matching_rows, "capital_after"] = capital_after

		try:
			trade_log.to_csv(self.trade_log_path, index=False)
		except (OSError, ValueError) as error:
			raise ValueError(f"Unable to save updated trade log: {error}") from error

		self.daily_pnl += pnl

	def log_error(self, error_msg):
		with open(self.error_log_path, "a", encoding="utf-8") as log_file:
			log_file.write(f"{datetime.now().isoformat()} | {error_msg}\n")

	def get_todays_stats(self):
		import pandas as pd

		stats = {
			"completed_trades": 0,
			"wins": 0,
			"losses": 0,
			"win_rate": 0.0,
			"daily_pnl": 0.0,
			"trades_today": 0,
		}

		try:
			trade_log = pd.read_csv(self.trade_log_path)
		except (pd.errors.EmptyDataError, FileNotFoundError):
			self.trades_today = 0
			self.daily_pnl = 0.0
			return stats

		if "timestamp" not in trade_log.columns or "outcome" not in trade_log.columns:
			raise ValueError("Malformed trade log; timestamp and outcome columns are required")

		timestamps = pd.to_datetime(trade_log["timestamp"], errors="coerce")
		today_rows = trade_log.loc[timestamps.dt.date == datetime.now().date()].copy()
		outcomes = today_rows["outcome"].fillna("").astype(str).str.strip().str.upper()
		completed = today_rows.loc[outcomes != ""].copy()
		completed_outcomes = completed["outcome"].fillna("").astype(str).str.strip().str.upper()
		wins = int((completed_outcomes == "WIN").sum())
		losses = int((completed_outcomes == "LOSS").sum())
		completed_trades = len(completed)
		pnl = pd.to_numeric(completed.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

		stats.update({
			"completed_trades": completed_trades,
			"wins": wins,
			"losses": losses,
			"win_rate": wins / completed_trades if completed_trades else 0.0,
			"daily_pnl": float(pnl.sum()),
			"trades_today": len(today_rows),
		})
		self.trades_today = stats["trades_today"]
		self.daily_pnl = stats["daily_pnl"]
		return stats

	def check_daily_limits(self, max_trades, max_loss_pct, capital):
		if self.trades_today >= max_trades:
			return True, "Maximum trades per day reached"
		if self.daily_pnl < -(capital * max_loss_pct):
			return True, "Maximum daily loss reached"
		return False, "OK"
