import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "force-lead-trading-system"

for path in (str(ROOT), str(PACKAGE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config.settings import settings
from utils.trade_logger import TradeLogger


class PaperTrader:
    def __init__(self, virtual_capital=None):
        if virtual_capital is None:
            virtual_capital = settings.CAPITAL

        self.virtual_capital = virtual_capital
        self.starting_capital = virtual_capital
        self.open_position = None
        self.trades = []
        self.logger = TradeLogger(log_dir='logs/paper_trading')
        self.max_position_pct = settings.MAX_POSITION_PCT
        self.max_daily_loss_pct = settings.MAX_DAILY_LOSS_PCT
        self.max_trades_per_day = settings.MAX_TRADES_PER_DAY
        self.no_trade_after = settings.NO_TRADE_AFTER
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Paper trading started. Virtual capital: ${self.virtual_capital:,.2f}")
        print("Paper trading mode enabled: no broker connection, no real orders placed.")

    def _open_virtual_trade(self, direction, entry_price, signal_data=None):
        if entry_price is None:
            return None

        try:
            entry_price = float(entry_price)
        except (TypeError, ValueError):
            return None

        if entry_price <= 0:
            return None

        signal_data = signal_data or {}
        direction = str(direction).upper()
        if direction not in {"BUY", "SELL"}:
            return None

        position_size = signal_data.get("position_size", {}) if isinstance(signal_data, dict) else {}
        if isinstance(position_size, dict):
            pct_of_capital = position_size.get("pct_of_capital", 0.0)
        else:
            pct_of_capital = position_size

        try:
            pct_of_capital = float(pct_of_capital)
        except (TypeError, ValueError):
            pct_of_capital = 0.0

        if pct_of_capital > 1:
            pct_of_capital = pct_of_capital / 100.0

        if pct_of_capital <= 0:
            pct_of_capital = self.max_position_pct

        capital_fraction = max(0.0, min(pct_of_capital, self.max_position_pct))

        today_trade_count = sum(
            1
            for trade in self.trades
            if isinstance(trade, dict)
            and trade.get("entry_time")
            and datetime.fromisoformat(str(trade["entry_time"])) .date() == datetime.now().date()
        )
        if today_trade_count >= self.max_trades_per_day:
            return None

        daily_loss = 0.0
        for trade in self.trades:
            if not isinstance(trade, dict):
                continue
            pnl = float(trade.get("pnl", 0.0)) if trade.get("reason") in {"TARGET", "SL", "FORCE_EXIT"} else 0.0
            if trade.get("exit_time"):
                try:
                    exit_date = datetime.fromisoformat(str(trade["exit_time"])) .date()
                    if exit_date == datetime.now().date() and pnl < 0:
                        daily_loss += abs(pnl)
                except ValueError:
                    pass
        if self.virtual_capital > 0 and daily_loss > 0 and (daily_loss / self.virtual_capital) > self.max_daily_loss_pct:
            return None

        virtual_position_value = self.virtual_capital * capital_fraction
        shares = int(virtual_position_value / entry_price)

        if shares <= 0:
            return None

        stop_loss = signal_data.get("stop_loss") if isinstance(signal_data, dict) else None
        target = signal_data.get("target") if isinstance(signal_data, dict) else None

        default_stop_loss_pct = settings.STOP_LOSS_PCT
        default_target_pct = settings.TARGET_PCT

        if stop_loss is None:
            stop_loss = entry_price * (1 - default_stop_loss_pct) if direction == "BUY" else entry_price * (1 + default_stop_loss_pct)
        if target is None:
            target = entry_price * (1 + default_target_pct) if direction == "BUY" else entry_price * (1 - default_target_pct)

        try:
            stop_loss = float(stop_loss)
            target = float(target)
        except (TypeError, ValueError):
            return None

        if stop_loss <= 0 or target <= 0:
            return None

        if direction == "BUY":
            if not (stop_loss < entry_price and target > entry_price):
                return None
        elif direction == "SELL":
            if not (stop_loss > entry_price and target < entry_price):
                return None

        entry_timestamp = datetime.now().isoformat()
        trade = {
            "direction": direction,
            "entry_price": entry_price,
            "shares": shares,
            "stop_loss": stop_loss,
            "target": target,
            "entry_time": entry_timestamp,
            "signal_data": signal_data,
            "virtual": True,
            "broker_connected": False,
            "status": "open",
        }

        self.open_position = trade
        self.trades.append(trade)

        try:
            self.logger.log_trade_entry({
                "timestamp": entry_timestamp,
                "symbol": signal_data.get("symbol", "") if isinstance(signal_data, dict) else "",
                "signal": direction,
                "entry_price": entry_price,
                "shares": shares,
                "stop_loss": stop_loss,
                "target": target,
                "notes": "Virtual-only paper trade",
                "force_score": signal_data.get("force_score", signal_data.get("combined_score", "")) if isinstance(signal_data, dict) else "",
                "entry_time": entry_timestamp,
            })
        except Exception:
            pass

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Virtual trade opened: direction={direction}, entry=${entry_price:,.2f}, shares={shares}, stop_loss=${stop_loss:,.2f}, target=${target:,.2f}, capital_fraction={capital_fraction:.4f}, no broker/order API called.")
        return trade

    def check_entry(self, signal_data, live_price):
        if self.open_position is not None:
            return None

        if signal_data is None:
            return None

        try:
            signal = str(signal_data.get("signal", "")).upper()
        except AttributeError:
            signal = str(signal_data).upper()

        if signal not in {"BUY", "SELL"}:
            return None

        try:
            price = float(live_price)
        except (TypeError, ValueError):
            return None

        if price <= 0:
            return None

        if signal == "BUY":
            return self._open_virtual_trade("BUY", price, signal_data)
        return self._open_virtual_trade("SELL", price, signal_data)

    def _print_stats(self):
        completed = [
            trade for trade in self.trades
            if isinstance(trade, dict) and trade.get("reason") in {"TARGET", "SL", "FORCE_EXIT"}
        ]

        if len(completed) < 3:
            return

        wins = sum(1 for trade in completed if trade.get("outcome") == "WIN")
        losses = sum(1 for trade in completed if trade.get("outcome") == "LOSS")
        total_pnl = sum(float(trade.get("pnl", 0.0)) for trade in completed)
        win_rate = (wins / len(completed)) * 100 if completed else 0.0

        total_return_pct = 0.0
        if self.starting_capital and self.starting_capital > 0:
            total_return_pct = ((self.virtual_capital - self.starting_capital) / self.starting_capital) * 100

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Paper trading summary: completed_trades={len(completed)}, win_rate={win_rate:.2f}%, total_return={total_return_pct:.2f}%, virtual_capital=${self.virtual_capital:,.2f}, total_pnl=${total_pnl:,.2f}, wins={wins}, losses={losses}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Paper results are historical simulation only and do not guarantee future performance.")

    def check_exit(self, current_price, current_time):
        if self.open_position is None:
            return None

        try:
            current_price = float(current_price)
        except (TypeError, ValueError):
            return None

        if current_price <= 0:
            return None

        position = self.open_position
        direction = str(position.get("direction", "")).upper()
        stop_loss = position.get("stop_loss")
        target = position.get("target")

        try:
            stop_loss = float(stop_loss)
            target = float(target)
        except (TypeError, ValueError):
            return None

        reason = None

        if direction == "BUY":
            if current_price <= stop_loss:
                reason = "SL"
            elif current_price >= target:
                reason = "TARGET"
        elif direction == "SELL":
            if current_price >= stop_loss:
                reason = "SL"
            elif current_price <= target:
                reason = "TARGET"
        else:
            return None

        if reason is None and current_time is not None:
            try:
                if isinstance(current_time, datetime):
                    current_time_obj = current_time.time()
                else:
                    current_time_obj = datetime.strptime(str(current_time), "%H:%M:%S").time()
            except ValueError:
                try:
                    current_time_obj = datetime.strptime(str(current_time), "%H:%M").time()
                except ValueError:
                    current_time_obj = None

            if current_time_obj is not None:
                force_exit_time = datetime.strptime(self.no_trade_after, "%H:%M").time()
                if current_time_obj >= force_exit_time:
                    reason = "FORCE_EXIT"

        if reason is None:
            return None

        entry_price = float(position.get("entry_price", 0.0))
        shares = int(position.get("shares", 0))
        transaction_cost = 0.0

        try:
            transaction_cost = float(settings.TRANSACTION_COST)
        except (AttributeError, TypeError, ValueError):
            transaction_cost = 0.0

        if direction == "BUY":
            pnl = (current_price - entry_price) * shares
        elif direction == "SELL":
            pnl = (entry_price - current_price) * shares
        else:
            return None

        pnl -= transaction_cost
        outcome = "WIN" if reason == "TARGET" else "LOSS"

        self.virtual_capital += pnl
        rounded_pnl = round(pnl, 2)
        rounded_capital = round(self.virtual_capital, 2)

        force_score = None
        signal_data = position.get("signal_data") if isinstance(position, dict) else None
        if isinstance(signal_data, dict):
            force_score = signal_data.get("force_score")
            if force_score is None:
                force_score = signal_data.get("combined_score")

        closed_trade = {
            "direction": direction,
            "entry": entry_price,
            "exit": current_price,
            "outcome": outcome,
            "pnl": rounded_pnl,
            "capital": rounded_capital,
            "reason": reason,
            "force_score": force_score,
        }
        self.trades.append(closed_trade)

        try:
            entry_timestamp = position.get("entry_time")
            if entry_timestamp is None:
                entry_timestamp = datetime.now().isoformat()
            self.logger.log_trade_exit(
                entry_timestamp,
                current_price,
                outcome,
                rounded_pnl,
                rounded_capital,
            )
        except Exception:
            pass

        self.open_position = None

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Paper trade closed: direction={direction}, entry=${entry_price:,.2f}, exit=${current_price:,.2f}, reason={reason}, outcome={outcome}, pnl=${rounded_pnl:,.2f}, capital=${rounded_capital:,.2f}, no broker/order API called.")
        self._print_stats()
        return closed_trade

    def _get_verdict(self, win_rate):
        try:
            win_rate = float(win_rate)
        except (TypeError, ValueError):
            return "POOR: Win rate could not be interpreted; this paper result is not enough to validate a real-money strategy. Evaluate drawdown, costs, sample size and out-of-sample performance before making any trading decision."

        if win_rate >= 0.65:
            verdict = "EXCELLENT"
            explanation = "This paper-trading win rate is strong, but a win rate alone is not enough to validate a strategy. Consider drawdown, transaction costs, sample size, and out-of-sample performance before any real-money decision."
        elif win_rate >= 0.55:
            verdict = "GOOD"
            explanation = "This win rate is encouraging, but it does not automatically justify real-money trading. Evaluate drawdown, costs, sample size, and out-of-sample performance before making any trading decision."
        elif win_rate >= 0.50:
            verdict = "MARGINAL"
            explanation = "This result is only marginally positive, and a win rate alone is not enough to validate a strategy. Review drawdown, costs, sample size and out-of-sample performance before any real-money trading decision."
        else:
            verdict = "POOR"
            explanation = "This win rate is weak for a paper-trading strategy and does not justify real-money trading on its own. Drawdown, costs, sample size and out-of-sample performance are also critical factors."

        return f"{verdict}: {explanation}"

    def generate_report(self):
        import json

        completed = [
            trade for trade in self.trades
            if isinstance(trade, dict) and trade.get("reason") in {"TARGET", "SL", "FORCE_EXIT"}
        ]

        if not completed:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No completed paper trades available to summarize.")
            return {}

        wins = sum(1 for trade in completed if trade.get("outcome") == "WIN")
        losses = sum(1 for trade in completed if trade.get("outcome") == "LOSS")
        total_trades = len(completed)
        win_rate = (wins / total_trades) * 100 if total_trades else 0.0

        gross_profit = sum(float(trade.get("pnl", 0.0)) for trade in completed if float(trade.get("pnl", 0.0)) > 0)
        gross_loss = abs(sum(float(trade.get("pnl", 0.0)) for trade in completed if float(trade.get("pnl", 0.0)) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit > 0 else 0.0)

        average_win = gross_profit / wins if wins else 0.0
        average_loss = gross_loss / losses if losses else 0.0
        reward_risk_ratio = average_win / average_loss if average_loss else (float("inf") if average_win > 0 else 0.0)

        total_return = 0.0
        if self.starting_capital and self.starting_capital > 0:
            total_return = ((self.virtual_capital - self.starting_capital) / self.starting_capital) * 100

        report = {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4) if isinstance(profit_factor, float) and profit_factor != float("inf") else ("inf" if profit_factor == float("inf") else 0.0),
            "reward_risk_ratio": round(reward_risk_ratio, 4) if isinstance(reward_risk_ratio, float) and reward_risk_ratio != float("inf") else ("inf" if reward_risk_ratio == float("inf") else 0.0),
            "total_return": round(total_return, 2),
            "final_capital": round(self.virtual_capital, 2),
            "starting_capital": round(self.starting_capital, 2),
            "generated_at": datetime.now().isoformat(),
            "verdict": self._get_verdict(win_rate / 100),
            "note": "Paper trading results are historical simulation only and do not guarantee future performance.",
        }

        log_dir = Path("logs/paper_trading")
        log_dir.mkdir(parents=True, exist_ok=True)
        report_path = log_dir / "final_report.json"
        with open(report_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Paper trading report saved to {report_path}")
        return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lightweight paper trading CLI")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate the paper-trading report and exit without live trading.",
    )
    args = parser.parse_args()

    trader = PaperTrader()
    if args.report:
        trader.generate_report()
        raise SystemExit(0)

    print("Paper trading mode is active. Use --report to generate a paper-trading summary without live execution.")
