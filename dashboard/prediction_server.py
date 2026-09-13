from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file
from flask_socketio import SocketIO, emit

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "force-lead-trading-system"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from config.settings import settings
from data.historical import HistoricalData
from statistics.backtester import Backtester

app = Flask(__name__, template_folder="templates")
socketio = SocketIO(app, cors_allowed_origins="*")

PROJECT_LOG_DIR = PACKAGE_ROOT / "logs"
PAPER_LOG_DIR = ROOT / "logs" / "paper_trading"
ENV_FILE = ROOT / ".env"
LATEST_PREDICTION: dict[str, Any] | None = None

ALLOWED_SETTINGS = {
    "STOCK_SYMBOL": (str, lambda value: bool(re.fullmatch(r"[A-Za-z0-9._-]{1,30}", value))),
    "TIMEFRAME": (int, lambda value: 1 <= value <= 1440),
    "CAPITAL": (float, lambda value: math.isfinite(value) and 1 <= value <= 1_000_000_000),
    "STOP_LOSS_PCT": (float, lambda value: 0 <= value <= 1),
    "TARGET_PCT": (float, lambda value: 0 <= value <= 1),
    "MAX_POSITION_PCT": (float, lambda value: 0 <= value <= 1),
    "MAX_DAILY_LOSS_PCT": (float, lambda value: 0 <= value <= 1),
    "MAX_TRADES_PER_DAY": (int, lambda value: 1 <= value <= 1000),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_candles() -> list[dict[str, Any]]:
    cache = PACKAGE_ROOT / "logs" / "historical_cache.csv"
    if cache.exists():
        try:
            with cache.open("r", newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
        except (OSError, csv.Error):
            pass
    try:
        return HistoricalData(None, symbol=settings.stock_symbol, limit=200).fetch_last_200_candles()
    except Exception:
        return []


def _load_weights() -> dict[str, float]:
    fallback = {"f1": .25, "f2": .20, "f3": .10, "f4": .20, "f5": .10, "f6": .05, "f7": .05, "f8": .03, "f9": .02}
    path = PACKAGE_ROOT / "config" / "optimal_weights.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return {key: float(value.get(key, fallback[key])) for key in fallback}
        if isinstance(value, list) and len(value) >= 9:
            return {f"f{index}": float(item) for index, item in enumerate(value[:9], 1)}
    except (OSError, ValueError, TypeError):
        pass
    return fallback


def _setting_value(name: str, raw: Any) -> Any:
    expected, valid = ALLOWED_SETTINGS[name]
    if isinstance(raw, bool):
        raise ValueError(f"{name} has an invalid type")
    try:
        value = expected(raw) if expected is not str else str(raw).strip()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} has an invalid type") from exc
    if not valid(value):
        raise ValueError(f"{name} has an unsafe or out-of-range value")
    return value


def _save_env(values: dict[str, Any]) -> None:
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines(True) if ENV_FILE.exists() else []
    seen: set[str] = set()
    output: list[str] = []
    for line in lines:
        match = re.match(r"^\s*([A-Z][A-Z0-9_]*)\s*=", line)
        key = match.group(1) if match else None
        if key in values:
            output.append(f"{key}={values[key]}\n")
            seen.add(key)
        else:
            output.append(line)
    for key, value in values.items():
        if key not in seen:
            output.append(f"{key}={value}\n")
    ENV_FILE.write_text("".join(output), encoding="utf-8")


def _log_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = (("TRADE", PAPER_LOG_DIR / "trade_log.csv"), ("INFO", PAPER_LOG_DIR / "signal_log.csv"), ("ERROR", PACKAGE_ROOT / "logs" / "errors.log"))
    for level, path in sources:
        if not path.exists():
            continue
        try:
            if path.suffix == ".csv":
                with path.open("r", newline="", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        rows.append({"level": level, "timestamp": row.get("timestamp", ""), "message": json.dumps(row, default=str)})
            else:
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.strip():
                        rows.append({"level": "ERROR", "timestamp": line[:24], "message": line})
        except (OSError, UnicodeError, csv.Error):
            continue
    return rows[-500:]


@app.route("/")
def index():
    return send_file(Path(__file__).resolve().parent / "templates" / "index.html", mimetype="text/html")


@app.post("/api/settings/save")
def save_settings():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or not payload:
        return jsonify({"ok": False, "error": "Expected a non-empty JSON object."}), 400
    unknown = set(payload) - set(ALLOWED_SETTINGS)
    if unknown:
        return jsonify({"ok": False, "error": "Unsupported setting."}), 400
    try:
        values = {key: _setting_value(key, value) for key, value in payload.items()}
        _save_env(values)
        for key, value in values.items():
            os.environ[key] = str(value)
            setattr(settings, key, value)
            setattr(settings, key.lower(), value)
        return jsonify({"ok": True, "settings": {key: values[key] for key in values}})
    except (OSError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@app.post("/api/backtest/run")
def run_backtest():
    payload = request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Expected a JSON object."}), 400
    try:
        capital = float(payload.get("capital", settings.CAPITAL))
        if not math.isfinite(capital) or capital <= 0:
            raise ValueError("capital must be greater than zero")
        candles = _read_candles()
        metrics = Backtester.run_full_backtest(candles, _load_weights(), capital=capital)
        final_value = float(metrics.get("final_capital", capital))
        result = _json_safe({
            "metrics": metrics,
            "equity_curve": [capital, final_value],
            "trades": [],
            "start_value": capital,
            "final_value": final_value,
            "data_points": len(candles),
            "note": "Equity/trade detail is unavailable from the repository Backtester API; summary metrics are authoritative.",
        })
        return jsonify({"ok": True, "result": result})
    except (TypeError, ValueError, OSError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception:
        return jsonify({"ok": False, "error": "Backtest unavailable."}), 500


@app.get("/api/system/status")
def system_status():
    return jsonify({"connected": LATEST_PREDICTION is not None, "mode": os.getenv("TRADING_MODE", "paper").upper(), "symbol": os.getenv("STOCK_SYMBOL", settings.STOCK_SYMBOL), "capital": float(os.getenv("CAPITAL", settings.CAPITAL)), "status": "PAPER / SIMULATION"})


@app.get("/api/logs")
def logs():
    return jsonify({"logs": _log_rows(), "count": len(_log_rows())})


@socketio.on("connect")
def handle_connect():
    emit("status", {"connected": True, "mode": "PAPER / SIMULATION", "msg": "Connected to Trading System"})
    if LATEST_PREDICTION is not None:
        emit("signal_update", _json_safe(LATEST_PREDICTION))


@socketio.on("request_logs")
def request_logs():
    emit("logs_update", {"logs": _log_rows()})


def push_update(signal_data: dict[str, Any]) -> None:
    global LATEST_PREDICTION
    if not isinstance(signal_data, dict):
        return
    LATEST_PREDICTION = dict(signal_data)
    socketio.emit("signal_update", _json_safe(LATEST_PREDICTION))


def update_prediction(signal_data: dict[str, Any]) -> None:
    push_update(signal_data)


def start_dashboard(host="0.0.0.0", port=5000):
    socketio.run(app, host=host, port=int(port), debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    start_dashboard()
