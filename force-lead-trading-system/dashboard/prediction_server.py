"""Small Flask prediction dashboard server.

This module intentionally stays separate from prediction calculations.
It exposes the dashboard HTML page, a JSON prediction endpoint,
and a health endpoint for the existing project's dashboard startup style.
"""

from __future__ import annotations

import copy
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException


app = Flask(__name__)

DASHBOARD_DIR = Path(__file__).resolve().parent
HTML_PAGE = DASHBOARD_DIR / "prediction_dashboard.html"

_STATE_LOCK = threading.RLock()
_STATE: dict[str, Any] = {
    "status": "no_prediction",
    "prediction": None,
    "updated_at": None,
    "error": None,
}


class PredictionStore:
    """In-memory store with thread-safe copy-on-read semantics."""

    @staticmethod
    def get_state() -> dict[str, Any]:
        with _STATE_LOCK:
            data = copy.deepcopy(_STATE)
        if data.get("prediction") is None:
            return {
                "status": "no_prediction",
                "prediction": None,
                "updated_at": None,
                "error": "No prediction data has been received yet.",
            }
        return data

    @staticmethod
    def set_state(data: dict[str, Any]) -> dict[str, Any]:
        payload = PredictionStore.normalize_prediction(data)
        with _STATE_LOCK:
            global _STATE
            _STATE = {
                "status": "ok",
                "prediction": payload,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": None,
            }
        return copy.deepcopy(_STATE)

    @staticmethod
    def clear_state() -> None:
        with _STATE_LOCK:
            global _STATE
            _STATE = {
                "status": "no_prediction",
                "prediction": None,
                "updated_at": None,
                "error": "No prediction data has been received yet.",
            }

    @staticmethod
    def normalize_prediction(payload: Any) -> dict[str, Any]:
        """Normalize arbitrary payloads into a safe prediction object.

        Accepts dictionaries from the application loop. Unknown or malformed
        values are dropped. This prevents accidentally exposing malicious or
        non-JSON objects in the endpoint response.
        """
        if not isinstance(payload, dict):
            raise ValueError("Prediction payload must be a JSON object.")

        # Keep only a safe key subset for the HTML dashboard and external clients.
        safe_fields = {
            "status",
            "symbol",
            "exchange",
            "price",
            "change",
            "changePct",
            "lastUpdate",
            "direction",
            "moveCategory",
            "expectedMovePercent",
            "expectedMovePoints",
            "targetPrice",
            "estimatedDuration",
            "confidence",
            "uncertainty",
            "factors",
            "buyerPressure",
            "sellerPressure",
            "volumeSpeed",
            "leadTimeStatus",
            "reversalVolume",
            "reversalVolumeLabel",
            "mcProbability",
            "mcSamples",
            "alerts",
            "error",
        }

        allowed = {}
        for key in safe_fields:
            if key in payload:
                allowed[key] = payload[key]

        # Minimal validation: direction is optional but must be a string if present.
        if "direction" in allowed and not isinstance(allowed["direction"], str):
            allowed["direction"] = str(allowed["direction"]).upper()

        if "alerts" in allowed:
            if not isinstance(allowed["alerts"], list):
                allowed["alerts"] = [str(allowed["alerts"])]

        if "factors" in allowed:
            if not isinstance(allowed["factors"], list):
                allowed["factors"] = []

        # Normalize status to avoid unsupported server state strings.
        text_status = str(allowed.get("status") or "SIMULATION").upper()
        if text_status not in {"PAPER", "SIMULATION", "LIVE"}:
            allowed["status"] = "SIMULATION"
        else:
            allowed["status"] = text_status

        # Ensure direction is one of the common supported labels safely.
        direction = str(allowed.get("direction") or "UNKNOWN").upper()
        if direction not in {"UP", "DOWN", "BUY", "SELL", "WAIT", "UNKNOWN", "NEUTRAL"}:
            direction = "UNKNOWN"
        allowed["direction"] = direction

        # Remove obvious non-serializable values from nested structures.
        try:
            json.dumps(allowed)
        except TypeError as exc:
            raise ValueError("Prediction payload is not JSON serializable.") from exc

        return allowed


@app.route("/")
def serve_prediction_dashboard() -> Any:
    """Serve the static dashboard HTML file for the root route."""
    try:
        if not HTML_PAGE.exists():
            return jsonify({"status": "error", "error": "Dashboard HTML file is missing."}), 500
        return send_file(HTML_PAGE, mimetype="text/html")
    except Exception as exc:
        return jsonify({"status": "error", "error": "Unable to serve prediction dashboard."}), 500


@app.route("/api/prediction", methods=["GET", "POST", "PUT"])
def prediction_endpoint() -> Any:
    """Return or update the current prediction state safely.

    GET returns the current prediction state.
    POST/PUT receive JSON prediction data from the application and update memory.
    """
    try:
        if request.method == "GET":
            return jsonify(PredictionStore.get_state())

        if not request.is_json:
            return jsonify({"status": "error", "error": "Expected JSON body."}), 400

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"status": "error", "error": "Prediction payload must be a JSON object."}), 400

        try:
            updated = PredictionStore.set_state(payload)
        except (TypeError, ValueError) as exc:
            return jsonify({"status": "error", "error": f"Malformed prediction: {exc}"}), 400

        return jsonify({"status": "ok", "prediction": updated["prediction"], "updated_at": updated["updated_at"]})
    except Exception as exc:
        return jsonify({"status": "error", "error": "Server error while handling prediction payload."}), 500


@app.route("/api/health", methods=["GET"])
def health_endpoint() -> Any:
    """Return a simple health response for the local dashboard service."""
    try:
        state = PredictionStore.get_state()
        return jsonify({
            "status": "ok",
            "service": "prediction_server",
            "framework": "flask",
            "prediction_available": state.get("prediction") is not None,
            "has_prediction": bool(state.get("prediction")),
            "updated_at": state.get("updated_at"),
        })
    except Exception as exc:
        return jsonify({"status": "error", "service": "prediction_server", "error": "Health check failed."}), 500


@app.errorhandler(HTTPException)
def handle_http_error(error: HTTPException) -> Any:
    return jsonify({"status": "error", "error": error.description}), error.code


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception) -> Any:
    # Keep this endpoint safe and non-secret by never exposing traceback internals.
    return jsonify({"status": "error", "error": "Server error while processing request."}), 500


def update_prediction(data: dict[str, Any]) -> dict[str, Any]:
    """Receive prediction data from the application and update the in-memory store.

    This function is separate from the prediction calculations and allows the
    existing main loop to call a single broadcast handler.
    """
    try:
        if not isinstance(data, dict):
            raise ValueError("Prediction data must be a mapping object.")
        return PredictionStore.set_state(data)
    except (TypeError, ValueError) as exc:
        # Preserve server safety. The application may log the exception separately.
        raise ValueError(f"Malformed prediction: {exc}") from exc


def broadcast_prediction(data: dict[str, Any]) -> dict[str, Any]:
    """Compatibility adapter expected by main.py.

    The application passes result dictionaries from main_loop. Save them in the
    in-memory state store and return the normalized safe prediction mapping.
    """
    try:
        return update_prediction(data)
    except Exception:
        # Do not crash the main loop. Keep the server separate from the analysis.
        return PredictionStore.get_state()


def start_dashboard(host: str = "0.0.0.0", port: int = 5000) -> None:
    """Run the Flask server using the same framework already used by the project."""
    app.run(host=host, port=int(port), debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    start_dashboard(host="0.0.0.0", port=5000)
