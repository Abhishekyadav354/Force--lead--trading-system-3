from __future__ import annotations

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    emit("status", {"msg": "Connected to Trading System"})


def push_update(signal_data):
    socketio.emit("signal_update", signal_data)


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5000)
