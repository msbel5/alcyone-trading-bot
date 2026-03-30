#!/usr/bin/env python3
"""
Trading Dashboard — lightweight HTTP server on Pi.
Serves real-time portfolio info at http://alcyone:8080
Auto-refreshes every 5 minutes.
"""
import sys
import os
import json
import time
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

log = logging.getLogger("dashboard")

PORT = 8080
LOG_DIR = Path("/home/msbel/.openclaw/workspace/trading/logs")
DASHBOARD_FILE = LOG_DIR / "dashboard_state.json"


def _read_state() -> dict:
    """Read latest bot state from shared JSON file."""
    try:
        if DASHBOARD_FILE.exists():
            with open(DASHBOARD_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _build_html(state: dict) -> str:
    """Build dashboard HTML from state."""
    positions = state.get("positions", {})
    equity = state.get("equity_curve", [])
    total_pnl = state.get("total_pnl", 0)
    daily_pnl = state.get("daily_pnl", 0)
    last_update = state.get("last_update", "unknown")
    uptime = state.get("uptime", "unknown")

    # Sparkline SVG
    recent = equity[-96:] if len(equity) > 96 else equity  # 48h at 5min = 576, show last 8h
    if recent and len(recent) > 2:
        min_eq = min(recent)
        max_eq = max(recent)
        rng = max_eq - min_eq if max_eq > min_eq else 1
        w = 600
        h = 100
        points = " ".join(f"{int(i * w / len(recent))},{h - int((v - min_eq) / rng * (h - 10) + 5)}" for i, v in enumerate(recent))
        color = "#4CAF50" if recent[-1] >= recent[0] else "#f44336"
        sparkline = f'<svg width="{w}" height="{h}" style="background:#1a1a2e;border-radius:8px;"><polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/></svg>'
    else:
        sparkline = '<p style="color:#666;">Collecting data...</p>'

    # Position rows
    pos_html = ""
    open_count = 0
    for sym, data in positions.items():
        if not isinstance(data, dict):
            continue
        coin = sym.replace("USDT", "")
        amount = data.get("amount", 0)
        value = data.get("value", 0)
        pnl = data.get("pnl_pct", 0)
        entry = data.get("entry_price", 0)
        sl = data.get("trailing_sl", 0)

        if amount > 0:
            open_count += 1
            color = "#4CAF50" if pnl >= 0 else "#f44336"
            sl_str = f"${sl:,.0f}" if sl else "-"
            pos_html += f"""<tr>
                <td><strong>{coin}</strong></td>
                <td>${value:.2f}</td>
                <td style="color:{color}">{pnl:+.2f}%</td>
                <td>${entry:,.2f}</td>
                <td>{sl_str}</td>
            </tr>"""
        else:
            pos_html += f'<tr><td>{coin}</td><td colspan="4" style="color:#555">idle</td></tr>'

    if not pos_html:
        pos_html = '<tr><td colspan="5" style="color:#555">No positions</td></tr>'

    # Model status
    models = state.get("models", {})
    model_html = ""
    for name, info in models.items():
        status = info.get("status", "unknown")
        color = "#4CAF50" if status == "active" else "#f44336"
        model_html += f'<span style="color:{color};margin-right:12px;">{name}: {status}</span>'

    pnl_color = "#4CAF50" if total_pnl >= 0 else "#f44336"
    daily_color = "#4CAF50" if daily_pnl >= 0 else "#f44336"

    return f"""<!DOCTYPE html>
<html><head>
<title>Alcyone Trading Dashboard</title>
<meta charset="utf-8">
<meta http-equiv="refresh" content="300">
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Courier New', monospace; background: #0d1117; color: #c9d1d9; padding: 24px; }}
h1 {{ color: #58a6ff; margin-bottom: 8px; font-size: 20px; }}
h2 {{ color: #8b949e; font-size: 14px; margin: 16px 0 8px; text-transform: uppercase; letter-spacing: 1px; }}
.metrics {{ display: flex; gap: 24px; margin: 16px 0; flex-wrap: wrap; }}
.metric {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px 24px; min-width: 140px; }}
.metric .value {{ font-size: 28px; font-weight: bold; }}
.metric .label {{ font-size: 11px; color: #8b949e; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; background: #161b22; border-radius: 8px; overflow: hidden; }}
th {{ background: #21262d; padding: 10px 12px; text-align: left; font-size: 12px; color: #8b949e; }}
td {{ padding: 10px 12px; border-bottom: 1px solid #21262d; font-size: 13px; }}
.footer {{ margin-top: 24px; font-size: 11px; color: #484f58; }}
.models {{ margin: 12px 0; font-size: 12px; }}
</style>
</head><body>
<h1>Alcyone Trading Bot</h1>
<p style="color:#484f58;font-size:12px;">7 coins | 6-layer ML strategy | Raspberry Pi 5 | Updated: {last_update}</p>

<div class="metrics">
    <div class="metric"><div class="value" style="color:{pnl_color}">${total_pnl:+.2f}</div><div class="label">Total PnL</div></div>
    <div class="metric"><div class="value" style="color:{daily_color}">${daily_pnl:+.2f}</div><div class="label">Today</div></div>
    <div class="metric"><div class="value">{open_count}/7</div><div class="label">Positions</div></div>
    <div class="metric"><div class="value">{uptime}</div><div class="label">Uptime</div></div>
</div>

<h2>Equity Curve</h2>
{sparkline}

<h2>Positions</h2>
<table>
<tr><th>Coin</th><th>Value</th><th>PnL</th><th>Entry</th><th>Trail SL</th></tr>
{pos_html}
</table>

<h2>Models</h2>
<div class="models">{model_html if model_html else '<span style="color:#555">Loading...</span>'}</div>

<div class="footer">
    Auto-refresh every 5 minutes | <a href="/api/state" style="color:#58a6ff;">JSON API</a> |
    Alcyone on Pi 5 | github.com/msbel5/alcyone-trading-bot
</div>
</body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        state = _read_state()
        if self.path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(state, indent=2).encode())
        else:
            html = _build_html(state)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())

    def log_message(self, format, *args):
        pass  # Suppress access logs


def start_dashboard(port=PORT):
    """Start dashboard server in a background thread."""
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info(f"Dashboard running at http://0.0.0.0:{port}")
    return server


def update_dashboard_state(positions: dict, equity_curve: list,
                            total_pnl: float, daily_pnl: float,
                            models: dict = None, uptime: str = ""):
    """Write current state to JSON for dashboard to read."""
    state = {
        "positions": positions,
        "equity_curve": equity_curve[-200:],  # Keep last 200 points
        "total_pnl": total_pnl,
        "daily_pnl": daily_pnl,
        "models": models or {},
        "uptime": uptime,
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(DASHBOARD_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        log.debug(f"Dashboard state write failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Starting dashboard on port {PORT}...")
    print(f"Open http://localhost:{PORT} or http://alcyone:{PORT}")
    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    server.serve_forever()
