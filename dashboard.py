#!/usr/bin/env python3
"""
Dashboard v5 — Interactive trading dashboard.
- Clickable coin selector (selected = orange)
- Positions + equity side by side at top
- All indicators update per selected coin
- 5s soft AJAX refresh, no page flash
- Real-time prices next to each coin
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

PORT = 8085
LOG_DIR = Path("/home/msbel/.openclaw/workspace/trading/logs")
DASHBOARD_FILE = LOG_DIR / "dashboard_state.json"


def _read_state() -> dict:
    try:
        if DASHBOARD_FILE.exists():
            with open(DASHBOARD_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _build_html(state: dict) -> str:
    positions = state.get("positions", {})
    equity = state.get("equity_curve", [])
    total_pnl = state.get("total_pnl", 0)
    daily_pnl = state.get("daily_pnl", 0)
    last_update = state.get("last_update", "unknown")
    uptime = state.get("uptime", "unknown")
    regime = state.get("regime", "unknown")
    vol_regime = state.get("vol_regime", "unknown")
    hurst = state.get("hurst", 0.5)
    risk = state.get("risk_metrics", {})
    ind = state.get("indicator_summary", {})
    pat = state.get("patterns", {})
    vol = state.get("volatility", {})
    stat = state.get("statistical", {})
    models = state.get("models", {})

    # Colors
    pnl_c = "#4CAF50" if total_pnl >= 0 else "#f44336"
    daily_c = "#4CAF50" if daily_pnl >= 0 else "#f44336"
    regime_colors = {"trending": "#4CAF50", "sideways": "#FF9800", "volatile": "#f44336"}
    vol_colors = {"low_vol": "#2196F3", "normal": "#4CAF50", "high_vol": "#FF9800", "extreme": "#f44336"}
    regime_c = regime_colors.get(regime, "#666")
    vol_c = vol_colors.get(vol_regime, "#666")
    hurst_c = "#4CAF50" if hurst > 0.6 else ("#2196F3" if hurst < 0.4 else "#FF9800")
    hurst_label = "Trending" if hurst > 0.6 else ("Mean-Revert" if hurst < 0.4 else "Random")

    # Sparkline
    recent = equity[-150:] if len(equity) > 150 else equity
    sparkline_svg = ""
    if recent and len(recent) > 2:
        mn, mx = min(recent), max(recent)
        rng = mx - mn if mx > mn else 1
        w, h = 100, 100  # percentage-based
        pts = " ".join(f"{i*100/len(recent):.1f},{100 - (v-mn)/rng*90 - 5:.1f}" for i, v in enumerate(recent))
        color = "#4CAF50" if recent[-1] >= recent[0] else "#f44336"
        sparkline_svg = f'<svg viewBox="0 0 100 100" preserveAspectRatio="none" style="width:100%;height:120px;background:#1a1a2e;border-radius:8px;"><polyline points="{pts}" fill="none" stroke="{color}" stroke-width="0.5" vector-effect="non-scaling-stroke"/></svg>'
    else:
        sparkline_svg = '<div style="height:120px;background:#1a1a2e;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#555;font-size:12px;">Collecting data...</div>'

    # Coin list with prices (for JS)
    coins_json = json.dumps({
        sym.replace("USDT", ""): {
            "symbol": sym,
            "amount": d.get("amount", 0) if isinstance(d, dict) else 0,
            "value": round(d.get("value", 0), 2) if isinstance(d, dict) else 0,
            "pnl_pct": round(d.get("pnl_pct", 0), 2) if isinstance(d, dict) else 0,
            "entry": round(d.get("entry_price", 0), 2) if isinstance(d, dict) else 0,
            "sl": round(d.get("trailing_sl", 0), 2) if isinstance(d, dict) else 0,
        }
        for sym, d in positions.items()
    })

    # Position rows
    coin_rows = ""
    for sym, data in positions.items():
        if not isinstance(data, dict):
            continue
        coin = sym.replace("USDT", "")
        amount = data.get("amount", 0)
        pnl = data.get("pnl_pct", 0)
        entry = data.get("entry_price", 0)

        if amount > 0:
            pc = "#4CAF50" if pnl >= 0 else "#f44336"
            status = f'<span style="color:{pc}">{pnl:+.2f}%</span>'
        else:
            status = '<span style="color:#555">idle</span>'

        coin_rows += f'''<div class="coin-row" data-coin="{coin}" onclick="selectCoin('{coin}')">
            <span class="coin-name">{coin}</span>
            <span class="coin-status">{status}</span>
        </div>'''

    # Indicator cards builder
    def ind_card(name, signal, detail=""):
        s = float(signal)
        c = "#4CAF50" if s > 0.1 else ("#f44336" if s < -0.1 else "#8b949e")
        return f'<div class="card-sm"><div class="cv" style="color:{c}">{s:+.2f}</div><div class="cl">{name}</div><div class="cd">{detail}</div></div>'

    indicator_cards = ""
    if ind:
        indicator_cards += ind_card("Ichimoku", ind.get("ichimoku", {}).get("signal", 0),
                                     f"Cloud: {ind.get('ichimoku', {}).get('cloud', 0):.4f}")
        indicator_cards += ind_card("MFI", ind.get("mfi", {}).get("signal", 0),
                                     f"Val: {ind.get('mfi', {}).get('value', 50):.0f}")
        indicator_cards += ind_card("VWAP", ind.get("vwap", {}).get("signal", 0),
                                     f"Dist: {ind.get('vwap', {}).get('distance', 0):+.3f}")
        indicator_cards += ind_card("Keltner", ind.get("keltner", {}).get("signal", 0),
                                     "SQUEEZE!" if ind.get("keltner", {}).get("squeeze") else "Open")
        indicator_cards += ind_card("Williams", ind.get("williams_r", {}).get("signal", 0),
                                     f"%R: {ind.get('williams_r', {}).get('value', -50):.0f}")
        indicator_cards += ind_card("CMF", ind.get("cmf", {}).get("signal", 0),
                                     f"Val: {ind.get('cmf', {}).get('value', 0):+.3f}")
        indicator_cards += ind_card("CCI", ind.get("cci", {}).get("signal", 0),
                                     f"Val: {ind.get('cci', {}).get('value', 0):.0f}")
        indicator_cards += ind_card("STC", ind.get("stc", {}).get("signal", 0),
                                     f"Val: {ind.get('stc', {}).get('value', 50):.0f}")
        indicator_cards += ind_card("Donchian", ind.get("donchian", {}).get("signal", 0), "Breakout")
        composite = ind.get("composite", 0)
        cc = "#4CAF50" if composite > 0.1 else ("#f44336" if composite < -0.1 else "#FF9800")
        indicator_cards += f'<div class="card-sm" style="border-color:{cc}"><div class="cv" style="color:{cc}">{composite:+.3f}</div><div class="cl">COMPOSITE</div><div class="cd">All combined</div></div>'
    else:
        indicator_cards = '<div class="card-sm"><div class="cv" style="color:#555">--</div><div class="cl">Computing...</div></div>'

    # Pattern section
    detected = pat.get("detected", [])
    pat_sig = pat.get("signal", 0)
    pat_conf = pat.get("confluence", 0)
    swing = pat.get("swing_position", 0.5)
    swing_label = "Near High" if swing > 0.7 else ("Near Low" if swing < 0.3 else "Mid")
    pat_color = "#4CAF50" if pat_sig > 0.1 else ("#f44336" if pat_sig < -0.1 else "#8b949e")
    pattern_html = ", ".join(detected) if detected else "None detected"

    # Vol section
    def vol_card(name, value, detail="", special=""):
        return f'<div class="card-sm"><div class="cv" {special}>{value}</div><div class="cl">{name}</div><div class="cd">{detail}</div></div>'

    vol_cards = ""
    if vol:
        vol_cards += vol_card("EWMA", f"{vol.get('ewma',0)*100:.3f}%", "RiskMetrics")
        vol_cards += vol_card("Parkinson", f"{vol.get('parkinson',0)*100:.3f}%", "Range-based")
        pct = vol.get('cone_pct', 50)
        pc = 'style="color:#f44336"' if pct > 75 else ('style="color:#2196F3"' if pct < 25 else "")
        vol_cards += vol_card("Vol Cone", f"p{pct:.0f}", "Percentile", pc)
        vol_cards += vol_card("Ratio", f"{vol.get('ratio',1):.2f}x", "Short/Long")
        sq = vol.get("squeeze", False)
        sq_style = 'style="color:#FF9800;font-weight:bold"' if sq else ""
        vol_cards += vol_card("Squeeze", "YES" if sq else "NO", "BB in Keltner", sq_style)
        vol_cards += vol_card("Term", f"{vol.get('term_structure',1):.2f}", "Short vs Long")

    # Stat section
    stat_cards = ""
    if stat:
        h = stat.get("hurst", 0.5)
        hl = "Trending" if h > 0.6 else ("Mean-Rev" if h < 0.4 else "Random")
        hc = "#4CAF50" if h > 0.6 else ("#2196F3" if h < 0.4 else "#FF9800")
        stat_cards += vol_card("Hurst", f"{h:.3f}", hl, f'style="color:{hc}"')
        stat_cards += vol_card("GARCH", f"{stat.get('garch_vol',0)*100:.3f}%", "Forecast")
        z = stat.get("zscore", 0)
        zc = "#4CAF50" if abs(z) > 2 else ""
        stat_cards += vol_card("Z-Score", f"{z:+.2f}", "Mean reversion", f'style="color:{zc}"' if zc else "")
        stat_cards += vol_card("Entropy", f"{stat.get('entropy',0.5):.3f}", "Disorder")
        stat_cards += vol_card("EGARCH", f"{stat.get('egarch_asym',0):+.3f}", "Asymmetry")

    # Model status
    model_items = ""
    for name, info in models.items():
        status = info.get("status", "?")
        mc = "#4CAF50" if status == "active" else "#f44336"
        model_items += f'<span class="model-tag" style="border-color:{mc};color:{mc}">{name}</span>'

    # Risk
    sqn = risk.get("sqn", 0)
    sqn_label = "Poor" if sqn < 1.7 else ("Avg" if sqn < 2.5 else ("Good" if sqn < 3 else "Excel"))
    sqn_c = "#f44336" if sqn < 1.7 else ("#FF9800" if sqn < 2.5 else "#4CAF50")

    return f"""<!DOCTYPE html>
<html><head>
<title>Alcyone v4</title>
<meta charset="utf-8">
<script src="https://unpkg.com/lightweight-charts@4.1.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
:root {{ --bg: #0d1117; --card: #161b22; --border: #30363d; --text: #c9d1d9; --dim: #484f58; --accent: #58a6ff; }}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, 'Segoe UI', monospace; background: var(--bg); color: var(--text); font-size: 13px; }}
.container {{ max-width: 1600px; margin: 0 auto; padding: 16px; }}

/* Header */
.header {{ display:flex; align-items:center; gap:12px; margin-bottom:12px; flex-wrap:wrap; }}
.header h1 {{ color: var(--accent); font-size: 16px; white-space:nowrap; }}
.badge {{ padding:2px 8px; border-radius:10px; font-size:10px; font-weight:bold; color:#fff; }}
.header-info {{ font-size:10px; color:var(--dim); }}

/* Top metrics row */
.metrics {{ display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; }}
.metric {{ background:var(--card); border:1px solid var(--border); border-radius:6px; padding:10px 16px; min-width:120px; flex:1; }}
.metric .mv {{ font-size:20px; font-weight:bold; }}
.metric .ml {{ font-size:9px; color:var(--dim); margin-top:2px; }}

/* Main layout: coins left, chart right */
.main {{ display:grid; grid-template-columns: 220px 1fr; gap:12px; margin-bottom:12px; }}

/* Coin selector */
.coin-list {{ background:var(--card); border:1px solid var(--border); border-radius:8px; overflow:hidden; }}
.coin-list-header {{ background:#21262d; padding:8px 12px; font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:1px; }}
.coin-row {{ display:flex; justify-content:space-between; align-items:center; padding:10px 12px; cursor:pointer; border-bottom:1px solid #21262d; transition: background 0.15s; }}
.coin-row:hover {{ background:#21262d; }}
.coin-row.selected {{ background:#1a2332; border-left:3px solid #FF9800; }}
.coin-row.selected .coin-name {{ color:#FF9800; font-weight:bold; }}
.coin-name {{ font-size:14px; font-weight:600; }}
.coin-status {{ font-size:12px; }}

/* Chart area */
.chart-area {{ background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px; }}
.chart-title {{ font-size:11px; color:var(--dim); text-transform:uppercase; margin-bottom:8px; }}

/* Section headers */
.section {{ margin-bottom:12px; }}
.section h2 {{ font-size:11px; color:var(--dim); text-transform:uppercase; letter-spacing:1px; margin-bottom:6px; padding-left:4px; }}

/* Small cards grid */
.cards {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap:6px; }}
.card-sm {{ background:var(--card); border:1px solid var(--border); border-radius:6px; padding:8px 10px; text-align:center; }}
.card-sm .cv {{ font-size:16px; font-weight:bold; }}
.card-sm .cl {{ font-size:9px; color:var(--dim); margin-top:1px; }}
.card-sm .cd {{ font-size:8px; color:#383d44; margin-top:1px; }}

/* Patterns bar */
.pat-bar {{ background:var(--card); border:1px solid var(--border); border-radius:6px; padding:8px 12px; font-size:11px; }}

/* Models */
.model-tags {{ display:flex; flex-wrap:wrap; gap:6px; }}
.model-tag {{ border:1px solid; border-radius:12px; padding:2px 8px; font-size:10px; }}

/* Footer */
.footer {{ text-align:center; font-size:9px; color:var(--dim); margin-top:12px; padding:8px; }}
.footer a {{ color:var(--accent); }}

/* Risk row */
.risk-row {{ display:grid; grid-template-columns: repeat(6, 1fr); gap:6px; }}
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<div class="header">
  <h1>Alcyone Trading Bot v4</h1>
  <span class="badge" style="background:{regime_c}">{regime}</span>
  <span class="badge" style="background:{vol_c}">{vol_regime}</span>
  <span class="badge" style="background:{hurst_c}">H={hurst:.2f} {hurst_label}</span>
  <span class="header-info">7 coins | 9 layers | 46 features | {len(models)} models | {last_update}</span>
</div>

<!-- Top Metrics -->
<div class="metrics">
  <div class="metric"><div class="mv" style="color:{pnl_c}">${total_pnl:+.2f}</div><div class="ml">Total PnL</div></div>
  <div class="metric"><div class="mv" style="color:{daily_c}">${daily_pnl:+.2f}</div><div class="ml">Today</div></div>
  <div class="metric"><div class="mv">{sum(1 for d in positions.values() if isinstance(d,dict) and d.get('amount',0)>0)}/7</div><div class="ml">Positions</div></div>
  <div class="metric"><div class="mv">{uptime}</div><div class="ml">Uptime</div></div>
  <div class="metric"><div class="mv" style="color:{sqn_c}">{sqn:.2f}</div><div class="ml">SQN ({sqn_label})</div></div>
  <div class="metric"><div class="mv">{risk.get('sortino',0):.2f}</div><div class="ml">Sortino</div></div>
  <div class="metric"><div class="mv">{risk.get('omega',0):.2f}</div><div class="ml">Omega</div></div>
  <div class="metric"><div class="mv">{risk.get('profit_factor',0):.2f}</div><div class="ml">Profit F.</div></div>
</div>

<!-- Main: Coins + Charts -->
<div style="display:grid;grid-template-columns:180px 1fr 1fr;gap:10px;margin-bottom:12px;">
  <div class="coin-list">
    <div class="coin-list-header">Positions</div>
    {coin_rows}
  </div>
  <div class="chart-area">
    <div class="chart-title">Candle Chart <span style="color:#FF9800" id="chart-coin-label">(BTC)</span></div>
    <div id="candle-chart" style="height:280px;background:#1a1a2e;border-radius:8px;"></div>
  </div>
  <div class="chart-area">
    <div class="chart-title">Equity Curve</div>
    {sparkline_svg}
  </div>
</div>

<!-- Indicators -->
<div class="section">
  <h2>Indicators <span style="color:#FF9800" id="selected-coin-label">(BTC)</span></h2>
  <div class="cards" id="indicators-grid">{indicator_cards}</div>
</div>

<!-- Patterns -->
<div class="section">
  <h2>Candlestick Patterns</h2>
  <div class="pat-bar" id="patterns-bar">
    <span style="color:{pat_color};font-weight:bold">Signal: {pat_sig:+.2f}</span> |
    Confluence: {pat_conf:+.2f} |
    Swing: {swing:.2f} ({swing_label}) |
    <span style="color:var(--dim)">Patterns: {pattern_html}</span>
  </div>
</div>

<!-- Volatility + Statistical side by side -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
  <div class="section">
    <h2>Volatility</h2>
    <div class="cards" id="vol-grid">{vol_cards if vol_cards else '<div class="card-sm"><div class="cv" style="color:#555">--</div><div class="cl">Computing</div></div>'}</div>
  </div>
  <div class="section">
    <h2>Statistical Models</h2>
    <div class="cards" id="stat-grid">{stat_cards if stat_cards else '<div class="card-sm"><div class="cv" style="color:#555">--</div><div class="cl">Computing</div></div>'}</div>
  </div>
</div>

<!-- Models -->
<div class="section">
  <h2>Active Models</h2>
  <div class="model-tags">{model_items}</div>
</div>

<div class="footer">
  Live refresh 5s | <a href="/api/state">JSON API</a> |
  v4: Ichimoku, GARCH, Hurst, CNN-LSTM, CPCV, Boruta, 46 features | Pi 5
</div>

</div><!-- /container -->

<script>
var selectedCoin = 'BTC';
var allCoinData = {{}};
var chart = null;
var candleSeries = null;

// Initialize TradingView chart
function initChart() {{
  var container = document.getElementById('candle-chart');
  if (!container || !window.LightweightCharts) return;
  chart = LightweightCharts.createChart(container, {{
    width: container.clientWidth,
    height: 280,
    layout: {{ background: {{ color: '#1a1a2e' }}, textColor: '#8b949e' }},
    grid: {{ vertLines: {{ color: '#21262d' }}, horzLines: {{ color: '#21262d' }} }},
    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
    timeScale: {{ timeVisible: true, borderColor: '#30363d' }},
    rightPriceScale: {{ borderColor: '#30363d' }},
  }});
  candleSeries = chart.addCandlestickSeries({{
    upColor: '#4CAF50', downColor: '#f44336',
    borderUpColor: '#4CAF50', borderDownColor: '#f44336',
    wickUpColor: '#4CAF50', wickDownColor: '#f44336',
  }});
}}

function updateChart(coin) {{
  var d = allCoinData[coin];
  if (!d || !d.candles || !candleSeries) return;
  candleSeries.setData(d.candles);
  chart.timeScale().fitContent();
  var label = document.getElementById('chart-coin-label');
  if (label) label.textContent = '(' + coin + ')';
}}

function sigColor(v) {{ return v > 0.1 ? '#4CAF50' : (v < -0.1 ? '#f44336' : '#8b949e'); }}

function makeCard(name, signal, detail) {{
  var c = sigColor(signal);
  return '<div class="card-sm"><div class="cv" style="color:' + c + '">' +
    (signal >= 0 ? '+' : '') + signal.toFixed(2) + '</div><div class="cl">' +
    name + '</div><div class="cd">' + (detail||'') + '</div></div>';
}}

function updateCoinDisplay(coin) {{
  var d = allCoinData[coin];
  if (!d) return;

  // Update label
  var label = document.getElementById('selected-coin-label');
  if (label) label.textContent = '(' + coin + ')';

  // Indicators
  var ind = d.indicators || {{}};
  var cards = '';
  if (ind.ichimoku) cards += makeCard('Ichimoku', ind.ichimoku.signal, 'Cloud: ' + (ind.ichimoku.cloud||0).toFixed(4));
  if (ind.mfi) cards += makeCard('MFI', ind.mfi.signal, 'Val: ' + (ind.mfi.value||50).toFixed(0));
  if (ind.vwap) cards += makeCard('VWAP', ind.vwap.signal, 'Dist: ' + (ind.vwap.distance||0).toFixed(3));
  if (ind.keltner) cards += makeCard('Keltner', ind.keltner.signal, ind.keltner.squeeze ? 'SQUEEZE!' : 'Open');
  if (ind.williams_r) cards += makeCard('Williams', ind.williams_r.signal, '%R: ' + (ind.williams_r.value||-50).toFixed(0));
  if (ind.cmf) cards += makeCard('CMF', ind.cmf.signal, 'Val: ' + (ind.cmf.value||0).toFixed(3));
  if (ind.cci) cards += makeCard('CCI', ind.cci.signal, 'Val: ' + (ind.cci.value||0).toFixed(0));
  if (ind.stc) cards += makeCard('STC', ind.stc.signal, 'Val: ' + (ind.stc.value||50).toFixed(0));
  if (ind.donchian) cards += makeCard('Donchian', ind.donchian.signal, 'Breakout');
  var comp = ind.composite || 0;
  var cc = sigColor(comp);
  cards += '<div class="card-sm" style="border-color:' + cc + '"><div class="cv" style="color:' + cc + '">' +
    (comp >= 0 ? '+' : '') + comp.toFixed(3) + '</div><div class="cl">COMPOSITE</div><div class="cd">All combined</div></div>';

  var indEl = document.getElementById('indicators-grid');
  if (indEl) indEl.innerHTML = cards;

  // Patterns
  var pat = d.patterns || {{}};
  var patEl = document.getElementById('patterns-bar');
  if (patEl) {{
    var ps = pat.signal || 0;
    var pc = sigColor(ps);
    var detected = (pat.detected && pat.detected.length) ? pat.detected.join(', ') : 'None detected';
    var sw = pat.swing_position || 0.5;
    var swl = sw > 0.7 ? 'Near High' : (sw < 0.3 ? 'Near Low' : 'Mid');
    patEl.innerHTML = '<span style="color:' + pc + ';font-weight:bold">Signal: ' + (ps>=0?'+':'') + ps.toFixed(2) + '</span> | ' +
      'Confluence: ' + (pat.confluence>=0?'+':'') + (pat.confluence||0).toFixed(2) + ' | ' +
      'Swing: ' + sw.toFixed(2) + ' (' + swl + ') | ' +
      '<span style="color:var(--dim)">Patterns: ' + detected + '</span>';
  }}

  // Volatility
  var vol = d.volatility || {{}};
  var volEl = document.getElementById('vol-grid');
  if (volEl && vol.ewma !== undefined) {{
    var vc = '';
    vc += makeCard('EWMA', 0, (vol.ewma*100).toFixed(3) + '%');
    vc += makeCard('Parkinson', 0, (vol.parkinson*100).toFixed(3) + '%');
    var cpct = vol.cone_pct || 50;
    vc += '<div class="card-sm"><div class="cv" style="color:' + (cpct>75?'#f44336':(cpct<25?'#2196F3':'#c9d1d9')) + '">p' + cpct.toFixed(0) + '</div><div class="cl">Vol Cone</div></div>';
    vc += makeCard('Ratio', 0, vol.ratio.toFixed(2) + 'x');
    vc += '<div class="card-sm"><div class="cv" style="color:' + (vol.squeeze?'#FF9800':'#c9d1d9') + '">' + (vol.squeeze?'YES':'NO') + '</div><div class="cl">Squeeze</div></div>';
    vc += makeCard('Term', 0, (vol.term_structure||1).toFixed(2));
    volEl.innerHTML = vc;
  }}

  // Statistical
  var stat = d.statistical || {{}};
  var statEl = document.getElementById('stat-grid');
  if (statEl && stat.hurst !== undefined) {{
    var sc = '';
    var h = stat.hurst;
    var hl = h > 0.6 ? 'Trending' : (h < 0.4 ? 'Mean-Rev' : 'Random');
    var hc = h > 0.6 ? '#4CAF50' : (h < 0.4 ? '#2196F3' : '#FF9800');
    sc += '<div class="card-sm"><div class="cv" style="color:' + hc + '">' + h.toFixed(3) + '</div><div class="cl">Hurst</div><div class="cd">' + hl + '</div></div>';
    sc += makeCard('GARCH', 0, (stat.garch_vol*100).toFixed(3) + '%');
    var z = stat.zscore || 0;
    sc += '<div class="card-sm"><div class="cv" style="color:' + (Math.abs(z)>2?'#4CAF50':'#c9d1d9') + '">' + (z>=0?'+':'') + z.toFixed(2) + '</div><div class="cl">Z-Score</div><div class="cd">Mean reversion</div></div>';
    sc += makeCard('Entropy', 0, (stat.entropy||0.5).toFixed(3));
    sc += makeCard('EGARCH', 0, (stat.egarch_asym>=0?'+':'') + (stat.egarch_asym||0).toFixed(3));
    statEl.innerHTML = sc;
  }}
}}

function selectCoin(coin) {{
  selectedCoin = coin;
  document.querySelectorAll('.coin-row').forEach(function(el) {{
    el.classList.toggle('selected', el.dataset.coin === coin);
  }});
  updateCoinDisplay(coin);
  updateChart(coin);
}}

// Fetch data and update every 5 seconds
function refreshData() {{
  fetch('/api/state').then(function(r) {{ return r.json(); }}).then(function(data) {{
    // Store per-coin data
    if (data.coins) allCoinData = data.coins;

    // Update header
    var badges = document.querySelectorAll('.badge');
    if (badges.length >= 1 && data.regime) badges[0].textContent = data.regime;
    if (badges.length >= 2 && data.vol_regime) badges[1].textContent = data.vol_regime;
    if (badges.length >= 3 && data.hurst) {{
      var h = data.hurst;
      badges[2].textContent = 'H=' + h.toFixed(2) + ' ' + (h>0.6?'Trending':(h<0.4?'Mean-Rev':'Random'));
    }}

    // Update selected coin display + chart
    updateCoinDisplay(selectedCoin);
    updateChart(selectedCoin);
  }}).catch(function() {{}});
}}

// Initial load
initChart();
selectCoin('BTC');
refreshData();
setInterval(refreshData, 5000);

// Resize chart on window resize
window.addEventListener('resize', function() {{
  if (chart) {{
    var c = document.getElementById('candle-chart');
    if (c) chart.resize(c.clientWidth, 280);
  }}
}});

// Full page refresh every 60s for layout changes
setInterval(function() {{
  fetch('/').then(function(r) {{ return r.text(); }}).then(function(html) {{
    var parser = new DOMParser();
    var doc = parser.parseFromString(html, 'text/html');
    var container = doc.querySelector('.container');
    if (container) {{
      document.querySelector('.container').innerHTML = container.innerHTML;
      selectCoin(selectedCoin);
      refreshData();
    }}
  }}).catch(function() {{}});
}}, 60000);
</script>
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
        pass


def start_dashboard(port=PORT):
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info(f"Dashboard running at http://0.0.0.0:{port}")
    return server


def update_dashboard_state(positions: dict, equity_curve: list,
                            total_pnl: float, daily_pnl: float,
                            models: dict = None, uptime: str = "",
                            regime: str = "unknown", vol_regime: str = "unknown",
                            hurst: float = 0.5, risk_metrics: dict = None,
                            features_count: int = 46, models_active: int = 0,
                            indicator_summary: dict = None,
                            patterns: dict = None,
                            volatility: dict = None,
                            statistical: dict = None,
                            coin_signals: dict = None,
                            coins: dict = None):
    state = {
        "positions": positions,
        "equity_curve": equity_curve[-200:],
        "total_pnl": total_pnl,
        "daily_pnl": daily_pnl,
        "models": models or {},
        "uptime": uptime,
        "regime": regime,
        "vol_regime": vol_regime,
        "hurst": hurst,
        "risk_metrics": risk_metrics or {},
        "features_count": features_count,
        "models_active": models_active,
        "indicator_summary": indicator_summary or {},
        "patterns": patterns or {},
        "volatility": volatility or {},
        "statistical": statistical or {},
        "coin_signals": coin_signals or {},
        "coins": coins or {},
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(DASHBOARD_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        log.debug(f"Dashboard write failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Starting dashboard v5 on port {PORT}...")
    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    server.serve_forever()
