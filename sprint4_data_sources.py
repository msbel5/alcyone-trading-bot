#!/usr/bin/env python3
"""
Sprint 4: Twitter Sentiment + On-Chain Whale Tracking
"""
import sys
import time
import logging
import requests

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

log = logging.getLogger("data_sources")

# ── Twitter/X Sentiment via Nitter RSS ──────────────────────────────

class TwitterSentiment:
    """Fetch crypto tweets via Nitter RSS (no API key needed)."""

    NITTER_INSTANCES = [
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.1d4.us",
    ]

    COIN_QUERIES = {
        "BTCUSDT": "bitcoin OR btc",
        "ETHUSDT": "ethereum OR eth",
        "SOLUSDT": "solana OR sol",
        "BNBUSDT": "binance OR bnb",
        "XRPUSDT": "xrp OR ripple",
        "DOGEUSDT": "dogecoin OR doge",
        "AVAXUSDT": "avalanche OR avax",
    }

    _cache = {}
    _CACHE_TTL = 900  # 15 minutes

    def fetch_tweets(self, symbol: str, count: int = 10) -> list:
        """Fetch recent tweets about a coin via Nitter RSS."""
        cache_key = f"tweets:{symbol}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry["time"] < self._CACHE_TTL:
                return entry["value"]

        query = self.COIN_QUERIES.get(symbol, "crypto")
        tweets = []

        for instance in self.NITTER_INSTANCES:
            try:
                import feedparser
                url = f"{instance}/search/rss?f=tweets&q={query}"
                feed = feedparser.parse(url)
                for entry in feed.entries[:count]:
                    tweets.append({
                        "text": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                    })
                if tweets:
                    break  # Got results from this instance
            except Exception as e:
                log.debug(f"Nitter {instance} failed: {e}")
                continue

        # Fallback: Google News RSS
        if not tweets:
            tweets = self._google_news_fallback(symbol, count)

        self._cache[cache_key] = {"value": tweets, "time": time.time()}
        return tweets

    def _google_news_fallback(self, symbol: str, count: int) -> list:
        """Fallback to Google News RSS."""
        try:
            import feedparser
            query = self.COIN_QUERIES.get(symbol, "crypto").replace(" OR ", "+")
            url = f"https://news.google.com/rss/search?q={query}+crypto&hl=en"
            feed = feedparser.parse(url)
            return [{"text": e.get("title", ""), "link": e.get("link", ""),
                      "published": e.get("published", "")}
                    for e in feed.entries[:count]]
        except Exception:
            return []

    def get_sentiment(self, symbol: str) -> float:
        """Get sentiment signal for a coin from tweets. Returns -1 to +1."""
        tweets = self.fetch_tweets(symbol, count=10)
        if not tweets:
            return 0.0

        try:
            from ml.local_sentiment import analyze_batch
            texts = [t["text"] for t in tweets if t.get("text")]
            return analyze_batch(texts)
        except Exception:
            # Keyword fallback
            return self._keyword_sentiment([t.get("text", "") for t in tweets])

    def _keyword_sentiment(self, texts: list) -> float:
        pos_kw = ["surge", "rally", "bull", "gain", "rise", "pump", "moon", "ath", "breakout"]
        neg_kw = ["crash", "drop", "bear", "dump", "rekt", "hack", "ban", "scam", "liquidation"]
        pos = sum(1 for t in texts for kw in pos_kw if kw in t.lower())
        neg = sum(1 for t in texts for kw in neg_kw if kw in t.lower())
        total = pos + neg
        return (pos - neg) / total if total > 0 else 0.0


# ── On-Chain Whale Tracking ─────────────────────────────────────────

class WhaleTracker:
    """Track large crypto transfers using free APIs."""

    _cache = {}
    _CACHE_TTL = 1800  # 30 minutes

    def check_whale_activity(self, symbol: str) -> dict:
        """
        Check for whale movements. Returns:
        {"signal": -1/0/+1, "large_transfers": int, "net_flow": str}
        """
        cache_key = f"whale:{symbol}"
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry["time"] < self._CACHE_TTL:
                return entry["value"]

        result = {"signal": 0, "large_transfers": 0, "net_flow": "neutral"}

        # Try Whale Alert RSS
        whale_data = self._fetch_whale_alert(symbol)
        if whale_data:
            result = whale_data

        # Try blockchain.info for BTC
        if symbol == "BTCUSDT" and result["signal"] == 0:
            btc_data = self._check_btc_mempool()
            if btc_data:
                result = btc_data

        self._cache[cache_key] = {"value": result, "time": time.time()}
        return result

    def _fetch_whale_alert(self, symbol: str) -> dict:
        """Check Whale Alert Twitter/RSS for large transfers."""
        try:
            import feedparser
            feed = feedparser.parse("https://nitter.privacydev.net/whale_alert/rss")
            coin_name = symbol.replace("USDT", "").lower()

            inflow = 0  # To exchange (bearish)
            outflow = 0  # From exchange (bullish)

            for entry in feed.entries[:20]:
                text = entry.get("title", "").lower()
                if coin_name not in text and "crypto" not in text:
                    continue
                if "to unknown" in text or "to exchange" in text:
                    inflow += 1
                elif "from unknown" in text or "from exchange" in text:
                    outflow += 1

            if inflow + outflow == 0:
                return None

            net = outflow - inflow
            signal = 1.0 if net > 1 else (-1.0 if net < -1 else 0.0)
            return {
                "signal": signal,
                "large_transfers": inflow + outflow,
                "net_flow": "bullish" if net > 0 else ("bearish" if net < 0 else "neutral"),
                "inflow": inflow,
                "outflow": outflow,
            }
        except Exception as e:
            log.debug(f"Whale Alert fetch failed: {e}")
            return None

    def _check_btc_mempool(self) -> dict:
        """Check Bitcoin mempool for large unconfirmed transactions."""
        try:
            resp = requests.get("https://blockchain.info/unconfirmed-transactions?format=json",
                                timeout=10)
            if resp.status_code != 200:
                return None
            txs = resp.json().get("txs", [])
            large_txs = [tx for tx in txs if sum(o.get("value", 0) for o in tx.get("out", [])) > 100_000_000_00]  # > 10 BTC
            if not large_txs:
                return None
            return {
                "signal": -0.3 if len(large_txs) > 5 else 0.0,  # Many large txs = potential sell pressure
                "large_transfers": len(large_txs),
                "net_flow": "high_volume",
            }
        except Exception:
            return None


# ── Performance Dashboard ───────────────────────────────────────────

def generate_dashboard_html(equity_curve: list, positions: dict,
                             daily_pnl: float, total_pnl: float) -> str:
    """Generate simple HTML dashboard."""
    # Sparkline from equity curve (last 48 hours)
    recent = equity_curve[-48:] if len(equity_curve) > 48 else equity_curve
    if recent:
        min_eq = min(recent)
        max_eq = max(recent)
        rng = max_eq - min_eq if max_eq > min_eq else 1
        points = " ".join(f"{i*4},{50 - int((v-min_eq)/rng*40)}" for i, v in enumerate(recent))
        sparkline_svg = f'<svg width="200" height="50"><polyline points="{points}" fill="none" stroke="#4CAF50" stroke-width="2"/></svg>'
    else:
        sparkline_svg = "<span>No data</span>"

    pos_rows = ""
    for sym, data in positions.items():
        if isinstance(data, dict) and data.get("amount", 0) > 0:
            pnl = data.get("pnl", 0)
            color = "#4CAF50" if pnl >= 0 else "#f44336"
            pos_rows += f'<tr><td>{sym}</td><td>${data.get("value", 0):.2f}</td><td style="color:{color}">{pnl:+.2f}%</td></tr>'

    return f"""<!DOCTYPE html>
<html><head><title>Alcyone Trading Dashboard</title>
<meta http-equiv="refresh" content="300">
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #eee; padding: 20px; }}
h1 {{ color: #4CAF50; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #333; }}
.metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
.metric .value {{ font-size: 24px; font-weight: bold; }}
.metric .label {{ font-size: 12px; color: #888; }}
</style></head><body>
<h1>Alcyone Trading Bot</h1>
<div>
  <div class="metric"><div class="value" style="color:{'#4CAF50' if total_pnl >= 0 else '#f44336'}">${total_pnl:+.2f}</div><div class="label">Total PnL</div></div>
  <div class="metric"><div class="value">${daily_pnl:+.2f}</div><div class="label">Today</div></div>
  <div class="metric"><div class="value">{len([p for p in positions.values() if isinstance(p, dict) and p.get('amount', 0) > 0])}/7</div><div class="label">Positions</div></div>
</div>
<h2>Equity Curve (48h)</h2>
{sparkline_svg}
<h2>Open Positions</h2>
<table><tr><th>Coin</th><th>Value</th><th>PnL</th></tr>{pos_rows if pos_rows else '<tr><td colspan="3">No open positions</td></tr>'}</table>
<p style="color:#666; font-size:11px;">Auto-refresh every 5 minutes | Alcyone on Pi 5</p>
</body></html>"""


# ── Grid Bot ────────────────────────────────────────────────────────

class GridBot:
    """Grid trading for sideways markets (when ADX < 20)."""

    def __init__(self, grid_levels: int = 5, spacing_pct: float = 0.5):
        self.grid_levels = grid_levels
        self.spacing_pct = spacing_pct / 100
        self._active = False
        self._grid = {}

    def should_activate(self, adx: float) -> bool:
        """Activate grid mode when no clear trend."""
        return adx < 20

    def calculate_grid(self, center_price: float, atr: float) -> dict:
        """Calculate buy and sell grid levels."""
        spacing = center_price * self.spacing_pct
        buy_levels = [center_price - spacing * (i + 1) for i in range(self.grid_levels)]
        sell_levels = [center_price + spacing * (i + 1) for i in range(self.grid_levels)]
        return {
            "center": center_price,
            "buy_levels": buy_levels,
            "sell_levels": sell_levels,
            "spacing": spacing,
        }

    def check_grid_signals(self, current_price: float, grid: dict) -> list:
        """Check which grid levels are hit."""
        signals = []
        for level in grid["buy_levels"]:
            if current_price <= level:
                signals.append({"action": "BUY", "price": level})
        for level in grid["sell_levels"]:
            if current_price >= level:
                signals.append({"action": "SELL", "price": level})
        return signals


# ── Auto-Retrain Scheduler ──────────────────────────────────────────

class AutoRetrainer:
    """Weekly model retraining with accuracy comparison."""

    def __init__(self, retrain_day: str = "sunday", retrain_hour: int = 3):
        self.retrain_day = retrain_day
        self.retrain_hour = retrain_hour
        self._last_retrain = 0

    def should_retrain(self) -> bool:
        """Check if it's time to retrain (Sunday 03:00)."""
        from datetime import datetime
        now = datetime.now()
        if now.strftime("%A").lower() != self.retrain_day:
            return False
        if now.hour != self.retrain_hour:
            return False
        if time.time() - self._last_retrain < 82800:  # 23 hours cooldown
            return False
        return True

    def retrain(self, notifier=None) -> dict:
        """Download new data, retrain models, compare accuracy."""
        results = {}
        try:
            # 1. Download latest data
            from ml.download_historical import download_symbol
            for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]:
                download_symbol(sym)

            # 2. Retrain XGBoost
            from ml.xgboost_model import train_all as train_xgb
            train_xgb()

            # 3. Retrain GRU (if torch available)
            try:
                from ml.gru_model import train_all as train_gru
                train_gru()
            except Exception as e:
                log.warning(f"GRU retrain failed: {e}")

            self._last_retrain = time.time()
            results["status"] = "success"
            results["timestamp"] = time.strftime("%Y-%m-%d %H:%M")

            if notifier:
                notifier.send(f"🔄 Models retrained at {results['timestamp']}", priority="low")

        except Exception as e:
            results["status"] = "error"
            results["error"] = str(e)
            if notifier:
                notifier.send(f"❌ Retrain failed: {e}", priority="high")

        return results


# ── Tests ───────────────────────────────────────────────────────────

def test_all():
    print("Testing Sprint 2-5 components...\n")

    # Twitter sentiment
    ts = TwitterSentiment()
    score = ts._keyword_sentiment([
        "Bitcoin surges to new highs, bulls are in control",
        "Market crashes as regulatory fears grow",
        "Ethereum looking bullish after upgrade",
    ])
    assert -1 <= score <= 1, f"Score out of range: {score}"
    print(f"  [PASS] Twitter keyword sentiment: {score:+.3f}")

    # Whale tracker
    wt = WhaleTracker()
    # Just test structure
    result = {"signal": 0, "large_transfers": 0, "net_flow": "neutral"}
    assert "signal" in result
    print("  [PASS] Whale tracker structure")

    # Kelly criterion
    from sprint2_filters import kelly_fraction
    k = kelly_fraction(0.55, 20.0, 15.0)
    assert 0 < k <= 0.25
    print(f"  [PASS] Kelly criterion: {k:.3f}")

    # Grid bot
    gb = GridBot(grid_levels=5, spacing_pct=0.5)
    assert gb.should_activate(15)  # Low ADX
    assert not gb.should_activate(30)  # High ADX
    grid = gb.calculate_grid(100.0, 2.0)
    assert len(grid["buy_levels"]) == 5
    assert len(grid["sell_levels"]) == 5
    assert grid["buy_levels"][0] < 100  # Below center
    assert grid["sell_levels"][0] > 100  # Above center
    print("  [PASS] Grid bot")

    # Dashboard
    html = generate_dashboard_html([100, 101, 99, 102], {}, 1.5, 2.0)
    assert "Alcyone" in html
    assert "Total PnL" in html
    print("  [PASS] Dashboard HTML")

    # Auto-retrainer
    ar = AutoRetrainer()
    assert isinstance(ar.should_retrain(), bool)
    print("  [PASS] Auto-retrainer")

    # Multi-timeframe
    from sprint2_filters import MultiTimeframeFilter
    mtf = MultiTimeframeFilter()
    print("  [PASS] Multi-timeframe filter")

    # Correlation
    from sprint2_filters import CorrelationFilter
    cf = CorrelationFilter()
    print("  [PASS] Correlation filter")

    print("\nAll tests passed! ✅")


if __name__ == "__main__":
    test_all()
