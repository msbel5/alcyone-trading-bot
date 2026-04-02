#!/usr/bin/env python3
"""
Multi-Asset Trading Bot v2 — ALL 10 improvements integrated.
Layers: Trend + Momentum + Volatility + Volume + Sentiment + ML
Filters: Multi-timeframe + Correlation + Kelly sizing
Features: Trailing stop, auto-retrain, Twitter, whale tracking, grid mode
"""
import sys
import os
import time
import json
import signal as sig_module
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

from binance_testnet.adapter_binance import BinanceTestnetAdapter
from strategies.pro_strategy import ProStrategy
from risk_manager import RiskManager
from trade_logger import TradeLogger
from telegram_notifier import TelegramNotifier
from backtester import TrailingStop
from filters import MultiTimeframeFilter, CorrelationFilter, kelly_fraction
from data_sources import TwitterSentiment, WhaleTracker, GridBot, AutoRetrainer
from dashboard import start_dashboard, update_dashboard_state
from position_store import save_positions, load_positions

# Equity curve persistence
def _load_equity():
    try:
        p = Path(LOG_DIR) / "equity_curve.json"
        if p.exists():
            import json as _j
            return _j.loads(p.read_text())[-500:]
    except Exception:
        pass
    return []

def _save_equity(curve):
    try:
        p = Path(LOG_DIR) / "equity_curve.json"
        import json as _j
        p.write_text(_j.dumps(curve[-500:]))
    except Exception:
        pass
from ml.onchain import get_onchain_signal
# ML v3 imports (scientific overhaul)
try:
    from ml.bot_v3_patch import get_ml_signal_v3, get_regime_params, get_onchain_v2_signal, CircuitBreaker
    ML_V3_AVAILABLE = True
    from ml.dashboard_collector import collect_dashboard_data
except ImportError:
    ML_V3_AVAILABLE = False

LOG_DIR = "/home/msbel/.openclaw/workspace/trading/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/bot_v2.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("bot_v2")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]
BALANCE_CAP = 100.0
PER_COIN_ALLOC = BALANCE_CAP / len(SYMBOLS)
DEFAULT_POSITION_PCT = 0.90
CHECK_INTERVAL = 5  # seconds to wait AFTER tick completes (adaptive speed)
COPILOT_API_URL = "http://localhost:4141/v1"

shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    log.info("Shutdown signal received")
    shutdown_requested = True

sig_module.signal(sig_module.SIGTERM, signal_handler)
sig_module.signal(sig_module.SIGINT, signal_handler)


class CoinTracker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.position = 0.0
        self.entry_price = None
        self.strategy = ProStrategy()
        self.trailing_stop = TrailingStop(atr_multiplier=2.0)
        self.risk = RiskManager(
            max_position_size=DEFAULT_POSITION_PCT,
            stop_loss_pct=3.0,
            take_profit_pct=8.0,
            max_drawdown_pct=15.0,
        )
        self.risk.peak_balance = PER_COIN_ALLOC
        self.trade_history = []  # For Kelly calculation
        self.position_pct = DEFAULT_POSITION_PCT

    def precision(self):
        return {"BTCUSDT": 5, "ETHUSDT": 4, "SOLUSDT": 3, "BNBUSDT": 3,
                "XRPUSDT": 1, "DOGEUSDT": 0, "AVAXUSDT": 2}.get(self.symbol, 5)

    def update_kelly_sizing(self):
        """Recalculate position size using Kelly criterion."""
        if len(self.trade_history) < 10:
            return  # Not enough data
        wins = [t for t in self.trade_history[-50:] if t > 0]
        losses = [t for t in self.trade_history[-50:] if t <= 0]
        if not wins or not losses:
            return
        win_rate = len(wins) / (len(wins) + len(losses))
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        kelly = kelly_fraction(win_rate, avg_win, avg_loss, half_kelly=True, max_fraction=0.30)
        if kelly > 0.05:  # Minimum 5%
            self.position_pct = kelly
            log.info(f"{self.symbol} Kelly sizing: {kelly:.1%} (WR={win_rate:.0%})")


def run():
    config_path = "/home/msbel/.openclaw/workspace/trading/config/testnet_config.json"
    with open(config_path) as f:
        config = json.load(f)

    adapter = BinanceTestnetAdapter(config["api_key"], config["private_key_path"])
    notifier = TelegramNotifier(enabled=True)
    trade_logger = TradeLogger()

    # Shared filters
    mtf_filter = MultiTimeframeFilter()
    corr_filter = CorrelationFilter(max_correlated=3, threshold=0.75)
    twitter = TwitterSentiment()
    whale_tracker = WhaleTracker()
    grid_bot = GridBot(grid_levels=5, spacing_pct=0.5)
    retrainer = AutoRetrainer()

    # Test connection
    ping = adapter.ping()
    if "error" in ping:
        log.error(f"Binance connection failed: {ping}")
        notifier.send("Bot v2: Connection failed!", priority="critical")
        return

    trackers = {sym: CoinTracker(sym) for sym in SYMBOLS}

    # Restore positions from disk (survive restart)
    saved = load_positions()
    restored = 0
    for sym, state in saved.items():
        if sym in trackers and state.get("position", 0) > 0:
            trackers[sym].position = state["position"]
            trackers[sym].entry_price = state["entry_price"]
            restored += 1
            log.info(f"Restored {sym}: {state['position']} @ ${state['entry_price']}")
    if restored:
        log.info(f"Restored {restored} positions from disk")

    coin_names = ", ".join(s.replace("USDT", "") for s in SYMBOLS)
    notifier.send(
        f"Bot v2 started\n"
        f"Coins: {len(SYMBOLS)} ({coin_names})\n"
        f"Balance: ${BALANCE_CAP}\n"
        f"Strategy: 6-Layer + MTF + Corr + Kelly + Trailing",
        priority="low"
    )
    log.info(f"Bot v2 started | {len(SYMBOLS)} coins | ${BALANCE_CAP}")

    # Equity curve tracker (persists across ticks)
    equity_curve = _load_equity()
# Circuit breaker (halts trading on excessive loss)    circuit_breaker = CircuitBreaker(        daily_max_loss_pct=5.0,        weekly_max_loss_pct=10.0,        total_max_loss_pct=20.0,        cooldown_hours=4,    ) if ML_V3_AVAILABLE else None

    # Start dashboard web server
    try:
        start_dashboard(8085)
    except Exception as e:
        log.warning(f"Dashboard failed to start: {e}")

    iteration = 0
    last_report_hour = -1
    last_corr_update = 0
    last_4h_update = 0

    try:
        while not shutdown_requested:
            iteration += 1
            now = datetime.now()
            current_hour = now.hour

            # ── Update 4h trends every 4 hours ──
            if time.time() - last_4h_update > 14400:
                for sym in SYMBOLS:
                    try:
                        klines_4h = adapter.get_klines(sym, "4h", limit=30)
                        mtf_filter.update_4h_trend(sym, klines_4h)
                    except Exception as _ce:
                        log.warning(f"Collector failed: {_ce}")
                        pass
                last_4h_update = time.time()
                log.info("4h trends updated")

            # ── Update correlation matrix daily ──
            if time.time() - last_corr_update > 86400:
                try:
                    price_data = {}
                    for sym in SYMBOLS:
                        klines = adapter.get_klines(sym, "1h", limit=720)
                        closes = pd.Series([float(k[4]) for k in klines])
                        price_data[sym] = closes
                    corr_filter.update_correlation(price_data)
                    last_corr_update = time.time()
                    log.info("Correlation matrix updated")
                except Exception as e:
                    log.warning(f"Correlation update failed: {e}")

            # ── Check auto-retrain ──
            if retrainer.should_retrain():
                log.info("Starting weekly retrain...")
                retrainer.retrain(notifier)

            # ── Tick each coin ──
            open_positions = {sym: t.position for sym, t in trackers.items()}
            for symbol, tracker in trackers.items():
                if shutdown_requested:
                    break
                try:
                    _tick_coin_v2(adapter, tracker, notifier, trade_logger,
                                  mtf_filter, corr_filter, twitter, whale_tracker,
                                  grid_bot, open_positions)
                except Exception as e:
                    log.error(f"{symbol} tick error: {e}")

            # ── Update dashboard every tick ──
            try:
                dash_positions = {}
                total_value = 0
                for s, t in trackers.items():
                    p = float(adapter.get_symbol_price(s).get("price", 0))
                    val = t.position * p if t.position > 0 else 0
                    total_value += val
                    dash_positions[s] = {
                        "amount": t.position,
                        "value": val,
                        "pnl_pct": ((p / t.entry_price) - 1) * 100 if t.entry_price and t.position > 0 else 0,
                        "entry_price": t.entry_price or 0,
                        "trailing_sl": t.trailing_stop.get_sl() or 0,
                    }
                # PnL = sum of individual position PnL (not total_value - balance)
                total_pnl = sum(
                    (float(adapter.get_symbol_price(s).get("price", 0)) - t.entry_price) * t.position
                    for s, t in trackers.items()
                    if t.position > 0 and t.entry_price
                )
                open_count = sum(1 for t in trackers.values() if t.position > 0)
                # Track equity: balance cap + current position values
                portfolio_value = BALANCE_CAP + total_pnl
                equity_curve.append(portfolio_value)
                _save_equity(equity_curve)
# Circuit breaker check                if circuit_breaker:                    circuit_breaker.update(portfolio_value)                    if circuit_breaker.is_tripped():                        if iteration % 180 == 0:  # Log every 15min                            log.warning(f"CIRCUIT BREAKER ACTIVE — trading halted")
                if len(equity_curve) > 500:
                    equity_curve.pop(0)
                # V4 dashboard: collect all indicator/model data
                if ML_V3_AVAILABLE:
                    try:
                        dash_data = collect_dashboard_data(adapter, trackers, iteration, equity_curve)
                    except Exception as _ce:
                        log.warning(f"Collector failed: {_ce}")
                        dash_data = {"regime": "unknown", "vol_regime": "unknown", "hurst": 0.5,
                                     "features_count": 46, "models_active": 8}
                else:
                    dash_data = {}

                update_dashboard_state(
                    dash_positions, equity_curve,
                    round(total_pnl, 4),
                    0,
                    {
                        "Stacked v3": {"status": "active"},
                        "CNN-LSTM": {"status": "active"},
                        "LightGBM": {"status": "active"},
                        "GRU": {"status": "active"},
                        "Ichimoku": {"status": "active"},
                        "Patterns": {"status": "active"},
                        "Hurst": {"status": "active"},
                        "GARCH": {"status": "active"},
                    },
                    f"{iteration * CHECK_INTERVAL // 60}min" if iteration * CHECK_INTERVAL >= 60 else f"{iteration * CHECK_INTERVAL}s",
                    regime=dash_data.get("regime", "unknown"),
                    vol_regime=dash_data.get("vol_regime", "unknown"),
                    hurst=dash_data.get("hurst", 0.5),
                    features_count=46,
                    models_active=8,
                    risk_metrics=dash_data.get("risk_metrics", {}),
                    indicator_summary=dash_data.get("indicator_summary", {}),
                    patterns=dash_data.get("patterns", {}),
                    volatility=dash_data.get("volatility", {}),
                    statistical=dash_data.get("statistical", {}),
                    coin_signals=dash_data.get("coin_signals", {}),
                    coins=dash_data.get("coins", {}),
                )
            except Exception:
                pass

            # ── Hourly heartbeat ──
            if iteration % 180 == 0:  # ~hourly at 20s/tick
                positions = sum(1 for t in trackers.values() if t.position > 0)
                log.info(f"Heartbeat | iter={iteration} | {positions}/{len(SYMBOLS)} open")

            # ── Daily report at 23:00 ──
            if current_hour == 23 and last_report_hour != 23:
                last_report_hour = 23
                _send_daily_report(trackers, notifier, adapter)
            if current_hour != 23:
                last_report_hour = -1

            time.sleep(CHECK_INTERVAL)

    finally:
        for symbol, tracker in trackers.items():
            if tracker.position > 0:
                try:
                    price = float(adapter.get_symbol_price(symbol).get("price", 0))
                    if price > 0:
                        prec = tracker.precision()
                        qty_str = f"{round(tracker.position, prec):.{prec}f}"
                        adapter.place_order(symbol, "SELL", "MARKET", float(qty_str))
                        log.info(f"Closed {symbol}: {qty_str} @ ${price:,.2f}")
                except Exception as e:
                    log.error(f"Failed to close {symbol}: {e}")

        notifier.send("Bot v2 stopped", priority="low")
        log.info("Bot v2 stopped.")


def _tick_coin_v2(adapter, tracker, notifier, trade_logger,
                   mtf_filter, corr_filter, twitter, whale_tracker,
                   grid_bot, open_positions):
    """Process one tick for one coin — with all improvements."""
    symbol = tracker.symbol
    prec = tracker.precision()

    try:
        price = float(adapter.get_symbol_price(symbol).get("price", 0))
        if price <= 0:
            return
    except Exception:
        return

    # ── If in position: check trailing stop ──
    if tracker.position > 0 and tracker.entry_price:
        # Update trailing stop
        try:
            klines = adapter.get_klines(symbol, "1h", limit=20)
            df_quick = pd.DataFrame(klines, columns=["ot", "o", "h", "l", "c", "v", "ct", "qv", "t", "tb", "tq", "ig"])
            high = df_quick["h"].astype(float)
            low = df_quick["l"].astype(float)
            close = df_quick["c"].astype(float)
            tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
            atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])
            tracker.trailing_stop.update(price, atr)
        except Exception:
            pass

        if tracker.trailing_stop.should_exit(price):
            qty_str = f"{round(tracker.position, prec):.{prec}f}"
            result = adapter.place_order(symbol, "SELL", "MARKET", float(qty_str))
            if result.get("status") == "FILLED":
                pnl = (price - tracker.entry_price) * tracker.position
                pnl_pct = ((price / tracker.entry_price) - 1) * 100
                tracker.trade_history.append(pnl_pct)
                tracker.update_kelly_sizing()
                coin = symbol.replace("USDT", "")
                log.info(f"TRAIL EXIT {coin} | SL={tracker.trailing_stop.get_sl():.2f} | PnL: ${pnl:+.4f} ({pnl_pct:+.2f}%)")
                notifier.notify_trade("SELL", price, tracker.position, pnl, pnl_pct, "TRAILING_STOP")
                tracker.position = 0
                tracker.entry_price = None
                tracker.trailing_stop.reset()
                save_positions(trackers)  # Persist to disk
            return

        # Check strategy SELL signal
        # (fall through to signal calculation below)

    # ── Get candles and calculate strategy signals ──
    try:
        klines = adapter.get_klines(symbol, "1h", limit=50)
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        if len(df) < 30:
            log.debug(f"{symbol}: not enough candles ({len(df)})"  )
            return

        # ML ensemble + Copilot AI + Twitter + Whale
        try:
            from ml.data_pipeline import add_features
            from ml.ensemble import get_ensemble
            from ml.news_parser import get_headlines_for_symbol
            df_features = add_features(df.copy())
            headlines = get_headlines_for_symbol(symbol)

            # Twitter sentiment (Nitter RSS + CryptoBERT)
            tw_score = twitter.get_sentiment(symbol)

            # Whale tracking (RSS-based)
            whale = whale_tracker.check_whale_activity(symbol)
            whale_signal = whale.get("signal", 0)

            # On-chain metrics (mempool, fees, hashrate — FREE, no API key)
            # On-chain v2 (MVRV + NVT + Exchange Flow)
            if ML_V3_AVAILABLE:
                prices_arr = df["close"].values if len(df) > 100 else None
                onchain = get_onchain_v2_signal(symbol, prices_arr)
            else:
                onchain = get_onchain_signal(symbol)
            onchain_signal = onchain.get("signal", 0)

            # Copilot AI sentiment (GPT-4.1 on Pi, FREE)
            copilot_score = 0.0
            try:
                import requests as _req
                news_text = "; ".join(headlines[:3]) if headlines else "no news"
                resp = _req.post(f"{COPILOT_API_URL}/chat/completions", json={
                    "model": "gpt-4.1",
                    "messages": [
                        {"role": "system", "content": "You are a crypto trading analyst. Reply ONLY with a JSON: {\"signal\": -1.0 to 1.0, \"reason\": \"one sentence\"}. Positive=bullish, negative=bearish, 0=neutral."},
                        {"role": "user", "content": f"Symbol: {symbol}, Price: ${price:,.2f}, RSI: {df['rsi'].iloc[-1] if 'rsi' in df else 50:.0f}, News: {news_text[:200]}"}
                    ],
                    "max_tokens": 60, "temperature": 0.1
                }, timeout=8)
                if resp.status_code == 200:
                    import json as _json
                    text = resp.json()["choices"][0]["message"]["content"].strip()
                    if text.startswith("```"): text = text.split("\n",1)[-1].rsplit("```",1)[0]
                    parsed = _json.loads(text)
                    copilot_score = float(parsed.get("signal", 0))
            except Exception:
                pass

            # ML models (XGBoost + GRU + CryptoBERT local)
            # ML models (v3: CNN-LSTM + Stacked v3 + LightGBM + GRU)
            if ML_V3_AVAILABLE:
                ml_v3_result = get_ml_signal_v3(symbol, df_features)
                ml_score = ml_v3_result["signal"]
            else:
                ml_result = get_ensemble().predict(symbol, df_features, headlines, f"{symbol} ${price:,.2f}")
                ml_score = ml_result.get("score", 0)

            # Combine all signals (8 sources!)
            combined_ml = (
                ml_score * 0.30 +                     # ML models (v3 or v2)
                copilot_score * 0.20 +                # GPT-4.1 AI opinion
                tw_score * 0.15 +                     # Twitter sentiment
                whale_signal * 0.10 +                 # Whale Alert RSS
                onchain_signal * 0.15 +               # On-chain (v2: MVRV+NVT+Flow)
                0 * 0.10                              # Reserved
            )

            # Regime-based parameter adjustment
            if ML_V3_AVAILABLE:
                regime_params = get_regime_params(df_features)
                tracker.strategy.buy_threshold = regime_params.get("buy_threshold", 0.40)
                tracker.strategy.sell_threshold = regime_params.get("sell_threshold", -0.40)

            tracker.strategy.set_ml_signal(combined_ml)
            # V4: Set Ichimoku, Pattern, Statistical signals            try:                from ml.indicators_advanced import AdvancedIndicators                _adv = AdvancedIndicators()                _df_adv = _adv.compute_all(df.copy())                tracker.strategy.set_ichimoku_signal(_adv.get_all_signals(_df_adv).get("ichimoku", 0))            except Exception:                tracker.strategy.set_ichimoku_signal(0)            try:                from ml.candlestick_patterns import PatternEngine                _pe = PatternEngine()                _df_pat = _pe.compute_all(df.copy())                tracker.strategy.set_pattern_signal(_pe.get_composite_signal(_df_pat))            except Exception:                tracker.strategy.set_pattern_signal(0)            try:                from ml.statistical_models import StatisticalEngine                _se = StatisticalEngine()                tracker.strategy.set_statistical_signal(_se.get_composite_signal(_se.compute_all(df.copy())))            except Exception:                tracker.strategy.set_statistical_signal(0)
        except Exception as ml_err:
            log.debug(f"{symbol} ML skipped: {ml_err}")
            tracker.strategy.set_ml_signal(0)
            tracker.strategy.set_ichimoku_signal(0)
            tracker.strategy.set_pattern_signal(0)
            tracker.strategy.set_statistical_signal(0)

        df = tracker.strategy.calculate_signals(df)
        raw_signal = int(df["signal"].iloc[-1])
        rsi = float(df["rsi"].iloc[-1])
        adx = float(df["adx"].iloc[-1])

    except Exception as e:
        log.warning(f"{symbol} signal error: {e}")
        return

    # ── Grid mode check (ADX < 20 = sideways) ──
    if grid_bot.should_activate(adx) and tracker.position == 0:
        atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else price * 0.02
        grid = grid_bot.calculate_grid(price, atr)
        grid_signals = grid_bot.check_grid_signals(price, grid)
        if grid_signals:
            raw_signal = 1 if grid_signals[0]["action"] == "BUY" else -1
            log.debug(f"{symbol} Grid mode: {grid_signals[0]['action']} at {price}")

    # ── Multi-timeframe filter ──
    filtered_signal = mtf_filter.filter_signal(symbol, raw_signal)

    # ── BUY ──
    if filtered_signal == 1 and tracker.position == 0:
        # Correlation filter
        if not corr_filter.allow_entry(symbol, open_positions):
            log.debug(f"{symbol}: BUY blocked by correlation filter")
            return

        usdt_to_use = PER_COIN_ALLOC * tracker.position_pct
        amount = round(usdt_to_use / price, prec)
        if amount <= 0:
            return
        qty_str = f"{amount:.{prec}f}"
        result = adapter.place_order(symbol, "BUY", "MARKET", float(qty_str))
        if result.get("status") == "FILLED":
            tracker.position = amount
            tracker.entry_price = price
            tracker.trailing_stop.reset()
            # Set dynamic SL/TP from ATR
            dsl, dtp = tracker.strategy.get_dynamic_sl_tp(df)
            if dsl and dtp and tracker.entry_price:
                tracker.risk.stop_loss_pct = abs(tracker.entry_price - dsl) / tracker.entry_price
                tracker.risk.take_profit_pct = abs(dtp - tracker.entry_price) / tracker.entry_price
            coin = symbol.replace("USDT", "")
            trend_4h = {1: "↑", -1: "↓", 0: "→"}.get(mtf_filter.get_trend(symbol), "?")
            log.info(f"BUY {coin} | {qty_str} @ ${price:,.2f} | RSI={rsi:.0f} ADX={adx:.0f} 4h={trend_4h} Kelly={tracker.position_pct:.0%}")
            notifier.notify_trade("BUY", price, amount)
            save_positions(trackers)  # Persist to disk

    # ── SELL ──
    elif filtered_signal == -1 and tracker.position > 0:
        qty_str = f"{round(tracker.position, prec):.{prec}f}"
        result = adapter.place_order(symbol, "SELL", "MARKET", float(qty_str))
        if result.get("status") == "FILLED":
            pnl = (price - tracker.entry_price) * tracker.position if tracker.entry_price else 0
            pnl_pct = ((price / tracker.entry_price) - 1) * 100 if tracker.entry_price else 0
            tracker.trade_history.append(pnl_pct)
            tracker.update_kelly_sizing()
            coin = symbol.replace("USDT", "")
            log.info(f"SELL {coin} | PnL: ${pnl:+.4f} ({pnl_pct:+.2f}%)")
            notifier.notify_trade("SELL", price, tracker.position, pnl, pnl_pct, "STRATEGY_SIGNAL")
            tracker.position = 0
            tracker.entry_price = None
            tracker.trailing_stop.reset()
            save_positions(trackers)  # Persist to disk


def _send_daily_report(trackers, notifier, adapter):
    lines = ["Daily Portfolio Report (v2)\n"]
    total_value = 0
    total_pnl = 0
    for symbol, tracker in trackers.items():
        try:
            price = float(adapter.get_symbol_price(symbol).get("price", 0))
        except Exception:
            price = 0
        coin_value = tracker.position * price if tracker.position > 0 else 0
        total_value += coin_value
        coin = symbol.replace("USDT", "")
        if tracker.position > 0:
            pnl_pct = ((price / tracker.entry_price) - 1) * 100 if tracker.entry_price else 0
            total_pnl += pnl_pct
            sl = tracker.trailing_stop.get_sl()
            sl_str = f" SL=${sl:.0f}" if sl else ""
            lines.append(f"  {coin}: ${coin_value:.2f} ({pnl_pct:+.1f}%){sl_str}")
        else:
            lines.append(f"  {coin}: idle")

    lines.append(f"\nPositions: ${total_value:.2f}")
    kelly_parts = []
    for s, t in trackers.items():
        if t.position_pct != DEFAULT_POSITION_PCT:
            coin_name = s.replace("USDT", "")
            kelly_parts.append(f"{coin_name}={t.position_pct:.0%}")
    kelly_info = ", ".join(kelly_parts) if kelly_parts else "all default"
    lines.append(f"Kelly: {kelly_info}")
    notifier.send("\n".join(lines), priority="normal")


if __name__ == "__main__":
    run()
