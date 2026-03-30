#!/usr/bin/env python3
"""
Multi-Asset Trading Bot — trades 7 coins with enhanced strategy.
Designed to run as systemd service, 24/7.
"""
import sys
import os
import time
import json
import signal
import logging
import pandas as pd
from datetime import datetime

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

from binance_testnet.adapter_binance import BinanceTestnetAdapter
from strategies.pro_strategy import ProStrategy
from risk_manager import RiskManager
from trade_logger import TradeLogger
from ml.ensemble import get_ensemble
from ml.news_parser import get_headlines_for_symbol
from ml.data_pipeline import add_features
from telegram_notifier import TelegramNotifier

LOG_DIR = "/home/msbel/.openclaw/workspace/trading/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/multi_bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("multi_bot")

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
]

BALANCE_CAP = 100.0
PER_COIN_ALLOC = BALANCE_CAP / len(SYMBOLS)
POSITION_PCT = 0.90
CHECK_INTERVAL = 300

shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    log.info("Shutdown signal received")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


class CoinTracker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.position = 0.0
        self.entry_price = None
        self.strategy = ProStrategy()
        self.risk = RiskManager(
            max_position_size=POSITION_PCT,
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            max_drawdown_pct=15.0,
        )
        self.risk.peak_balance = PER_COIN_ALLOC

    def precision(self):
        precisions = {
            "BTCUSDT": 5,
            "ETHUSDT": 4,
            "SOLUSDT": 3,
            "BNBUSDT": 3,
            "XRPUSDT": 1,
            "DOGEUSDT": 0,
            "AVAXUSDT": 2,
        }
        return precisions.get(self.symbol, 5)


def run():
    config_path = "/home/msbel/.openclaw/workspace/trading/config/testnet_config.json"
    with open(config_path) as f:
        config = json.load(f)

    adapter = BinanceTestnetAdapter(config["api_key"], config["private_key_path"])
    notifier = TelegramNotifier(enabled=True)
    trade_logger = TradeLogger()

    ping = adapter.ping()
    if "error" in ping:
        log.error(f"Binance connection failed: {ping}")
        notifier.send("Multi-bot: Binance connection failed!", priority="critical")
        return

    trackers = {sym: CoinTracker(sym) for sym in SYMBOLS}

    coin_names = ", ".join(s.replace("USDT", "") for s in SYMBOLS)
    notifier.send(
        f"Multi-Asset Bot started\n"
        f"Coins: {len(SYMBOLS)} ({coin_names})\n"
        f"Balance cap: ${BALANCE_CAP}\n"
        f"Per coin: ${PER_COIN_ALLOC:.2f}\n"
        f"Strategy: Enhanced EMA+RSI+Volume+ATR",
        priority="low"
    )
    log.info(f"Multi-bot started | {len(SYMBOLS)} coins | ${BALANCE_CAP} cap")

    iteration = 0
    last_report_hour = -1

    try:
        while not shutdown_requested:
            iteration += 1
            current_hour = datetime.now().hour

            for symbol, tracker in trackers.items():
                if shutdown_requested:
                    break
                try:
                    _tick_coin(adapter, tracker, notifier, trade_logger)
                except Exception as e:
                    log.error(f"{symbol} tick error: {e}")

            if iteration % 12 == 0:
                positions = sum(1 for t in trackers.values() if t.position > 0)
                log.info(f"Heartbeat | iter={iteration} | {positions}/{len(SYMBOLS)} positions open")

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

        notifier.send("Multi-Asset Bot stopped", priority="low")
        log.info("Multi-bot stopped.")


def _tick_coin(adapter, tracker, notifier, trade_logger):
    symbol = tracker.symbol
    prec = tracker.precision()

    try:
        price = float(adapter.get_symbol_price(symbol).get("price", 0))
        if price <= 0:
            return
    except Exception:
        return

    if tracker.position > 0 and tracker.entry_price:
        should_exit, reason = tracker.risk.should_exit(tracker.entry_price, price)
        if should_exit:
            qty_str = f"{round(tracker.position, prec):.{prec}f}"
            result = adapter.place_order(symbol, "SELL", "MARKET", float(qty_str))
            if result.get("status") == "FILLED":
                pnl = (price - tracker.entry_price) * tracker.position
                pnl_pct = ((price / tracker.entry_price) - 1) * 100
                log.info(f"EXIT {symbol} | {reason} | PnL: ${pnl:+.4f} ({pnl_pct:+.2f}%)")
                notifier.notify_trade("SELL", price, tracker.position, pnl, pnl_pct, reason)
                tracker.position = 0
                tracker.entry_price = None
            return

    try:
        klines = adapter.get_klines(symbol, "1h", limit=50)
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        # Add features for ML models
        try:
            df_features = add_features(df.copy())
            headlines = get_headlines_for_symbol(symbol)
            price_summary = f"{symbol} at ${price:,.2f}"
            ml_result = get_ensemble().predict(symbol, df_features, headlines, price_summary)
            tracker.strategy.set_ml_signal(ml_result.get("score", 0))
        except Exception as ml_err:
            log.debug(f"{symbol} ML prediction skipped: {ml_err}")
            tracker.strategy.set_ml_signal(0)

        df = tracker.strategy.calculate_signals(df)
        sig = int(df["signal"].iloc[-1])
        rsi = float(df["rsi"].iloc[-1])
    except Exception as e:
        log.warning(f"{symbol} signal error: {e}")
        return

    if sig == 1 and tracker.position == 0:
        usdt_to_use = PER_COIN_ALLOC * POSITION_PCT
        amount = round(usdt_to_use / price, prec)
        if amount <= 0:
            return
        qty_str = f"{amount:.{prec}f}"
        result = adapter.place_order(symbol, "BUY", "MARKET", float(qty_str))
        if result.get("status") == "FILLED":
            tracker.position = amount
            tracker.entry_price = price
            # Set dynamic SL/TP from ATR
            dsl, dtp = tracker.strategy.get_dynamic_sl_tp(df)
            if dsl and dtp and tracker.entry_price:
                tracker.risk.stop_loss_pct = abs(tracker.entry_price - dsl) / tracker.entry_price
                tracker.risk.take_profit_pct = abs(dtp - tracker.entry_price) / tracker.entry_price
            coin = symbol.replace("USDT", "")
            log.info(f"BUY {coin} | {qty_str} @ ${price:,.2f} | RSI={rsi:.1f}")
            notifier.notify_trade("BUY", price, amount)

    elif sig == -1 and tracker.position > 0:
        qty_str = f"{round(tracker.position, prec):.{prec}f}"
        result = adapter.place_order(symbol, "SELL", "MARKET", float(qty_str))
        if result.get("status") == "FILLED":
            pnl = (price - tracker.entry_price) * tracker.position if tracker.entry_price else 0
            pnl_pct = ((price / tracker.entry_price) - 1) * 100 if tracker.entry_price else 0
            coin = symbol.replace("USDT", "")
            log.info(f"SELL {coin} | PnL: ${pnl:+.4f} ({pnl_pct:+.2f}%) | RSI={rsi:.1f}")
            notifier.notify_trade("SELL", price, tracker.position, pnl, pnl_pct, "STRATEGY_SIGNAL")
            tracker.position = 0
            tracker.entry_price = None


def _send_daily_report(trackers, notifier, adapter):
    lines = ["Daily Portfolio Report\n"]
    total_value = 0
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
            lines.append(f"  {coin}: ${coin_value:.2f} ({pnl_pct:+.1f}%)")
        else:
            lines.append(f"  {coin}: idle")

    lines.append(f"\nPositions: ${total_value:.2f}")
    lines.append(f"Available: ${BALANCE_CAP - total_value:.2f}")
    notifier.send("\n".join(lines), priority="normal")


if __name__ == "__main__":
    run()
