#!/usr/bin/env python3
"""
Trading Bot Service — 24/7 continuous operation.
Designed to run as a systemd service with auto-restart.
"""
import sys
import os
import time
import json
import signal
import logging
from datetime import datetime

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

from live_trader import LiveTrader
from balance_cap import apply_balance_cap

# Setup logging
LOG_DIR = "/home/msbel/.openclaw/workspace/trading/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/bot_service.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("trading_bot")

# Graceful shutdown
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    log.info("Shutdown signal received, closing positions...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def run():
    config_path = "/home/msbel/.openclaw/workspace/trading/config/testnet_config.json"
    with open(config_path) as f:
        config = json.load(f)

    trader = LiveTrader(
        api_key=config["api_key"],
        private_key_path=config["private_key_path"],
        dry_run=False,  # Real testnet orders (no real money)
        telegram_enabled=True
    )

    # Cap balance to $100 (simulate small account)
    apply_balance_cap(trader, cap=100.0)
    # Override position size: 15% of $100 = $15 per trade (meets Binance $10 min notional)
    trader.risk_manager.max_position_size = 0.15

    # Notify start
    balance = trader.get_balance() or 0
    log.info(f"Bot started | Balance: ${balance:.2f} | Strategy: EMA {trader.strategy.fast_period}/{trader.strategy.slow_period}")
    trader.notifier.notify_session_start(
        mode="TESTNET LIVE",
        balance=balance,
        strategy=f"EMA {trader.strategy.fast_period}/{trader.strategy.slow_period}"
    )

    CHECK_INTERVAL = 300  # 5 minutes
    iteration = 0

    try:
        while not shutdown_requested:
            iteration += 1
            try:
                _tick(trader, iteration)
            except Exception as e:
                log.error(f"Tick error: {e}", exc_info=True)
                trader.notifier.notify_risk_event("ERROR", {"message": str(e)})
                time.sleep(60)  # Wait 1min on error before retry
                continue

            time.sleep(CHECK_INTERVAL)

    finally:
        # Close open positions on shutdown
        if trader.position > 0:
            price = trader.get_current_price()
            if price:
                log.info(f"Closing position: {trader.position} BTC @ ${price}")
                trader.execute_sell(price, "SERVICE_SHUTDOWN")

        summary = trader.logger.get_daily_summary()
        trader.logger.print_summary()
        trader.notifier.notify_session_end(summary)
        log.info("Bot service stopped cleanly.")


def _tick(trader, iteration):
    """Single trading tick — called every CHECK_INTERVAL."""
    import pandas as pd

    if not trader.is_active:
        log.warning("Trading paused (max drawdown). Waiting for manual reset.")
        return

    price = trader.get_current_price()
    balance = trader.get_balance()

    if price is None or balance is None:
        log.warning("Failed to fetch price/balance")
        return

    # Fetch candles and calculate signals
    klines = trader.adapter.get_klines(trader.symbol, "1h", limit=50)
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["close"] = df["close"].astype(float)

    df_signals = trader.strategy.calculate_signals(df)
    current_signal = df_signals["signal"].iloc[-1]
    ema_fast = df_signals["ema_fast"].iloc[-1]
    ema_slow = df_signals["ema_slow"].iloc[-1]

    # Check drawdown
    current_value = balance + (trader.position * price if trader.position > 0 else 0)
    if trader.risk_manager.check_drawdown(current_value):
        trader.is_active = False
        log.error(f"MAX DRAWDOWN at ${current_value:.2f}")
        trader.notifier.notify_risk_event("MAX_DRAWDOWN", {"balance": current_value})
        return

    # Check SL/TP
    if trader.position > 0:
        should_exit, reason = trader.risk_manager.should_exit(trader.entry_price, price)
        if should_exit:
            log.info(f"Risk exit: {reason}")
            trader.execute_sell(price, reason)
            return

    # Process signals
    if current_signal == 1 and trader.position == 0:
        log.info(f"BUY SIGNAL | ${price:,.2f} | EMA: {ema_fast:.2f}/{ema_slow:.2f}")
        result = trader.execute_buy(price)
        if result:
            log.info(f"BUY EXECUTED | {trader.position:.5f} BTC @ ${price:,.2f}")
        else:
            log.warning(f"BUY FAILED | balance=${trader.get_balance()}, price=${price}")
    elif current_signal == -1 and trader.position > 0:
        log.info(f"SELL SIGNAL | ${price:,.2f} | EMA: {ema_fast:.2f}/{ema_slow:.2f}")
        result = trader.execute_sell(price, "STRATEGY_SIGNAL")
        if result:
            log.info(f"SELL EXECUTED | PnL logged")
        else:
            log.warning(f"SELL FAILED | position={trader.position}")
    else:
        pos_str = f"{trader.position:.5f} BTC" if trader.position > 0 else "no position"
        signal_str = "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "HOLD"
        if iteration % 12 == 0:  # Log every hour (12 * 5min)
            log.info(f"Heartbeat | {signal_str} | ${price:,.2f} | {pos_str} | Balance: ${balance:.2f}")


if __name__ == "__main__":
    run()
