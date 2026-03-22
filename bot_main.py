#!/usr/bin/env python3
"""
Trading Bot Main Loop
Continuously monitors market, generates signals, and executes trades.
"""

import sys
import time
import json
import pandas as pd
from datetime import datetime
sys.path.append('/home/msbel/.openclaw/workspace/trading')

from live_trader import LiveTrader


def run_trading_loop(trader, interval_seconds=300, max_iterations=None):
    """
    Main trading loop.
    :param trader: LiveTrader instance
    :param interval_seconds: Check interval (default 5 minutes)
    :param max_iterations: Max iterations (None = infinite)
    """
    iteration = 0
    
    print("="*70)
    print("TRADING BOT STARTED")
    print("="*70)
    print(f"Mode: {'DRY RUN' if trader.dry_run else 'LIVE TRADING ⚠️'}")
    print(f"Symbol: {trader.symbol}")
    print(f"Strategy: EMA {trader.strategy.fast_period}/{trader.strategy.slow_period}")
    print(f"Check interval: {interval_seconds}s")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    print()
    
    try:
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            if not trader.is_active:
                print(f"[{timestamp}] ⚠️ Trading paused (max drawdown reached)")
                break
            
            # Get current state
            price = trader.get_current_price()
            balance = trader.get_balance()
            
            if price is None or balance is None:
                print(f"[{timestamp}] ❌ Failed to fetch data, retrying...")
                time.sleep(10)
                continue
            
            # Fetch recent candles for signal calculation
            try:
                klines = trader.adapter.get_klines(trader.symbol, '1h', limit=50)
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                df['close'] = df['close'].astype(float)
                
                # Calculate signals
                df_with_signals = trader.strategy.calculate_signals(df)
                current_signal = df_with_signals['signal'].iloc[-1]
                ema_fast = df_with_signals['ema_fast'].iloc[-1]
                ema_slow = df_with_signals['ema_slow'].iloc[-1]
                
            except Exception as e:
                trader.logger.log_error(f'Signal calculation failed: {e}')
                print(f"[{timestamp}] ❌ Signal error: {e}")
                time.sleep(10)
                continue
            
            # Check drawdown
            current_value = balance + (trader.position * price if trader.position > 0 else 0)
            if trader.risk_manager.check_drawdown(current_value):
                trader.is_active = False
                trader.notifier.notify_risk_event('MAX_DRAWDOWN', {
                    'balance': current_value,
                    'threshold': trader.risk_manager.max_drawdown_pct * 100
                })
                print(f"[{timestamp}] 🛑 Max drawdown reached at ${current_value:.2f}")
                break
            
            # Check risk exit (SL/TP)
            if trader.position > 0:
                should_exit, reason = trader.risk_manager.should_exit(trader.entry_price, price)
                if should_exit:
                    print(f"[{timestamp}] 🔔 Risk exit: {reason}")
                    trader.execute_sell(price, reason)
                    time.sleep(interval_seconds)
                    continue
            
            # Process strategy signals
            signal_str = "BUY" if current_signal == 1 else "SELL" if current_signal == -1 else "HOLD"
            
            if current_signal == 1 and trader.position == 0:  # Buy signal
                print(f"[{timestamp}] 🟢 BUY SIGNAL | Price: ${price:,.2f} | EMA: {ema_fast:.2f}/{ema_slow:.2f}")
                trader.execute_buy(price)
            
            elif current_signal == -1 and trader.position > 0:  # Sell signal
                print(f"[{timestamp}] 🔴 SELL SIGNAL | Price: ${price:,.2f} | EMA: {ema_fast:.2f}/{ema_slow:.2f}")
                trader.execute_sell(price, 'STRATEGY_SIGNAL')
            
            else:
                # No signal, just monitor
                pos_str = f"{trader.position:.5f} BTC" if trader.position > 0 else "No position"
                print(f"[{timestamp}] 📊 {signal_str:4s} | Price: ${price:,.2f} | {pos_str} | Balance: ${balance:.2f}")
            
            # Wait for next iteration
            time.sleep(interval_seconds)
    
    except KeyboardInterrupt:
        print("\n⏹️  Bot stopped by user")
    
    finally:
        # Close any open position
        if trader.position > 0:
            price = trader.get_current_price()
            if price:
                print(f"\nClosing open position...")
                trader.execute_sell(price, 'BOT_STOPPED')
        
        # Print final summary
        print("\n" + "="*70)
        print("TRADING SESSION ENDED")
        print("="*70)
        summary = trader.logger.get_daily_summary()
        trader.logger.print_summary()
        trader.notifier.notify_session_end(summary)
        print("="*70)


if __name__ == '__main__':
    import os
    
    # Load config
    config_path = '/home/msbel/.openclaw/workspace/trading/config/testnet_config.json'
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize trader
    trader = LiveTrader(
        api_key=config['api_key'],
        private_key_path=config['private_key_path'],
        dry_run=True,  # Change to False for live trading
        telegram_enabled=True
    )
    
    # Send start notification
    balance = trader.get_balance()
    trader.notifier.notify_session_start(
        mode='DRY RUN' if trader.dry_run else 'LIVE',
        balance=balance,
        strategy=f'EMA {trader.strategy.fast_period}/{trader.strategy.slow_period}'
    )
    
    # Test mode: run 3 iterations with 10s interval
    print("Running in TEST MODE (3 cycles, 10s interval)")
    run_trading_loop(trader, interval_seconds=10, max_iterations=3)
