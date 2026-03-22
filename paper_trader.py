#!/usr/bin/env python3
"""
Paper Trading Engine
Simulates live trading without real orders.
"""

import sys
import time
import pandas as pd
sys.path.append('/home/msbel/.openclaw/workspace/trading')

from strategies.ma_crossover import MACrossover
from risk_manager import RiskManager
from trade_logger import TradeLogger


class PaperTrader:
    def __init__(
        self,
        initial_balance=1000,
        strategy=None,
        risk_manager=None,
        logger=None
    ):
        """
        :param initial_balance: Starting USDT balance
        :param strategy: Trading strategy instance
        :param risk_manager: Risk manager instance
        :param logger: Trade logger instance
        """
        self.balance = initial_balance
        self.position = 0  # BTC held
        self.entry_price = None
        
        self.strategy = strategy or MACrossover(fast_period=12, slow_period=30)
        self.risk_manager = risk_manager or RiskManager()
        self.logger = logger or TradeLogger()
        
        self.trade_count = 0
        self.is_active = True

    def process_candle(self, candle):
        """
        Process a single price candle.
        :param candle: Dict with timestamp, close, ema_fast, ema_slow, signal
        """
        if not self.is_active:
            return
        
        timestamp = candle['timestamp']
        price = candle['close']
        signal = candle.get('signal', 0)
        
        # Check drawdown
        current_value = self.balance + (self.position * price if self.position > 0 else 0)
        if self.risk_manager.check_drawdown(current_value):
            self.logger.log_risk_event('MAX_DRAWDOWN_REACHED', {
                'balance': current_value,
                'threshold': self.risk_manager.max_drawdown_pct * 100
            })
            self.is_active = False
            print(f"⚠️ Trading paused: max drawdown reached at ${current_value:.2f}")
            return
        
        # Check risk exit (SL/TP)
        if self.position > 0:
            should_exit, reason = self.risk_manager.should_exit(self.entry_price, price)
            if should_exit:
                self._execute_sell(timestamp, price, reason)
                return
        
        # Process strategy signals
        if signal == 1 and self.position == 0:  # Buy signal
            self._execute_buy(timestamp, price)
        
        elif signal == -1 and self.position > 0:  # Sell signal
            self._execute_sell(timestamp, price, 'STRATEGY_SIGNAL')

    def _execute_buy(self, timestamp, price):
        """Execute a buy order."""
        amount = self.risk_manager.calculate_position_size(self.balance, price)
        cost = amount * price * 1.001  # 0.1% fee
        
        if cost > self.balance:
            self.logger.log_error('Insufficient balance for buy', {
                'required': cost,
                'available': self.balance
            })
            return
        
        self.position = amount
        self.entry_price = price
        self.balance -= cost
        self.trade_count += 1
        
        self.logger.log_buy(price, amount, self.balance, strategy='MA_CROSS_12_30')
        
        print(f"🟢 BUY  @ ${price:,.2f} | Amount: {amount:.6f} BTC | Balance: ${self.balance:.2f}")

    def _execute_sell(self, timestamp, price, reason='SIGNAL'):
        """Execute a sell order."""
        revenue = self.position * price * 0.999  # 0.1% fee
        cost = self.position * self.entry_price
        pnl = revenue - cost
        pnl_pct = (pnl / cost) * 100
        
        self.balance += revenue
        self.position = 0
        
        self.logger.log_sell(price, self.position, pnl, pnl_pct, self.balance, reason)
        
        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"{emoji} SELL @ ${price:,.2f} | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Balance: ${self.balance:.2f} | Reason: {reason}")
        
        self.entry_price = None

    def get_status(self):
        """Get current trading status."""
        return {
            'balance': round(self.balance, 2),
            'position': round(self.position, 6),
            'entry_price': round(self.entry_price, 2) if self.entry_price else None,
            'trades': self.trade_count,
            'active': self.is_active
        }


def run_paper_trading_simulation(data_path, fast=12, slow=30, verbose=True):
    """
    Run paper trading simulation on historical data.
    :param data_path: Path to CSV with OHLCV data
    :param fast: Fast EMA period
    :param slow: Slow EMA period
    :param verbose: Print trade details
    """
    # Load data
    df = pd.read_csv(data_path)
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Generate signals
    strategy = MACrossover(fast_period=fast, slow_period=slow)
    df = strategy.calculate_signals(df)
    
    # Initialize paper trader
    trader = PaperTrader(
        initial_balance=1000,
        strategy=strategy,
        risk_manager=RiskManager(
            max_position_size=0.95,
            stop_loss_pct=2.0,
            take_profit_pct=5.0,
            max_drawdown_pct=10.0
        )
    )
    
    print("="*70)
    print(f"PAPER TRADING SIMULATION (EMA {fast}/{slow})")
    print("="*70)
    print(f"Initial Balance: ${trader.balance:.2f}")
    print(f"Data: {len(df)} candles ({df.index[0]} → {df.index[-1]})")
    print()
    
    # Process each candle
    for idx, row in df.iterrows():
        candle = {
            'timestamp': idx,
            'close': row['close'],
            'signal': row['signal']
        }
        trader.process_candle(candle)
    
    # Close any open position at end
    if trader.position > 0:
        last_price = df['close'].iloc[-1]
        trader._execute_sell(df.index[-1], last_price, 'END_OF_DATA')
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    status = trader.get_status()
    print(f"Final Balance:   ${status['balance']:.2f}")
    print(f"Total Return:    ${status['balance'] - 1000:+.2f} ({((status['balance']/1000) - 1)*100:+.2f}%)")
    print(f"Total Trades:    {status['trades']}")
    print("="*70)
    
    trader.logger.print_summary()


if __name__ == '__main__':
    run_paper_trading_simulation(
        data_path='/home/msbel/.openclaw/workspace/trading/data/btcusdt_1h.csv',
        fast=12,
        slow=30
    )
