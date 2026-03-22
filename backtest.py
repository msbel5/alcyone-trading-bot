#!/usr/bin/env python3
"""
Backtest Engine
Tests a trading strategy on historical data and calculates PnL.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class Backtester:
    def __init__(self, initial_balance=1000, fee_rate=0.001):
        """
        :param initial_balance: Starting capital (USDT)
        :param fee_rate: Trading fee (0.1% = 0.001)
        """
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.reset()

    def reset(self):
        """Reset backtest state."""
        self.balance = self.initial_balance
        self.position = 0  # BTC held
        self.position_value = 0
        self.trades = []
        self.equity_curve = []

    def run(self, df, signals):
        """
        Execute backtest on historical data.
        :param df: DataFrame with price data (must have 'close' column)
        :param signals: List of trade signals from strategy
        :return: Dict with performance metrics
        """
        self.reset()
        
        for signal in signals:
            timestamp = signal['timestamp']
            action = signal['action']
            price = signal['price']
            
            if action == 'BUY' and self.position == 0:
                # Buy with all available balance
                cost = self.balance * (1 - self.fee_rate)
                amount = cost / price
                self.position = amount
                self.position_value = cost
                self.balance = 0
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': price,
                    'amount': amount,
                    'cost': cost,
                    'balance': self.balance,
                    'equity': cost
                })
            
            elif action == 'SELL' and self.position > 0:
                # Sell all position
                revenue = self.position * price * (1 - self.fee_rate)
                pnl = revenue - self.position_value
                pnl_pct = (pnl / self.position_value) * 100
                
                self.balance = revenue
                self.position = 0
                self.position_value = 0
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': price,
                    'amount': 0,
                    'revenue': revenue,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'balance': self.balance,
                    'equity': self.balance
                })
        
        # Calculate final equity (if still holding)
        if self.position > 0:
            last_price = df['close'].iloc[-1]
            final_value = self.position * last_price * (1 - self.fee_rate)
            self.balance = final_value
            self.position = 0
        
        return self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if len(self.trades) < 2:
            return {
                'total_trades': 0,
                'final_balance': self.balance,
                'total_return': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        # Extract completed trades (BUY → SELL pairs)
        completed = [t for t in self.trades if t['action'] == 'SELL']
        
        wins = [t['pnl'] for t in completed if t['pnl'] > 0]
        losses = [t['pnl'] for t in completed if t['pnl'] <= 0]
        
        return {
            'total_trades': len(completed),
            'final_balance': round(self.balance, 2),
            'total_return': round(self.balance - self.initial_balance, 2),
            'total_return_pct': round(((self.balance / self.initial_balance) - 1) * 100, 2),
            'win_rate': round((len(wins) / len(completed)) * 100, 2) if completed else 0,
            'avg_win': round(np.mean(wins), 2) if wins else 0,
            'avg_loss': round(np.mean(losses), 2) if losses else 0,
            'trades': self.trades
        }

    def print_report(self, metrics):
        """Print formatted backtest report."""
        print("\n" + "="*60)
        print("BACKTEST REPORT")
        print("="*60)
        print(f"Initial Balance:     ${self.initial_balance:.2f}")
        print(f"Final Balance:       ${metrics['final_balance']:.2f}")
        print(f"Total Return:        ${metrics['total_return']:.2f} ({metrics['total_return_pct']:+.2f}%)")
        print(f"Total Trades:        {metrics['total_trades']}")
        print(f"Win Rate:            {metrics['win_rate']:.2f}%")
        print(f"Avg Win:             ${metrics['avg_win']:.2f}")
        print(f"Avg Loss:            ${metrics['avg_loss']:.2f}")
        print("="*60)
        
        if metrics['trades']:
            print("\nTrade History:")
            print(f"{'Timestamp':<20} {'Action':<6} {'Price':<10} {'PnL':<12} {'Balance':<12}")
            print("-" * 60)
            for t in metrics['trades']:
                pnl_str = f"${t.get('pnl', 0):+.2f}" if 'pnl' in t else "-"
                print(f"{str(t['timestamp']):<20} {t['action']:<6} ${t['price']:<9.2f} {pnl_str:<12} ${t.get('balance', 0):<11.2f}")


if __name__ == '__main__':
    # Test backtest engine
    import sys
    sys.path.append('/home/msbel/.openclaw/workspace/trading')
    from strategies.ma_crossover import MACrossover
    
    # Load data
    df = pd.read_csv('/home/msbel/.openclaw/workspace/trading/data/btcusdt_1h.csv')
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Generate signals
    strategy = MACrossover(fast_period=7, slow_period=25)
    signals = strategy.get_trade_signals(df)
    
    # Run backtest
    backtester = Backtester(initial_balance=1000, fee_rate=0.001)
    metrics = backtester.run(df, signals)
    backtester.print_report(metrics)
