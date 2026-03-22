#!/usr/bin/env python3
"""
MA Crossover Strategy
Generates buy/sell signals based on EMA 7/25 crossover.
"""

import pandas as pd
import numpy as np


class MACrossover:
    def __init__(self, fast_period=7, slow_period=25):
        """
        :param fast_period: Short EMA period (default 7)
        :param slow_period: Long EMA period (default 25)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period

    def calculate_signals(self, df):
        """
        Calculate EMA crossover signals.
        :param df: DataFrame with 'close' column
        :return: DataFrame with ema_fast, ema_slow, signal columns
        """
        df = df.copy()
        
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        
        # Generate signals: 1 = buy, -1 = sell, 0 = hold
        df['signal'] = 0
        df.loc[df['ema_fast'] > df['ema_slow'], 'signal'] = 1  # Buy when fast > slow
        df.loc[df['ema_fast'] < df['ema_slow'], 'signal'] = -1  # Sell when fast < slow
        
        # Detect crossovers (position change)
        df['position'] = df['signal'].diff()
        # position == 2: bullish crossover (buy signal)
        # position == -2: bearish crossover (sell signal)
        
        return df

    def get_trade_signals(self, df):
        """
        Extract only the actual trade entry points.
        :param df: DataFrame with signal column
        :return: List of dicts with timestamp, action, price
        """
        df = self.calculate_signals(df)
        trades = []
        
        for i in range(1, len(df)):
            if df['position'].iloc[i] == 2:  # Bullish crossover
                trades.append({
                    'timestamp': df.index[i],
                    'action': 'BUY',
                    'price': df['close'].iloc[i],
                    'ema_fast': df['ema_fast'].iloc[i],
                    'ema_slow': df['ema_slow'].iloc[i]
                })
            elif df['position'].iloc[i] == -2:  # Bearish crossover
                trades.append({
                    'timestamp': df.index[i],
                    'action': 'SELL',
                    'price': df['close'].iloc[i],
                    'ema_fast': df['ema_fast'].iloc[i],
                    'ema_slow': df['ema_slow'].iloc[i]
                })
        
        return trades


if __name__ == '__main__':
    # Test with sample data
    import sys
    sys.path.append('/home/msbel/.openclaw/workspace/trading')
    
    # Load recent BTC data
    df = pd.read_csv('/home/msbel/.openclaw/workspace/trading/data/btcusdt_1h.csv')
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Calculate signals
    strategy = MACrossover(fast_period=7, slow_period=25)
    trades = strategy.get_trade_signals(df)
    
    print(f"Total signals: {len(trades)}")
    print("\nLast 5 signals:")
    for trade in trades[-5:]:
        print(f"{trade['timestamp']} | {trade['action']:4s} @ ${trade['price']:.2f} | EMA7: {trade['ema_fast']:.2f} | EMA25: {trade['ema_slow']:.2f}")
