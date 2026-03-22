#!/usr/bin/env python3
"""
Strategy Optimizer
Grid search over EMA periods to find best parameters.
"""

import pandas as pd
import sys
sys.path.append('/home/msbel/.openclaw/workspace/trading')

from strategies.ma_crossover import MACrossover
from backtest import Backtester


def optimize_strategy(df, fast_range, slow_range, initial_balance=1000):
    """
    Grid search over EMA periods.
    :param df: Price data
    :param fast_range: List of fast EMA periods to test
    :param slow_range: List of slow EMA periods to test
    :return: DataFrame with all results, sorted by return
    """
    results = []
    
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:  # Skip invalid combinations
                continue
            
            strategy = MACrossover(fast_period=fast, slow_period=slow)
            signals = strategy.get_trade_signals(df)
            
            if len(signals) < 2:  # Need at least 1 complete trade
                continue
            
            backtester = Backtester(initial_balance=initial_balance)
            metrics = backtester.run(df, signals)
            
            results.append({
                'fast': fast,
                'slow': slow,
                'trades': metrics['total_trades'],
                'return': metrics['total_return'],
                'return_pct': metrics['total_return_pct'],
                'win_rate': metrics['win_rate'],
                'avg_win': metrics['avg_win'],
                'avg_loss': metrics['avg_loss']
            })
    
    return pd.DataFrame(results).sort_values('return_pct', ascending=False)


if __name__ == '__main__':
    # Load data
    df = pd.read_csv('/home/msbel/.openclaw/workspace/trading/data/btcusdt_1h.csv')
    df['close'] = df['close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    print("Starting optimization...")
    print("Testing fast: [5, 7, 10, 12] × slow: [20, 25, 30, 35]")
    print()
    
    results = optimize_strategy(
        df,
        fast_range=[5, 7, 10, 12],
        slow_range=[20, 25, 30, 35]
    )
    
    print("="*80)
    print("TOP 10 PARAMETER COMBINATIONS")
    print("="*80)
    print(results.head(10).to_string(index=False))
    
    print("\n" + "="*80)
    print("WORST 5 PARAMETER COMBINATIONS")
    print("="*80)
    print(results.tail(5).to_string(index=False))
    
    # Save full results
    results.to_csv('/home/msbel/.openclaw/workspace/trading/optimization_results.csv', index=False)
    print(f"\nFull results saved to: trading/optimization_results.csv")
