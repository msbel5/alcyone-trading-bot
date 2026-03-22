#!/usr/bin/env python3
"""
Test Suite for Trading Bot
Run all component tests before deployment.
"""

import sys
sys.path.append('/home/msbel/.openclaw/workspace/trading')

from strategies.ma_crossover import MACrossover
from backtest import Backtester
import pandas as pd


def test_ma_crossover():
    """Test MA Crossover signal generation."""
    print("Testing MA Crossover Strategy...")
    
    # Create sample data
    data = {
        'close': [100, 102, 105, 103, 101, 104, 108, 110, 107, 109]
    }
    df = pd.DataFrame(data)
    
    strategy = MACrossover(fast_period=3, slow_period=5)
    result = strategy.calculate_signals(df)
    
    assert 'ema_fast' in result.columns, "Missing ema_fast column"
    assert 'ema_slow' in result.columns, "Missing ema_slow column"
    assert 'signal' in result.columns, "Missing signal column"
    
    print("✅ MA Crossover test passed")


def test_backtester():
    """Test backtest engine with known data."""
    print("\nTesting Backtest Engine...")
    
    # Create simple uptrend data
    data = {
        'close': [100 + i for i in range(20)]
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range('2026-01-01', periods=20, freq='h')
    
    # Create simple buy-hold signals
    signals = [
        {'timestamp': df.index[0], 'action': 'BUY', 'price': 100},
        {'timestamp': df.index[-1], 'action': 'SELL', 'price': 119}
    ]
    
    backtester = Backtester(initial_balance=1000, fee_rate=0.001)
    metrics = backtester.run(df, signals)
    
    assert metrics['total_trades'] == 1, "Should have 1 completed trade"
    assert metrics['final_balance'] > 1000, "Should profit in uptrend"
    assert metrics['win_rate'] == 100.0, "Single winning trade should be 100%"
    
    print(f"✅ Backtest test passed (Return: +{metrics['total_return_pct']:.2f}%)")


def test_optimization_output():
    """Verify optimization results exist."""
    print("\nChecking Optimization Results...")
    
    import os
    path = '/home/msbel/.openclaw/workspace/trading/optimization_results.csv'
    
    assert os.path.exists(path), "Optimization results not found"
    
    results = pd.read_csv(path)
    assert len(results) > 0, "Optimization results are empty"
    assert 'return_pct' in results.columns, "Missing return_pct column"
    
    best = results.iloc[0]
    print(f"✅ Optimization results valid (Best: EMA {int(best['fast'])}/{int(best['slow'])}, {best['return_pct']:+.2f}%)")


def test_data_integrity():
    """Verify market data is valid."""
    print("\nChecking Market Data...")
    
    df = pd.read_csv('/home/msbel/.openclaw/workspace/trading/data/btcusdt_1h.csv')
    
    assert len(df) > 100, "Insufficient data points"
    assert 'close' in df.columns, "Missing close column"
    assert df['close'].notna().all(), "Null values in close price"
    
    print(f"✅ Market data valid ({len(df)} candles)")


if __name__ == '__main__':
    print("="*60)
    print("TRADING BOT TEST SUITE")
    print("="*60)
    
    try:
        test_ma_crossover()
        test_backtester()
        test_optimization_output()
        test_data_integrity()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✅")
        print("="*60)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
