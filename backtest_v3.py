#!/usr/bin/env python3
"""
Backtest v3 — Compare old (v2) vs new (v3) ML performance.
Tests on 2 years of data, 7 coins, trailing stop + regime detection.
"""
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("backtest_v3")


def backtest_single(symbol: str, df: pd.DataFrame, strategy_fn, initial_balance: float = 100.0,
                     position_pct: float = 0.90, sl_pct: float = 3.0, tp_pct: float = 8.0,
                     use_trailing: bool = True, atr_mult: float = 2.0) -> dict:
    """
    Backtest a single symbol with given strategy function.
    strategy_fn(df, idx) → signal: -1, 0, 1
    """
    balance = initial_balance
    position = 0.0
    entry_price = None
    trailing_sl = None
    peak_balance = initial_balance
    trades = []
    equity_curve = [balance]

    for i in range(50, len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        atr = float(row.get("atr", price * 0.02))

        current_value = balance + position * price

        # Trailing stop check
        if position > 0 and use_trailing:
            new_sl = price - atr * atr_mult
            if trailing_sl is None:
                trailing_sl = new_sl
            else:
                trailing_sl = max(trailing_sl, new_sl)

            if price <= trailing_sl:
                # Trailing stop hit
                pnl = position * (price - entry_price)
                balance += position * price
                trades.append({"type": "trailing_stop", "pnl": pnl, "price": price})
                position = 0.0
                entry_price = None
                trailing_sl = None
                equity_curve.append(balance)
                continue

        signal = strategy_fn(df, i)

        if signal == 1 and position == 0:
            # BUY
            invest = balance * position_pct
            position = invest / price
            balance -= invest
            entry_price = price
            trailing_sl = price - atr * atr_mult

        elif signal == -1 and position > 0:
            # SELL
            pnl = position * (price - entry_price)
            balance += position * price
            trades.append({"type": "sell", "pnl": pnl, "price": price})
            position = 0.0
            entry_price = None
            trailing_sl = None

        current_value = balance + position * price
        equity_curve.append(current_value)
        peak_balance = max(peak_balance, current_value)

    # Close any open position
    if position > 0:
        price = float(df.iloc[-1]["close"])
        pnl = position * (price - entry_price)
        balance += position * price
        trades.append({"type": "close", "pnl": pnl, "price": price})

    final_value = balance
    total_return = (final_value / initial_balance - 1) * 100
    max_drawdown = 0
    peak = initial_balance
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100
        max_drawdown = max(max_drawdown, dd)

    # Sharpe ratio (hourly)
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe = np.mean(returns) / max(np.std(returns), 1e-8) * np.sqrt(8760)  # Annualized

    win_trades = [t for t in trades if t["pnl"] > 0]
    lose_trades = [t for t in trades if t["pnl"] <= 0]

    return {
        "symbol": symbol,
        "initial": initial_balance,
        "final": round(final_value, 2),
        "return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "sharpe": round(sharpe, 3),
        "n_trades": len(trades),
        "win_rate": round(len(win_trades) / max(len(trades), 1) * 100, 1),
        "avg_pnl": round(np.mean([t["pnl"] for t in trades]), 4) if trades else 0,
    }


def strategy_v2(df, idx):
    """v2 strategy: simple composite threshold."""
    row = df.iloc[idx]
    composite = float(row.get("composite", 0))
    if composite >= 0.40:
        return 1
    elif composite <= -0.40:
        return -1
    return 0


def strategy_v3(df, idx, regime_params=None):
    """v3 strategy: regime-adjusted thresholds."""
    row = df.iloc[idx]
    composite = float(row.get("composite", 0))

    # Regime-adjusted thresholds
    buy_th = 0.40
    sell_th = -0.40
    if regime_params:
        buy_th = regime_params.get("buy_threshold", 0.40)
        sell_th = regime_params.get("sell_threshold", -0.40)

    if composite >= buy_th:
        return 1
    elif composite <= sell_th:
        return -1
    return 0


def run_full_backtest():
    """Compare v2 vs v3 across all symbols."""
    from ml.data_pipeline import load_ohlcv, add_labels
    from strategies.pro_strategy import ProStrategy

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]

    print("=" * 80)
    print("BACKTEST v3 — Old vs New Comparison")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)

    v2_results = []
    v3_results = []

    strategy = ProStrategy()

    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"  {sym}")
        print(f"{'='*60}")

        try:
            df = load_ohlcv(sym)

            # Add strategy signals
            df = strategy.calculate_signals(df)

            # v2 backtest (fixed thresholds)
            r2 = backtest_single(sym, df, strategy_v2, use_trailing=True)
            v2_results.append(r2)
            print(f"  v2: return={r2['return_pct']:+.1f}%, DD={r2['max_drawdown_pct']:.1f}%, "
                  f"Sharpe={r2['sharpe']:.3f}, trades={r2['n_trades']}, WR={r2['win_rate']:.0f}%")

            # v3 backtest (regime-adjusted)
            try:
                from ml.ml_v3 import RegimeDetector, add_features_v3
                df_v3 = add_features_v3(df.copy())
                df_v3 = strategy.calculate_signals(df_v3)
                detector = RegimeDetector()
                regime = detector.detect(df_v3)
                regime_params = detector.get_params(regime)

                def strat_v3(d, i):
                    return strategy_v3(d, i, regime_params)

                r3 = backtest_single(sym, df_v3, strat_v3, use_trailing=True)
                v3_results.append(r3)
                print(f"  v3: return={r3['return_pct']:+.1f}%, DD={r3['max_drawdown_pct']:.1f}%, "
                      f"Sharpe={r3['sharpe']:.3f}, trades={r3['n_trades']}, WR={r3['win_rate']:.0f}% "
                      f"[regime={regime}]")
            except Exception as e:
                print(f"  v3 failed: {e}")
                v3_results.append(r2)  # Use v2 as fallback

        except Exception as e:
            print(f"  ERROR: {e}")

    # Portfolio comparison
    print(f"\n{'='*80}")
    print("PORTFOLIO COMPARISON")
    print(f"{'='*80}")

    v2_total = sum(r["final"] for r in v2_results) if v2_results else 0
    v3_total = sum(r["final"] for r in v3_results) if v3_results else 0
    initial_total = 100 * len(symbols)

    v2_return = (v2_total / initial_total - 1) * 100
    v3_return = (v3_total / initial_total - 1) * 100

    v2_avg_dd = np.mean([r["max_drawdown_pct"] for r in v2_results]) if v2_results else 0
    v3_avg_dd = np.mean([r["max_drawdown_pct"] for r in v3_results]) if v3_results else 0

    v2_avg_sharpe = np.mean([r["sharpe"] for r in v2_results]) if v2_results else 0
    v3_avg_sharpe = np.mean([r["sharpe"] for r in v3_results]) if v3_results else 0

    print(f"\n{'Metric':<25} {'v2 (Old)':<20} {'v3 (New)':<20} {'Target':<15}")
    print("-" * 80)
    print(f"{'Portfolio Return':<25} {v2_return:>+7.2f}%{'':<11} {v3_return:>+7.2f}%{'':<11} >10%")
    print(f"{'Avg Max Drawdown':<25} {v2_avg_dd:>7.1f}%{'':<11} {v3_avg_dd:>7.1f}%{'':<11} <15%")
    print(f"{'Avg Sharpe Ratio':<25} {v2_avg_sharpe:>7.3f}{'':<12} {v3_avg_sharpe:>7.3f}{'':<12} >0.5")
    print(f"{'Total Value ($700)':<25} ${v2_total:>7.2f}{'':<11} ${v3_total:>7.2f}{'':<11}")

    improved = v3_return > v2_return
    print(f"\n{'✅ v3 IMPROVED' if improved else '⚠️ v3 NOT BETTER'} — "
          f"Δreturn={v3_return - v2_return:+.2f}%, Δdrawdown={v3_avg_dd - v2_avg_dd:+.1f}%")

    if improved:
        print("\n🚀 RECOMMENDATION: Deploy v3 models to production bot.")
    else:
        print("\n⚠️ RECOMMENDATION: Keep v2, investigate v3 underperformance.")

    return {"v2": v2_results, "v3": v3_results}


if __name__ == "__main__":
    run_full_backtest()
