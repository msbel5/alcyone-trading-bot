#!/usr/bin/env python3
"""
Backtest Framework — simulates ProStrategy on historical data.
Calculates: return, Sharpe, max drawdown, win rate, profit factor.
"""
import sys
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

log = logging.getLogger("backtest")


@dataclass
class Trade:
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    amount: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    symbol: str = ""


@dataclass
class BacktestResult:
    symbol: str
    strategy_name: str
    initial_balance: float
    final_balance: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


class TrailingStop:
    """ATR-based trailing stop loss."""

    def __init__(self, atr_multiplier: float = 2.0):
        self.atr_multiplier = atr_multiplier
        self.highest_since_entry = None
        self.trailing_sl = None

    def reset(self):
        self.highest_since_entry = None
        self.trailing_sl = None

    def update(self, current_price: float, current_atr: float):
        """Update trailing stop. Call every candle while in position."""
        if self.highest_since_entry is None or current_price > self.highest_since_entry:
            self.highest_since_entry = current_price
            new_sl = current_price - current_atr * self.atr_multiplier
            # SL only goes up, never down
            if self.trailing_sl is None or new_sl > self.trailing_sl:
                self.trailing_sl = new_sl

    def should_exit(self, current_price: float) -> bool:
        """Check if trailing stop is hit."""
        if self.trailing_sl is None:
            return False
        return current_price <= self.trailing_sl

    def get_sl(self) -> Optional[float]:
        return self.trailing_sl


class Backtester:
    """Run strategy on historical data and calculate performance."""

    def __init__(self, initial_balance: float = 100.0, commission_pct: float = 0.1,
                 position_pct: float = 0.90, use_trailing_stop: bool = True,
                 trailing_atr_mult: float = 2.0, fixed_sl_pct: float = 2.0,
                 fixed_tp_pct: float = 5.0):
        self.initial_balance = initial_balance
        self.commission_pct = commission_pct / 100
        self.position_pct = position_pct
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop = TrailingStop(trailing_atr_mult)
        self.fixed_sl_pct = fixed_sl_pct / 100
        self.fixed_tp_pct = fixed_tp_pct / 100

    def run(self, df: pd.DataFrame, symbol: str = "UNKNOWN",
            strategy_name: str = "ProStrategy") -> BacktestResult:
        """Run backtest on a DataFrame with 'signal', 'close', 'atr' columns."""
        balance = self.initial_balance
        position = 0.0
        entry_price = 0.0
        entry_time = ""
        trades: List[Trade] = []
        equity_curve = [balance]

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            price = float(row["close"])
            signal = int(prev.get("signal", 0))  # Use previous candle's signal
            atr = float(row.get("atr", price * 0.02))
            timestamp = str(row.name) if hasattr(row, "name") else str(i)

            # If in position: check exits
            if position > 0:
                # Trailing stop update
                if self.use_trailing_stop:
                    self.trailing_stop.update(price, atr)
                    if self.trailing_stop.should_exit(price):
                        pnl, pnl_pct = self._calc_pnl(entry_price, price, position)
                        balance += position * price * (1 - self.commission_pct)
                        trades.append(Trade(entry_time, timestamp, entry_price, price,
                                            position, pnl, pnl_pct, "TRAILING_STOP", symbol))
                        position = 0.0
                        self.trailing_stop.reset()
                        equity_curve.append(balance)
                        continue

                # Fixed SL/TP
                if not self.use_trailing_stop:
                    sl_price = entry_price * (1 - self.fixed_sl_pct)
                    tp_price = entry_price * (1 + self.fixed_tp_pct)
                    if price <= sl_price:
                        pnl, pnl_pct = self._calc_pnl(entry_price, price, position)
                        balance += position * price * (1 - self.commission_pct)
                        trades.append(Trade(entry_time, timestamp, entry_price, price,
                                            position, pnl, pnl_pct, "STOP_LOSS", symbol))
                        position = 0.0
                        equity_curve.append(balance)
                        continue
                    if price >= tp_price:
                        pnl, pnl_pct = self._calc_pnl(entry_price, price, position)
                        balance += position * price * (1 - self.commission_pct)
                        trades.append(Trade(entry_time, timestamp, entry_price, price,
                                            position, pnl, pnl_pct, "TAKE_PROFIT", symbol))
                        position = 0.0
                        equity_curve.append(balance)
                        continue

                # Strategy SELL signal
                if signal == -1:
                    pnl, pnl_pct = self._calc_pnl(entry_price, price, position)
                    balance += position * price * (1 - self.commission_pct)
                    trades.append(Trade(entry_time, timestamp, entry_price, price,
                                        position, pnl, pnl_pct, "SIGNAL_SELL", symbol))
                    position = 0.0
                    self.trailing_stop.reset()

            # If no position: check BUY
            elif signal == 1 and position == 0:
                usdt_to_use = balance * self.position_pct
                position = (usdt_to_use * (1 - self.commission_pct)) / price
                entry_price = price
                entry_time = timestamp
                balance -= usdt_to_use
                self.trailing_stop.reset()

            # Track equity
            current_equity = balance + position * price
            equity_curve.append(current_equity)

        # Close any remaining position
        if position > 0:
            final_price = float(df.iloc[-1]["close"])
            pnl, pnl_pct = self._calc_pnl(entry_price, final_price, position)
            balance += position * final_price * (1 - self.commission_pct)
            trades.append(Trade(entry_time, "END", entry_price, final_price,
                                position, pnl, pnl_pct, "END_OF_DATA", symbol))
            equity_curve.append(balance)

        return self._build_result(symbol, strategy_name, balance, trades, equity_curve)

    def _calc_pnl(self, entry: float, exit: float, amount: float):
        pnl = (exit - entry) * amount
        pnl_pct = ((exit / entry) - 1) * 100 if entry > 0 else 0
        return pnl, pnl_pct

    def _build_result(self, symbol, strategy_name, final_balance, trades, equity_curve):
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total_trades = len(trades)

        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        total_return = ((final_balance / self.initial_balance) - 1) * 100
        days = len(equity_curve) / 24  # hours to days
        ann_return = total_return * (365 / max(days, 1))

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (hourly returns annualized)
        returns = pd.Series(equity_curve).pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(24 * 365)
        else:
            sharpe = 0

        avg_trade_pnl = np.mean([t.pnl for t in trades]) if trades else 0

        return BacktestResult(
            symbol=symbol, strategy_name=strategy_name,
            initial_balance=self.initial_balance, final_balance=final_balance,
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(ann_return, 2),
            max_drawdown_pct=round(max_dd, 2),
            sharpe_ratio=round(sharpe, 3),
            win_rate=round(win_rate, 1),
            profit_factor=round(profit_factor, 2),
            total_trades=total_trades,
            avg_trade_pnl=round(avg_trade_pnl, 4),
            avg_win=round(avg_win, 4),
            avg_loss=round(avg_loss, 4),
            trades=trades,
            equity_curve=equity_curve,
        )


def run_full_backtest():
    """Run backtest on all 7 coins with ProStrategy."""
    from ml.data_pipeline import load_ohlcv, add_features
    from strategies.pro_strategy import ProStrategy

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]
    strategy = ProStrategy()

    print("=" * 70)
    print("BACKTEST: ProStrategy (6-Layer) on 8 months data")
    print("=" * 70)

    all_results = []

    for sym in symbols:
        try:
            df = load_ohlcv(sym)
            df = add_features(df)
            df = strategy.calculate_signals(df)
            df.dropna(inplace=True)

            # Test with trailing stop
            bt_trailing = Backtester(initial_balance=14.28, use_trailing_stop=True)
            result_trailing = bt_trailing.run(df, sym, "ProStrategy+TrailingStop")

            # Compare: fixed SL/TP
            bt_fixed = Backtester(initial_balance=14.28, use_trailing_stop=False)
            result_fixed = bt_fixed.run(df, sym, "ProStrategy+FixedSLTP")

            all_results.append((result_trailing, result_fixed))

            coin = sym.replace("USDT", "")
            print(f"\n{coin}:")
            print(f"  Trailing: ${result_trailing.final_balance:.2f} ({result_trailing.total_return_pct:+.1f}%) "
                  f"Sharpe={result_trailing.sharpe_ratio:.2f} DD={result_trailing.max_drawdown_pct:.1f}% "
                  f"Trades={result_trailing.total_trades} WR={result_trailing.win_rate:.0f}%")
            print(f"  Fixed:    ${result_fixed.final_balance:.2f} ({result_fixed.total_return_pct:+.1f}%) "
                  f"Sharpe={result_fixed.sharpe_ratio:.2f} DD={result_fixed.max_drawdown_pct:.1f}% "
                  f"Trades={result_fixed.total_trades} WR={result_fixed.win_rate:.0f}%")

        except Exception as e:
            print(f"\n{sym}: ERROR — {e}")

    # Portfolio summary
    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY ($100 across 7 coins)")
    print("=" * 70)

    total_trailing = sum(r[0].final_balance for r in all_results)
    total_fixed = sum(r[1].final_balance for r in all_results)
    print(f"  Trailing Stop: ${total_trailing:.2f} ({((total_trailing/100)-1)*100:+.1f}%)")
    print(f"  Fixed SL/TP:   ${total_fixed:.2f} ({((total_fixed/100)-1)*100:+.1f}%)")
    print(f"  Buy & Hold BTC: check manually")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_full_backtest()
