#!/usr/bin/env python3
"""
Sprint 2: Multi-Timeframe Filter + Coin Correlation Filter
"""
import sys
import numpy as np
import pandas as pd
import logging

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

log = logging.getLogger("filters")


# ── Multi-Timeframe Filter ──────────────────────────────────────────

class MultiTimeframeFilter:
    """Reject signals that conflict with higher timeframe trend."""

    def __init__(self, ema_fast=12, ema_slow=26):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self._4h_trends = {}  # symbol -> trend direction

    def update_4h_trend(self, symbol: str, klines_4h: list):
        """Calculate 4h trend from kline data."""
        if not klines_4h or len(klines_4h) < self.ema_slow:
            return
        closes = [float(k[4]) for k in klines_4h]
        series = pd.Series(closes)
        ema_f = series.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
        ema_s = series.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]
        if ema_f > ema_s * 1.001:
            self._4h_trends[symbol] = 1  # Bullish
        elif ema_f < ema_s * 0.999:
            self._4h_trends[symbol] = -1  # Bearish
        else:
            self._4h_trends[symbol] = 0  # Neutral

    def filter_signal(self, symbol: str, signal_1h: int) -> int:
        """Filter 1h signal against 4h trend."""
        trend_4h = self._4h_trends.get(symbol, 0)
        # BUY rejected if 4h bearish
        if signal_1h == 1 and trend_4h == -1:
            log.debug(f"{symbol}: BUY rejected (4h bearish)")
            return 0
        # SELL rejected if 4h bullish
        if signal_1h == -1 and trend_4h == 1:
            log.debug(f"{symbol}: SELL rejected (4h bullish)")
            return 0
        return signal_1h

    def get_trend(self, symbol: str) -> int:
        return self._4h_trends.get(symbol, 0)


# ── Correlation Filter ──────────────────────────────────────────────

class CorrelationFilter:
    """Limit positions in highly correlated coins."""

    def __init__(self, max_correlated: int = 3, threshold: float = 0.75,
                 lookback_hours: int = 720):
        self.max_correlated = max_correlated
        self.threshold = threshold
        self.lookback_hours = lookback_hours
        self._corr_matrix = None
        self._last_update = 0

    def update_correlation(self, price_data: dict):
        """
        Update correlation matrix from recent price data.
        price_data: {symbol: pd.Series of close prices}
        """
        if not price_data:
            return
        df = pd.DataFrame(price_data)
        returns = df.pct_change().dropna()
        if len(returns) < 30:
            return
        self._corr_matrix = returns.corr()

    def allow_entry(self, symbol: str, open_positions: dict) -> bool:
        """Check if we can open a new position for this symbol."""
        if self._corr_matrix is None:
            return True  # No data, allow

        open_symbols = [s for s, pos in open_positions.items() if pos > 0]
        if len(open_symbols) < self.max_correlated:
            return True  # Below limit, allow

        # Count highly correlated open positions
        correlated_count = 0
        for open_sym in open_symbols:
            if symbol in self._corr_matrix.index and open_sym in self._corr_matrix.columns:
                corr = abs(self._corr_matrix.loc[symbol, open_sym])
                if corr > self.threshold:
                    correlated_count += 1

        if correlated_count >= self.max_correlated:
            log.debug(f"{symbol}: Entry blocked ({correlated_count} correlated positions)")
            return False
        return True

    def get_matrix(self) -> pd.DataFrame:
        return self._corr_matrix


# ── Kelly Criterion ─────────────────────────────────────────────────

def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                   half_kelly: bool = True, max_fraction: float = 0.25) -> float:
    """
    Calculate optimal position size using Kelly Criterion.
    Returns fraction of bankroll to bet (0 to max_fraction).
    """
    if avg_loss == 0 or win_rate <= 0:
        return 0.0
    b = abs(avg_win / avg_loss)  # Win/loss ratio
    p = win_rate
    q = 1 - p
    kelly = (b * p - q) / b
    if half_kelly:
        kelly *= 0.5
    return max(0.0, min(kelly, max_fraction))


# ── Tests ───────────────────────────────────────────────────────────

def test_all():
    print("Testing Sprint 2 components...\n")

    # Multi-timeframe
    mtf = MultiTimeframeFilter()
    # Simulate 4h bullish trend
    fake_4h = [[0, 0, 0, 0, str(100 + i * 0.5), 0] for i in range(30)]
    mtf.update_4h_trend("BTCUSDT", fake_4h)
    assert mtf.get_trend("BTCUSDT") == 1, "Should be bullish"
    assert mtf.filter_signal("BTCUSDT", 1) == 1, "BUY should pass in bullish 4h"
    assert mtf.filter_signal("BTCUSDT", -1) == 0, "SELL should be rejected in bullish 4h"
    print("  [PASS] Multi-timeframe filter")

    # Correlation
    corr = CorrelationFilter(max_correlated=2, threshold=0.7)
    prices = {
        "BTCUSDT": pd.Series(np.random.randn(100).cumsum() + 100),
        "ETHUSDT": pd.Series(np.random.randn(100).cumsum() + 100),
        "DOGEUSDT": pd.Series(np.random.randn(100).cumsum() + 10),
    }
    corr.update_correlation(prices)
    assert corr._corr_matrix is not None, "Matrix should exist"
    assert corr.allow_entry("BTCUSDT", {}), "Should allow with no positions"
    print("  [PASS] Correlation filter")

    # Kelly
    k = kelly_fraction(0.55, 20.0, 15.0, half_kelly=True)
    assert 0 < k < 0.25, f"Kelly should be positive and capped: {k}"
    k_neg = kelly_fraction(0.3, 10.0, 15.0)
    assert k_neg == 0, "Negative edge should return 0"
    print(f"  [PASS] Kelly criterion (f={k:.3f})")

    # Trailing stop
    from backtester import TrailingStop
    ts = TrailingStop(atr_multiplier=2.0)
    ts.update(100.0, 2.0)  # Price 100, ATR 2 → SL = 96
    assert ts.trailing_sl == 96.0, f"SL should be 96, got {ts.trailing_sl}"
    ts.update(105.0, 2.0)  # Price rises → SL = 101
    assert ts.trailing_sl == 101.0, f"SL should rise to 101, got {ts.trailing_sl}"
    ts.update(103.0, 2.0)  # Price drops but SL stays
    assert ts.trailing_sl == 101.0, "SL should not decrease"
    assert not ts.should_exit(102.0), "102 > 101, no exit"
    assert ts.should_exit(100.5), "100.5 < 101, should exit"
    print("  [PASS] Trailing stop")

    print("\nAll Sprint 2 tests passed!")


if __name__ == "__main__":
    test_all()
