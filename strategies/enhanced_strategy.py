#!/usr/bin/env python3
"""
Enhanced Trading Strategy — EMA + RSI + Volume + ATR
Multi-layer signal confirmation for better win rate.
"""
import pandas as pd
import numpy as np


class EnhancedStrategy:
    def __init__(self, ema_fast=12, ema_slow=30, rsi_period=14,
                 rsi_buy_low=30, rsi_buy_high=50,
                 rsi_sell_low=60, rsi_sell_high=80,
                 volume_mult=1.2, atr_period=14,
                 atr_sl_mult=1.5, atr_tp_mult=2.5):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_buy_low = rsi_buy_low
        self.rsi_buy_high = rsi_buy_high
        self.rsi_sell_low = rsi_sell_low
        self.rsi_sell_high = rsi_sell_high
        self.volume_mult = volume_mult
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult

    def calculate_indicators(self, df):
        """Calculate all indicators on a DataFrame with OHLCV columns."""
        df = df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float) if "high" in df.columns else close
        low = df["low"].astype(float) if "low" in df.columns else close
        volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0, index=df.index)

        # EMA
        df["ema_fast"] = close.ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self.ema_slow, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"].fillna(50)

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=self.atr_period, adjust=False).mean()

        # Volume average
        df["vol_avg"] = volume.rolling(window=5, min_periods=1).mean()
        df["vol_ok"] = volume >= df["vol_avg"] * self.volume_mult

        # Dynamic SL/TP
        df["dynamic_sl"] = close - df["atr"] * self.atr_sl_mult
        df["dynamic_tp"] = close + df["atr"] * self.atr_tp_mult

        return df

    def calculate_signals(self, df):
        """Generate buy/sell signals with multi-layer confirmation."""
        df = self.calculate_indicators(df)

        # EMA trend
        ema_bull = df["ema_fast"] > df["ema_slow"]
        ema_bear = df["ema_fast"] < df["ema_slow"]

        # RSI filter
        rsi_buy_zone = (df["rsi"] >= self.rsi_buy_low) & (df["rsi"] <= self.rsi_buy_high)
        rsi_sell_zone = (df["rsi"] >= self.rsi_sell_low) & (df["rsi"] <= self.rsi_sell_high)
        rsi_strong_buy = df["rsi"] < 25
        rsi_strong_sell = df["rsi"] > 75

        # Combined signal
        df["signal"] = 0

        # BUY: EMA bullish + RSI in buy zone + volume OK
        buy_normal = ema_bull & rsi_buy_zone & df["vol_ok"]
        buy_strong = ema_bull & rsi_strong_buy  # Strong buy even without volume
        df.loc[buy_normal | buy_strong, "signal"] = 1

        # SELL: EMA bearish + RSI in sell zone
        sell_normal = ema_bear & rsi_sell_zone
        sell_strong = rsi_strong_sell  # Strong sell regardless of EMA
        df.loc[sell_normal | sell_strong, "signal"] = -1

        # Crossover detection
        df["position"] = df["signal"].diff()

        return df

    def get_dynamic_sl_tp(self, df):
        """Return latest dynamic SL and TP."""
        if df.empty:
            return None, None
        last = df.iloc[-1]
        return float(last.get("dynamic_sl", 0)), float(last.get("dynamic_tp", 0))

    # Compatibility: same API as MACrossover
    @property
    def fast_period(self):
        return self.ema_fast

    @property
    def slow_period(self):
        return self.ema_slow
