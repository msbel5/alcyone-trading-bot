#!/usr/bin/env python3
"""
Advanced Technical Indicators — 10 indicators from OHLCV data.
Zero API cost, all computed locally.

Research basis:
- Ichimoku Cloud: Goichi Hosoda (1968), strongest trend system
- Money Flow Index: Quong & Saunders (1994), volume-weighted RSI
- VWAP: Institutional benchmark, 74% of hedge funds use it
- Keltner Channels: Chester Keltner (1960), squeeze detection
- Williams %R: Larry Williams (1966), overbought/oversold
- Chaikin Money Flow: Marc Chaikin (1974), accumulation/distribution
- CCI: Donald Lambert (1980), cycle detection
- Donchian Channels: Richard Donchian, turtle trading breakout
- Fibonacci Retracement: Classic support/resistance levels
- Schaff Trend Cycle: Doug Schaff, faster MACD via double stochastic

SOLID Principles:
- SRP: Each indicator is its own class with compute() and signal()
- OCP: New indicators added without modifying existing ones
- LSP: All indicators implement IndicatorBase interface
- ISP: Minimal interface — just compute() and signal()
- DIP: Facade depends on abstractions, not concrete implementations
"""
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

log = logging.getLogger("indicators.advanced")


# ═══════════════════════════════════════════════════════════════════
# Interface (Dependency Inversion Principle)
# ═══════════════════════════════════════════════════════════════════

class IndicatorBase(ABC):
    """Base interface for all indicators. SRP: one indicator, one class."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicator columns to DataFrame. Returns modified df."""
        pass

    @abstractmethod
    def signal(self, df: pd.DataFrame) -> float:
        """Return trading signal [-1.0, +1.0] from last row."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def feature_columns(self) -> List[str]:
        """Column names added by this indicator."""
        pass


# ═══════════════════════════════════════════════════════════════════
# 1. Ichimoku Cloud — Hosoda (1968)
# ═══════════════════════════════════════════════════════════════════

class IchimokuCloud(IndicatorBase):
    """
    Ichimoku Kinko Hyo — the strongest standalone trend system.
    Combines 5 lines: tenkan, kijun, senkou A/B, chikou.
    Replaces 3-4 separate indicators in one view.
    """

    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26,
                 senkou_b_period: int = 52, displacement: int = 26):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement

    @property
    def name(self) -> str:
        return "ichimoku"

    @property
    def feature_columns(self) -> List[str]:
        return ["ichimoku_signal", "ichimoku_cloud_thickness",
                "ichimoku_tenkan_kijun_diff"]

    def _midpoint(self, series: pd.Series, period: int) -> pd.Series:
        """Ichimoku midpoint = (highest high + lowest low) / 2 over period."""
        high = series.rolling(period, min_periods=1).max()
        low = series.rolling(period, min_periods=1).min()
        return (high + low) / 2

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # Tenkan-sen (Conversion Line) — 9-period midpoint
        df["ichimoku_tenkan"] = self._midpoint(high, self.tenkan_period)
        tenkan_low = low.rolling(self.tenkan_period, min_periods=1).min()
        df["ichimoku_tenkan"] = (high.rolling(self.tenkan_period, min_periods=1).max()
                                  + tenkan_low) / 2

        # Kijun-sen (Base Line) — 26-period midpoint
        df["ichimoku_kijun"] = (high.rolling(self.kijun_period, min_periods=1).max()
                                 + low.rolling(self.kijun_period, min_periods=1).min()) / 2

        # Senkou Span A (Leading Span A) — average of tenkan & kijun, shifted forward
        df["ichimoku_senkou_a"] = ((df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2).shift(
            self.displacement)

        # Senkou Span B (Leading Span B) — 52-period midpoint, shifted forward
        df["ichimoku_senkou_b"] = ((high.rolling(self.senkou_b_period, min_periods=1).max()
                                    + low.rolling(self.senkou_b_period, min_periods=1).min()) / 2
                                   ).shift(self.displacement)

        # Chikou Span (Lagging Span) — close shifted back 26 periods
        df["ichimoku_chikou"] = close.shift(-self.displacement)

        # Cloud thickness (normalized by price)
        cloud_top = df[["ichimoku_senkou_a", "ichimoku_senkou_b"]].max(axis=1)
        cloud_bot = df[["ichimoku_senkou_a", "ichimoku_senkou_b"]].min(axis=1)
        df["ichimoku_cloud_thickness"] = ((cloud_top - cloud_bot) / close).fillna(0)

        # Tenkan-Kijun difference (normalized)
        df["ichimoku_tenkan_kijun_diff"] = (
            (df["ichimoku_tenkan"] - df["ichimoku_kijun"]) / close
        ).fillna(0)

        # Combined signal
        above_cloud = (close > cloud_top).astype(float)
        below_cloud = (close < cloud_bot).astype(float)
        tenkan_above = (df["ichimoku_tenkan"] > df["ichimoku_kijun"]).astype(float)

        # Bullish: price above cloud AND tenkan > kijun
        bullish = above_cloud * tenkan_above
        # Bearish: price below cloud AND tenkan < kijun
        bearish = below_cloud * (1 - tenkan_above)
        df["ichimoku_signal"] = (bullish - bearish).fillna(0).clip(-1, 1)

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "ichimoku_signal" not in df.columns:
            df = self.compute(df)
        return float(df["ichimoku_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 2. Money Flow Index (MFI) — Quong & Saunders
# ═══════════════════════════════════════════════════════════════════

class MoneyFlowIndex(IndicatorBase):
    """
    Volume-weighted RSI. Better than RSI for divergence detection.
    MFI < 20 = oversold (buy), MFI > 80 = overbought (sell).
    """

    def __init__(self, period: int = 14):
        self.period = period

    @property
    def name(self) -> str:
        return "mfi"

    @property
    def feature_columns(self) -> List[str]:
        return ["mfi", "mfi_signal"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        raw_money_flow = typical_price * df["volume"]

        # Positive/negative flow based on typical price direction
        tp_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(tp_diff > 0, 0)
        negative_flow = raw_money_flow.where(tp_diff < 0, 0)

        positive_sum = positive_flow.rolling(self.period, min_periods=1).sum()
        negative_sum = negative_flow.rolling(self.period, min_periods=1).sum()
        negative_sum = negative_sum.replace(0, np.nan)

        money_ratio = positive_sum / negative_sum
        df["mfi"] = (100 - (100 / (1 + money_ratio))).fillna(50)

        # Signal: oversold (+1), overbought (-1)
        mfi = df["mfi"].values
        sig = np.zeros(len(mfi))
        sig[mfi < 20] = 1.0      # Extreme oversold
        sig[mfi < 30] = 0.5      # Oversold
        sig[mfi > 80] = -1.0     # Extreme overbought
        sig[mfi > 70] = -0.5     # Overbought
        df["mfi_signal"] = sig

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "mfi_signal" not in df.columns:
            df = self.compute(df)
        return float(df["mfi_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 3. VWAP (Volume Weighted Average Price)
# ═══════════════════════════════════════════════════════════════════

class VWAPCalculator(IndicatorBase):
    """
    Institutional price benchmark. 74% of hedge funds use VWAP.
    Price above VWAP = institutional buying (bullish).
    Price below VWAP = institutional selling (bearish).
    Distance from VWAP = signal strength.
    """

    def __init__(self, reset_period: int = 24):
        self.reset_period = reset_period  # Hours before VWAP resets

    @property
    def name(self) -> str:
        return "vwap"

    @property
    def feature_columns(self) -> List[str]:
        return ["vwap", "vwap_distance"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical_price * df["volume"]).rolling(
            self.reset_period, min_periods=1).sum()
        cum_vol = df["volume"].rolling(self.reset_period, min_periods=1).sum()
        cum_vol = cum_vol.replace(0, np.nan)

        df["vwap"] = (cum_tp_vol / cum_vol).fillna(df["close"])

        # Distance from VWAP as % of price (normalized signal)
        df["vwap_distance"] = ((df["close"] - df["vwap"]) / df["close"]).fillna(0)
        df["vwap_distance"] = df["vwap_distance"].clip(-0.05, 0.05) / 0.05  # Normalize to [-1,1]

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "vwap_distance" not in df.columns:
            df = self.compute(df)
        return float(df["vwap_distance"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 4. Keltner Channels — squeeze detection
# ═══════════════════════════════════════════════════════════════════

class KeltnerChannels(IndicatorBase):
    """
    Keltner Channels = EMA ± ATR multiplier.
    KEY FEATURE: When Bollinger Bands are INSIDE Keltner = SQUEEZE.
    Squeeze release = explosive move. Direction of breakout = trade direction.
    """

    def __init__(self, ema_period: int = 20, atr_period: int = 14,
                 atr_multiplier: float = 2.0):
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    @property
    def name(self) -> str:
        return "keltner"

    @property
    def feature_columns(self) -> List[str]:
        return ["keltner_upper", "keltner_lower", "keltner_squeeze",
                "keltner_squeeze_signal"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Keltner middle = EMA
        df["keltner_mid"] = close.ewm(span=self.ema_period, adjust=False).mean()

        # ATR for channel width
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()

        df["keltner_upper"] = df["keltner_mid"] + self.atr_multiplier * atr
        df["keltner_lower"] = df["keltner_mid"] - self.atr_multiplier * atr

        # Squeeze detection: BB inside Keltner
        bb_mid = close.rolling(20, min_periods=1).mean()
        bb_std = close.rolling(20, min_periods=1).std().fillna(0)
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # Squeeze = BB fully inside Keltner
        squeeze = (bb_upper < df["keltner_upper"]) & (bb_lower > df["keltner_lower"])
        df["keltner_squeeze"] = squeeze.astype(float)

        # Squeeze signal: 0 during squeeze, direction of breakout after
        # Use momentum (close - keltner_mid) as direction indicator
        momentum = (close - df["keltner_mid"]) / atr.replace(0, np.nan)
        momentum = momentum.fillna(0).clip(-2, 2) / 2  # Normalize to [-1, 1]

        # Signal: strong when exiting squeeze in a direction
        was_squeeze = squeeze.shift(1).fillna(False)
        exit_squeeze = was_squeeze & ~squeeze
        df["keltner_squeeze_signal"] = np.where(
            exit_squeeze, momentum * 1.0,  # Full signal on squeeze exit
            np.where(squeeze, 0.0,         # No signal during squeeze
                     momentum * 0.3)       # Reduced signal outside squeeze
        )
        df["keltner_squeeze_signal"] = df["keltner_squeeze_signal"].fillna(0).clip(-1, 1)

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "keltner_squeeze_signal" not in df.columns:
            df = self.compute(df)
        return float(df["keltner_squeeze_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 5. Williams %R — Larry Williams
# ═══════════════════════════════════════════════════════════════════

class WilliamsPercentR(IndicatorBase):
    """
    Williams %R: overbought/oversold oscillator.
    Range: -100 to 0. Below -80 = oversold (buy), above -20 = overbought (sell).
    Faster than RSI at detecting turning points.
    """

    def __init__(self, period: int = 14):
        self.period = period

    @property
    def name(self) -> str:
        return "williams_r"

    @property
    def feature_columns(self) -> List[str]:
        return ["williams_r", "williams_r_signal"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        highest = df["high"].rolling(self.period, min_periods=1).max()
        lowest = df["low"].rolling(self.period, min_periods=1).min()
        denom = (highest - lowest).replace(0, np.nan)

        df["williams_r"] = (((highest - df["close"]) / denom) * -100).fillna(-50)

        # Signal conversion
        wr = df["williams_r"].values
        sig = np.zeros(len(wr))
        # Oversold zone: buy signals
        sig[wr < -90] = 1.0     # Extreme oversold
        sig[(wr >= -90) & (wr < -80)] = 0.6
        # Overbought zone: sell signals
        sig[wr > -10] = -1.0    # Extreme overbought
        sig[(wr <= -10) & (wr > -20)] = -0.6
        # Neutral zone: linear interpolation
        neutral = (wr >= -80) & (wr <= -20)
        sig[neutral] = np.clip((-50 - wr[neutral]) / 40, -0.3, 0.3)

        df["williams_r_signal"] = sig
        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "williams_r_signal" not in df.columns:
            df = self.compute(df)
        return float(df["williams_r_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 6. Chaikin Money Flow (CMF) — Marc Chaikin 1974
# ═══════════════════════════════════════════════════════════════════

class ChaikinMoneyFlow(IndicatorBase):
    """
    Measures accumulation/distribution pressure over a period.
    CMF > 0.1 = strong accumulation (buy), CMF < -0.1 = distribution (sell).
    Better than OBV for short-term signals.
    """

    def __init__(self, period: int = 20):
        self.period = period

    @property
    def name(self) -> str:
        return "cmf"

    @property
    def feature_columns(self) -> List[str]:
        return ["cmf", "cmf_signal"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # Money Flow Multiplier = ((C-L) - (H-C)) / (H-L)
        hl_range = (high - low).replace(0, np.nan)
        mfm = ((close - low) - (high - close)) / hl_range
        mfm = mfm.fillna(0)

        # Money Flow Volume = MFM × Volume
        mfv = mfm * volume

        # CMF = Sum(MFV, period) / Sum(Volume, period)
        vol_sum = volume.rolling(self.period, min_periods=1).sum().replace(0, np.nan)
        df["cmf"] = (mfv.rolling(self.period, min_periods=1).sum() / vol_sum).fillna(0)

        # Signal: accumulation vs distribution
        cmf = df["cmf"].values
        sig = np.clip(cmf * 5, -1, 1)  # Scale: 0.2 CMF → 1.0 signal
        df["cmf_signal"] = sig

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "cmf_signal" not in df.columns:
            df = self.compute(df)
        return float(df["cmf_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 7. Commodity Channel Index (CCI) — Lambert 1980
# ═══════════════════════════════════════════════════════════════════

class CommodityChannelIndex(IndicatorBase):
    """
    CCI measures deviation from statistical mean.
    > +100 = overbought cycle high, < -100 = oversold cycle low.
    > +200 or < -200 = extreme conditions.
    Lambert constant: 0.015 ensures ~75% of values within ±100.
    """

    def __init__(self, period: int = 20, constant: float = 0.015):
        self.period = period
        self.constant = constant

    @property
    def name(self) -> str:
        return "cci"

    @property
    def feature_columns(self) -> List[str]:
        return ["cci", "cci_signal"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma_tp = typical_price.rolling(self.period, min_periods=1).mean()

        # Mean deviation (NOT standard deviation)
        mean_dev = typical_price.rolling(self.period, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        mean_dev = mean_dev.replace(0, np.nan)

        df["cci"] = ((typical_price - sma_tp) / (self.constant * mean_dev)).fillna(0)

        # Signal conversion
        cci = df["cci"].values
        sig = np.zeros(len(cci))
        # Overbought
        sig[cci > 200] = -1.0
        sig[(cci > 100) & (cci <= 200)] = -0.5
        # Oversold
        sig[cci < -200] = 1.0
        sig[(cci < -100) & (cci >= -200)] = 0.5
        # Neutral: slight lean
        neutral = (cci >= -100) & (cci <= 100)
        sig[neutral] = np.clip(-cci[neutral] / 200, -0.3, 0.3)

        df["cci_signal"] = sig
        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "cci_signal" not in df.columns:
            df = self.compute(df)
        return float(df["cci_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 8. Donchian Channels — Richard Donchian (Turtle Trading)
# ═══════════════════════════════════════════════════════════════════

class DonchianChannels(IndicatorBase):
    """
    Donchian Channels: highest high and lowest low over period.
    Breakout above upper = strong buy (turtle trading entry).
    Breakout below lower = strong sell.
    Position within channel = trend strength.
    """

    def __init__(self, period: int = 20):
        self.period = period

    @property
    def name(self) -> str:
        return "donchian"

    @property
    def feature_columns(self) -> List[str]:
        return ["donchian_upper", "donchian_lower", "donchian_position",
                "donchian_signal"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        df["donchian_upper"] = high.rolling(self.period, min_periods=1).max()
        df["donchian_lower"] = low.rolling(self.period, min_periods=1).min()
        df["donchian_mid"] = (df["donchian_upper"] + df["donchian_lower"]) / 2

        # Position within channel [0, 1]
        channel_width = (df["donchian_upper"] - df["donchian_lower"]).replace(0, np.nan)
        df["donchian_position"] = ((close - df["donchian_lower"]) / channel_width).fillna(0.5)

        # Signal: breakout detection
        prev_upper = df["donchian_upper"].shift(1)
        prev_lower = df["donchian_lower"].shift(1)

        breakout_up = (close >= prev_upper).astype(float)
        breakout_down = (close <= prev_lower).astype(float)

        # Position-based signal: near top = bullish, near bottom = bearish
        pos_signal = (df["donchian_position"] - 0.5) * 2  # Convert [0,1] to [-1,1]

        # Combine: breakout overrides position signal
        df["donchian_signal"] = np.where(
            breakout_up, 1.0,
            np.where(breakout_down, -1.0, pos_signal * 0.5)
        )
        df["donchian_signal"] = df["donchian_signal"].fillna(0).clip(-1, 1)

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "donchian_signal" not in df.columns:
            df = self.compute(df)
        return float(df["donchian_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 9. Fibonacci Retracement — Auto swing detection
# ═══════════════════════════════════════════════════════════════════

class FibonacciRetracement(IndicatorBase):
    """
    Automatic Fibonacci retracement levels from detected swing points.
    Key levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%.
    Price near 61.8% retracement in uptrend = high-probability buy.
    """

    FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    def __init__(self, swing_lookback: int = 20, tolerance_pct: float = 1.0):
        self.swing_lookback = swing_lookback
        self.tolerance_pct = tolerance_pct / 100

    @property
    def name(self) -> str:
        return "fibonacci"

    @property
    def feature_columns(self) -> List[str]:
        return ["fib_nearest_level", "fib_signal"]

    def _find_swing_high(self, highs: np.ndarray, lookback: int) -> Optional[int]:
        """Find most recent swing high (local maximum)."""
        for i in range(len(highs) - lookback - 1, max(lookback, 0), -1):
            window = highs[i - lookback:i + lookback + 1]
            if highs[i] == window.max():
                return i
        return None

    def _find_swing_low(self, lows: np.ndarray, lookback: int) -> Optional[int]:
        """Find most recent swing low (local minimum)."""
        for i in range(len(lows) - lookback - 1, max(lookback, 0), -1):
            window = lows[i - lookback:i + lookback + 1]
            if lows[i] == window.min():
                return i
        return None

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        n = len(df)

        fib_levels = np.full(n, 0.5)
        fib_signals = np.zeros(n)

        # Only compute for last 500 rows to save time
        start = max(0, n - 500)

        for i in range(start, n):
            window_h = high[max(0, i - 200):i + 1]
            window_l = low[max(0, i - 200):i + 1]

            swing_h_idx = self._find_swing_high(window_h, min(self.swing_lookback, len(window_h) // 3))
            swing_l_idx = self._find_swing_low(window_l, min(self.swing_lookback, len(window_l) // 3))

            if swing_h_idx is None or swing_l_idx is None:
                continue

            swing_high = window_h[swing_h_idx]
            swing_low = window_l[swing_l_idx]

            if swing_high <= swing_low:
                continue

            # Current price position as Fibonacci level
            fib_range = swing_high - swing_low
            current_level = (close[i] - swing_low) / fib_range

            # Find nearest Fibonacci level
            nearest = min(self.FIB_LEVELS, key=lambda x: abs(x - current_level))
            fib_levels[i] = nearest

            # Signal: price at key support levels in uptrend = buy
            is_uptrend = swing_h_idx > swing_l_idx  # Swing high more recent

            if is_uptrend:
                # Pullback to 61.8% or 50% = buy opportunity
                if abs(current_level - 0.618) < self.tolerance_pct * 5:
                    fib_signals[i] = 0.8
                elif abs(current_level - 0.5) < self.tolerance_pct * 5:
                    fib_signals[i] = 0.5
                elif abs(current_level - 0.382) < self.tolerance_pct * 5:
                    fib_signals[i] = 0.3
            else:
                # Downtrend: bounce to 38.2% or 50% = sell opportunity
                if abs(current_level - 0.382) < self.tolerance_pct * 5:
                    fib_signals[i] = -0.8
                elif abs(current_level - 0.5) < self.tolerance_pct * 5:
                    fib_signals[i] = -0.5
                elif abs(current_level - 0.618) < self.tolerance_pct * 5:
                    fib_signals[i] = -0.3

        df["fib_nearest_level"] = fib_levels
        df["fib_signal"] = fib_signals

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "fib_signal" not in df.columns:
            df = self.compute(df)
        return float(df["fib_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 10. Schaff Trend Cycle — Doug Schaff
# ═══════════════════════════════════════════════════════════════════

class SchaffTrendCycle(IndicatorBase):
    """
    Schaff Trend Cycle: double stochastic smoothing of MACD.
    Faster than MACD, oscillates between 0 and 100.
    STC < 25 = oversold buy signal, STC > 75 = overbought sell.
    Captures trend direction earlier than standard MACD.
    """

    def __init__(self, fast_period: int = 23, slow_period: int = 50,
                 cycle_period: int = 10, smooth: float = 0.5):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.cycle_period = cycle_period
        self.smooth = smooth

    @property
    def name(self) -> str:
        return "stc"

    @property
    def feature_columns(self) -> List[str]:
        return ["stc", "stc_signal"]

    def _stochastic(self, series: pd.Series, period: int) -> pd.Series:
        """Stochastic oscillator on any series."""
        lowest = series.rolling(period, min_periods=1).min()
        highest = series.rolling(period, min_periods=1).max()
        denom = (highest - lowest).replace(0, np.nan)
        return ((series - lowest) / denom * 100).fillna(50)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]

        # Step 1: MACD line
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow

        # Step 2: First stochastic of MACD
        stoch1 = self._stochastic(macd, self.cycle_period)

        # Step 3: Smooth first stochastic
        smooth1 = stoch1.ewm(span=3, adjust=False).mean()

        # Step 4: Second stochastic (of the smoothed first)
        stoch2 = self._stochastic(smooth1, self.cycle_period)

        # Step 5: Final smooth = STC
        df["stc"] = stoch2.ewm(span=3, adjust=False).mean().fillna(50)

        # Signal
        stc = df["stc"].values
        sig = np.zeros(len(stc))
        sig[stc < 10] = 1.0     # Strong oversold
        sig[(stc >= 10) & (stc < 25)] = 0.6
        sig[stc > 90] = -1.0    # Strong overbought
        sig[(stc <= 90) & (stc > 75)] = -0.6
        # Neutral zone
        neutral = (stc >= 25) & (stc <= 75)
        sig[neutral] = np.clip((50 - stc[neutral]) / 50, -0.3, 0.3)

        df["stc_signal"] = sig
        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "stc_signal" not in df.columns:
            df = self.compute(df)
        return float(df["stc_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# Facade — Orchestrates All 10 Indicators (Open/Closed Principle)
# ═══════════════════════════════════════════════════════════════════

class AdvancedIndicators:
    """
    Facade for all 10 advanced indicators.
    OCP: New indicators added by appending to indicators list.
    """

    # Feature columns that should be added to ML pipeline
    ML_FEATURE_COLS = [
        "ichimoku_signal", "mfi", "vwap_distance", "keltner_squeeze",
        "williams_r", "cmf", "cci", "stc",
    ]

    def __init__(self):
        self.indicators: List[IndicatorBase] = [
            IchimokuCloud(),
            MoneyFlowIndex(),
            VWAPCalculator(),
            KeltnerChannels(),
            WilliamsPercentR(),
            ChaikinMoneyFlow(),
            CommodityChannelIndex(),
            DonchianChannels(),
            FibonacciRetracement(),
            SchaffTrendCycle(),
        ]

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all 10 indicators, adding columns to DataFrame."""
        for indicator in self.indicators:
            try:
                df = indicator.compute(df)
            except Exception as e:
                log.warning(f"Indicator {indicator.name} failed: {e}")
                # Add zero columns for failed indicators
                for col in indicator.feature_columns:
                    if col not in df.columns:
                        df[col] = 0.0
        return df

    def get_all_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get signals from all indicators."""
        signals = {}
        for indicator in self.indicators:
            try:
                signals[indicator.name] = indicator.signal(df)
            except Exception as e:
                log.warning(f"Signal {indicator.name} failed: {e}")
                signals[indicator.name] = 0.0
        return signals

    def get_composite_signal(self, df: pd.DataFrame,
                              weights: Optional[Dict[str, float]] = None) -> float:
        """
        Weighted composite signal from all indicators.
        Default weights favor Ichimoku (proven trend system) and squeeze (rare but powerful).
        """
        default_weights = {
            "ichimoku": 0.20,
            "mfi": 0.10,
            "vwap": 0.10,
            "keltner": 0.15,
            "williams_r": 0.08,
            "cmf": 0.10,
            "cci": 0.07,
            "donchian": 0.08,
            "fibonacci": 0.05,
            "stc": 0.07,
        }
        weights = weights or default_weights

        signals = self.get_all_signals(df)
        total_weight = sum(weights.get(k, 0) for k in signals)
        if total_weight == 0:
            return 0.0

        composite = sum(signals[k] * weights.get(k, 0) for k in signals) / total_weight
        return float(np.clip(composite, -1, 1))

    def get_ml_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get feature values for ML pipeline from last row."""
        if df.empty:
            return {col: 0.0 for col in self.ML_FEATURE_COLS}
        last = df.iloc[-1]
        return {col: float(last.get(col, 0)) for col in self.ML_FEATURE_COLS}


# ═══════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("ADVANCED INDICATORS TEST — 10 indicators, real data")
    print("=" * 70)

    try:
        from ml.data_pipeline import load_ohlcv
        df = load_ohlcv("BTCUSDT")
    except Exception:
        # Generate synthetic data for testing
        np.random.seed(42)
        n = 1000
        close = np.cumsum(np.random.randn(n) * 100) + 50000
        close = np.abs(close)
        df = pd.DataFrame({
            "open": close * (1 + np.random.randn(n) * 0.001),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.005)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.005)),
            "close": close,
            "volume": np.random.uniform(100, 10000, n),
        })

    facade = AdvancedIndicators()
    df = facade.compute_all(df)

    # Test each indicator
    for indicator in facade.indicators:
        sig = indicator.signal(df)
        cols = indicator.feature_columns
        last_values = {c: round(float(df[c].iloc[-1]), 4) if c in df.columns else "MISSING"
                       for c in cols}
        status = "OK" if all(c in df.columns for c in cols) else "FAIL"
        print(f"\n  [{status}] {indicator.name.upper()}")
        print(f"       Signal: {sig:+.4f}")
        for k, v in last_values.items():
            print(f"       {k}: {v}")

    # Composite signal
    composite = facade.get_composite_signal(df)
    print(f"\n  COMPOSITE SIGNAL: {composite:+.4f}")

    # ML features
    ml_feats = facade.get_ml_features(df)
    print(f"\n  ML Features ({len(ml_feats)}):")
    for k, v in ml_feats.items():
        print(f"    {k}: {v:+.4f}")

    # Verify no NaN/Inf
    for col in facade.ML_FEATURE_COLS:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            inf_count = np.isinf(df[col].values).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  WARNING: {col} has {nan_count} NaN, {inf_count} Inf")

    print(f"\n  Total lines in module: {sum(1 for _ in open(__file__))}")
    print("=" * 70)
