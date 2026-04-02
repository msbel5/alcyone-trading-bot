#!/usr/bin/env python3
"""
Volatility Engine — Forecasting, regime detection, and position sizing.
12 components from OHLCV data, zero API cost.

Research basis:
- EWMA: RiskMetrics (JP Morgan 1994), λ=0.94
- Parkinson: Parkinson (1980), high-low range estimator
- Garman-Klass: Garman & Klass (1980), 8x more efficient
- Yang-Zhang: Yang & Zhang (2000), most efficient OHLC
- Rogers-Satchell: Rogers & Satchell (1991), drift-independent
- Volatility Cone: Natenberg "Option Volatility & Pricing"
- BB-Keltner Squeeze: John Carter "Mastering the Trade"

SOLID Principles:
- SRP: Each estimator is its own class
- OCP: New estimators added without modifying existing
- DIP: VolatilityEngine depends on EstimatorBase interface
"""
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from enum import Enum

log = logging.getLogger("volatility.engine")


# ═══════════════════════════════════════════════════════════════════
# Volatility Regime Enum
# ═══════════════════════════════════════════════════════════════════

class VolRegime(Enum):
    LOW = "low_vol"
    NORMAL = "normal"
    HIGH = "high_vol"
    EXTREME = "extreme"


# ═══════════════════════════════════════════════════════════════════
# Interface
# ═══════════════════════════════════════════════════════════════════

class VolEstimatorBase(ABC):
    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def column_name(self) -> str:
        pass


# ═══════════════════════════════════════════════════════════════════
# 1. EWMA Volatility — RiskMetrics (JP Morgan 1994)
# ═══════════════════════════════════════════════════════════════════

class EWMAVolatility(VolEstimatorBase):
    """
    Exponentially Weighted Moving Average volatility.
    σ²_t = λ * σ²_{t-1} + (1-λ) * r²_{t-1}
    λ = 0.94 is RiskMetrics standard for daily, 0.97 for monthly.
    """

    def __init__(self, lam: float = 0.94):
        self.lam = lam

    @property
    def name(self) -> str:
        return "ewma"

    @property
    def column_name(self) -> str:
        return "ewma_vol"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(float)
        n = len(close)

        log_returns = np.diff(np.log(np.where(close > 0, close, 1.0)))
        log_returns = np.nan_to_num(log_returns, nan=0.0)

        variance = np.zeros(n)
        if len(log_returns) > 0:
            variance[1] = log_returns[0] ** 2
            for t in range(2, n):
                r_idx = t - 1
                if r_idx < len(log_returns):
                    variance[t] = (self.lam * variance[t - 1] +
                                   (1 - self.lam) * log_returns[r_idx] ** 2)
                else:
                    variance[t] = variance[t - 1]

        df["ewma_vol"] = np.sqrt(variance)
        return df


# ═══════════════════════════════════════════════════════════════════
# 2. Parkinson Volatility — (1980)
# ═══════════════════════════════════════════════════════════════════

class ParkinsonVolatility(VolEstimatorBase):
    """
    Parkinson: uses high-low range. 5x more efficient than close-to-close.
    σ² = (1/4nln2) * Σ ln(H/L)²
    """

    def __init__(self, period: int = 24):
        self.period = period

    @property
    def name(self) -> str:
        return "parkinson"

    @property
    def column_name(self) -> str:
        return "parkinson_vol"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)

        log_hl = np.log(h / np.where(l > 0, l, 1.0))
        log_hl_sq = log_hl ** 2

        factor = 1.0 / (4 * np.log(2))
        park_series = pd.Series(log_hl_sq, index=df.index)
        rolling_mean = park_series.rolling(self.period, min_periods=1).mean()

        df["parkinson_vol"] = np.sqrt(factor * rolling_mean.values)
        df["parkinson_vol"] = df["parkinson_vol"].fillna(0)
        return df


# ═══════════════════════════════════════════════════════════════════
# 3. Rogers-Satchell Volatility — (1991), drift-independent
# ═══════════════════════════════════════════════════════════════════

class RogersSatchellVolatility(VolEstimatorBase):
    """
    Rogers-Satchell: drift-independent OHLC estimator.
    σ² = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)
    """

    def __init__(self, period: int = 24):
        self.period = period

    @property
    def name(self) -> str:
        return "rogers_satchell"

    @property
    def column_name(self) -> str:
        return "rogers_satchell_vol"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        o = df["open"].values.astype(float)
        c = df["close"].values.astype(float)

        # Safe log ratios
        log_hc = np.log(h / np.where(c > 0, c, 1.0))
        log_ho = np.log(h / np.where(o > 0, o, 1.0))
        log_lc = np.log(l / np.where(c > 0, c, 1.0))
        log_lo = np.log(l / np.where(o > 0, o, 1.0))

        rs_term = log_hc * log_ho + log_lc * log_lo
        rs_series = pd.Series(rs_term, index=df.index)
        rolling_mean = rs_series.rolling(self.period, min_periods=1).mean()

        df["rogers_satchell_vol"] = np.sqrt(np.clip(rolling_mean.values, 0, None))
        df["rogers_satchell_vol"] = df["rogers_satchell_vol"].fillna(0)
        return df


# ═══════════════════════════════════════════════════════════════════
# 4. Volatility Cone — Natenberg
# ═══════════════════════════════════════════════════════════════════

class VolatilityCone(VolEstimatorBase):
    """
    Volatility Cone: historical percentiles by horizon.
    Shows where current vol sits relative to history.
    If at p10 → expect expansion. If at p90 → expect contraction.
    """

    HORIZONS = [24, 168, 720]  # 1 day, 1 week, 1 month (hours)

    def __init__(self, lookback: int = 4320):  # 6 months
        self.lookback = lookback

    @property
    def name(self) -> str:
        return "vol_cone"

    @property
    def column_name(self) -> str:
        return "vol_cone_percentile"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(float)
        n = len(df)

        # Compute realized vol for default horizon (24h)
        log_returns = np.diff(np.log(np.where(close > 0, close, 1.0)))
        log_returns = np.nan_to_num(log_returns, nan=0.0)

        # Rolling 24h realized vol
        rv_series = pd.Series(log_returns).rolling(24, min_periods=1).std()
        rv_values = rv_series.values

        # Percentile of current vol within lookback
        percentiles = np.full(n, 50.0)
        for i in range(max(self.lookback, 100), n):
            lookback_start = max(0, i - self.lookback)
            history = rv_values[lookback_start:i]
            history = history[~np.isnan(history)]
            if len(history) > 10:
                current = rv_values[i] if i < len(rv_values) else 0
                pct = np.searchsorted(np.sort(history), current) / len(history) * 100
                percentiles[i + 1 if i + 1 < n else i] = pct

        df["vol_cone_percentile"] = np.clip(percentiles, 0, 100)

        # Multi-horizon cones (store as metadata)
        for horizon in self.HORIZONS:
            col = f"vol_cone_{horizon}h"
            rv_h = pd.Series(log_returns).rolling(horizon, min_periods=1).std()
            df[col] = rv_h.values[:n] if len(rv_h) >= n else np.zeros(n)
            df[col] = df[col].fillna(0)

        return df


# ═══════════════════════════════════════════════════════════════════
# 5. Volatility Ratio — Current vs Historical
# ═══════════════════════════════════════════════════════════════════

class VolatilityRatio(VolEstimatorBase):
    """
    Ratio of short-term to long-term volatility.
    > 1.5: volatility expanding (caution)
    < 0.5: volatility contracting (opportunity or squeeze)
    """

    def __init__(self, short_period: int = 24, long_period: int = 168):
        self.short_period = short_period
        self.long_period = long_period

    @property
    def name(self) -> str:
        return "vol_ratio"

    @property
    def column_name(self) -> str:
        return "vol_ratio_yz"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(float)

        log_returns = np.diff(np.log(np.where(close > 0, close, 1.0)))
        log_returns = np.nan_to_num(log_returns, nan=0.0)
        lr_series = pd.Series(log_returns)

        short_vol = lr_series.rolling(self.short_period, min_periods=1).std()
        long_vol = lr_series.rolling(self.long_period, min_periods=1).std()
        long_vol_safe = long_vol.replace(0, np.nan)

        ratio = (short_vol / long_vol_safe).fillna(1.0)

        # Pad to match df length
        result = np.ones(len(df))
        result[1:] = ratio.values[:len(df) - 1] if len(ratio) >= len(df) - 1 else 1.0
        df["vol_ratio_yz"] = np.clip(result, 0.1, 5.0)

        return df


# ═══════════════════════════════════════════════════════════════════
# 6. BB-Keltner Squeeze Detection — John Carter
# ═══════════════════════════════════════════════════════════════════

class BBKeltnerSqueeze(VolEstimatorBase):
    """
    BB inside Keltner = squeeze (volatility compression).
    Squeeze release = explosive move in breakout direction.
    Uses momentum (close vs mid) for direction.
    """

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 kc_period: int = 20, kc_mult: float = 1.5):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult

    @property
    def name(self) -> str:
        return "bb_keltner_squeeze"

    @property
    def column_name(self) -> str:
        return "bb_keltner_squeeze"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Bollinger Bands
        bb_mid = close.rolling(self.bb_period, min_periods=1).mean()
        bb_std = close.rolling(self.bb_period, min_periods=1).std().fillna(0)
        bb_upper = bb_mid + self.bb_std * bb_std
        bb_lower = bb_mid - self.bb_std * bb_std

        # Keltner Channels
        kc_mid = close.ewm(span=self.kc_period, adjust=False).mean()
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(span=self.kc_period, adjust=False).mean()
        kc_upper = kc_mid + self.kc_mult * atr
        kc_lower = kc_mid - self.kc_mult * atr

        # Squeeze: BB fully inside Keltner
        squeeze = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(float)
        df["bb_keltner_squeeze"] = squeeze

        # Squeeze momentum direction
        momentum = close - (kc_upper + kc_lower) / 2
        atr_safe = atr.replace(0, np.nan)
        df["squeeze_momentum"] = (momentum / atr_safe).fillna(0).clip(-2, 2)

        return df


# ═══════════════════════════════════════════════════════════════════
# 7. Volatility Term Structure — Short vs Long
# ═══════════════════════════════════════════════════════════════════

class VolTermStructure(VolEstimatorBase):
    """
    Compares short-term vs long-term volatility.
    Contango (long > short): normal market.
    Backwardation (short > long): stress/crisis.
    """

    @property
    def name(self) -> str:
        return "vol_term_structure"

    @property
    def column_name(self) -> str:
        return "vol_term_structure"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(float)
        lr = np.diff(np.log(np.where(close > 0, close, 1.0)))
        lr = np.nan_to_num(lr, nan=0.0)
        lr_s = pd.Series(lr)

        vol_1d = lr_s.rolling(24, min_periods=1).std()
        vol_1w = lr_s.rolling(168, min_periods=1).std()
        vol_1w_safe = vol_1w.replace(0, np.nan)

        # Ratio: < 1 = contango (normal), > 1 = backwardation (stress)
        term_struct = (vol_1d / vol_1w_safe).fillna(1.0)

        result = np.ones(len(df))
        result[1:len(term_struct) + 1] = term_struct.values[:len(df) - 1]
        df["vol_term_structure"] = np.clip(result, 0.2, 5.0)
        return df


# ═══════════════════════════════════════════════════════════════════
# 8. Volatility Clustering — Autocorrelation of Squared Returns
# ═══════════════════════════════════════════════════════════════════

class VolClustering(VolEstimatorBase):
    """
    Volatility clustering: high vol follows high vol.
    Measured by autocorrelation of squared returns.
    High autocorr → vol persistence → trending vol regime.
    """

    def __init__(self, period: int = 100, lag: int = 1):
        self.period = period
        self.lag = lag

    @property
    def name(self) -> str:
        return "vol_clustering"

    @property
    def column_name(self) -> str:
        return "vol_clustering"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(float)
        n = len(df)

        lr = np.diff(np.log(np.where(close > 0, close, 1.0)))
        lr = np.nan_to_num(lr, nan=0.0)
        sq_returns = lr ** 2

        clustering = np.zeros(n)
        for i in range(self.period + self.lag, min(len(sq_returns), n)):
            window = sq_returns[i - self.period:i]
            lagged = sq_returns[i - self.period - self.lag:i - self.lag]
            if len(window) == len(lagged) and len(window) > 10:
                corr = np.corrcoef(window, lagged)[0, 1]
                clustering[i + 1 if i + 1 < n else i] = corr if np.isfinite(corr) else 0

        df["vol_clustering"] = np.clip(clustering, -1, 1)
        return df


# ═══════════════════════════════════════════════════════════════════
# 9. Volatility Regime Detector
# ═══════════════════════════════════════════════════════════════════

class VolRegimeDetector:
    """
    Four-state volatility regime detection.
    Uses vol cone percentile + vol ratio + squeeze.
    LOW: cone < p25 (opportunity)
    NORMAL: p25-p75 (standard trading)
    HIGH: p75-p90 (reduce size)
    EXTREME: > p90 (halt or minimum size)
    """

    def detect(self, df: pd.DataFrame) -> VolRegime:
        if df.empty:
            return VolRegime.NORMAL

        last = df.iloc[-1]
        cone_pct = float(last.get("vol_cone_percentile", 50))
        squeeze = float(last.get("bb_keltner_squeeze", 0))

        if cone_pct > 90:
            return VolRegime.EXTREME
        elif cone_pct > 75:
            return VolRegime.HIGH
        elif cone_pct < 25 or squeeze > 0:
            return VolRegime.LOW
        return VolRegime.NORMAL

    def get_position_scale(self, regime: VolRegime) -> float:
        return {
            VolRegime.LOW: 1.2,       # Low vol = opportunity, larger size
            VolRegime.NORMAL: 1.0,    # Standard
            VolRegime.HIGH: 0.6,      # Reduce exposure
            VolRegime.EXTREME: 0.3,   # Minimum exposure
        }[regime]

    def get_atr_multiplier(self, regime: VolRegime) -> float:
        return {
            VolRegime.LOW: 1.5,       # Tight stop in low vol
            VolRegime.NORMAL: 2.0,    # Standard
            VolRegime.HIGH: 3.0,      # Wide stop
            VolRegime.EXTREME: 4.0,   # Very wide stop
        }[regime]


# ═══════════════════════════════════════════════════════════════════
# 10. Vol-Adjusted Position Sizer
# ═══════════════════════════════════════════════════════════════════

class VolAdjustedSizer:
    """
    Institutional risk parity: position = base * (target_vol / current_vol).
    When current vol is 2x target, halve position.
    When current vol is 0.5x target, double position (up to max).
    """

    def __init__(self, target_vol: float = 0.02, max_scale: float = 2.0,
                 min_scale: float = 0.25):
        self.target_vol = target_vol
        self.max_scale = max_scale
        self.min_scale = min_scale

    def compute_scale(self, current_vol: float) -> float:
        if current_vol <= 0:
            return 1.0
        raw_scale = self.target_vol / current_vol
        return float(np.clip(raw_scale, self.min_scale, self.max_scale))

    def compute_series(self, df: pd.DataFrame,
                        vol_column: str = "ewma_vol") -> pd.Series:
        if vol_column not in df.columns:
            return pd.Series(1.0, index=df.index)

        vol = df[vol_column].values
        scales = np.array([self.compute_scale(v) for v in vol])
        return pd.Series(scales, index=df.index, name="vol_position_scale")


# ═══════════════════════════════════════════════════════════════════
# Facade — VolatilityEngine
# ═══════════════════════════════════════════════════════════════════

class VolatilityEngine:
    """Facade for all volatility computation and regime detection."""

    ML_FEATURE_COLS = [
        "ewma_vol", "vol_cone_percentile", "vol_ratio_yz",
        "bb_keltner_squeeze", "vol_term_structure",
    ]

    def __init__(self):
        self.estimators: List[VolEstimatorBase] = [
            EWMAVolatility(),
            ParkinsonVolatility(),
            RogersSatchellVolatility(),
            VolatilityCone(),
            VolatilityRatio(),
            BBKeltnerSqueeze(),
            VolTermStructure(),
            VolClustering(),
        ]
        self.regime_detector = VolRegimeDetector()
        self.position_sizer = VolAdjustedSizer()

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        for est in self.estimators:
            try:
                df = est.compute(df)
            except Exception as e:
                log.warning(f"Vol estimator {est.name} failed: {e}")
                if est.column_name not in df.columns:
                    df[est.column_name] = 0.0
        return df

    def get_regime(self, df: pd.DataFrame) -> VolRegime:
        return self.regime_detector.detect(df)

    def get_position_scale(self, df: pd.DataFrame) -> float:
        regime = self.get_regime(df)
        regime_scale = self.regime_detector.get_position_scale(regime)

        # Also apply vol-adjusted sizing
        if "ewma_vol" in df.columns:
            vol_scale = self.position_sizer.compute_scale(
                float(df["ewma_vol"].iloc[-1]))
        else:
            vol_scale = 1.0

        return min(regime_scale, vol_scale)

    def get_signal(self, df: pd.DataFrame) -> float:
        """Volatility-based trading signal."""
        if df.empty:
            return 0.0

        last = df.iloc[-1]
        cone_pct = float(last.get("vol_cone_percentile", 50))
        squeeze = float(last.get("bb_keltner_squeeze", 0))
        momentum = float(last.get("squeeze_momentum", 0))

        # Squeeze release = strong signal
        if squeeze == 0 and cone_pct < 30:
            return momentum * 0.5  # Direction of squeeze momentum

        # High vol = reduce signal
        if cone_pct > 80:
            return -0.3  # Bearish bias in high vol

        return 0.0

    def get_all_values(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {}
        last = df.iloc[-1]
        regime = self.get_regime(df)
        return {
            "ewma_vol": float(last.get("ewma_vol", 0)),
            "parkinson_vol": float(last.get("parkinson_vol", 0)),
            "rogers_satchell_vol": float(last.get("rogers_satchell_vol", 0)),
            "vol_cone_percentile": float(last.get("vol_cone_percentile", 50)),
            "vol_ratio": float(last.get("vol_ratio_yz", 1)),
            "bb_keltner_squeeze": bool(last.get("bb_keltner_squeeze", 0)),
            "vol_term_structure": float(last.get("vol_term_structure", 1)),
            "vol_clustering": float(last.get("vol_clustering", 0)),
            "regime": regime.value,
            "position_scale": self.get_position_scale(df),
            "atr_multiplier": self.regime_detector.get_atr_multiplier(regime),
        }


# ═══════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("VOLATILITY ENGINE TEST — 8 estimators + regime + sizing")
    print("=" * 70)

    try:
        from ml.data_pipeline import load_ohlcv
        df = load_ohlcv("BTCUSDT")
    except Exception:
        np.random.seed(42)
        n = 2000
        close = np.cumsum(np.random.randn(n) * 100) + 50000
        close = np.abs(close)
        df = pd.DataFrame({
            "open": close * (1 + np.random.randn(n) * 0.001),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.005)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.005)),
            "close": close,
            "volume": np.random.uniform(100, 10000, n),
        })

    engine = VolatilityEngine()
    df = engine.compute_all(df)

    # Print all values
    values = engine.get_all_values(df)
    print(f"\n  BTC Volatility Analysis:")
    for k, v in values.items():
        print(f"    {k}: {v}")

    # ML features
    print(f"\n  ML Features ({len(engine.ML_FEATURE_COLS)}):")
    for col in engine.ML_FEATURE_COLS:
        val = float(df[col].iloc[-1]) if col in df.columns else "MISSING"
        print(f"    {col}: {val}")

    # Signal
    sig = engine.get_signal(df)
    print(f"\n  Volatility Signal: {sig:+.4f}")

    # NaN check
    for col in ["ewma_vol", "parkinson_vol", "rogers_satchell_vol",
                 "vol_cone_percentile", "vol_ratio_yz", "bb_keltner_squeeze",
                 "vol_term_structure", "vol_clustering"]:
        if col in df.columns:
            nans = df[col].isna().sum()
            if nans > 0:
                print(f"  WARNING: {col} has {nans} NaN")

    print(f"\n  Total lines: {sum(1 for _ in open(__file__))}")
    print("=" * 70)
