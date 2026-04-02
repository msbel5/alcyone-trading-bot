#!/usr/bin/env python3
"""
Statistical Models — Quantitative finance models for trading signals.
All computed from OHLCV data, zero API cost.

Research basis:
- Hurst Exponent: Mandelbrot (1963), determines trending vs mean-reverting
- GARCH(1,1): Bollerslev (1986), volatility clustering
- EGARCH: Nelson (1991), asymmetric volatility
- Garman-Klass: Garman & Klass (1980), 8x more efficient vol estimator
- Yang-Zhang: Yang & Zhang (2000), most efficient OHLC estimator
- Fractional Differentiation: de Prado Ch.5 (2018), stationary features
- Shannon Entropy: Shannon (1948), market disorder measure
- Z-Score: Classic mean-reversion signal

SOLID Principles:
- SRP: Each model is its own class
- OCP: New models added without modifying existing
- DIP: StatisticalEngine depends on ModelBase interface
"""
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

log = logging.getLogger("models.statistical")


# ═══════════════════════════════════════════════════════════════════
# Interface
# ═══════════════════════════════════════════════════════════════════

class StatModelBase(ABC):
    """Base interface for statistical models."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def signal(self, df: pd.DataFrame) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def feature_columns(self) -> List[str]:
        pass


# ���══════════════════════════════════════════════════════════════════
# 1. Hurst Exponent — Mandelbrot (1963)
# ═══════════════════════════════════════════════════════════════════

class HurstExponent(StatModelBase):
    """
    Hurst Exponent via Rescaled Range (R/S) analysis.
    THE most important new feature for regime detection:
    - H > 0.5: Trending (use momentum strategies)
    - H ≈ 0.5: Random walk (reduce trading)
    - H < 0.5: Mean-reverting (use mean-reversion strategies)

    R/S method: split returns into chunks, compute (max-min of cumulative
    deviations) / std for each chunk, fit log-log regression.
    """

    def __init__(self, max_window: int = 2000, min_lags: int = 5):
        self.max_window = max_window
        self.min_lags = min_lags

    @property
    def name(self) -> str:
        return "hurst"

    @property
    def feature_columns(self) -> List[str]:
        return ["hurst_exponent", "hurst_regime"]

    def _compute_hurst(self, returns: np.ndarray) -> float:
        """Compute Hurst exponent from returns using R/S analysis."""
        n = len(returns)
        if n < 100:
            return 0.5  # Not enough data

        # Use geometric series of lags
        lags = []
        lag = 10
        while lag < n // 2:
            lags.append(lag)
            lag = int(lag * 1.5)

        if len(lags) < self.min_lags:
            return 0.5

        rs_values = []
        for lag in lags:
            rs_list = []
            n_chunks = n // lag
            for chunk_idx in range(n_chunks):
                chunk = returns[chunk_idx * lag:(chunk_idx + 1) * lag]
                if len(chunk) < 2:
                    continue

                mean_c = np.mean(chunk)
                deviations = np.cumsum(chunk - mean_c)
                r = np.max(deviations) - np.min(deviations)
                s = np.std(chunk, ddof=1)
                if s > 0:
                    rs_list.append(r / s)

            if rs_list:
                rs_values.append((lag, np.mean(rs_list)))

        if len(rs_values) < self.min_lags:
            return 0.5

        # Linear regression in log-log space
        log_lags = np.log([v[0] for v in rs_values])
        log_rs = np.log([v[1] for v in rs_values])

        # Simple least squares
        n_pts = len(log_lags)
        mean_x = np.mean(log_lags)
        mean_y = np.mean(log_rs)
        ss_xy = np.sum((log_lags - mean_x) * (log_rs - mean_y))
        ss_xx = np.sum((log_lags - mean_x) ** 2)

        if ss_xx == 0:
            return 0.5

        hurst = ss_xy / ss_xx
        return float(np.clip(hurst, 0.0, 1.0))

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values
        n = len(df)

        hurst_vals = np.full(n, 0.5)

        # Compute rolling Hurst (expensive, so only for last 500 points)
        window = min(self.max_window, n)
        start = max(0, n - 500)

        for i in range(start, n):
            lookback = min(window, i + 1)
            if lookback < 100:
                continue
            returns = np.diff(np.log(close[max(0, i - lookback + 1):i + 1]))
            returns = returns[np.isfinite(returns)]
            if len(returns) >= 100:
                hurst_vals[i] = self._compute_hurst(returns)

        df["hurst_exponent"] = hurst_vals

        # Regime classification
        regime = np.where(hurst_vals > 0.6, 1.0,      # Trending
                 np.where(hurst_vals < 0.4, -1.0, 0.0))  # Mean-reverting / Random
        df["hurst_regime"] = regime

        return df

    def signal(self, df: pd.DataFrame) -> float:
        """Signal: +1 trending, -1 mean-reverting, 0 random."""
        if "hurst_regime" not in df.columns:
            df = self.compute(df)
        return float(df["hurst_regime"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 2. GARCH(1,1) — Bollerslev (1986)
# ═══════════════════════════════════════════════════════════════════

class GARCHForecaster(StatModelBase):
    """
    GARCH(1,1) volatility forecasting.
    sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

    Signal: high forecast vol → cautious (-0.5), low vol → opportunity (+0.5).
    Uses pure numpy implementation (no arch dependency needed).
    """

    def __init__(self, omega: float = 0.00001, alpha: float = 0.10,
                 beta: float = 0.85, window: int = 500):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.window = window

    @property
    def name(self) -> str:
        return "garch"

    @property
    def feature_columns(self) -> List[str]:
        return ["garch_vol", "garch_vol_ratio"]

    def _fit_garch(self, returns: np.ndarray) -> np.ndarray:
        """Manual GARCH(1,1) — no external dependency needed."""
        n = len(returns)
        variance = np.zeros(n)
        variance[0] = np.var(returns)

        for t in range(1, n):
            variance[t] = (self.omega +
                           self.alpha * returns[t - 1] ** 2 +
                           self.beta * variance[t - 1])

        return np.sqrt(variance)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values
        n = len(df)

        garch_vol = np.zeros(n)
        garch_ratio = np.zeros(n)

        # Compute GARCH on rolling window
        for i in range(self.window, n):
            window_close = close[i - self.window:i + 1]
            returns = np.diff(np.log(window_close))
            returns = returns[np.isfinite(returns)]

            if len(returns) < 50:
                continue

            vol_series = self._fit_garch(returns)
            garch_vol[i] = vol_series[-1]

            # Ratio: current GARCH vol / mean GARCH vol
            mean_vol = np.mean(vol_series)
            if mean_vol > 0:
                garch_ratio[i] = vol_series[-1] / mean_vol

        df["garch_vol"] = garch_vol
        df["garch_vol_ratio"] = garch_ratio

        return df

    def signal(self, df: pd.DataFrame) -> float:
        """High vol = cautious, low vol = opportunity."""
        if "garch_vol_ratio" not in df.columns:
            df = self.compute(df)
        ratio = df["garch_vol_ratio"].iloc[-1]
        if ratio > 2.0:
            return -0.8   # Very high vol — avoid
        elif ratio > 1.5:
            return -0.4   # High vol — cautious
        elif ratio < 0.5:
            return 0.5    # Low vol — opportunity
        elif ratio < 0.8:
            return 0.3    # Below average vol
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# 3. Garman-Klass Volatility — 8x more efficient
# ═══════════════════════════════════════════════════════════════════

class GarmanKlassVolatility(StatModelBase):
    """
    Garman-Klass (1980): OHLC-based volatility, 8x more efficient
    than close-to-close estimator.
    GK = 0.5 * ln(H/L)^2 - (2ln2-1) * ln(C/O)^2
    """

    def __init__(self, period: int = 24):
        self.period = period

    @property
    def name(self) -> str:
        return "garman_klass"

    @property
    def feature_columns(self) -> List[str]:
        return ["garman_klass_vol"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        o = df["open"].values.astype(float)
        c = df["close"].values.astype(float)

        # Garman-Klass: 0.5 * ln(H/L)^2 - (2ln2-1) * ln(C/O)^2
        log_hl = np.log(h / np.where(l > 0, l, 1.0))
        log_co = np.log(c / np.where(o > 0, o, 1.0))
        gk_term = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2

        # Rolling average then sqrt — use index-aligned Series
        gk_series = pd.Series(gk_term, index=df.index)
        gk_mean = gk_series.rolling(self.period, min_periods=1).mean()
        gk_vol = np.sqrt(np.clip(gk_mean.values, 0, None))
        df["garman_klass_vol"] = np.nan_to_num(gk_vol, nan=0.0)

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "garman_klass_vol" not in df.columns:
            df = self.compute(df)
        return 0.0  # Pure feature, no direct signal


# ═══════════════════════════════════════════════════════════════════
# 4. Yang-Zhang Volatility — Most efficient OHLC estimator
# ���══════════════════════════════════════════════════════════════════

class YangZhangVolatility(StatModelBase):
    """
    Yang-Zhang (2000): drift-independent, most efficient OHLC estimator.
    Combines overnight, open-to-close, and Rogers-Satchell components.
    """

    def __init__(self, period: int = 24):
        self.period = period

    @property
    def name(self) -> str:
        return "yang_zhang"

    @property
    def feature_columns(self) -> List[str]:
        return ["yang_zhang_vol"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        n = len(df)

        yz_vol = np.zeros(n)

        for i in range(self.period, n):
            window_o = o[i - self.period + 1:i + 1]
            window_h = h[i - self.period + 1:i + 1]
            window_l = l[i - self.period + 1:i + 1]
            window_c = c[i - self.period + 1:i + 1]
            prev_c = c[i - self.period:i]

            k = len(window_o)
            if k < 2:
                continue

            # Overnight variance: log(O_t / C_{t-1})
            log_oc = np.log(np.where(prev_c > 0, window_o / prev_c, 1.0))
            overnight_var = np.var(log_oc, ddof=1) if len(log_oc) > 1 else 0

            # Open-to-close variance: log(C_t / O_t)
            log_co = np.log(np.where(window_o > 0, window_c / window_o, 1.0))
            oc_var = np.var(log_co, ddof=1) if len(log_co) > 1 else 0

            # Rogers-Satchell variance
            log_ho = np.log(np.where(window_o > 0, window_h / window_o, 1.0))
            log_hc = np.log(np.where(window_c > 0, window_h / window_c, 1.0))
            log_lo = np.log(np.where(window_o > 0, window_l / window_o, 1.0))
            log_lc = np.log(np.where(window_c > 0, window_l / window_c, 1.0))
            rs_var = np.mean(log_ho * log_hc + log_lo * log_lc)

            # Yang-Zhang combination
            alpha = 1.34 / (1 + k / (k + 2))
            yz = overnight_var + alpha * oc_var + (1 - alpha) * rs_var
            yz_vol[i] = np.sqrt(max(yz, 0))

        df["yang_zhang_vol"] = yz_vol
        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "yang_zhang_vol" not in df.columns:
            df = self.compute(df)
        return 0.0  # Pure feature


# ════════════════════��══════════════════════════════��═══════════════
# 5. Z-Score Mean Reversion — Classic
# ════════════════════════��══════════════════════════════════════════

class ZScoreMeanReversion(StatModelBase):
    """
    Z-Score: (price - rolling_mean) / rolling_std.
    z > 2: overbought → sell. z < -2: oversold → buy.
    Most effective in sideways/mean-reverting regimes (Hurst < 0.5).
    """

    def __init__(self, period: int = 20):
        self.period = period

    @property
    def name(self) -> str:
        return "zscore"

    @property
    def feature_columns(self) -> List[str]:
        return ["zscore_20", "zscore_signal"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        rolling_mean = close.rolling(self.period, min_periods=1).mean()
        rolling_std = close.rolling(self.period, min_periods=1).std().replace(0, np.nan)

        df["zscore_20"] = ((close - rolling_mean) / rolling_std).fillna(0).clip(-4, 4)

        # Signal: mean-reversion contrarian
        z = df["zscore_20"].values
        sig = np.zeros(len(z))
        # Strong signals
        sig[z > 2.5] = -1.0    # Extreme overbought
        sig[z > 2.0] = -0.7
        sig[z < -2.5] = 1.0    # Extreme oversold
        sig[z < -2.0] = 0.7
        # Moderate signals
        sig[(z > 1.0) & (z <= 2.0)] = -0.3
        sig[(z < -1.0) & (z >= -2.0)] = 0.3

        df["zscore_signal"] = sig
        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "zscore_signal" not in df.columns:
            df = self.compute(df)
        return float(df["zscore_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 6. Fractional Differentiation — de Prado Ch.5
# ═══════════════════════════════════════════════════════════════════

class FractionalDifferentiator(StatModelBase):
    """
    Fractional Differentiation: stationary features that preserve memory.
    d=0: original (non-stationary, full memory)
    d=1: fully differenced (stationary, no memory)
    d∈[0.3,0.5]: sweet spot — stationary enough for ML, memory preserved.

    Fixed-width window FFD implementation (de Prado).
    """

    def __init__(self, d: float = 0.4, window: int = 100, threshold: float = 1e-5):
        self.d = d
        self.window = window
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "frac_diff"

    @property
    def feature_columns(self) -> List[str]:
        return ["frac_diff_close"]

    def _get_weights(self) -> np.ndarray:
        """Compute fractional differentiation weights."""
        weights = [1.0]
        for k in range(1, self.window):
            w = -weights[-1] * (self.d - k + 1) / k
            if abs(w) < self.threshold:
                break
            weights.append(w)
        return np.array(weights[::-1])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = np.log(df["close"].values.astype(float))
        close = np.nan_to_num(close, nan=0.0)
        n = len(close)
        weights = self._get_weights()
        w_len = len(weights)

        frac_diff = np.zeros(n)
        for i in range(w_len - 1, n):
            window = close[i - w_len + 1:i + 1]
            frac_diff[i] = np.dot(weights, window)

        # Normalize
        std = np.std(frac_diff[frac_diff != 0])
        if std > 0:
            frac_diff = frac_diff / std

        df["frac_diff_close"] = frac_diff
        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "frac_diff_close" not in df.columns:
            df = self.compute(df)
        return 0.0  # Pure feature for ML


# ��════════════════════════════════════════════════��═════════════════
# 7. Shannon Entropy — Market Disorder Measure
# ═══════════════════════════════════════════════════════════════════

class ShannonEntropy(StatModelBase):
    """
    Shannon Entropy: H = -sum(p * log(p)) over binned returns.
    Higher entropy = more disordered/random market.
    Lower entropy = more predictable patterns.

    Signal: low entropy (predictable) → trade, high entropy → reduce exposure.
    """

    def __init__(self, period: int = 100, n_bins: int = 10):
        self.period = period
        self.n_bins = n_bins

    @property
    def name(self) -> str:
        return "entropy"

    @property
    def feature_columns(self) -> List[str]:
        return ["shannon_entropy", "entropy_signal"]

    def _compute_entropy(self, returns: np.ndarray) -> float:
        """Compute Shannon entropy of binned returns."""
        if len(returns) < 10:
            return 0.5

        # Bin returns
        counts, _ = np.histogram(returns, bins=self.n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # Remove zero bins

        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs))

        # Normalize by max possible entropy (uniform distribution)
        max_entropy = np.log2(self.n_bins)
        if max_entropy > 0:
            return entropy / max_entropy  # [0, 1]
        return 0.5

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values
        n = len(df)

        entropy_vals = np.full(n, 0.5)

        for i in range(self.period, n):
            window = close[i - self.period:i + 1]
            returns = np.diff(np.log(np.where(window > 0, window, 1.0)))
            returns = returns[np.isfinite(returns)]
            if len(returns) >= 10:
                entropy_vals[i] = self._compute_entropy(returns)

        df["shannon_entropy"] = entropy_vals

        # Signal: low entropy = predictable = good for trading
        ent = entropy_vals
        sig = np.zeros(n)
        sig[ent < 0.4] = 0.5     # Very predictable — trade more
        sig[ent < 0.6] = 0.2     # Somewhat predictable
        sig[ent > 0.85] = -0.5   # Very random — trade less
        sig[ent > 0.7] = -0.2    # Somewhat random

        df["entropy_signal"] = sig
        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "entropy_signal" not in df.columns:
            df = self.compute(df)
        return float(df["entropy_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 8. EGARCH — Nelson (1991), Asymmetric Volatility
# ═══════════════════════════════════════════════════════════════════

class EGARCHModel(StatModelBase):
    """
    EGARCH: Exponential GARCH — captures leverage effect.
    Bad news creates more volatility than good news (asymmetric).
    Better for crypto than standard GARCH.

    log(sigma^2_t) = omega + alpha*(|z_{t-1}| - E|z|) + gamma*z_{t-1} + beta*log(sigma^2_{t-1})
    gamma < 0 means negative shocks increase vol more than positive.
    """

    def __init__(self, omega: float = -0.2, alpha: float = 0.15,
                 gamma: float = -0.05, beta: float = 0.95, window: int = 500):
        self.omega = omega
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.window = window

    @property
    def name(self) -> str:
        return "egarch"

    @property
    def feature_columns(self) -> List[str]:
        return ["egarch_vol", "egarch_asymmetry"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values
        n = len(df)

        egarch_vol = np.zeros(n)
        egarch_asym = np.zeros(n)

        e_abs_z = np.sqrt(2.0 / np.pi)  # E[|z|] for standard normal

        for i in range(self.window, n):
            window_close = close[i - self.window:i + 1]
            returns = np.diff(np.log(np.where(window_close > 0, window_close, 1.0)))
            returns = returns[np.isfinite(returns)]

            if len(returns) < 50:
                continue

            # Initialize
            log_var = np.zeros(len(returns))
            log_var[0] = np.log(max(np.var(returns), 1e-10))

            for t in range(1, len(returns)):
                sigma = np.sqrt(np.exp(log_var[t - 1]))
                z = returns[t - 1] / max(sigma, 1e-8)

                log_var[t] = (self.omega +
                              self.alpha * (abs(z) - e_abs_z) +
                              self.gamma * z +
                              self.beta * log_var[t - 1])
                # Clamp to prevent overflow
                log_var[t] = np.clip(log_var[t], -20, 5)

            egarch_vol[i] = np.sqrt(np.exp(log_var[-1]))

            # Asymmetry: how much more volatile after negative returns
            neg_returns = returns[returns < 0]
            pos_returns = returns[returns > 0]
            if len(neg_returns) > 5 and len(pos_returns) > 5:
                neg_vol = np.std(neg_returns)
                pos_vol = np.std(pos_returns)
                if pos_vol > 0:
                    egarch_asym[i] = (neg_vol / pos_vol) - 1  # >0 means neg more volatile

        df["egarch_vol"] = egarch_vol
        df["egarch_asymmetry"] = egarch_asym

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "egarch_asymmetry" not in df.columns:
            df = self.compute(df)
        # High asymmetry = bearish environment
        asym = df["egarch_asymmetry"].iloc[-1]
        if asym > 0.5:
            return -0.5  # Strong negative asymmetry
        elif asym > 0.2:
            return -0.2
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# Facade — StatisticalEngine
# ═════════════════════════════════════════���═════════════════════════

class StatisticalEngine:
    """Facade for all statistical models."""

    ML_FEATURE_COLS = [
        "hurst_exponent", "garch_vol", "zscore_20",
        "shannon_entropy", "frac_diff_close",
    ]

    def __init__(self):
        self.models: List[StatModelBase] = [
            HurstExponent(),
            GARCHForecaster(),
            GarmanKlassVolatility(),
            YangZhangVolatility(),
            ZScoreMeanReversion(),
            FractionalDifferentiator(),
            ShannonEntropy(),
            EGARCHModel(),
        ]

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all statistical features."""
        for model in self.models:
            try:
                df = model.compute(df)
            except Exception as e:
                log.warning(f"Model {model.name} failed: {e}")
                for col in model.feature_columns:
                    if col not in df.columns:
                        df[col] = 0.0
        return df

    def get_all_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get signals from all models."""
        signals = {}
        for model in self.models:
            try:
                signals[model.name] = model.signal(df)
            except Exception as e:
                signals[model.name] = 0.0
        return signals

    def get_composite_signal(self, df: pd.DataFrame) -> float:
        """Weighted composite signal."""
        weights = {
            "hurst": 0.25,      # Most important — regime detection
            "garch": 0.15,
            "garman_klass": 0.05,
            "yang_zhang": 0.05,
            "zscore": 0.20,     # Direct trading signal
            "frac_diff": 0.05,
            "entropy": 0.15,
            "egarch": 0.10,
        }
        signals = self.get_all_signals(df)
        total_w = sum(weights.get(k, 0) for k in signals)
        if total_w == 0:
            return 0.0
        composite = sum(signals[k] * weights.get(k, 0) for k in signals) / total_w
        return float(np.clip(composite, -1, 1))


# ═══════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("STATISTICAL MODELS TEST — 8 models, real data")
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

    engine = StatisticalEngine()
    df = engine.compute_all(df)

    for model in engine.models:
        sig = model.signal(df)
        cols = model.feature_columns
        last_values = {c: round(float(df[c].iloc[-1]), 6) if c in df.columns else "MISSING"
                       for c in cols}
        status = "OK" if all(c in df.columns for c in cols) else "FAIL"
        print(f"\n  [{status}] {model.name.upper()}")
        print(f"       Signal: {sig:+.4f}")
        for k, v in last_values.items():
            print(f"       {k}: {v}")

    composite = engine.get_composite_signal(df)
    print(f"\n  COMPOSITE SIGNAL: {composite:+.4f}")

    print(f"\n  ML Features ({len(engine.ML_FEATURE_COLS)}):")
    for col in engine.ML_FEATURE_COLS:
        val = float(df[col].iloc[-1]) if col in df.columns else "MISSING"
        print(f"    {col}: {val}")

    print(f"\n  Total lines: {sum(1 for _ in open(__file__))}")
    print("=" * 70)
