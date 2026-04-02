#!/usr/bin/env python3
"""
Data Pipeline v4 — Integrates all 6 new modules into feature engineering.
Expands from 25 (v3) to 46 ML features.

Modules integrated:
- indicators_advanced: Ichimoku, MFI, VWAP, Keltner, Williams %R, CMF, CCI, STC
- statistical_models: Hurst, GARCH, Z-Score, Entropy, Fractional Diff
- candlestick_patterns: Pattern signal, confluence score, swing position
- volatility_engine: EWMA vol, vol cone percentile, vol ratio, squeeze
"""
import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

log = logging.getLogger("pipeline.v4")


# ═══════════════════════════════════════════════════════════════════
# V4 Feature Columns — 46 total
# ═══════════════════════════════════════════════════════════════════

FEATURE_COLS_V4 = [
    # ── V3 Original (25) ──
    "ema_diff", "rsi", "macd_hist", "bb_pct", "atr_pct", "adx",
    "stoch_k", "stoch_d", "obv_diff", "vol_ratio",
    "momentum_1h", "momentum_4h", "momentum_24h",
    "hour", "day_of_week",
    "momentum_12h", "volume_momentum_4h", "rsi_divergence",
    "bb_width", "close_ema_ratio",
    "mvrv_zscore", "nvt_zscore", "exchange_flow_proxy",
    "mvrv_roc", "vol_price_divergence",
    # ── V4 Advanced Indicators (8) ──
    "ichimoku_signal", "mfi", "vwap_distance", "keltner_squeeze",
    "williams_r", "cmf", "cci", "stc",
    # ── V4 Statistical Models (5) ──
    "hurst_exponent", "garch_vol", "zscore_20",
    "shannon_entropy", "frac_diff_close",
    # ── V4 Candlestick Patterns (3) ──
    "pattern_signal", "pattern_confluence_score", "swing_position",
    # ── V4 Volatility Engine (5) ──
    "ewma_vol", "vol_cone_percentile", "vol_ratio_yz",
    "bb_keltner_squeeze", "vol_term_structure",
]


def add_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full v4 feature pipeline: v3 (25) + indicators (8) + statistical (5)
    + patterns (3) + volatility (5) = 46 features.
    Graceful fallback: if any module fails, fills zeros and continues.
    """
    # Step 1: V3 features (25)
    try:
        from ml.ml_v3 import add_features_v3
        df = add_features_v3(df)
    except Exception as e:
        log.warning(f"V3 features failed: {e}")
        from ml.data_pipeline import add_features
        df = add_features(df)

    # Step 2: Advanced Indicators (8 new features)
    try:
        from ml.indicators_advanced import AdvancedIndicators
        adv = AdvancedIndicators()
        df = adv.compute_all(df)
    except Exception as e:
        log.warning(f"Advanced indicators failed: {e}")
        for col in ["ichimoku_signal", "mfi", "vwap_distance", "keltner_squeeze",
                     "williams_r", "cmf", "cci", "stc"]:
            if col not in df.columns:
                df[col] = 0.0

    # Step 3: Statistical Models (5 new features)
    try:
        from ml.statistical_models import StatisticalEngine
        stats = StatisticalEngine()
        df = stats.compute_all(df)
    except Exception as e:
        log.warning(f"Statistical models failed: {e}")
        for col in ["hurst_exponent", "garch_vol", "zscore_20",
                     "shannon_entropy", "frac_diff_close"]:
            if col not in df.columns:
                df[col] = 0.0

    # Step 4: Candlestick Patterns (3 new features)
    try:
        from ml.candlestick_patterns import PatternEngine
        patterns = PatternEngine()
        df = patterns.compute_all(df)
    except Exception as e:
        log.warning(f"Candlestick patterns failed: {e}")
        for col in ["pattern_signal", "pattern_confluence_score", "swing_position"]:
            if col not in df.columns:
                df[col] = 0.0 if col != "swing_position" else 0.5

    # Step 5: Volatility Engine (5 new features)
    try:
        from ml.volatility_engine import VolatilityEngine
        vol_eng = VolatilityEngine()
        df = vol_eng.compute_all(df)
    except Exception as e:
        log.warning(f"Volatility engine failed: {e}")
        for col in ["ewma_vol", "vol_cone_percentile", "vol_ratio_yz",
                     "bb_keltner_squeeze", "vol_term_structure"]:
            if col not in df.columns:
                df[col] = 0.0

    # Ensure all V4 columns exist (safety net)
    for col in FEATURE_COLS_V4:
        if col not in df.columns:
            df[col] = 0.0

    # Clean NaN/Inf
    for col in FEATURE_COLS_V4:
        df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)

    return df


# ═══════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("DATA PIPELINE V4 TEST — 46 features")
    print("=" * 70)

    from ml.data_pipeline import load_ohlcv
    df = load_ohlcv("BTCUSDT")
    print(f"  Raw data: {len(df)} candles")

    df = add_features_v4(df)
    df.dropna(inplace=True)

    print(f"  After v4 features: {len(df)} candles")
    print(f"  Total features: {len(FEATURE_COLS_V4)}")

    missing = [c for c in FEATURE_COLS_V4 if c not in df.columns]
    print(f"  Missing: {missing or 'none'}")

    nan_cols = [c for c in FEATURE_COLS_V4 if df[c].isna().any()]
    print(f"  NaN columns: {nan_cols or 'none'}")

    inf_cols = [c for c in FEATURE_COLS_V4 if np.isinf(df[c].values).any()]
    print(f"  Inf columns: {inf_cols or 'none'}")

    print(f"\n  Feature values (last row):")
    last = df.iloc[-1]
    for i, col in enumerate(FEATURE_COLS_V4):
        val = last[col]
        category = ""
        if i < 25:
            category = "v3"
        elif i < 33:
            category = "indicator"
        elif i < 38:
            category = "stat"
        elif i < 41:
            category = "pattern"
        else:
            category = "vol"
        print(f"    [{category:>9}] {col:30s}: {val:+.6f}")

    print("=" * 70)
