#!/usr/bin/env python3
"""
Dashboard Data Collector v2 — computes indicators for EACH coin separately.
Called once per tick cycle. Caches per-coin data for 30 seconds.
"""
import sys
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

log = logging.getLogger("dashboard.collector")

_cache = {}
_cache_time = 0
_CACHE_TTL = 30

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]


def _get_coin_df(symbol: str, adapter) -> pd.DataFrame:
    """Get OHLCV data for a coin — live or historical fallback."""
    try:
        klines = adapter.get_klines(symbol, "1h", limit=500)
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        if len(df) >= 50:
            return df
    except Exception:
        pass

    # Fallback to historical CSV
    try:
        from ml.data_pipeline import load_ohlcv
        return load_ohlcv(symbol).tail(500).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _compute_coin_indicators(df: pd.DataFrame) -> Dict:
    """Compute all indicators for a single coin's DataFrame."""
    if df.empty or len(df) < 20:
        return {}

    result = {}

    # Advanced Indicators
    try:
        from ml.indicators_advanced import AdvancedIndicators
        adv = AdvancedIndicators()
        df_ind = adv.compute_all(df.copy())
        signals = adv.get_all_signals(df_ind)
        last = df_ind.iloc[-1]

        result["indicators"] = {
            "ichimoku": {"signal": round(signals.get("ichimoku", 0), 2),
                          "cloud": round(float(last.get("ichimoku_cloud_thickness", 0)), 4)},
            "mfi": {"value": round(float(last.get("mfi", 50)), 1),
                     "signal": round(signals.get("mfi", 0), 2)},
            "vwap": {"distance": round(float(last.get("vwap_distance", 0)), 4),
                      "signal": round(signals.get("vwap", 0), 2)},
            "keltner": {"squeeze": bool(last.get("keltner_squeeze", 0)),
                         "signal": round(signals.get("keltner", 0), 2)},
            "williams_r": {"value": round(float(last.get("williams_r", -50)), 1),
                            "signal": round(signals.get("williams_r", 0), 2)},
            "cmf": {"value": round(float(last.get("cmf", 0)), 4),
                     "signal": round(signals.get("cmf", 0), 2)},
            "cci": {"value": round(float(last.get("cci", 0)), 1),
                     "signal": round(signals.get("cci", 0), 2)},
            "stc": {"value": round(float(last.get("stc", 50)), 1),
                     "signal": round(signals.get("stc", 0), 2)},
            "donchian": {"signal": round(signals.get("donchian", 0), 2)},
            "fibonacci": {"signal": round(signals.get("fibonacci", 0), 2)},
            "composite": round(adv.get_composite_signal(df_ind), 4),
        }
    except Exception as e:
        log.debug(f"Indicators failed: {e}")

    # Candlestick Patterns
    try:
        from ml.candlestick_patterns import PatternEngine
        pe = PatternEngine()
        df_pat = pe.compute_all(df.copy())
        summary = pe.get_pattern_summary(df_pat)
        result["patterns"] = {
            "detected": [p["pattern"] for p in summary.get("patterns", [])],
            "signal": round(summary.get("signal", 0), 2),
            "confluence": round(summary.get("confluence_score", 0), 2),
            "swing_position": round(summary.get("swing_position", 0.5), 3),
        }
    except Exception as e:
        log.debug(f"Patterns failed: {e}")

    # Volatility
    try:
        from ml.volatility_engine import VolatilityEngine
        ve = VolatilityEngine()
        df_vol = ve.compute_all(df.copy())
        last_v = df_vol.iloc[-1]
        vol_reg = ve.get_regime(df_vol)
        result["volatility"] = {
            "ewma": round(float(last_v.get("ewma_vol", 0)), 6),
            "parkinson": round(float(last_v.get("parkinson_vol", 0)), 6),
            "cone_pct": round(float(last_v.get("vol_cone_percentile", 50)), 1),
            "ratio": round(float(last_v.get("vol_ratio_yz", 1)), 3),
            "squeeze": bool(last_v.get("bb_keltner_squeeze", 0)),
            "term_structure": round(float(last_v.get("vol_term_structure", 1)), 3),
        }
        result["vol_regime"] = vol_reg.value
    except Exception as e:
        log.debug(f"Volatility failed: {e}")

    # Statistical
    try:
        from ml.statistical_models import StatisticalEngine
        se = StatisticalEngine()
        df_stat = se.compute_all(df.copy())
        last_s = df_stat.iloc[-1]
        result["statistical"] = {
            "hurst": round(float(last_s.get("hurst_exponent", 0.5)), 3),
            "garch_vol": round(float(last_s.get("garch_vol", 0)), 6),
            "zscore": round(float(last_s.get("zscore_20", 0)), 3),
            "entropy": round(float(last_s.get("shannon_entropy", 0.5)), 3),
            "egarch_asym": round(float(last_s.get("egarch_asymmetry", 0)), 3),
        }
        result["hurst"] = result["statistical"]["hurst"]
    except Exception as e:
        log.debug(f"Statistical failed: {e}")

    # Regime (from strategy)
    try:
        from ml.bot_v3_patch import get_regime_params
        rp = get_regime_params(df)
        result["regime"] = rp.get("regime", "unknown")
    except Exception:
        pass

    # Current price
    try:
        result["price"] = round(float(df["close"].iloc[-1]), 2)
    except Exception:
        pass

    # OHLCV candle data for chart (last 100 candles)
    try:
        n = min(100, len(df))
        candles = []
        for i in range(len(df) - n, len(df)):
            row = df.iloc[i]
            # Use open_time if available, else index
            t = 0
            if "open_time" in df.columns:
                t = int(row["open_time"]) // 1000  # ms → seconds
            elif hasattr(df.index, "astype"):
                try:
                    t = int(pd.Timestamp(df.index[i]).timestamp())
                except Exception:
                    t = int(time.time()) - (n - i) * 3600

            candles.append({
                "time": t,
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
            })
        result["candles"] = candles
    except Exception:
        pass

    return result


def collect_dashboard_data(adapter, trackers: dict, iteration: int,
                            equity_curve: list) -> Dict:
    """
    Collect data for ALL coins. Returns dict with per-coin data.
    Dashboard reads selected coin's data via JavaScript.
    """
    global _cache, _cache_time

    now = time.time()
    if now - _cache_time < _CACHE_TTL and _cache.get("coins"):
        return _cache

    result = {
        "regime": "unknown",
        "vol_regime": "unknown",
        "hurst": 0.5,
        "risk_metrics": {},
        "features_count": 46,
        "models_active": 8,
        "coins": {},  # Per-coin indicator data
    }

    # Compute indicators for each coin
    for symbol in SYMBOLS:
        coin = symbol.replace("USDT", "")
        try:
            df = _get_coin_df(symbol, adapter)
            if not df.empty:
                coin_data = _compute_coin_indicators(df)
                result["coins"][coin] = coin_data
        except Exception as e:
            log.debug(f"Collector {coin} failed: {e}")

    # Use BTC as the "global" regime/hurst reference
    btc = result["coins"].get("BTC", {})
    result["regime"] = btc.get("regime", "unknown")
    result["vol_regime"] = btc.get("vol_regime", "unknown")
    result["hurst"] = btc.get("hurst", 0.5)

    # For backward compatibility, also set top-level fields from BTC
    result["indicator_summary"] = btc.get("indicators", {})
    result["patterns"] = btc.get("patterns", {})
    result["volatility"] = btc.get("volatility", {})
    result["statistical"] = btc.get("statistical", {})

    log.info(f"Collector: {len(result['coins'])} coins computed")

    _cache = result
    _cache_time = now
    return result
