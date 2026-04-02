#!/usr/bin/env python3
"""Test EVERY model individually for EVERY coin with real data."""
import sys
import time
sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

from ml.data_pipeline import load_ohlcv, add_labels, SEQUENCE_FEATURES
from ml.ml_v3 import (add_features_v3, FEATURE_COLS_V3, predict_stacked_v3,
                       predict_cnn_lstm, RegimeDetector, boruta_select,
                       cpcv_split)
from ml.ml_v2 import predict_lightgbm, predict_stacked, FEATURE_COLS_V2
from ml.gru_model import predict_gru
from ml.onchain_v2 import (compute_mvrv_zscore, mvrv_signal, fetch_btc_nvt,
                            fetch_exchange_flow, fetch_funding_rate,
                            fetch_liquidation_proxy, get_onchain_signal_v2)
from ml.bot_v3_patch import get_ml_signal_v3, get_regime_params, CircuitBreaker
import numpy as np

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]

print("=" * 70)
print("FULL MODEL TEST — Every model, every coin, real numbers")
print("=" * 70)

# ════════════════════════════════════════════════════════
# TEST 1: On-chain v2 signals (API calls)
# ════════════════════════════════════════════════════════
print("\n╔══ TEST 1: On-Chain v2 API Signals ══╗")

print("\n  [1a] BTC NVT Ratio:")
nvt = fetch_btc_nvt()
print(f"       NVT ratio = {nvt.get('nvt_ratio', 'FAIL')}")
print(f"       Market cap = ${nvt.get('market_cap', 'FAIL'):,.0f}" if isinstance(nvt.get('market_cap'), (int, float)) else f"       Market cap = {nvt.get('market_cap', 'FAIL')}")
print(f"       Signal = {nvt.get('signal', 'FAIL')}")
print(f"       STATUS: {'OK' if 'error' not in nvt else 'FAIL: ' + nvt['error']}")

print("\n  [1b] Exchange Flow (mempool):")
flow = fetch_exchange_flow()
print(f"       Fastest fee = {flow.get('fastest_fee', 'FAIL')} sat/vB")
print(f"       Mempool count = {flow.get('mempool_count', 'FAIL')}")
print(f"       Signal = {flow.get('inflow_signal', 'FAIL')}")
print(f"       STATUS: {'OK' if 'error' not in flow else 'FAIL: ' + flow['error']}")

print("\n  [1c] Funding Rate (BTC):")
fr = fetch_funding_rate("BTCUSDT")
print(f"       Rate = {fr.get('funding_rate', 'FAIL')}")
print(f"       Signal = {fr.get('signal', 'FAIL')}")
print(f"       STATUS: {'OK' if fr.get('funding_rate') is not None else 'FAIL'}")

print("\n  [1d] Long/Short Ratio (BTC):")
lq = fetch_liquidation_proxy("BTCUSDT")
print(f"       Open Interest = {lq.get('open_interest', 'FAIL')}")
print(f"       L/S Ratio = {lq.get('long_short_ratio', 'FAIL')}")
print(f"       Signal = {lq.get('signal', 'FAIL')}")
print(f"       STATUS: {'OK' if lq.get('open_interest', 0) > 0 else 'FAIL'}")

print("\n  [1e] Combined On-Chain v2 (BTC):")
prices = load_ohlcv("BTCUSDT")["close"].values
combined = get_onchain_signal_v2("BTCUSDT", prices)
print(f"       Combined signal = {combined.get('signal', 'FAIL')}")
print(f"       Components = {combined.get('signals', {})}")
print(f"       STATUS: {'OK' if combined.get('signal') is not None else 'FAIL'}")

# ════════════════════════════════════════════════════════
# TEST 2: Feature engineering (v3 — 25 features)
# ════════════════════════════════════════════════════════
print("\n╔══ TEST 2: Feature Engineering v3 ══╗")
for sym in SYMBOLS:
    try:
        df = load_ohlcv(sym)
        df = add_features_v3(df)
        df = add_labels(df)
        df.dropna(inplace=True)
        missing = [c for c in FEATURE_COLS_V3 if c not in df.columns]
        last = df.iloc[-1]
        print(f"  {sym:10s}: {len(df):5d} candles, {len(FEATURE_COLS_V3)} features, "
              f"missing={missing or 'none'}, "
              f"mvrv={last['mvrv_zscore']:.3f}, nvt={last['nvt_zscore']:.3f}")
    except Exception as e:
        print(f"  {sym:10s}: FAIL — {e}")

# ════════════════════════════════════════════════════════
# TEST 3: Each ML model individually
# ════════════════════════════════════════════════════════
print("\n╔══ TEST 3: Individual Model Predictions ══╗")

for sym in SYMBOLS:
    try:
        df = load_ohlcv(sym)
        df = add_features_v3(df)
        df = add_labels(df)
        df.dropna(inplace=True)
        recent = df.tail(100)
        coin = sym.replace("USDT", "")
        print(f"\n  ── {coin} ──")

        # LightGBM
        try:
            r = predict_lightgbm(sym, recent, FEATURE_COLS_V3)
            sig, conf = r["signal"], r["confidence"]
            label = {-1: "SELL", 0: "HOLD", 1: "BUY"}[sig]
            print(f"    LightGBM:   {label:4s} (signal={sig:+d}, conf={conf:.3f}, "
                  f"probs=[D:{r['probs']['down']:.3f} F:{r['probs']['flat']:.3f} U:{r['probs']['up']:.3f}])")
        except Exception as e:
            print(f"    LightGBM:   FAIL — {e}")

        # GRU
        try:
            r = predict_gru(sym, recent, SEQUENCE_FEATURES)
            sig, conf = r["signal"], r["confidence"]
            label = {-1: "SELL", 0: "HOLD", 1: "BUY"}[sig]
            print(f"    GRU:        {label:4s} (signal={sig:+d}, conf={conf:.3f}, "
                  f"probs=[D:{r['probs']['down']:.3f} F:{r['probs']['flat']:.3f} U:{r['probs']['up']:.3f}])")
        except Exception as e:
            print(f"    GRU:        FAIL — {e}")

        # CNN-LSTM
        try:
            r = predict_cnn_lstm(sym, recent, FEATURE_COLS_V3, lookback=48)
            if "error" in r and r["error"]:
                print(f"    CNN-LSTM:   NOT TRAINED YET (model file missing)")
            else:
                sig, conf = r["signal"], r["confidence"]
                label = {-1: "SELL", 0: "HOLD", 1: "BUY"}[sig]
                print(f"    CNN-LSTM:   {label:4s} (signal={sig:+d}, conf={conf:.3f}, "
                      f"probs=[D:{r['probs']['down']:.3f} F:{r['probs']['flat']:.3f} U:{r['probs']['up']:.3f}])")
        except Exception as e:
            print(f"    CNN-LSTM:   FAIL — {e}")

        # Stacked v3
        try:
            r = predict_stacked_v3(sym, recent, FEATURE_COLS_V3)
            sig, conf = r["signal"], r["confidence"]
            label = {-1: "SELL", 0: "HOLD", 1: "BUY"}[sig]
            print(f"    Stacked v3: {label:4s} (signal={sig:+d}, conf={conf:.3f}, "
                  f"probs=[D:{r['probs']['down']:.3f} F:{r['probs']['flat']:.3f} U:{r['probs']['up']:.3f}])")
        except Exception as e:
            print(f"    Stacked v3: FAIL — {e}")

        # Stacked v2 (old — compare)
        try:
            r = predict_stacked(sym, recent, FEATURE_COLS_V2)
            sig, conf = r["signal"], r["confidence"]
            label = {-1: "SELL", 0: "HOLD", 1: "BUY"}[sig]
            print(f"    Stacked v2: {label:4s} (signal={sig:+d}, conf={conf:.3f}) [OLD]")
        except Exception as e:
            print(f"    Stacked v2: FAIL — {e}")

        # Combined v3 signal (what bot actually uses)
        try:
            ml = get_ml_signal_v3(sym, df)
            print(f"    COMBINED:   signal={ml['signal']:+.4f}, models={ml['n_models']}, "
                  f"components={ml['components']}")
        except Exception as e:
            print(f"    COMBINED:   FAIL — {e}")

    except Exception as e:
        print(f"  {sym}: LOAD FAIL — {e}")

# ════════════════════════════════════════════════════════
# TEST 4: Regime Detection
# ════════════════════════════════════════════════════════
print("\n╔══ TEST 4: Regime Detection ══╗")
detector = RegimeDetector()
for sym in SYMBOLS:
    try:
        df = load_ohlcv(sym)
        df = add_features_v3(df)
        df.dropna(inplace=True)
        regime = detector.detect(df)
        params = detector.get_params(regime)
        last = df.iloc[-1]
        print(f"  {sym:10s}: regime={regime:10s}  ADX={last['adx']:.1f}  "
              f"ATR%={last['atr_pct']*100:.2f}%  "
              f"→ buy={params['buy_threshold']}, sell={params['sell_threshold']}, "
              f"pos_scale={params['position_scale']}")
    except Exception as e:
        print(f"  {sym:10s}: FAIL — {e}")

# ════════════════════════════════════════════════════════
# TEST 5: CPCV split verification
# ════════════════════════════════════════════════════════
print("\n╔══ TEST 5: CPCV Split Verification ══╗")
df = load_ohlcv("BTCUSDT")
df = add_features_v3(df)
df = add_labels(df)
df.dropna(inplace=True)
folds = cpcv_split(df, n_groups=6, n_test_groups=2)
print(f"  BTC data: {len(df)} candles")
print(f"  CPCV folds: {len(folds)} (expected C(6,2)=15)")
for i, (train, test) in enumerate(folds[:3]):
    print(f"    Fold {i}: train={len(train)}, test={len(test)}, "
          f"train_range=[{min(train)}..{max(train)}], test_range=[{min(test)}..{max(test)}]")
print(f"    ... ({len(folds)-3} more folds)")

# ════════════════════════════════════════════════════════
# TEST 6: Circuit Breaker
# ════════════════════════════════════════════════════════
print("\n╔══ TEST 6: Circuit Breaker ══╗")
cb = CircuitBreaker(daily_max_loss_pct=5.0, weekly_max_loss_pct=10.0)
cb.update(100.0)
print(f"  Start $100: tripped={cb.is_tripped()}, scale={cb.position_scale(100.0)}")
cb.update(96.0)
print(f"  Drop to $96 (4% loss): tripped={cb.is_tripped()}, scale={cb.position_scale(96.0)}")
cb.update(94.0)
print(f"  Drop to $94 (6% loss): tripped={cb.is_tripped()}, scale={cb.position_scale(94.0)}")

# ════════════════════════════════════════════════════════
# TEST 7: Boruta (quick check, not full run)
# ════════════════════════════════════════════════════════
print("\n╔══ TEST 7: Boruta Import + binomtest ══╗")
try:
    from scipy.stats import binomtest
    result = binomtest(25, 30, 0.5, alternative="greater")
    print(f"  binomtest(25/30, p=0.5): pvalue={result.pvalue:.6f}")
    print(f"  STATUS: OK (scipy.stats.binomtest works)")
except Exception as e:
    print(f"  STATUS: FAIL — {e}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
