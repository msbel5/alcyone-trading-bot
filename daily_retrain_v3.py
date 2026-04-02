#!/usr/bin/env python3
"""
Daily CPCV Retrain v3 — runs every night at 03:00 via systemd timer.
Uses CPCV instead of walk-forward, Boruta features, PBO check.
Only swaps model if PBO < 0.5 and accuracy >= old - 2%.
"""
import sys
import os
import time
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

LOG_DIR = Path("/home/msbel/.openclaw/workspace/trading/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "retrain_v3.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("retrain_v3")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]
MODEL_DIR = Path("/home/msbel/.openclaw/workspace/trading/ml/models")
ACCURACY_LOG = LOG_DIR / "retrain_v3_accuracy.json"


def load_accuracy_log() -> dict:
    if ACCURACY_LOG.exists():
        try:
            return json.loads(ACCURACY_LOG.read_text())
        except Exception:
            pass
    return {}


def save_accuracy_log(data: dict):
    ACCURACY_LOG.write_text(json.dumps(data, indent=2))


def load_boruta_features() -> dict:
    """Load Boruta-selected features for each symbol."""
    path = MODEL_DIR / "boruta_features.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def run_retrain():
    """Full daily retrain with CPCV + PBO gating."""
    start = time.time()
    log.info("=" * 60)
    log.info("DAILY RETRAIN v3 — CPCV + Boruta + PBO")
    log.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    accuracy_log = load_accuracy_log()
    boruta_features = load_boruta_features()
    results = {}

    # Step 1: Download latest data
    log.info("\n[1/5] Downloading latest data...")
    try:
        from ml.download_historical import download_symbol
        for sym in SYMBOLS:
            download_symbol(sym)
        log.info("Data download complete")
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return

    # Step 2: CPCV evaluation + LightGBM retrain
    log.info("\n[2/5] CPCV evaluation + LightGBM retrain...")
    try:
        from ml.ml_v3 import add_features_v3, FEATURE_COLS_V3, cpcv_evaluate, RegimeDetector
        from ml.ml_v2 import train_lightgbm
        from ml.data_pipeline import load_ohlcv, add_labels
        import lightgbm as lgb

        regime_detector = RegimeDetector()

        for sym in SYMBOLS:
            try:
                df = load_ohlcv(sym)
                df = add_features_v3(df)
                df = add_labels(df)
                df.dropna(inplace=True)

                # Use Boruta features if available
                feature_cols = boruta_features.get(sym, FEATURE_COLS_V3)
                # Ensure all feature cols exist
                feature_cols = [c for c in feature_cols if c in df.columns]
                if len(feature_cols) < 5:
                    feature_cols = FEATURE_COLS_V3

                # Detect current regime
                regime = regime_detector.detect(df)

                # CPCV evaluation
                cpcv = cpcv_evaluate(
                    lgb.LGBMClassifier, df, feature_cols,
                    n_groups=6, n_test_groups=2,
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    verbose=-1, n_jobs=4, random_state=42
                )
                new_acc = cpcv["avg_accuracy"]
                new_pbo = cpcv["pbo"]

                # Compare with previous
                old_data = accuracy_log.get(f"lgbm_{sym}", {})
                old_acc = old_data.get("accuracy", 0)

                # PBO gate: only swap if PBO < 0.5 AND accuracy within tolerance
                should_swap = new_pbo < 0.5 and new_acc >= old_acc - 0.02
                coin = sym.replace("USDT", "")

                if should_swap:
                    split = int(len(df) * 0.8)
                    train_lightgbm(sym, df.iloc[:split], df.iloc[split:], feature_cols)
                    accuracy_log[f"lgbm_{sym}"] = {
                        "accuracy": new_acc,
                        "pbo": new_pbo,
                        "dsr": cpcv.get("dsr", 0),
                        "regime": regime,
                        "features": feature_cols,
                        "date": datetime.now().isoformat(),
                    }
                    log.info(f"  {coin}: acc={new_acc:.4f} PBO={new_pbo:.2f} regime={regime} → SWAPPED")
                else:
                    reason = f"PBO={new_pbo:.2f}" if new_pbo >= 0.5 else f"acc drop"
                    log.info(f"  {coin}: acc={new_acc:.4f} PBO={new_pbo:.2f} → KEPT ({reason})")

                results[f"lgbm_{sym}"] = {
                    "new_acc": new_acc, "old_acc": old_acc,
                    "pbo": new_pbo, "swapped": should_swap, "regime": regime,
                }

            except Exception as e:
                log.error(f"  {sym} CPCV failed: {e}")

    except ImportError as e:
        log.error(f"CPCV evaluation failed (import): {e}")

    # Step 3: Retrain GRU models
    log.info("\n[3/5] GRU retrain...")
    try:
        from ml.gru_model import train_gru
        from ml.data_pipeline import load_ohlcv, add_labels, SEQUENCE_FEATURES
        from ml.ml_v3 import add_features_v3

        for sym in SYMBOLS:
            try:
                df = load_ohlcv(sym)
                df = add_features_v3(df)
                df = add_labels(df)
                df.dropna(inplace=True)

                split = int(len(df) * 0.8)
                _, gru_acc = train_gru(sym, df.iloc[:split], df.iloc[split:],
                                        SEQUENCE_FEATURES, epochs=15)
                log.info(f"  {sym.replace('USDT','')}: GRU={gru_acc:.4f}")
                results[f"gru_{sym}"] = {"accuracy": gru_acc}
            except Exception as e:
                log.error(f"  {sym} GRU failed: {e}")
    except ImportError as e:
        log.error(f"GRU retrain failed (import): {e}")

    # Step 4: Retrain CNN-LSTM
    log.info("\n[4/5] CNN-LSTM retrain...")
    try:
        from ml.ml_v3 import train_cnn_lstm, add_features_v3
        from ml.data_pipeline import load_ohlcv, add_labels

        for sym in SYMBOLS:
            try:
                df = load_ohlcv(sym)
                df = add_features_v3(df)
                df = add_labels(df)
                df.dropna(inplace=True)

                feature_cols = boruta_features.get(sym, FEATURE_COLS_V3)
                feature_cols = [c for c in feature_cols if c in df.columns]

                split = int(len(df) * 0.8)
                _, cnn_acc = train_cnn_lstm(
                    sym, df.iloc[:split], df.iloc[split:],
                    feature_cols, lookback=48, epochs=25
                )
                log.info(f"  {sym.replace('USDT','')}: CNN-LSTM={cnn_acc:.4f}")
                results[f"cnn_lstm_{sym}"] = {"accuracy": cnn_acc}
            except Exception as e:
                log.error(f"  {sym} CNN-LSTM failed: {e}")
    except ImportError as e:
        log.error(f"CNN-LSTM retrain failed (import): {e}")

    # Step 5: Retrain stacked v3
    log.info("\n[5/5] Stacked v3 retrain...")
    try:
        from ml.ml_v3 import train_stacked_v3, add_features_v3
        from ml.data_pipeline import load_ohlcv, add_labels

        for sym in SYMBOLS:
            try:
                df = load_ohlcv(sym)
                df = add_features_v3(df)
                df = add_labels(df)
                df.dropna(inplace=True)

                feature_cols = boruta_features.get(sym, FEATURE_COLS_V3)
                feature_cols = [c for c in feature_cols if c in df.columns]

                stack_result = train_stacked_v3(sym, df, feature_cols)
                log.info(f"  {sym.replace('USDT','')}: Stacked={stack_result.get('accuracy',0):.4f}")
                results[f"stacked_{sym}"] = stack_result
            except Exception as e:
                log.error(f"  {sym} Stacked failed: {e}")
    except ImportError as e:
        log.error(f"Stacked retrain failed (import): {e}")

    # Save accuracy log
    save_accuracy_log(accuracy_log)

    # Send Telegram notification
    elapsed = time.time() - start
    try:
        from telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier(enabled=True)

        lines = ["📊 Daily Retrain v3 Complete\n"]
        for key, val in sorted(results.items()):
            if isinstance(val, dict):
                if "new_acc" in val:
                    status = "✅" if val.get("swapped") else "⏭️"
                    lines.append(f"  {status} {key}: {val['new_acc']:.3f} PBO={val.get('pbo','?')}")
                elif "accuracy" in val:
                    lines.append(f"  📈 {key}: {val['accuracy']:.3f}")

        lines.append(f"\n⏱️ Duration: {elapsed/60:.1f}min")
        notifier.send("\n".join(lines), priority="low")
    except Exception:
        pass

    log.info(f"\n{'='*60}")
    log.info(f"RETRAIN v3 COMPLETE — {elapsed/60:.1f} minutes")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    run_retrain()
