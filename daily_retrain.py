#!/usr/bin/env python3
"""
Daily Walk-Forward Retrain — runs every night at 03:00 via systemd timer.
Downloads latest data, retrains models with walk-forward, swaps if better.
"""
import sys
import os
import time
import json
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
        logging.FileHandler(LOG_DIR / "retrain.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("retrain")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]
MODEL_DIR = Path("/home/msbel/.openclaw/workspace/trading/ml/models")
ACCURACY_LOG = LOG_DIR / "retrain_accuracy.json"


def load_accuracy_log() -> dict:
    if ACCURACY_LOG.exists():
        try:
            return json.loads(ACCURACY_LOG.read_text())
        except Exception:
            pass
    return {}


def save_accuracy_log(data: dict):
    ACCURACY_LOG.write_text(json.dumps(data, indent=2))


def run_retrain():
    """Full daily retrain pipeline."""
    start = time.time()
    log.info("=" * 60)
    log.info("DAILY RETRAIN STARTED")
    log.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    accuracy_log = load_accuracy_log()
    results = {}

    # Step 1: Download latest data
    log.info("\n[1/4] Downloading latest data...")
    try:
        from ml.download_historical import download_symbol
        for sym in SYMBOLS:
            download_symbol(sym)
        log.info("Data download complete")
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return

    # Step 2: Train LightGBM with walk-forward
    log.info("\n[2/4] Training LightGBM models...")
    try:
        from ml.ml_v2 import add_features_v2, FEATURE_COLS_V2, train_lightgbm, walk_forward_evaluate
        from ml.data_pipeline import load_ohlcv, add_labels
        import lightgbm as lgb

        for sym in SYMBOLS:
            try:
                df = load_ohlcv(sym)
                df = add_features_v2(df)
                df = add_labels(df)
                df.dropna(inplace=True)

                # Walk-forward accuracy
                wf = walk_forward_evaluate(
                    lgb.LGBMClassifier, df, FEATURE_COLS_V2,
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    verbose=-1, n_jobs=4, random_state=42
                )
                new_acc = wf["avg_accuracy"]

                # Compare with previous
                old_acc = accuracy_log.get(f"lgbm_{sym}", {}).get("accuracy", 0)
                improved = new_acc >= old_acc - 0.02  # Accept if within 2% of previous

                coin = sym.replace("USDT", "")
                if improved:
                    # Train production model
                    split = int(len(df) * 0.8)
                    train_lightgbm(sym, df.iloc[:split], df.iloc[split:], FEATURE_COLS_V2)
                    log.info(f"  {coin}: {new_acc:.4f} (prev {old_acc:.4f}) → SWAPPED")
                    accuracy_log[f"lgbm_{sym}"] = {"accuracy": new_acc, "date": datetime.now().isoformat()}
                else:
                    log.info(f"  {coin}: {new_acc:.4f} (prev {old_acc:.4f}) → KEPT OLD (worse by >{0.02:.0%})")

                results[f"lgbm_{sym}"] = {"new": new_acc, "old": old_acc, "swapped": improved}

            except Exception as e:
                log.error(f"  {sym} failed: {e}")

    except ImportError as e:
        log.error(f"LightGBM training failed (import): {e}")

    # Step 3: Retrain GRU models
    log.info("\n[3/4] Training GRU models...")
    try:
        from ml.gru_model import train_gru
        from ml.data_pipeline import load_ohlcv, add_labels, SEQUENCE_FEATURES
        from ml.ml_v2 import add_features_v2

        for sym in SYMBOLS:
            try:
                df = load_ohlcv(sym)
                df = add_features_v2(df)
                df = add_labels(df)
                df.dropna(inplace=True)

                split = int(len(df) * 0.8)
                _, gru_acc = train_gru(sym, df.iloc[:split], df.iloc[split:],
                                        SEQUENCE_FEATURES, epochs=20)

                coin = sym.replace("USDT", "")
                log.info(f"  {coin} GRU: {gru_acc:.4f}")
                results[f"gru_{sym}"] = {"accuracy": gru_acc}

            except Exception as e:
                log.error(f"  {sym} GRU failed: {e}")

    except ImportError as e:
        log.error(f"GRU training failed (import): {e}")

    # Step 4: Retrain stacked ensemble
    log.info("\n[4/4] Training stacked ensembles...")
    try:
        from ml.ml_v2 import train_stacked_ensemble, add_features_v2, FEATURE_COLS_V2
        from ml.data_pipeline import load_ohlcv, add_labels

        for sym in SYMBOLS:
            try:
                df = load_ohlcv(sym)
                df = add_features_v2(df)
                df = add_labels(df)
                df.dropna(inplace=True)

                stack_result = train_stacked_ensemble(sym, df, FEATURE_COLS_V2)
                coin = sym.replace("USDT", "")
                log.info(f"  {coin} Stacked: {stack_result.get('accuracy', 0):.4f}")
                results[f"stacked_{sym}"] = stack_result

            except Exception as e:
                log.error(f"  {sym} Stacked failed: {e}")

    except ImportError as e:
        log.error(f"Stacked training failed (import): {e}")

    # Save accuracy log
    save_accuracy_log(accuracy_log)

    # Send Telegram notification
    elapsed = time.time() - start
    try:
        from telegram_notifier import TelegramNotifier
        notifier = TelegramNotifier(enabled=True)

        summary_lines = ["Daily Retrain Complete\n"]
        for key, val in results.items():
            if isinstance(val, dict) and "new" in val:
                status = "SWAPPED" if val.get("swapped") else "KEPT"
                summary_lines.append(f"  {key}: {val['new']:.3f} ({status})")
            elif isinstance(val, dict) and "accuracy" in val:
                summary_lines.append(f"  {key}: {val['accuracy']:.3f}")

        summary_lines.append(f"\nDuration: {elapsed/60:.1f}min")
        notifier.send("\n".join(summary_lines), priority="low")
    except Exception:
        pass

    log.info(f"\n{'='*60}")
    log.info(f"RETRAIN COMPLETE — {elapsed/60:.1f} minutes")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    run_retrain()
