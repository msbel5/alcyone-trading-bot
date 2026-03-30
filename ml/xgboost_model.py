#!/usr/bin/env python3
"""
XGBoost classifier for crypto price direction.
Fast, interpretable, works great with engineered features.
"""
import os
import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger("ml.xgboost")

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

MODEL_DIR = Path("/home/msbel/.openclaw/workspace/trading/ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_xgboost(symbol: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                   feature_cols: list):
    """Train XGBoost classifier for a symbol."""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, classification_report

    X_train = train_df[feature_cols].values
    y_train = train_df["label_1h"].values + 1  # -1,0,1 → 0,1,2

    X_val = val_df[feature_cols].values
    y_val = val_df["label_1h"].values + 1

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]

    # Save
    save_path = MODEL_DIR / f"xgboost_{symbol.lower()}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    log.info(f"  {symbol} XGBoost: val_acc={acc:.4f}, top_features={[f[0] for f in top_features]}")
    return model, acc


def predict_xgboost(symbol: str, recent_df: pd.DataFrame, feature_cols: list) -> dict:
    """Predict with trained XGBoost model."""
    save_path = MODEL_DIR / f"xgboost_{symbol.lower()}.pkl"
    if not save_path.exists():
        return {"signal": 0, "confidence": 0, "error": "model not found"}

    with open(save_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    cols = data["feature_cols"]

    X = recent_df[cols].values[-1:].astype(np.float32)
    probs = model.predict_proba(X)[0]

    pred_class = int(np.argmax(probs))
    signal = pred_class - 1  # 0→-1, 1→0, 2→+1
    confidence = float(probs[pred_class])

    return {
        "signal": signal,
        "confidence": confidence,
        "probs": {"down": float(probs[0]), "flat": float(probs[1]), "up": float(probs[2])},
    }


def train_all():
    """Train XGBoost for all symbols."""
    from ml.data_pipeline import prepare_all_symbols, FEATURE_COLS

    print("Training XGBoost models for all symbols...\n")
    results = prepare_all_symbols()

    accuracies = {}
    for symbol, data in results.items():
        print(f"\nTraining {symbol}...")
        _, acc = train_xgboost(symbol, data["train"], data["val"], FEATURE_COLS)
        accuracies[symbol] = acc
        print(f"  {symbol}: val_acc = {acc:.4f}")

    print("\n=== XGBoost Training Summary ===")
    for sym, acc in accuracies.items():
        status = "GOOD" if acc > 0.45 else "WEAK"
        print(f"  {sym}: {acc:.4f} [{status}]")
    avg = np.mean(list(accuracies.values()))
    print(f"  Average: {avg:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_all()
