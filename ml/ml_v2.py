#!/usr/bin/env python3
"""
ML v2 — Scientific best practices upgrade.
Walk-forward validation, LightGBM, stacked ensemble, feature selection,
dynamic thresholds. Replaces old train-once approach.
"""
import sys
import logging
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

log = logging.getLogger("ml.v2")

MODEL_DIR = Path("/home/msbel/.openclaw/workspace/trading/ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─── Step 3: Enhanced Feature Engineering ───────────────────────────

def add_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """20 features — optimal set per 2026 research."""
    from ml.data_pipeline import add_features
    df = add_features(df)

    close = df["close"]
    volume = df["volume"]

    # 5 NEW features (15 → 20)
    df["momentum_12h"] = close.pct_change(12).fillna(0)
    df["volume_momentum_4h"] = volume.pct_change(4).fillna(0)

    # RSI divergence: price makes new high but RSI doesn't
    price_high_20 = close.rolling(20, min_periods=1).max()
    rsi_at_price_high = df["rsi"].where(close == price_high_20).ffill().fillna(50)
    df["rsi_divergence"] = (df["rsi"] - rsi_at_price_high).fillna(0)

    # Bollinger Band width (volatility measure)
    if "bb_upper" in df.columns and "bb_lower" in df.columns:
        df["bb_width"] = ((df["bb_upper"] - df["bb_lower"]) / close).fillna(0)
    else:
        bb_mid = close.rolling(20, min_periods=1).mean()
        bb_std = close.rolling(20, min_periods=1).std().fillna(0)
        df["bb_width"] = (4 * bb_std / close).fillna(0)

    # Close to EMA ratio (mean reversion signal)
    df["close_ema_ratio"] = (close / df["ema_12"] - 1).fillna(0) if "ema_12" in df.columns else 0

    return df


FEATURE_COLS_V2 = [
    "ema_diff", "rsi", "macd_hist", "bb_pct", "atr_pct", "adx",
    "stoch_k", "stoch_d", "obv_diff", "vol_ratio",
    "momentum_1h", "momentum_4h", "momentum_24h",
    "hour", "day_of_week",
    # New 5
    "momentum_12h", "volume_momentum_4h", "rsi_divergence",
    "bb_width", "close_ema_ratio",
]


def feature_importance_mi(df: pd.DataFrame, label_col: str = "label_1h") -> pd.Series:
    """Mutual Information ranking of features."""
    from sklearn.feature_selection import mutual_info_classif

    features = df[FEATURE_COLS_V2].fillna(0).values
    labels = df[label_col].fillna(0).values + 1  # Shift to 0,1,2

    mi_scores = mutual_info_classif(features, labels, random_state=42)
    return pd.Series(mi_scores, index=FEATURE_COLS_V2).sort_values(ascending=False)


# ─── Step 4: Walk-Forward Validation ────────────────────────────────

def walk_forward_split(df: pd.DataFrame, train_months: int = 6, test_months: int = 1) -> List[Tuple]:
    """Generate walk-forward folds. Train 6mo, test 1mo, slide."""
    total_hours = len(df)
    train_size = train_months * 30 * 24
    test_size = test_months * 30 * 24

    folds = []
    start = 0
    while start + train_size + test_size <= total_hours:
        train_end = start + train_size
        test_end = train_end + test_size
        folds.append((
            df.iloc[start:train_end],
            df.iloc[train_end:test_end],
        ))
        start += test_size  # Slide by test window

    return folds


def walk_forward_evaluate(model_class, df: pd.DataFrame, feature_cols: list,
                           label_col: str = "label_1h", **model_kwargs) -> Dict:
    """Run walk-forward validation and return average metrics."""
    from sklearn.metrics import accuracy_score

    folds = walk_forward_split(df)
    accuracies = []

    for i, (train, test) in enumerate(folds):
        X_train = train[feature_cols].fillna(0).values
        y_train = train[label_col].values + 1
        X_test = test[feature_cols].fillna(0).values
        y_test = test[label_col].values + 1

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)

    return {
        "avg_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "min_accuracy": np.min(accuracies),
        "max_accuracy": np.max(accuracies),
        "n_folds": len(folds),
        "fold_accuracies": accuracies,
    }


# ─── Step 5: LightGBM Model ────────────────────────────────────────

def train_lightgbm(symbol: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                    feature_cols: list) -> Tuple:
    """Train LightGBM classifier."""
    import lightgbm as lgb

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["label_1h"].values + 1
    X_val = val_df[feature_cols].fillna(0).values
    y_val = val_df["label_1h"].values + 1

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        num_class=3,
        objective="multiclass",
        metric="multi_logloss",
        n_jobs=4,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    from sklearn.metrics import accuracy_score
    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)

    save_path = MODEL_DIR / f"lgbm_{symbol.lower()}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "feature_cols": feature_cols}, f)

    return model, acc


def predict_lightgbm(symbol: str, recent_df: pd.DataFrame, feature_cols: list) -> Dict:
    """Predict with LightGBM."""
    save_path = MODEL_DIR / f"lgbm_{symbol.lower()}.pkl"
    if not save_path.exists():
        return {"signal": 0, "confidence": 0, "probs": {"down": 0.33, "flat": 0.34, "up": 0.33}, "error": "model not found"}

    with open(save_path, "rb") as f:
        data = pickle.load(f)

    model = data["model"]
    X = recent_df[data["feature_cols"]].fillna(0).values[-1:]
    probs = model.predict_proba(X)[0]
    pred = int(np.argmax(probs))

    return {
        "signal": pred - 1,
        "confidence": float(probs[pred]),
        "probs": {"down": float(probs[0]), "flat": float(probs[1]), "up": float(probs[2])},
    }


# ─── Step 6: Stacked Ensemble (Meta-Learner) ───────────────────────

def train_stacked_ensemble(symbol: str, df: pd.DataFrame, feature_cols: list) -> Dict:
    """
    Train a stacked ensemble: base models predict on validation folds,
    meta-learner learns from their predictions.
    """
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score

    folds = walk_forward_split(df, train_months=6, test_months=1)
    if len(folds) < 3:
        return {"error": "Not enough data for stacked ensemble"}

    # Collect base model predictions across folds
    meta_features = []
    meta_labels = []

    for train_fold, test_fold in folds:
        X_train = train_fold[feature_cols].fillna(0).values
        y_train = train_fold["label_1h"].values + 1
        X_test = test_fold[feature_cols].fillna(0).values
        y_test = test_fold["label_1h"].values + 1

        # Base model 1: LightGBM
        lgbm = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                    verbose=-1, n_jobs=4, random_state=42)
        lgbm.fit(X_train, y_train)
        lgbm_probs = lgbm.predict_proba(X_test)

        # Base model 2: XGBoost
        import xgboost as xgb
        xgbm = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                   tree_method="hist", n_jobs=4, random_state=42,
                                   eval_metric="mlogloss")
        xgbm.fit(X_train, y_train)
        xgb_probs = xgbm.predict_proba(X_test)

        # Stack: [lgbm_down, lgbm_flat, lgbm_up, xgb_down, xgb_flat, xgb_up]
        stacked = np.hstack([lgbm_probs, xgb_probs])
        meta_features.append(stacked)
        meta_labels.append(y_test)

    X_meta = np.vstack(meta_features)
    y_meta = np.concatenate(meta_labels)

    # Meta-learner: lightweight LightGBM
    meta = lgb.LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                verbose=-1, n_jobs=4, random_state=42)
    meta.fit(X_meta, y_meta)

    # Save
    save_path = MODEL_DIR / f"stacked_{symbol.lower()}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "meta_model": meta,
            "feature_cols": feature_cols,
        }, f)

    acc = accuracy_score(y_meta, meta.predict(X_meta))
    log.info(f"{symbol} Stacked ensemble: acc={acc:.4f}")
    return {"accuracy": acc, "n_folds": len(folds)}


def predict_stacked(symbol: str, recent_df: pd.DataFrame, feature_cols: list) -> Dict:
    """Predict with stacked ensemble."""
    save_path = MODEL_DIR / f"stacked_{symbol.lower()}.pkl"
    if not save_path.exists():
        # Fallback to LightGBM alone
        return predict_lightgbm(symbol, recent_df, feature_cols)

    with open(save_path, "rb") as f:
        data = pickle.load(f)

    meta = data["meta_model"]
    cols = data["feature_cols"]
    X = recent_df[cols].fillna(0).values[-1:]

    # Get base model predictions
    lgbm_path = MODEL_DIR / f"lgbm_{symbol.lower()}.pkl"
    xgb_path = MODEL_DIR / f"xgboost_{symbol.lower()}.pkl"

    lgbm_probs = np.array([[0.33, 0.34, 0.33]])
    xgb_probs = np.array([[0.33, 0.34, 0.33]])

    if lgbm_path.exists():
        with open(lgbm_path, "rb") as f:
            lgbm_data = pickle.load(f)
        lgbm_cols = lgbm_data.get("feature_cols", cols)
        X_lgbm = recent_df[lgbm_cols].fillna(0).values[-1:]
        lgbm_probs = lgbm_data["model"].predict_proba(X_lgbm)

    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            xgb_data = pickle.load(f)
        xgb_cols = xgb_data.get("feature_cols", cols)
        X_xgb = recent_df[xgb_cols].fillna(0).values[-1:]
        xgb_probs = xgb_data["model"].predict_proba(X_xgb)

    stacked = np.hstack([lgbm_probs, xgb_probs])
    probs = meta.predict_proba(stacked)[0]
    pred = int(np.argmax(probs))

    return {
        "signal": pred - 1,
        "confidence": float(probs[pred]),
        "probs": {"down": float(probs[0]), "flat": float(probs[1]), "up": float(probs[2])},
    }


# ─── Step 7: Dynamic Confidence Threshold ───────────────────────────

def dynamic_threshold(atr_pct: float, base_buy: float = 0.40, base_sell: float = -0.40) -> Tuple[float, float]:
    """
    Adjust buy/sell thresholds based on current volatility.
    High volatility → higher threshold (more conservative)
    Low volatility → lower threshold (more opportunities)
    """
    # ATR as % of price — typical range 0.5% to 5%
    vol_norm = np.clip(atr_pct * 100, 0.5, 5.0)

    # Scale: low vol (0.5%) → threshold * 0.7, high vol (5%) → threshold * 1.4
    scale = 0.7 + (vol_norm - 0.5) / (5.0 - 0.5) * 0.7

    return base_buy * scale, base_sell * scale


# ─── Full Training Pipeline ─────────────────────────────────────────

def train_all_v2():
    """Train all models with scientific best practices."""
    from ml.data_pipeline import load_ohlcv, add_labels

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]

    print("=" * 70)
    print("ML v2 TRAINING — Scientific Best Practices")
    print("=" * 70)

    for sym in symbols:
        print(f"\n{'='*50}")
        print(f"  {sym}")
        print(f"{'='*50}")

        try:
            df = load_ohlcv(sym)
            df = add_features_v2(df)
            df = add_labels(df)
            df.dropna(inplace=True)
            print(f"  Data: {len(df)} candles")

            # Feature importance
            mi = feature_importance_mi(df)
            print(f"  Top 5 features: {list(mi.head(5).index)}")

            # Walk-forward: LightGBM
            try:
                import lightgbm as lgb
                wf_lgbm = walk_forward_evaluate(
                    lgb.LGBMClassifier, df, FEATURE_COLS_V2,
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    verbose=-1, n_jobs=4, random_state=42
                )
                print(f"  WF LightGBM: avg={wf_lgbm['avg_accuracy']:.4f} ±{wf_lgbm['std_accuracy']:.4f} ({wf_lgbm['n_folds']} folds)")
            except Exception as e:
                print(f"  WF LightGBM failed: {e}")
                wf_lgbm = None

            # Walk-forward: XGBoost
            import xgboost as xgb
            wf_xgb = walk_forward_evaluate(
                xgb.XGBClassifier, df, FEATURE_COLS_V2,
                n_estimators=200, max_depth=6, learning_rate=0.05,
                tree_method="hist", n_jobs=4, random_state=42,
                eval_metric="mlogloss"
            )
            print(f"  WF XGBoost:  avg={wf_xgb['avg_accuracy']:.4f} ±{wf_xgb['std_accuracy']:.4f} ({wf_xgb['n_folds']} folds)")

            # Train production models (70/30 split on full data)
            split = int(len(df) * 0.8)
            train_df = df.iloc[:split]
            val_df = df.iloc[split:]

            # LightGBM
            try:
                _, lgbm_acc = train_lightgbm(sym, train_df, val_df, FEATURE_COLS_V2)
                print(f"  LightGBM trained: val_acc={lgbm_acc:.4f}")
            except Exception as e:
                print(f"  LightGBM train failed: {e}")

            # Stacked ensemble
            try:
                stack_result = train_stacked_ensemble(sym, df, FEATURE_COLS_V2)
                print(f"  Stacked ensemble: acc={stack_result.get('accuracy',0):.4f}")
            except Exception as e:
                print(f"  Stacked ensemble failed: {e}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_all_v2()
