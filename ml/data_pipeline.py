#!/usr/bin/env python3
"""
ML Data Pipeline — Feature engineering + labeling for crypto prediction.
Reads historical OHLCV CSVs, adds technical indicators, creates labels.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/msbel/.openclaw/workspace/trading/data/historical")
FEATURE_DIR = Path("/home/msbel/.openclaw/workspace/trading/data/features")


def load_ohlcv(symbol: str) -> pd.DataFrame:
    """Load raw OHLCV CSV for a symbol."""
    path = DATA_DIR / f"{symbol.lower()}_1h.csv"
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol}: {path}")
    df = pd.read_csv(path)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicator features to OHLCV dataframe."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── EMA ──
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["ema_diff"] = df["ema_12"] - df["ema_26"]

    # ── RSI ──
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    # ── MACD ──
    ema_f = close.ewm(span=12, adjust=False).mean()
    ema_s = close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema_f - ema_s
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # ── Bollinger Bands ──
    bb_mid = close.rolling(20, min_periods=1).mean()
    bb_std = close.rolling(20, min_periods=1).std().fillna(0)
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"] = ((close - df["bb_lower"]) / bb_width).fillna(0.5)

    # ── ATR ──
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()
    df["atr_pct"] = df["atr"] / close  # ATR as percentage of price

    # ── ADX ──
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx"] = dx.ewm(span=14, adjust=False).mean().fillna(0)

    # ── Stochastic ──
    lowest = low.rolling(14, min_periods=1).min()
    highest = high.rolling(14, min_periods=1).max()
    df["stoch_k"] = ((close - lowest) / (highest - lowest).replace(0, np.nan) * 100).fillna(50)
    df["stoch_d"] = df["stoch_k"].rolling(3, min_periods=1).mean()

    # ── OBV ──
    obv_dir = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
    df["obv"] = (volume * obv_dir).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=20, adjust=False).mean()
    df["obv_diff"] = df["obv"] - df["obv_ema"]

    # ── Volume ratio ──
    df["vol_ratio"] = volume / volume.rolling(10, min_periods=1).mean().replace(0, 1)

    # ── Price momentum ──
    df["momentum_1h"] = close.pct_change(1).fillna(0)
    df["momentum_4h"] = close.pct_change(4).fillna(0)
    df["momentum_24h"] = close.pct_change(24).fillna(0)

    # ── Time features ──
    df["hour"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek

    return df


def add_labels(df: pd.DataFrame, threshold_pct: float = 0.3) -> pd.DataFrame:
    """
    Add prediction labels: next 1h price direction.
    - up (1): next close > current close * (1 + threshold)
    - down (-1): next close < current close * (1 - threshold)
    - flat (0): otherwise
    """
    future_close = df["close"].shift(-1)
    pct_change = (future_close / df["close"] - 1) * 100

    df["label_1h"] = 0
    df.loc[pct_change >= threshold_pct, "label_1h"] = 1
    df.loc[pct_change <= -threshold_pct, "label_1h"] = -1

    # 4h direction (for Bi-LSTM secondary target)
    future_4h = df["close"].shift(-4)
    pct_4h = (future_4h / df["close"] - 1) * 100
    df["label_4h"] = 0
    df.loc[pct_4h >= threshold_pct * 2, "label_4h"] = 1
    df.loc[pct_4h <= -threshold_pct * 2, "label_4h"] = -1

    return df


# ── Feature columns used for ML ──
FEATURE_COLS = [
    "ema_diff", "rsi", "macd_hist", "bb_pct", "atr_pct", "adx",
    "stoch_k", "stoch_d", "obv_diff", "vol_ratio",
    "momentum_1h", "momentum_4h", "momentum_24h",
    "hour", "day_of_week",
]

SEQUENCE_FEATURES = [
    "close", "volume", "ema_diff", "rsi", "macd_hist", "bb_pct",
    "atr_pct", "obv_diff", "vol_ratio",
]


def prepare_dataset(symbol: str, lookback: int = 24):
    """Full pipeline: load → features → labels → train/val/test split."""
    df = load_ohlcv(symbol)
    df = add_features(df)
    df = add_labels(df)
    df.dropna(inplace=True)

    # Time-based split (no shuffle!)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(f"  {symbol}: total={n}, train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"  Label dist (train): up={sum(train['label_1h']==1)}, flat={sum(train['label_1h']==0)}, down={sum(train['label_1h']==-1)}")

    return {
        "train": train,
        "val": val,
        "test": test,
        "feature_cols": FEATURE_COLS,
        "sequence_features": SEQUENCE_FEATURES,
        "lookback": lookback,
    }


def prepare_all_symbols():
    """Prepare datasets for all 7 coins."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]

    results = {}
    for sym in symbols:
        try:
            data = prepare_dataset(sym)
            results[sym] = data
            # Save feature-enhanced CSV
            full_df = pd.concat([data["train"], data["val"], data["test"]])
            full_df.to_csv(FEATURE_DIR / f"{sym.lower()}_features.csv")
        except FileNotFoundError as e:
            print(f"  {sym}: SKIPPED ({e})")

    return results


if __name__ == "__main__":
    print("Preparing ML datasets for all symbols...\n")
    results = prepare_all_symbols()
    print(f"\nDone! {len(results)} symbols prepared.")
