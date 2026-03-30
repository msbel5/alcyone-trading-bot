#!/usr/bin/env python3
"""
GRU Model for crypto price direction prediction.
Lightweight, runs on Pi 5 (CPU, <20MB).
"""
import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

log = logging.getLogger("ml.gru")

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

MODEL_DIR = Path("/home/msbel/.openclaw/workspace/trading/ml/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_sequences(df, feature_cols, label_col, lookback=24):
    """Create sequences for GRU input."""
    features = df[feature_cols].values.astype(np.float32)
    labels = df[label_col].values.astype(np.int64)

    # Normalize features (per-sequence z-score)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1
    features = (features - mean) / std

    X, y = [], []
    for i in range(lookback, len(features) - 1):
        X.append(features[i - lookback:i])
        y.append(labels[i])

    return np.array(X), np.array(y), mean, std


def train_gru(symbol: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
              feature_cols: list, lookback: int = 24, epochs: int = 30,
              hidden_size: int = 64, num_layers: int = 2, lr: float = 0.001):
    """Train GRU model for a symbol."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Prepare data
    X_train, y_train, mean, std = _prepare_sequences(train_df, feature_cols, "label_1h", lookback)
    X_val, y_val, _, _ = _prepare_sequences(val_df, feature_cols, "label_1h", lookback)

    # Shift labels: -1,0,1 → 0,1,2 for CrossEntropy
    y_train = y_train + 1
    y_val = y_val + 1

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    # Model
    class GRUClassifier(nn.Module):
        def __init__(self, input_size, hidden, layers, num_classes=3):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden, layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden, num_classes)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    model = GRUClassifier(len(feature_cols), hidden_size, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                correct += (pred.argmax(1) == yb).sum().item()
                total += len(yb)

        val_acc = correct / max(total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            log.info(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

    # Restore best
    if best_state:
        model.load_state_dict(best_state)

    # Save model + normalization params
    save_path = MODEL_DIR / f"gru_{symbol.lower()}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "input_size": len(feature_cols),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "lookback": lookback,
            "feature_cols": feature_cols,
        },
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
    }, save_path)

    log.info(f"  {symbol} GRU saved: val_acc={best_val_acc:.4f}, path={save_path}")
    return model, best_val_acc


def predict_gru(symbol: str, recent_df: pd.DataFrame, feature_cols: list, lookback: int = 24) -> dict:
    """Load trained GRU and predict direction."""
    import torch
    import torch.nn as nn

    save_path = MODEL_DIR / f"gru_{symbol.lower()}.pt"
    if not save_path.exists():
        return {"signal": 0, "confidence": 0, "error": "model not found"}

    checkpoint = torch.load(save_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    mean = np.array(checkpoint["norm_mean"])
    std = np.array(checkpoint["norm_std"])

    class GRUClassifier(nn.Module):
        def __init__(self, input_size, hidden, layers, num_classes=3):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden, layers, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden, num_classes)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    model = GRUClassifier(cfg["input_size"], cfg["hidden_size"], cfg["num_layers"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Prepare input
    features = recent_df[feature_cols].values[-lookback:].astype(np.float32)
    std[std == 0] = 1
    features = (features - mean) / std
    x = torch.tensor(features).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    # probs: [down, flat, up]
    pred_class = int(np.argmax(probs))
    signal = pred_class - 1  # 0→-1, 1→0, 2→+1
    confidence = float(probs[pred_class])

    return {
        "signal": signal,
        "confidence": confidence,
        "probs": {"down": float(probs[0]), "flat": float(probs[1]), "up": float(probs[2])},
    }


def train_all():
    """Train GRU for all symbols."""
    from ml.data_pipeline import prepare_all_symbols, SEQUENCE_FEATURES

    print("Training GRU models for all symbols...\n")
    results = prepare_all_symbols()

    feature_cols = SEQUENCE_FEATURES
    accuracies = {}

    for symbol, data in results.items():
        print(f"\nTraining {symbol}...")
        _, acc = train_gru(symbol, data["train"], data["val"], feature_cols)
        accuracies[symbol] = acc
        print(f"  {symbol}: val_acc = {acc:.4f}")

    print("\n=== GRU Training Summary ===")
    for sym, acc in accuracies.items():
        status = "GOOD" if acc > 0.45 else "WEAK"
        print(f"  {sym}: {acc:.4f} [{status}]")
    avg = np.mean(list(accuracies.values()))
    print(f"  Average: {avg:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_all()
