#!/usr/bin/env python3
"""
Bot v3 Patch — Integrates ML v3 (CNN-LSTM, Stacked v3, Regime Detection,
On-chain v2) into the existing bot.py.

This script patches the main bot loop to use:
1. v3 features (25 features including on-chain)
2. CNN-LSTM + Stacked v3 predictions
3. Regime-based parameter adjustment
4. On-chain v2 signal (MVRV + NVT + Exchange Flow)

Usage: Replace the ML signal computation section in bot.py main loop.
"""
import sys
import logging
import numpy as np

sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

log = logging.getLogger("bot_v3")


def get_ml_signal_v3(symbol: str, df, feature_cols=None) -> dict:
    """
    Get ML signal from all v3 models + ensemble.
    Combines: LightGBM, XGBoost, GRU, CNN-LSTM, Stacked v3.
    Returns weighted consensus signal.
    """
    signals = {}
    weights = {}

    # Try v3 features
    try:
        from ml.ml_v3 import add_features_v3, FEATURE_COLS_V3, predict_stacked_v3, predict_cnn_lstm
        df_v3 = add_features_v3(df.copy())
        cols = feature_cols or FEATURE_COLS_V3
        cols = [c for c in cols if c in df_v3.columns]
    except ImportError:
        from ml.ml_v2 import add_features_v2, FEATURE_COLS_V2
        df_v3 = add_features_v2(df.copy())
        cols = FEATURE_COLS_V2

    # 1. Stacked v3 ensemble (highest weight — meta-learner)
    try:
        from ml.ml_v3 import predict_stacked_v3
        pred = predict_stacked_v3(symbol, df_v3, cols)
        if pred.get("confidence", 0) > 0:
            signals["stacked_v3"] = pred["signal"]
            weights["stacked_v3"] = 0.35
    except Exception:
        pass

    # 2. CNN-LSTM
    try:
        from ml.ml_v3 import predict_cnn_lstm
        pred = predict_cnn_lstm(symbol, df_v3, cols)
        if pred.get("confidence", 0) > 0:
            signals["cnn_lstm"] = pred["signal"]
            weights["cnn_lstm"] = 0.25
    except Exception:
        pass

    # 3. LightGBM (fallback, always available)
    try:
        from ml.ml_v2 import predict_lightgbm
        pred = predict_lightgbm(symbol, df_v3, cols)
        if pred.get("confidence", 0) > 0:
            signals["lgbm"] = pred["signal"]
            weights["lgbm"] = 0.20
    except Exception:
        pass

    # 4. GRU
    try:
        from ml.gru_model import predict_gru
        from ml.data_pipeline import SEQUENCE_FEATURES
        pred = predict_gru(symbol, df_v3, SEQUENCE_FEATURES)
        if pred.get("confidence", 0) > 0:
            signals["gru"] = pred["signal"]
            weights["gru"] = 0.20
    except Exception:
        pass

    # Weighted consensus
    if weights:
        total_w = sum(weights.values())
        consensus = sum(signals[k] * weights[k] for k in signals) / total_w
    else:
        consensus = 0.0

    return {
        "signal": float(np.clip(consensus, -1, 1)),
        "components": signals,
        "weights": weights,
        "n_models": len(signals),
    }


def get_regime_params(df) -> dict:
    """Get regime-adjusted strategy parameters."""
    try:
        from ml.ml_v3 import RegimeDetector, add_features_v3
        detector = RegimeDetector()
        df_v3 = add_features_v3(df.copy()) if "mvrv_zscore" not in df.columns else df
        regime = detector.detect(df_v3)
        params = detector.get_params(regime)
        params["regime"] = regime
        return params
    except Exception as e:
        log.debug(f"Regime detection failed: {e}")
        return {"regime": "unknown", "buy_threshold": 0.40, "sell_threshold": -0.40,
                "position_scale": 1.0}


def get_onchain_v2_signal(symbol: str, prices=None) -> dict:
    """Get enhanced on-chain signal."""
    try:
        from ml.onchain_v2 import get_onchain_signal_v2
        return get_onchain_signal_v2(symbol, prices)
    except ImportError:
        from ml.onchain import get_onchain_signal
        return get_onchain_signal(symbol)


# ═══════════════════════════════════════════════════════════════════
# Circuit Breaker — halts trading when drawdown exceeds limits
# ═══════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Risk management circuit breaker.
    Halts trading when drawdown exceeds daily or weekly limits.
    Reduces position size proportionally as drawdown increases.
    """

    def __init__(self, daily_max_loss_pct: float = 5.0,
                 weekly_max_loss_pct: float = 10.0,
                 total_max_loss_pct: float = 20.0,
                 cooldown_hours: int = 4):
        self.daily_max_loss_pct = daily_max_loss_pct
        self.weekly_max_loss_pct = weekly_max_loss_pct
        self.total_max_loss_pct = total_max_loss_pct
        self.cooldown_hours = cooldown_hours

        self._initial_balance = None
        self._day_start_balance = None
        self._week_start_balance = None
        self._tripped = False
        self._trip_time = 0
        self._current_day = None
        self._current_week = None

    def update(self, current_balance: float, now=None):
        """Call every tick with current portfolio value."""
        import time
        from datetime import datetime

        if now is None:
            now = datetime.now()

        if self._initial_balance is None:
            self._initial_balance = current_balance

        # Reset daily tracker at midnight
        today = now.strftime("%Y-%m-%d")
        if self._current_day != today:
            self._current_day = today
            self._day_start_balance = current_balance

        # Reset weekly tracker on Monday
        week_key = now.strftime("%Y-W%W")
        if self._current_week != week_key:
            self._current_week = week_key
            self._week_start_balance = current_balance

        # Check cooldown
        if self._tripped:
            elapsed_h = (time.time() - self._trip_time) / 3600
            if elapsed_h >= self.cooldown_hours:
                self._tripped = False
                log.info("Circuit breaker reset after cooldown")
            else:
                return

        # Check daily loss
        if self._day_start_balance and self._day_start_balance > 0:
            daily_loss = (self._day_start_balance - current_balance) / self._day_start_balance * 100
            if daily_loss >= self.daily_max_loss_pct:
                self._tripped = True
                self._trip_time = time.time()
                log.warning(f"CIRCUIT BREAKER: Daily loss {daily_loss:.1f}% >= {self.daily_max_loss_pct}%")
                return

        # Check weekly loss
        if self._week_start_balance and self._week_start_balance > 0:
            weekly_loss = (self._week_start_balance - current_balance) / self._week_start_balance * 100
            if weekly_loss >= self.weekly_max_loss_pct:
                self._tripped = True
                self._trip_time = time.time()
                log.warning(f"CIRCUIT BREAKER: Weekly loss {weekly_loss:.1f}% >= {self.weekly_max_loss_pct}%")
                return

        # Check total loss
        if self._initial_balance and self._initial_balance > 0:
            total_loss = (self._initial_balance - current_balance) / self._initial_balance * 100
            if total_loss >= self.total_max_loss_pct:
                self._tripped = True
                self._trip_time = time.time()
                log.warning(f"CIRCUIT BREAKER: Total loss {total_loss:.1f}% >= {self.total_max_loss_pct}%")

    def is_tripped(self) -> bool:
        """Returns True if trading should be halted."""
        return self._tripped

    def position_scale(self, current_balance: float) -> float:
        """
        Drawdown-adjusted position scaling.
        At 10% drawdown → 75% of normal size.
        At 15% drawdown → 50% of normal size.
        """
        if self._initial_balance is None or self._initial_balance <= 0:
            return 1.0
        dd_pct = (self._initial_balance - current_balance) / self._initial_balance * 100
        if dd_pct <= 5:
            return 1.0
        elif dd_pct <= 10:
            return 0.75
        elif dd_pct <= 15:
            return 0.50
        else:
            return 0.25  # Severely reduced
