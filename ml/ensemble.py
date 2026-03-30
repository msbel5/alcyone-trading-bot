#!/usr/bin/env python3
"""
ML Ensemble Predictor — combines GRU, XGBoost, CryptoBERT, CryptoTrader-LM
into a single prediction signal for the trading bot.
"""
import logging
import numpy as np
from typing import Dict, Optional

log = logging.getLogger("ml.ensemble")


class EnsemblePredictor:
    """Combines multiple ML models into a single prediction signal."""

    def __init__(self):
        self.weights = {
            "gru": 0.30,
            "xgboost": 0.25,
            "hf_sentiment": 0.25,
            "news": 0.20,
        }
        self._gru_available = True
        self._xgb_available = True
        self._hf_available = True
        self._news_available = True

    def predict(self, symbol: str, recent_df, headlines: list = None,
                price_summary: str = "") -> Dict:
        """
        Get ensemble ML prediction for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            recent_df: DataFrame with recent OHLCV + features (min 24 rows)
            headlines: List of news headline strings
            price_summary: Short price summary for CryptoTrader-LM

        Returns:
            {"signal": -1/0/+1, "score": -1.0 to +1.0, "confidence": 0-1,
             "breakdown": {model_name: score}}
        """
        signals = {}
        active_weights = {}

        # 1. GRU prediction (local)
        if self._gru_available:
            try:
                from ml.gru_model import predict_gru
                from ml.data_pipeline import SEQUENCE_FEATURES
                result = predict_gru(symbol, recent_df, SEQUENCE_FEATURES)
                if "error" not in result:
                    signals["gru"] = result["signal"] * result["confidence"]
                    active_weights["gru"] = self.weights["gru"]
            except Exception as e:
                log.debug(f"GRU unavailable for {symbol}: {e}")
                self._gru_available = False

        # 2. XGBoost prediction (local)
        if self._xgb_available:
            try:
                from ml.xgboost_model import predict_xgboost
                from ml.data_pipeline import FEATURE_COLS
                result = predict_xgboost(symbol, recent_df, FEATURE_COLS)
                if "error" not in result:
                    signals["xgboost"] = result["signal"] * result["confidence"]
                    active_weights["xgboost"] = self.weights["xgboost"]
            except Exception as e:
                log.debug(f"XGBoost unavailable for {symbol}: {e}")
                self._xgb_available = False

        # 3. CryptoBERT sentiment (local on Pi, no API needed)
        if self._hf_available and headlines:
            try:
                from ml.local_sentiment import analyze_batch
                local_signal = analyze_batch(headlines)
                signals["cryptobert_local"] = local_signal
                active_weights["cryptobert_local"] = self.weights["hf_sentiment"]
            except Exception as e:
                log.debug(f"Local CryptoBERT unavailable: {e}")
                self._hf_available = False

        # 4. News-based signal (keyword analysis as fallback)
        if headlines:
            try:
                news_signal = self._keyword_sentiment(headlines)
                signals["news"] = news_signal
                active_weights["news"] = self.weights["news"]
            except Exception as e:
                log.debug(f"News analysis failed: {e}")

        # Combine with dynamic weight redistribution
        if not signals:
            return {"signal": 0, "score": 0.0, "confidence": 0.0, "breakdown": {}}

        total_weight = sum(active_weights.values())
        if total_weight == 0:
            return {"signal": 0, "score": 0.0, "confidence": 0.0, "breakdown": signals}

        # Weighted average
        score = sum(signals[k] * active_weights[k] / total_weight for k in signals)
        score = float(np.clip(score, -1, 1))

        # Convert to discrete signal
        if score > 0.25:
            signal = 1
        elif score < -0.25:
            signal = -1
        else:
            signal = 0

        confidence = abs(score)

        return {
            "signal": signal,
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "breakdown": {k: round(v, 4) for k, v in signals.items()},
            "active_models": list(signals.keys()),
            "total_models": len(self.weights),
        }

    def _keyword_sentiment(self, headlines: list) -> float:
        """Simple keyword-based sentiment as news fallback."""
        positive_keywords = [
            "surge", "rally", "bull", "gain", "rise", "high", "record",
            "adoption", "etf", "approval", "partnership", "upgrade",
            "growth", "breakout", "momentum",
        ]
        negative_keywords = [
            "crash", "drop", "bear", "fall", "low", "hack", "ban",
            "regulation", "fear", "sell-off", "decline", "loss",
            "fraud", "scam", "dump", "liquidation",
        ]

        pos_count = 0
        neg_count = 0
        for headline in headlines:
            text = headline.lower()
            pos_count += sum(1 for kw in positive_keywords if kw in text)
            neg_count += sum(1 for kw in negative_keywords if kw in text)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return float(np.clip((pos_count - neg_count) / total, -1, 1))


# Global singleton
_ensemble: Optional[EnsemblePredictor] = None


def get_ensemble() -> EnsemblePredictor:
    global _ensemble
    if _ensemble is None:
        _ensemble = EnsemblePredictor()
    return _ensemble


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Ensemble Predictor initialized")
    print(f"Weights: {EnsemblePredictor().weights}")
    print(f"Models: GRU (local), XGBoost (local), CryptoBERT (HF API), News (keyword)")
