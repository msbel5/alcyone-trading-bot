#!/usr/bin/env python3
"""
Professional 5-Layer Trading Strategy
Layer 1: Trend (EMA crossover + ADX trend strength)
Layer 2: Momentum (RSI + MACD + Stochastic)
Layer 3: Volatility (Bollinger Bands + ATR dynamic SL/TP)
Layer 4: Volume (OBV trend + Volume spike detection)
Layer 5: Sentiment (Fear & Greed Index from API)

Signal = weighted vote from all 5 layers.
"""
import numpy as np
import pandas as pd
import requests
import logging

log = logging.getLogger("strategy")


class ProStrategy:
    """5-layer professional trading strategy."""

    def __init__(self,
                 # Trend
                 ema_fast=12, ema_slow=26, adx_period=14, adx_threshold=25,
                 # Momentum
                 rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9,
                 stoch_k=14, stoch_d=3,
                 # Volatility
                 bb_period=20, bb_std=2.0, atr_period=14,
                 atr_sl_mult=1.5, atr_tp_mult=2.5,
                 # Volume
                 obv_ema=20, volume_spike_mult=1.5,
                 # Weights
                 weight_trend=0.15, weight_momentum=0.15,
                 weight_volatility=0.10, weight_volume=0.08,
                 weight_sentiment=0.10, weight_ml=0.18,
                 weight_ichimoku=0.10, weight_patterns=0.07,
                 weight_statistical=0.07,
                 # Thresholds
                 buy_threshold=0.25, sell_threshold=-0.25):
        # Trend params
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        # Momentum params
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        # Volatility params
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.atr_period = atr_period
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        # Volume params
        self.obv_ema = obv_ema
        self.volume_spike_mult = volume_spike_mult
        # Weights (must sum to 1.0)
        self.weight_trend = weight_trend
        self.weight_momentum = weight_momentum
        self.weight_volatility = weight_volatility
        self.weight_volume = weight_volume
        self.weight_sentiment = weight_sentiment
        self.weight_ml = weight_ml
        self.weight_ichimoku = weight_ichimoku
        self.weight_patterns = weight_patterns
        self.weight_statistical = weight_statistical
        # External signals (set by bot before calculate_signals)
        self._ml_signal = 0.0
        self._ichimoku_signal = 0.0
        self._pattern_signal = 0.0
        self._statistical_signal = 0.0
        # Signal thresholds
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        # Cached sentiment
        self._fng_cache = None
        self._fng_cache_time = 0

    # ─── LAYER 1: TREND (EMA + ADX) ────────────────────────────────
    def _calc_trend(self, df):
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # EMA crossover
        df["ema_fast"] = close.ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = close.ewm(span=self.ema_slow, adjust=False).mean()
        df["ema_diff"] = df["ema_fast"] - df["ema_slow"]

        # ADX (Average Directional Index) — measures trend strength
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        # When +DM > -DM, keep +DM; else 0 (and vice versa)
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        atr_smooth = tr.ewm(span=self.adx_period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=self.adx_period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(span=self.adx_period, adjust=False).mean() / atr_smooth.replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df["adx"] = dx.ewm(span=self.adx_period, adjust=False).mean().fillna(0)
        df["plus_di"] = plus_di.fillna(0)
        df["minus_di"] = minus_di.fillna(0)

        # Trend score: -1 to +1
        # EMA bullish = positive, ADX strong = amplify
        ema_signal = np.where(df["ema_diff"] > 0, 1.0, -1.0)
        adx_strength = np.clip(df["adx"].values / 50.0, 0, 1)  # 0-1 scale
        df["trend_score"] = ema_signal * (0.5 + 0.5 * adx_strength)
        return df

    # ─── LAYER 2: MOMENTUM (RSI + MACD + Stochastic) ───────────────
    def _calc_momentum(self, df):
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

        # MACD
        ema_f = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_s = close.ewm(span=self.macd_slow, adjust=False).mean()
        df["macd_line"] = ema_f - ema_s
        df["macd_signal"] = df["macd_line"].ewm(span=self.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]

        # Stochastic %K and %D
        lowest_low = low.rolling(window=self.stoch_k, min_periods=1).min()
        highest_high = high.rolling(window=self.stoch_k, min_periods=1).max()
        denom = (highest_high - lowest_low).replace(0, np.nan)
        df["stoch_k"] = ((close - lowest_low) / denom * 100).fillna(50)
        df["stoch_d"] = df["stoch_k"].rolling(window=self.stoch_d, min_periods=1).mean()

        # Momentum score: -1 to +1
        # RSI: <30 = +1 (oversold buy), >70 = -1 (overbought sell), 50 = 0
        rsi_score = np.clip((50 - df["rsi"].values) / 30, -1, 1)
        # MACD: histogram positive = bullish
        macd_max = df["macd_hist"].abs().rolling(20, min_periods=1).max().replace(0, 1)
        macd_score = np.clip(df["macd_hist"].values / macd_max.values, -1, 1)
        # Stochastic: <20 = buy, >80 = sell
        stoch_score = np.clip((50 - df["stoch_k"].values) / 40, -1, 1)

        df["momentum_score"] = (rsi_score * 0.4 + macd_score * 0.35 + stoch_score * 0.25)
        return df

    # ─── LAYER 3: VOLATILITY (Bollinger Bands + ATR) ────────────────
    def _calc_volatility(self, df):
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Bollinger Bands
        df["bb_mid"] = close.rolling(window=self.bb_period, min_periods=1).mean()
        bb_std = close.rolling(window=self.bb_period, min_periods=1).std().fillna(0)
        df["bb_upper"] = df["bb_mid"] + self.bb_std * bb_std
        df["bb_lower"] = df["bb_mid"] - self.bb_std * bb_std
        bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        df["bb_pct"] = ((close - df["bb_lower"]) / bb_width).fillna(0.5)

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.ewm(span=self.atr_period, adjust=False).mean()

        # Dynamic SL/TP
        df["dynamic_sl"] = close - df["atr"] * self.atr_sl_mult
        df["dynamic_tp"] = close + df["atr"] * self.atr_tp_mult

        # Volatility score: price near lower BB = buy, near upper = sell
        df["volatility_score"] = np.clip(1.0 - 2.0 * df["bb_pct"].values, -1, 1)
        return df

    # ─── LAYER 4: VOLUME (OBV + Spike) ─────────────────────────────
    def _calc_volume(self, df):
        close = df["close"]
        volume = df["volume"]

        # OBV (On-Balance Volume)
        obv_direction = np.where(close.diff() > 0, 1, np.where(close.diff() < 0, -1, 0))
        df["obv"] = (volume * obv_direction).cumsum()
        df["obv_ema"] = df["obv"].ewm(span=self.obv_ema, adjust=False).mean()
        df["obv_trend"] = df["obv"] - df["obv_ema"]

        # Volume spike
        vol_avg = volume.rolling(window=10, min_periods=1).mean()
        df["vol_spike"] = volume > vol_avg * self.volume_spike_mult

        # Volume score
        obv_max = df["obv_trend"].abs().rolling(20, min_periods=1).max().replace(0, 1)
        obv_score = np.clip(df["obv_trend"].values / obv_max.values, -1, 1)
        # Volume spike amplifies the signal
        spike_mult = np.where(df["vol_spike"].values, 1.3, 0.8)
        df["volume_score"] = obv_score * spike_mult
        df["volume_score"] = df["volume_score"].clip(-1, 1)
        return df

    # ─── LAYER 5: SENTIMENT (Fear & Greed Index) ───────────────────
    def _get_fear_greed(self):
        """Fetch Fear & Greed Index from alternative.me API (cached 15min)."""
        import time
        now = time.time()
        if self._fng_cache is not None and (now - self._fng_cache_time) < 900:
            return self._fng_cache

        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            data = resp.json()
            value = int(data["data"][0]["value"])
            self._fng_cache = value
            self._fng_cache_time = now
            return value
        except Exception as e:
            log.warning(f"Fear & Greed API failed: {e}")
            return self._fng_cache if self._fng_cache is not None else 50

    def _calc_sentiment_score(self):
        """Convert Fear & Greed (0-100) to signal score (-1 to +1).
        Extreme fear (0-25) = strong buy (+1)
        Fear (25-40) = buy (+0.5)
        Neutral (40-60) = no signal (0)
        Greed (60-75) = sell (-0.5)
        Extreme greed (75-100) = strong sell (-1)
        """
        fng = self._get_fear_greed()
        if fng <= 25:
            return 1.0
        elif fng <= 40:
            return 0.5
        elif fng <= 60:
            return 0.0
        elif fng <= 75:
            return -0.5
        else:
            return -1.0

    # ─── COMBINED SIGNAL ────────────────────────────────────────────
    def calculate_signals(self, df):
        """Calculate all 5 layers and produce weighted signal."""
        df = df.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        df = self._calc_trend(df)
        df = self._calc_momentum(df)
        df = self._calc_volatility(df)
        df = self._calc_volume(df)

        sentiment = self._calc_sentiment_score()
        df["sentiment_score"] = sentiment

        # Weighted composite score (6 layers)
        df["ml_score"] = self._ml_signal
        df["ichimoku_score"] = self._ichimoku_signal
        df["pattern_score"] = self._pattern_signal
        df["statistical_score"] = self._statistical_signal

        # 9-layer weighted composite
        df["composite"] = (
            df["trend_score"] * self.weight_trend +
            df["momentum_score"] * self.weight_momentum +
            df["volatility_score"] * self.weight_volatility +
            df["volume_score"] * self.weight_volume +
            sentiment * self.weight_sentiment +
            self._ml_signal * self.weight_ml +
            self._ichimoku_signal * self.weight_ichimoku +
            self._pattern_signal * self.weight_patterns +
            self._statistical_signal * self.weight_statistical
        )

        # Signal: 1 = buy, -1 = sell, 0 = hold
        df["signal"] = 0
        df.loc[df["composite"] >= self.buy_threshold, "signal"] = 1
        df.loc[df["composite"] <= self.sell_threshold, "signal"] = -1

        # ADX filter: only trade when trend is strong enough
        weak_trend = df["adx"] < self.adx_threshold
        df.loc[weak_trend & (df["signal"] != 0), "signal"] = 0

        return df

    def set_ichimoku_signal(self, signal: float):
        self._ichimoku_signal = max(-1.0, min(1.0, float(signal)))

    def set_pattern_signal(self, signal: float):
        self._pattern_signal = max(-1.0, min(1.0, float(signal)))

    def set_statistical_signal(self, signal: float):
        self._statistical_signal = max(-1.0, min(1.0, float(signal)))

    def set_ml_signal(self, signal: float):
        """Set ML prediction signal from ensemble. Call before calculate_signals."""
        self._ml_signal = max(-1.0, min(1.0, float(signal)))

    def get_dynamic_sl_tp(self, df):
        """Return latest dynamic SL and TP from ATR."""
        if df.empty or "dynamic_sl" not in df.columns:
            return None, None
        last = df.iloc[-1]
        return float(last["dynamic_sl"]), float(last["dynamic_tp"])

    def get_signal_breakdown(self, df):
        """Return human-readable breakdown of last signal."""
        if df.empty:
            return {}
        last = df.iloc[-1]
        return {
            "composite": round(float(last.get("composite", 0)), 3),
            "trend": round(float(last.get("trend_score", 0)), 3),
            "momentum": round(float(last.get("momentum_score", 0)), 3),
            "volatility": round(float(last.get("volatility_score", 0)), 3),
            "volume": round(float(last.get("volume_score", 0)), 3),
            "sentiment": round(float(last.get("sentiment_score", 0)), 3),
            "ml_prediction": round(self._ml_signal, 3),
            "ichimoku": round(self._ichimoku_signal, 3),
            "patterns": round(self._pattern_signal, 3),
            "statistical": round(self._statistical_signal, 3),
            "rsi": round(float(last.get("rsi", 50)), 1),
            "adx": round(float(last.get("adx", 0)), 1),
            "macd_hist": round(float(last.get("macd_hist", 0)), 4),
            "bb_pct": round(float(last.get("bb_pct", 0.5)), 3),
            "signal": int(last.get("signal", 0)),
        }

    # Compatibility properties
    @property
    def fast_period(self):
        return self.ema_fast

    @property
    def slow_period(self):
        return self.ema_slow
