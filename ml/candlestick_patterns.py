#!/usr/bin/env python3
"""
Candlestick Pattern Recognition — 15 patterns + swing detection + confluence.
All computed from OHLCV data, zero API cost.

Research basis:
- "Japanese Candlestick Charting Techniques" — Steve Nison (1991)
- Pattern reliability weights from Bulkowski "Encyclopedia of Candlestick Charts"

SOLID Principles:
- SRP: Each pattern category is a separate class
- OCP: New patterns added by creating new class implementing PatternDetector
- LSP: All detectors return same signal format
- DIP: PatternEngine depends on PatternDetector interface, not implementations

Pattern Weights (Bulkowski research):
- Morning/Evening Star: 0.9 (strongest reversal)
- Engulfing: 0.8 (very reliable)
- Three White Soldiers/Black Crows: 0.7
- Hammer/Shooting Star: 0.6
- Harami/Tweezer: 0.5
- Marubozu: 0.5
- Inside/Outside Bar: 0.4-0.5
- Doji/Spinning Top: 0.3 (context-dependent)
"""
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

log = logging.getLogger("patterns.candlestick")


# ═══════════════════════════════════════════════════════════════════
# Interface
# ═══════════════════════════════════════════════════════════════════

class PatternDetector(ABC):
    """Base interface for pattern detection classes."""

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern columns to DataFrame. Returns modified df."""
        pass

    @abstractmethod
    def signal(self, df: pd.DataFrame) -> float:
        """Return combined signal [-1.0, +1.0] from last row."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


# ═══════════════════════════════════════════════════════════════════
# Helper Functions (shared across all pattern classes)
# ═══════════════════════════════════════════════════════════════════

def body_size(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Absolute body size."""
    return np.abs(close - open_)


def upper_shadow(high: np.ndarray, open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Upper shadow (wick) size."""
    return high - np.maximum(open_, close)


def lower_shadow(open_: np.ndarray, close: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Lower shadow (tail) size."""
    return np.minimum(open_, close) - low


def candle_range(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Total range of candle."""
    return high - low


def is_bullish(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Boolean: close > open."""
    return close > open_


def is_bearish(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Boolean: close < open."""
    return close < open_


def avg_body(open_: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Rolling average body size for relative comparison."""
    bodies = body_size(open_, close)
    result = np.zeros_like(bodies)
    for i in range(period, len(bodies)):
        result[i] = np.mean(bodies[max(0, i - period):i])
    result[:period] = np.mean(bodies[:period]) if period <= len(bodies) else 1.0
    return np.where(result == 0, 1.0, result)


def trend_direction(close: np.ndarray, lookback: int = 10) -> np.ndarray:
    """
    Recent trend: +1 uptrend, -1 downtrend, 0 sideways.
    Uses simple slope of last N closes.
    """
    n = len(close)
    trend = np.zeros(n)
    for i in range(lookback, n):
        window = close[i - lookback:i + 1]
        slope = (window[-1] - window[0]) / (window[0] if window[0] != 0 else 1)
        if slope > 0.02:
            trend[i] = 1.0
        elif slope < -0.02:
            trend[i] = -1.0
    return trend


# ═══════════════════════════════════════════════════════════════════
# 1. Single Candle Patterns
# ═══════════════════════════════════════════════════════════════════

class SingleCandlePatterns(PatternDetector):
    """
    5 single-candle patterns:
    - Doji: indecision (small body, weight 0.3)
    - Hammer: bullish reversal at bottom (long lower shadow, weight 0.6)
    - Shooting Star: bearish reversal at top (long upper shadow, weight 0.6)
    - Spinning Top: indecision (small body, large shadows, weight 0.3)
    - Marubozu: strong conviction (no shadows, weight 0.5)
    """

    @property
    def name(self) -> str:
        return "single_candle"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)

        bs = body_size(o, c)
        us = upper_shadow(h, o, c)
        ls = lower_shadow(o, c, l)
        cr = candle_range(h, l)
        cr_safe = np.where(cr == 0, 1.0, cr)
        ab = avg_body(o, c)
        trend = trend_direction(c)

        # Doji: body < 10% of range
        doji = (bs / cr_safe) < 0.10
        df["pattern_doji"] = doji.astype(float)

        # Hammer: in downtrend, lower shadow >= 2x body, small upper shadow
        hammer = ((trend < 0) &
                  (ls >= 2 * bs) &
                  (us < bs * 0.5) &
                  (bs > 0))
        df["pattern_hammer"] = hammer.astype(float)

        # Shooting Star: in uptrend, upper shadow >= 2x body, small lower shadow
        shooting_star = ((trend > 0) &
                         (us >= 2 * bs) &
                         (ls < bs * 0.5) &
                         (bs > 0))
        df["pattern_shooting_star"] = shooting_star.astype(float)

        # Spinning Top: small body (< 30% range), large shadows on both sides
        spinning_top = ((bs / cr_safe < 0.30) &
                        (us > bs * 0.5) &
                        (ls > bs * 0.5) &
                        (cr > 0))
        df["pattern_spinning_top"] = spinning_top.astype(float)

        # Marubozu: body >= 95% of range (no shadows)
        marubozu_bull = ((bs / cr_safe >= 0.95) & is_bullish(o, c))
        marubozu_bear = ((bs / cr_safe >= 0.95) & is_bearish(o, c))
        df["pattern_marubozu_bull"] = marubozu_bull.astype(float)
        df["pattern_marubozu_bear"] = marubozu_bear.astype(float)

        # Combined single-candle signal
        sig = np.zeros(len(df))
        # Bullish patterns
        sig += hammer * 0.6          # Hammer = buy
        sig += marubozu_bull * 0.5   # Bullish marubozu = buy
        # Bearish patterns
        sig -= shooting_star * 0.6   # Shooting star = sell
        sig -= marubozu_bear * 0.5   # Bearish marubozu = sell
        # Doji/spinning top: context-dependent (multiply by -trend for reversal)
        sig -= doji * trend * 0.3
        sig -= spinning_top * trend * 0.3

        df["single_candle_signal"] = np.clip(sig, -1, 1)
        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "single_candle_signal" not in df.columns:
            df = self.detect(df)
        return float(df["single_candle_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 2. Double Candle Patterns
# ═══════════════════════════════════════════════════════════════════

class DoubleCandlePatterns(PatternDetector):
    """
    3 double-candle patterns:
    - Bullish/Bearish Engulfing (weight 0.8 — most reliable double)
    - Tweezer Top/Bottom (weight 0.5)
    - Bullish/Bearish Harami (weight 0.5)
    """

    @property
    def name(self) -> str:
        return "double_candle"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        n = len(df)

        bull_engulf = np.zeros(n)
        bear_engulf = np.zeros(n)
        tweezer_top = np.zeros(n)
        tweezer_bot = np.zeros(n)
        bull_harami = np.zeros(n)
        bear_harami = np.zeros(n)

        for i in range(1, n):
            prev_o, prev_c = o[i - 1], c[i - 1]
            curr_o, curr_c = o[i], c[i]
            prev_h, prev_l = h[i - 1], l[i - 1]
            curr_h, curr_l = h[i], l[i]

            prev_body = abs(prev_c - prev_o)
            curr_body = abs(curr_c - curr_o)

            # Bullish Engulfing: bearish candle → bullish candle that fully engulfs
            if (prev_c < prev_o and          # Previous bearish
                curr_c > curr_o and          # Current bullish
                curr_o <= prev_c and         # Opens at/below previous close
                curr_c >= prev_o and         # Closes at/above previous open
                curr_body > prev_body):      # Current body larger
                bull_engulf[i] = 1.0

            # Bearish Engulfing: bullish → bearish that fully engulfs
            if (prev_c > prev_o and
                curr_c < curr_o and
                curr_o >= prev_c and
                curr_c <= prev_o and
                curr_body > prev_body):
                bear_engulf[i] = 1.0

            # Tweezer Top: matching highs in uptrend
            high_match = abs(prev_h - curr_h) / max(prev_h, 1) < 0.001
            if high_match and prev_c > prev_o and curr_c < curr_o:
                tweezer_top[i] = 1.0

            # Tweezer Bottom: matching lows in downtrend
            low_match = abs(prev_l - curr_l) / max(prev_l, 1) < 0.001
            if low_match and prev_c < prev_o and curr_c > curr_o:
                tweezer_bot[i] = 1.0

            # Bullish Harami: large bearish → small bullish inside
            if (prev_c < prev_o and curr_c > curr_o and
                curr_body < prev_body * 0.5 and
                curr_o > prev_c and curr_c < prev_o):
                bull_harami[i] = 1.0

            # Bearish Harami: large bullish → small bearish inside
            if (prev_c > prev_o and curr_c < curr_o and
                curr_body < prev_body * 0.5 and
                curr_o < prev_c and curr_c > prev_o):
                bear_harami[i] = 1.0

        df["pattern_bull_engulf"] = bull_engulf
        df["pattern_bear_engulf"] = bear_engulf
        df["pattern_tweezer_top"] = tweezer_top
        df["pattern_tweezer_bot"] = tweezer_bot
        df["pattern_bull_harami"] = bull_harami
        df["pattern_bear_harami"] = bear_harami

        # Combined signal
        sig = (bull_engulf * 0.8 - bear_engulf * 0.8 +
               tweezer_bot * 0.5 - tweezer_top * 0.5 +
               bull_harami * 0.5 - bear_harami * 0.5)
        df["double_candle_signal"] = np.clip(sig, -1, 1)

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "double_candle_signal" not in df.columns:
            df = self.detect(df)
        return float(df["double_candle_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 3. Triple Candle Patterns
# ═══════════════════════════════════════════════════════════════════

class TripleCandlePatterns(PatternDetector):
    """
    4 triple-candle patterns:
    - Morning Star (weight 0.9 — strongest bullish reversal)
    - Evening Star (weight 0.9 — strongest bearish reversal)
    - Three White Soldiers (weight 0.7)
    - Three Black Crows (weight 0.7)
    """

    @property
    def name(self) -> str:
        return "triple_candle"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        n = len(df)

        morning_star = np.zeros(n)
        evening_star = np.zeros(n)
        three_white = np.zeros(n)
        three_black = np.zeros(n)

        for i in range(2, n):
            o1, c1 = o[i - 2], c[i - 2]
            o2, c2 = o[i - 1], c[i - 1]
            o3, c3 = o[i], c[i]

            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)
            body3 = abs(c3 - o3)
            avg_b = (body1 + body2 + body3) / 3 if (body1 + body2 + body3) > 0 else 1

            # Morning Star: bearish → small body gap down → bullish gap up
            if (c1 < o1 and              # First: bearish
                body2 < body1 * 0.3 and  # Second: small body (star)
                c3 > o3 and              # Third: bullish
                c3 > (o1 + c1) / 2 and   # Third closes above midpoint of first
                body3 > body2):          # Third body > star body
                morning_star[i] = 1.0

            # Evening Star: bullish → small body gap up → bearish gap down
            if (c1 > o1 and
                body2 < body1 * 0.3 and
                c3 < o3 and
                c3 < (o1 + c1) / 2 and
                body3 > body2):
                evening_star[i] = 1.0

            # Three White Soldiers: 3 consecutive bullish with higher closes
            if (c1 > o1 and c2 > o2 and c3 > o3 and  # All bullish
                c2 > c1 and c3 > c2 and                # Each closes higher
                body1 > avg_b * 0.5 and                 # Decent body sizes
                body2 > avg_b * 0.5 and
                body3 > avg_b * 0.5):
                three_white[i] = 1.0

            # Three Black Crows: 3 consecutive bearish with lower closes
            if (c1 < o1 and c2 < o2 and c3 < o3 and
                c2 < c1 and c3 < c2 and
                body1 > avg_b * 0.5 and
                body2 > avg_b * 0.5 and
                body3 > avg_b * 0.5):
                three_black[i] = 1.0

        df["pattern_morning_star"] = morning_star
        df["pattern_evening_star"] = evening_star
        df["pattern_three_white"] = three_white
        df["pattern_three_black"] = three_black

        sig = (morning_star * 0.9 - evening_star * 0.9 +
               three_white * 0.7 - three_black * 0.7)
        df["triple_candle_signal"] = np.clip(sig, -1, 1)

        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "triple_candle_signal" not in df.columns:
            df = self.detect(df)
        return float(df["triple_candle_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 4. Multi-Candle Patterns (Inside/Outside Bar)
# ═══════════════════════════════════════════════════════════════════

class MultiCandlePatterns(PatternDetector):
    """
    2 multi-candle patterns:
    - Inside Bar: consolidation before breakout (weight 0.4)
    - Outside Bar: expansion, engulfs previous range (weight 0.5)
    """

    @property
    def name(self) -> str:
        return "multi_candle"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        o = df["open"].values.astype(float)
        n = len(df)

        inside_bar = np.zeros(n)
        outside_bar = np.zeros(n)

        for i in range(1, n):
            # Inside Bar: current range fully inside previous range
            if h[i] <= h[i - 1] and l[i] >= l[i - 1]:
                inside_bar[i] = 1.0

            # Outside Bar: current range fully contains previous range
            if h[i] > h[i - 1] and l[i] < l[i - 1]:
                outside_bar[i] = 1.0

        df["pattern_inside_bar"] = inside_bar
        df["pattern_outside_bar"] = outside_bar

        # Signal: inside bar → neutral (waiting for breakout)
        # Outside bar → direction of close relative to midpoint
        trend = trend_direction(c)
        sig = np.zeros(n)
        for i in range(1, n):
            if outside_bar[i]:
                mid = (h[i - 1] + l[i - 1]) / 2
                if c[i] > mid:
                    sig[i] = 0.5
                else:
                    sig[i] = -0.5
            elif inside_bar[i]:
                # Inside bar: slight lean toward trend continuation
                sig[i] = trend[i] * 0.2

        df["multi_candle_signal"] = np.clip(sig, -1, 1)
        return df

    def signal(self, df: pd.DataFrame) -> float:
        if "multi_candle_signal" not in df.columns:
            df = self.detect(df)
        return float(df["multi_candle_signal"].iloc[-1])


# ═══════════════════════════════════════════════════════════════════
# 5. Swing Point Detector
# ═══════════════════════════════════════════════════════════════════

class SwingDetector:
    """
    Detects local swing highs and swing lows.
    Used by Fibonacci, pattern confluence, and as ML feature.
    """

    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        n = len(df)
        lb = self.lookback

        swing_high = np.zeros(n)
        swing_low = np.zeros(n)
        swing_position = np.full(n, 0.5)

        last_swing_h = h[0]
        last_swing_l = l[0]

        for i in range(lb, n - lb):
            # Swing high: highest point in window
            window_h = h[i - lb:i + lb + 1]
            if h[i] == window_h.max() and h[i] > h[i - 1] and h[i] > h[i + 1]:
                swing_high[i] = 1.0
                last_swing_h = h[i]

            # Swing low: lowest point in window
            window_l = l[i - lb:i + lb + 1]
            if l[i] == window_l.min() and l[i] < l[i - 1] and l[i] < l[i + 1]:
                swing_low[i] = 1.0
                last_swing_l = l[i]

            # Position between last swing low and swing high [0, 1]
            sw_range = last_swing_h - last_swing_l
            if sw_range > 0:
                swing_position[i] = np.clip(
                    (c[i] - last_swing_l) / sw_range, 0, 1
                )

        df["swing_high"] = swing_high
        df["swing_low"] = swing_low
        df["swing_position"] = swing_position

        return df


# ═══════════════════════════════════════════════════════════════════
# 6. Pattern Confluence — Multiple Patterns = Stronger Signal
# ═══════════════════════════════════════════════════════════════════

class PatternConfluence:
    """
    Combines all pattern signals with diminishing returns.
    Multiple patterns at same candle = stronger confirmation.
    Formula: first + 0.5*second + 0.25*third + ...
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        signal_cols = [c for c in df.columns if c.endswith("_signal") and
                       "candle" in c]

        if not signal_cols:
            df["pattern_confluence_score"] = 0.0
            df["pattern_signal"] = 0.0
            return df

        # Collect all signals
        signals = df[signal_cols].values  # (n_rows, n_signals)
        n = len(df)

        confluence = np.zeros(n)
        for i in range(n):
            row_sigs = signals[i]
            # Separate bullish and bearish signals
            bullish = sorted([s for s in row_sigs if s > 0], reverse=True)
            bearish = sorted([abs(s) for s in row_sigs if s < 0], reverse=True)

            # Diminishing returns
            bull_total = sum(s * (0.5 ** j) for j, s in enumerate(bullish))
            bear_total = sum(s * (0.5 ** j) for j, s in enumerate(bearish))

            confluence[i] = bull_total - bear_total

        df["pattern_confluence_score"] = np.clip(confluence, -2, 2)
        # Normalize to [-1, 1] for ML
        df["pattern_signal"] = np.clip(confluence, -1, 1)

        return df


# ═══════════════════════════════════════════════════════════════════
# Facade — PatternEngine (orchestrates all pattern detection)
# ═══════════════════════════════════════════════════════════════════

class PatternEngine:
    """
    Facade for all candlestick pattern detection.
    Computes all patterns, swing points, and confluence in one call.
    """

    ML_FEATURE_COLS = [
        "pattern_signal", "pattern_confluence_score", "swing_position",
    ]

    def __init__(self):
        self.detectors: List[PatternDetector] = [
            SingleCandlePatterns(),
            DoubleCandlePatterns(),
            TripleCandlePatterns(),
            MultiCandlePatterns(),
        ]
        self.swing_detector = SwingDetector()
        self.confluence = PatternConfluence()

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect all patterns and compute confluence."""
        for detector in self.detectors:
            try:
                df = detector.detect(df)
            except Exception as e:
                log.warning(f"Pattern {detector.name} failed: {e}")

        # Swing points
        try:
            df = self.swing_detector.detect(df)
        except Exception as e:
            log.warning(f"Swing detection failed: {e}")
            df["swing_position"] = 0.5

        # Confluence
        try:
            df = self.confluence.compute(df)
        except Exception as e:
            log.warning(f"Confluence failed: {e}")
            df["pattern_confluence_score"] = 0.0
            df["pattern_signal"] = 0.0

        return df

    def get_all_signals(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get individual pattern signals from last row."""
        signals = {}
        for detector in self.detectors:
            signals[detector.name] = detector.signal(df)
        if "pattern_signal" in df.columns:
            signals["confluence"] = float(df["pattern_signal"].iloc[-1])
        return signals

    def get_composite_signal(self, df: pd.DataFrame) -> float:
        """Get the final confluence signal."""
        if "pattern_signal" not in df.columns:
            df = self.compute_all(df)
        return float(df["pattern_signal"].iloc[-1])

    def get_pattern_summary(self, df: pd.DataFrame) -> Dict:
        """Get human-readable summary of detected patterns at last candle."""
        if df.empty:
            return {"patterns": [], "signal": 0}

        last = df.iloc[-1]
        detected = []

        pattern_map = {
            "pattern_doji": ("Doji", "indecision"),
            "pattern_hammer": ("Hammer", "bullish reversal"),
            "pattern_shooting_star": ("Shooting Star", "bearish reversal"),
            "pattern_spinning_top": ("Spinning Top", "indecision"),
            "pattern_marubozu_bull": ("Bullish Marubozu", "strong buy"),
            "pattern_marubozu_bear": ("Bearish Marubozu", "strong sell"),
            "pattern_bull_engulf": ("Bullish Engulfing", "strong buy"),
            "pattern_bear_engulf": ("Bearish Engulfing", "strong sell"),
            "pattern_tweezer_top": ("Tweezer Top", "bearish"),
            "pattern_tweezer_bot": ("Tweezer Bottom", "bullish"),
            "pattern_bull_harami": ("Bullish Harami", "buy"),
            "pattern_bear_harami": ("Bearish Harami", "sell"),
            "pattern_morning_star": ("Morning Star", "strong buy"),
            "pattern_evening_star": ("Evening Star", "strong sell"),
            "pattern_three_white": ("Three White Soldiers", "buy"),
            "pattern_three_black": ("Three Black Crows", "sell"),
            "pattern_inside_bar": ("Inside Bar", "consolidation"),
            "pattern_outside_bar": ("Outside Bar", "expansion"),
        }

        for col, (name, meaning) in pattern_map.items():
            if col in df.columns and last.get(col, 0) > 0:
                detected.append({"pattern": name, "meaning": meaning})

        return {
            "patterns": detected,
            "confluence_score": float(last.get("pattern_confluence_score", 0)),
            "signal": float(last.get("pattern_signal", 0)),
            "swing_position": float(last.get("swing_position", 0.5)),
        }


# ═══════════════════════════════════════════════════════════════════
# Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/msbel/.openclaw/workspace/trading")

    logging.basicConfig(level=logging.INFO)
    print("=" * 70)
    print("CANDLESTICK PATTERNS TEST — 15 patterns + swing + confluence")
    print("=" * 70)

    try:
        from ml.data_pipeline import load_ohlcv
        df = load_ohlcv("BTCUSDT")
    except Exception:
        np.random.seed(42)
        n = 1000
        close = np.cumsum(np.random.randn(n) * 100) + 50000
        close = np.abs(close)
        df = pd.DataFrame({
            "open": close * (1 + np.random.randn(n) * 0.002),
            "high": close * (1 + np.abs(np.random.randn(n) * 0.005)),
            "low": close * (1 - np.abs(np.random.randn(n) * 0.005)),
            "close": close,
            "volume": np.random.uniform(100, 10000, n),
        })

    engine = PatternEngine()
    df = engine.compute_all(df)

    # Count detected patterns
    pattern_cols = [c for c in df.columns if c.startswith("pattern_") and
                    c not in ["pattern_signal", "pattern_confluence_score"]]

    print(f"\n  Total candles: {len(df)}")
    print(f"\n  Pattern Detection Summary:")
    for col in sorted(pattern_cols):
        count = int(df[col].sum())
        if count > 0:
            print(f"    {col}: {count} occurrences")

    # Signals
    signals = engine.get_all_signals(df)
    print(f"\n  Current Signals:")
    for name, sig in signals.items():
        print(f"    {name}: {sig:+.4f}")

    # Swing detection
    swing_highs = int(df["swing_high"].sum())
    swing_lows = int(df["swing_low"].sum())
    print(f"\n  Swing Points: {swing_highs} highs, {swing_lows} lows")
    print(f"  Current Swing Position: {df['swing_position'].iloc[-1]:.3f} "
          f"(0=low, 0.5=mid, 1=high)")

    # Pattern summary
    summary = engine.get_pattern_summary(df)
    print(f"\n  Patterns at Last Candle: {len(summary['patterns'])}")
    for p in summary["patterns"]:
        print(f"    → {p['pattern']}: {p['meaning']}")
    print(f"  Confluence Score: {summary['confluence_score']:+.4f}")
    print(f"  Final Signal: {summary['signal']:+.4f}")

    # ML features
    print(f"\n  ML Features:")
    for col in engine.ML_FEATURE_COLS:
        val = float(df[col].iloc[-1]) if col in df.columns else "MISSING"
        print(f"    {col}: {val}")

    print(f"\n  Total lines: {sum(1 for _ in open(__file__))}")
    print("=" * 70)
