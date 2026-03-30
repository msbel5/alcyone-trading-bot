#!/usr/bin/env python3
"""
Local CryptoBERT sentiment — runs on Pi, no API calls needed.
First run downloads the model (~500MB), then runs from cache.
"""
import time
import logging
from typing import Optional

log = logging.getLogger("ml.local_sentiment")

_pipeline = None
_cache = {}
_CACHE_TTL = 300


def _get_pipeline():
    """Lazy-load the sentiment pipeline. First call downloads the model."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from transformers import pipeline
        log.info("Loading CryptoBERT model (first time may download ~500MB)...")
        _pipeline = pipeline(
            "text-classification",
            model="ElKulako/cryptobert",
            device=-1,  # CPU
            top_k=None,  # Return all labels with scores
        )
        log.info("CryptoBERT loaded successfully")
        return _pipeline
    except Exception as e:
        log.error(f"Failed to load CryptoBERT: {e}")
        return None


def analyze_sentiment(text: str) -> dict:
    """
    Analyze crypto sentiment locally with CryptoBERT.
    Returns: {"label": str, "score": float, "signal": float}
    """
    cache_key = f"local:{text[:80]}"
    if cache_key in _cache:
        entry = _cache[cache_key]
        if time.time() - entry["time"] < _CACHE_TTL:
            return entry["value"]

    pipe = _get_pipeline()
    if pipe is None:
        return {"label": "NEUTRAL", "score": 0.5, "signal": 0}

    try:
        results = pipe(text[:512])  # CryptoBERT max 512 tokens
        if isinstance(results, list) and results:
            if isinstance(results[0], list):
                results = results[0]

            # Find best label
            best = max(results, key=lambda x: x.get("score", 0))
            label = best.get("label", "Neutral")
            score = float(best.get("score", 0.5))

            # CryptoBERT labels: Bearish, Neutral, Bullish
            signal = 0.0
            label_upper = label.upper()
            if "BULL" in label_upper or "POS" in label_upper:
                signal = score
            elif "BEAR" in label_upper or "NEG" in label_upper:
                signal = -score

            output = {"label": label, "score": score, "signal": signal}
            _cache[cache_key] = {"value": output, "time": time.time()}
            return output

    except Exception as e:
        log.warning(f"CryptoBERT inference error: {e}")

    return {"label": "Neutral", "score": 0.5, "signal": 0}


def analyze_batch(headlines: list) -> float:
    """Analyze multiple headlines, return combined signal (-1 to +1)."""
    if not headlines:
        return 0.0

    signals = []
    for headline in headlines[:5]:
        result = analyze_sentiment(headline)
        signals.append(result["signal"])

    return sum(signals) / len(signals) if signals else 0.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Local CryptoBERT on Pi...\n")

    headlines = [
        "Bitcoin surges past $70,000 as institutional demand grows",
        "Crypto market crashes amid regulatory crackdown fears",
        "Ethereum upgrade boosts network speed significantly",
        "Dogecoin whale moves 500M tokens to exchange",
        "SEC approves new Bitcoin ETF application",
    ]

    for h in headlines:
        start = time.time()
        r = analyze_sentiment(h)
        elapsed = time.time() - start
        print(f"  [{elapsed:.2f}s] {h[:50]:50s} → {r['label']:10s} signal={r['signal']:+.3f}")

    print(f"\nBatch signal: {analyze_batch(headlines):+.3f}")
