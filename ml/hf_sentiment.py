#!/usr/bin/env python3
"""
HuggingFace Sentiment Models — CryptoBERT + CryptoTrader-LM
Uses HF Inference API (free tier: ~30 req/min).
Caches results to minimize API calls.
"""
import os
import time
import json
import logging
import requests
from pathlib import Path

log = logging.getLogger("ml.hf_sentiment")

HF_API_URL = "https://router.huggingface.co/hf-inference/models"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY", "")

# Model IDs
CRYPTOBERT_MODEL = "kk08/CryptoBERT"
CRYPTOTRADER_MODEL = "agarkovv/CryptoTrader-LM"

# Cache
_cache = {}
_CACHE_TTL = 300  # 5 minutes


def _hf_request(model_id: str, payload: dict, timeout: int = 30) -> dict:
    """Make a request to HuggingFace Inference API."""
    if not HF_TOKEN:
        log.warning("No HF_TOKEN set, skipping HF API call")
        return {"error": "No HF_TOKEN"}

    url = f"{HF_API_URL}/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 503:
            # Model loading — wait and retry once
            log.info(f"Model {model_id} loading, waiting 20s...")
            time.sleep(20)
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        log.warning(f"HF API error {resp.status_code}: {resp.text[:200]}")
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        log.warning(f"HF API request failed: {e}")
        return {"error": str(e)}


def _get_cached(key: str):
    """Get cached result if not expired."""
    if key in _cache:
        entry = _cache[key]
        if time.time() - entry["time"] < _CACHE_TTL:
            return entry["value"]
    return None


def _set_cache(key: str, value):
    """Cache a result."""
    _cache[key] = {"value": value, "time": time.time()}


def cryptobert_sentiment(text: str) -> dict:
    """
    Get crypto sentiment from CryptoBERT.
    Returns: {"label": "POSITIVE/NEGATIVE/NEUTRAL", "score": 0.0-1.0, "signal": -1/0/+1}
    """
    cache_key = f"cryptobert:{text[:100]}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    result = _hf_request(CRYPTOBERT_MODEL, {"inputs": text})

    if "error" in result:
        return {"label": "NEUTRAL", "score": 0.5, "signal": 0, "error": result["error"]}

    # CryptoBERT returns [[{"label": "POSITIVE", "score": 0.8}, ...]]
    try:
        predictions = result[0] if isinstance(result, list) and result else result
        if isinstance(predictions, list):
            best = max(predictions, key=lambda x: x.get("score", 0))
        else:
            best = predictions

        label = best.get("label", "NEUTRAL").upper()
        score = float(best.get("score", 0.5))

        signal = 0
        if "POS" in label:
            signal = score  # 0 to +1
        elif "NEG" in label:
            signal = -score  # -1 to 0

        output = {"label": label, "score": score, "signal": signal}
        _set_cache(cache_key, output)
        return output

    except Exception as e:
        log.warning(f"CryptoBERT parse error: {e}")
        return {"label": "NEUTRAL", "score": 0.5, "signal": 0}


def cryptotrader_decision(news_text: str, price_info: str) -> dict:
    """
    Get trading decision from CryptoTrader-LM.
    Returns: {"decision": "BUY/SELL/HOLD", "confidence": 0.0-1.0, "signal": -1/0/+1}
    """
    prompt = f"""Based on the following crypto market information, what is the best trading decision?

News: {news_text}
Price Info: {price_info}

Decision (BUY, SELL, or HOLD):"""

    cache_key = f"cryptotrader:{news_text[:50]}:{price_info[:50]}"
    cached = _get_cached(cache_key)
    if cached is not None:
        return cached

    result = _hf_request(CRYPTOTRADER_MODEL, {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 50, "temperature": 0.3}
    })

    if "error" in result:
        return {"decision": "HOLD", "confidence": 0.5, "signal": 0, "error": result["error"]}

    try:
        generated = ""
        if isinstance(result, list) and result:
            generated = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            generated = result.get("generated_text", str(result))

        generated_upper = generated.upper()
        if "BUY" in generated_upper:
            decision, signal = "BUY", 1.0
        elif "SELL" in generated_upper:
            decision, signal = "SELL", -1.0
        else:
            decision, signal = "HOLD", 0.0

        output = {"decision": decision, "confidence": 0.7, "signal": signal, "raw": generated[:200]}
        _set_cache(cache_key, output)
        return output

    except Exception as e:
        log.warning(f"CryptoTrader parse error: {e}")
        return {"decision": "HOLD", "confidence": 0.5, "signal": 0}


def get_combined_hf_signal(headlines: list, price_summary: str) -> float:
    """
    Combine CryptoBERT sentiment + CryptoTrader decision into single signal.
    Returns: -1.0 to +1.0
    """
    signals = []
    weights = []

    # CryptoBERT on each headline
    for headline in headlines[:5]:  # Max 5 to respect rate limits
        result = cryptobert_sentiment(headline)
        if "error" not in result:
            signals.append(result["signal"])
            weights.append(0.15)

    # CryptoTrader on combined news
    if headlines:
        combined_news = ". ".join(headlines[:3])
        trader_result = cryptotrader_decision(combined_news, price_summary)
        if "error" not in trader_result:
            signals.append(trader_result["signal"])
            weights.append(0.25)

    if not signals:
        return 0.0  # No data

    # Weighted average
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    return sum(s * w for s, w in zip(signals, weights)) / total_weight


# ── Test ──
if __name__ == "__main__":
    print(f"HF Token: {'SET' if HF_TOKEN else 'NOT SET'}")

    if HF_TOKEN:
        print("\nTesting CryptoBERT...")
        test_texts = [
            "Bitcoin surges past $70,000 as institutional demand grows",
            "Crypto market crashes amid regulatory crackdown fears",
            "Ethereum network upgrade completed successfully",
        ]
        for text in test_texts:
            result = cryptobert_sentiment(text)
            print(f"  '{text[:50]}...' → {result}")

        print("\nTesting CryptoTrader-LM...")
        result = cryptotrader_decision(
            "Bitcoin ETF sees record inflows, institutional adoption growing",
            "BTC at $66,000, up 3% in 24h, RSI=45"
        )
        print(f"  Decision: {result}")
    else:
        print("Set HF_TOKEN to test HuggingFace models")
