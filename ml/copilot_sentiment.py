#!/usr/bin/env python3
"""
Sentiment analysis via Copilot API (free GPT-4.1).
Uses copilot-api on localhost:4141 as OpenAI-compatible endpoint.
Zero cost, no rate limit concerns.
"""
import os
import json
import time
import logging
import requests

log = logging.getLogger("ml.copilot_sentiment")

COPILOT_API_URL = os.environ.get("COPILOT_API_URL", "http://localhost:4141/v1")
MODEL = "gpt-4.1"

_cache = {}
_CACHE_TTL = 300  # 5 minutes


def _copilot_chat(system_prompt: str, user_prompt: str, max_tokens: int = 100) -> str:
    """Call copilot-api with a chat completion request."""
    try:
        resp = requests.post(
            f"{COPILOT_API_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.1,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        log.warning(f"Copilot API error {resp.status_code}: {resp.text[:100]}")
        return ""
    except Exception as e:
        log.debug(f"Copilot API unavailable: {e}")
        return ""


def analyze_sentiment(text: str) -> dict:
    """
    Analyze crypto sentiment of a text using GPT-4.1 via copilot-api.
    Returns: {"label": "POSITIVE/NEGATIVE/NEUTRAL", "score": 0.0-1.0, "signal": -1 to +1}
    """
    cache_key = f"sent:{text[:80]}"
    if cache_key in _cache:
        entry = _cache[cache_key]
        if time.time() - entry["time"] < _CACHE_TTL:
            return entry["value"]

    system = (
        "You are a crypto market sentiment analyzer. "
        "Respond with EXACTLY one JSON object, no other text. "
        'Format: {"label": "POSITIVE" or "NEGATIVE" or "NEUTRAL", "confidence": 0.0 to 1.0}'
    )

    result_text = _copilot_chat(system, f"Analyze sentiment: {text}")

    if not result_text:
        return {"label": "NEUTRAL", "score": 0.5, "signal": 0}

    try:
        # Parse JSON from response
        # Handle cases where GPT wraps in markdown
        clean = result_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)

        label = data.get("label", "NEUTRAL").upper()
        confidence = float(data.get("confidence", 0.5))

        signal = 0.0
        if "POS" in label:
            signal = confidence
        elif "NEG" in label:
            signal = -confidence

        output = {"label": label, "score": confidence, "signal": signal}
        _cache[cache_key] = {"value": output, "time": time.time()}
        return output

    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: keyword detection
        upper = result_text.upper()
        if "POSITIVE" in upper:
            output = {"label": "POSITIVE", "score": 0.7, "signal": 0.7}
        elif "NEGATIVE" in upper:
            output = {"label": "NEGATIVE", "score": 0.7, "signal": -0.7}
        else:
            output = {"label": "NEUTRAL", "score": 0.5, "signal": 0}
        _cache[cache_key] = {"value": output, "time": time.time()}
        return output


def analyze_batch(headlines: list) -> float:
    """
    Analyze multiple headlines and return combined signal (-1 to +1).
    """
    if not headlines:
        return 0.0

    signals = []
    for headline in headlines[:5]:  # Max 5 to be fast
        result = analyze_sentiment(headline)
        signals.append(result["signal"])

    if not signals:
        return 0.0
    return sum(signals) / len(signals)


def get_trading_opinion(symbol: str, price: float, rsi: float, trend: str, news_summary: str) -> dict:
    """
    Get a trading opinion from GPT-4.1 — acts like CryptoTrader-LM but free.
    Returns: {"decision": "BUY/SELL/HOLD", "confidence": 0-1, "signal": -1/0/+1, "reasoning": str}
    """
    cache_key = f"opinion:{symbol}:{int(price)}:{int(rsi)}"
    if cache_key in _cache:
        entry = _cache[cache_key]
        if time.time() - entry["time"] < _CACHE_TTL:
            return entry["value"]

    system = (
        "You are a crypto trading analyst. Based on the data, give a trading decision. "
        "Respond with EXACTLY one JSON object: "
        '{"decision": "BUY" or "SELL" or "HOLD", "confidence": 0.0 to 1.0, "reasoning": "one sentence"}'
    )

    user = (
        f"Symbol: {symbol}\n"
        f"Price: ${price:,.2f}\n"
        f"RSI(14): {rsi:.1f}\n"
        f"Trend: {trend}\n"
        f"Recent news: {news_summary[:200]}\n"
        f"Decision?"
    )

    result_text = _copilot_chat(system, user, max_tokens=150)

    if not result_text:
        return {"decision": "HOLD", "confidence": 0.5, "signal": 0, "reasoning": "API unavailable"}

    try:
        clean = result_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(clean)

        decision = data.get("decision", "HOLD").upper()
        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning", "")

        signal_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        signal = signal_map.get(decision, 0.0) * confidence

        output = {"decision": decision, "confidence": confidence, "signal": signal, "reasoning": reasoning}
        _cache[cache_key] = {"value": output, "time": time.time()}
        return output

    except (json.JSONDecodeError, KeyError):
        return {"decision": "HOLD", "confidence": 0.5, "signal": 0, "reasoning": "parse error"}


if __name__ == "__main__":
    print("Testing Copilot Sentiment (GPT-4.1 via copilot-api)...\n")

    headlines = [
        "Bitcoin surges past $70,000 as institutional demand grows",
        "Crypto market crashes amid regulatory crackdown fears",
        "Ethereum upgrade boosts network speed by 50%",
    ]

    for h in headlines:
        r = analyze_sentiment(h)
        print(f"  {h[:55]:55s} → {r['label']:10s} signal={r['signal']:+.2f}")

    print(f"\nBatch signal: {analyze_batch(headlines):+.3f}")

    print("\nTrading opinion:")
    opinion = get_trading_opinion("BTCUSDT", 66300, 32.0, "bearish EMA, strong ADX",
                                  "Bitcoin ETF sees record inflows despite market uncertainty")
    print(f"  {opinion}")
