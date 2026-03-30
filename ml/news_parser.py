#!/usr/bin/env python3
"""
Crypto News RSS Parser — fetches headlines from major crypto news sources.
Used by CryptoBERT and CryptoTrader-LM for sentiment analysis.
"""
import time
import logging
import hashlib
from typing import List, Dict, Optional

log = logging.getLogger("ml.news")

RSS_FEEDS = [
    {"name": "CoinDesk", "url": "https://www.coindesk.com/arc/outboundfeeds/rss/"},
    {"name": "CoinTelegraph", "url": "https://cointelegraph.com/rss"},
    {"name": "CryptoSlate", "url": "https://cryptoslate.com/feed/"},
]

# Coin keywords for filtering
COIN_KEYWORDS = {
    "BTCUSDT": ["bitcoin", "btc", "crypto market"],
    "ETHUSDT": ["ethereum", "eth", "defi"],
    "SOLUSDT": ["solana", "sol"],
    "BNBUSDT": ["binance", "bnb"],
    "XRPUSDT": ["xrp", "ripple"],
    "DOGEUSDT": ["dogecoin", "doge", "meme coin"],
    "AVAXUSDT": ["avalanche", "avax"],
}

# Cache
_headline_cache: Dict[str, dict] = {}
_CACHE_TTL = 900  # 15 minutes


def fetch_headlines(max_per_feed: int = 10) -> List[dict]:
    """Fetch latest headlines from all RSS feeds."""
    cache_key = "all_headlines"
    if cache_key in _headline_cache:
        entry = _headline_cache[cache_key]
        if time.time() - entry["time"] < _CACHE_TTL:
            return entry["value"]

    try:
        import feedparser
    except ImportError:
        log.warning("feedparser not installed, skipping RSS")
        return []

    all_headlines = []
    for feed_info in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"])
            for entry in feed.entries[:max_per_feed]:
                headline = {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "source": feed_info["name"],
                    "published": entry.get("published", ""),
                    "summary": entry.get("summary", "")[:200],
                }
                all_headlines.append(headline)
        except Exception as e:
            log.warning(f"RSS fetch failed for {feed_info['name']}: {e}")

    _headline_cache[cache_key] = {"value": all_headlines, "time": time.time()}
    log.info(f"Fetched {len(all_headlines)} headlines from {len(RSS_FEEDS)} feeds")
    return all_headlines


def filter_headlines_for_coin(headlines: List[dict], symbol: str) -> List[str]:
    """Filter headlines relevant to a specific coin."""
    keywords = COIN_KEYWORDS.get(symbol, [])
    if not keywords:
        return [h["title"] for h in headlines[:3]]  # Generic top 3

    relevant = []
    for h in headlines:
        text = (h.get("title", "") + " " + h.get("summary", "")).lower()
        if any(kw in text for kw in keywords):
            relevant.append(h["title"])

    # Always include general crypto headlines too
    if len(relevant) < 3:
        general = [h["title"] for h in headlines if "crypto" in h.get("title", "").lower()
                    or "bitcoin" in h.get("title", "").lower()]
        relevant.extend(general[:3 - len(relevant)])

    return relevant[:5]


def get_headlines_for_symbol(symbol: str) -> List[str]:
    """Get relevant headlines for a trading symbol."""
    all_headlines = fetch_headlines()
    return filter_headlines_for_coin(all_headlines, symbol)


if __name__ == "__main__":
    print("Fetching crypto news headlines...\n")
    headlines = fetch_headlines()
    print(f"Total: {len(headlines)} headlines\n")

    for sym in ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]:
        filtered = filter_headlines_for_coin(headlines, sym)
        coin = sym.replace("USDT", "")
        print(f"{coin}: {len(filtered)} relevant headlines")
        for h in filtered[:3]:
            print(f"  - {h[:80]}")
        print()
