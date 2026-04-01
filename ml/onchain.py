#!/usr/bin/env python3
"""
On-Chain Data — free APIs for exchange flow, active addresses, mempool.
Sources: blockchain.info, mempool.space, Glassnode free tier.
No paid API keys needed.
"""
import time
import logging
import requests
from typing import Dict, Optional

log = logging.getLogger("ml.onchain")

_cache: Dict[str, dict] = {}
_CACHE_TTL = 1800  # 30 minutes


def _cached(key: str) -> Optional[dict]:
    if key in _cache and time.time() - _cache[key]["time"] < _CACHE_TTL:
        return _cache[key]["value"]
    return None


def _set_cache(key: str, value: dict):
    _cache[key] = {"value": value, "time": time.time()}


# ── Blockchain.info (BTC, free, no key) ─────────────────────────────

def btc_mempool_stats() -> Dict:
    """Get BTC mempool stats — pending transactions indicate network activity."""
    cached = _cached("btc_mempool")
    if cached:
        return cached

    try:
        resp = requests.get("https://blockchain.info/q/unconfirmedcount", timeout=10)
        unconfirmed = int(resp.text.strip())

        resp2 = requests.get("https://blockchain.info/q/24hrbtcsent", timeout=10)
        btc_sent_24h = float(resp2.text.strip()) / 1e8  # satoshi to BTC

        result = {
            "unconfirmed_tx": unconfirmed,
            "btc_sent_24h": btc_sent_24h,
            # High unconfirmed = high activity = potential volatility
            "activity_signal": 1.0 if unconfirmed > 100000 else (0.5 if unconfirmed > 50000 else 0.0),
        }
        _set_cache("btc_mempool", result)
        return result
    except Exception as e:
        log.debug(f"blockchain.info failed: {e}")
        return {"unconfirmed_tx": 0, "btc_sent_24h": 0, "activity_signal": 0}


def btc_hashrate() -> Dict:
    """Get BTC hashrate — declining hashrate can signal miner stress."""
    cached = _cached("btc_hashrate")
    if cached:
        return cached

    try:
        resp = requests.get("https://blockchain.info/q/hashrate", timeout=10)
        hashrate = float(resp.text.strip())

        result = {
            "hashrate_gh": hashrate,
            # We'd need historical to compare, so just report raw
            "signal": 0,
        }
        _set_cache("btc_hashrate", result)
        return result
    except Exception as e:
        log.debug(f"Hashrate fetch failed: {e}")
        return {"hashrate_gh": 0, "signal": 0}


# ── Mempool.space (BTC, free, no key) ───────────────────────────────

def mempool_fees() -> Dict:
    """Get current BTC fee estimates — high fees = high demand."""
    cached = _cached("mempool_fees")
    if cached:
        return cached

    try:
        resp = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=10)
        data = resp.json()

        fastest = data.get("fastestFee", 0)
        half_hour = data.get("halfHourFee", 0)
        hour = data.get("hourFee", 0)
        economy = data.get("economyFee", 0)

        # High fees = high demand = potential price movement
        fee_signal = 0.0
        if fastest > 100:
            fee_signal = 0.8  # Very high demand
        elif fastest > 50:
            fee_signal = 0.4
        elif fastest < 5:
            fee_signal = -0.3  # Very low demand

        result = {
            "fastest_fee": fastest,
            "half_hour_fee": half_hour,
            "hour_fee": hour,
            "economy_fee": economy,
            "fee_signal": fee_signal,
        }
        _set_cache("mempool_fees", result)
        return result
    except Exception as e:
        log.debug(f"Mempool fees failed: {e}")
        return {"fastest_fee": 0, "fee_signal": 0}


def mempool_blocks() -> Dict:
    """Get mempool block info — large mempool = congestion."""
    cached = _cached("mempool_blocks")
    if cached:
        return cached

    try:
        resp = requests.get("https://mempool.space/api/mempool", timeout=10)
        data = resp.json()

        count = data.get("count", 0)
        vsize = data.get("vsize", 0)

        result = {
            "tx_count": count,
            "vsize_bytes": vsize,
            "congestion_signal": 1.0 if count > 50000 else (0.5 if count > 20000 else 0.0),
        }
        _set_cache("mempool_blocks", result)
        return result
    except Exception as e:
        log.debug(f"Mempool blocks failed: {e}")
        return {"tx_count": 0, "congestion_signal": 0}


# ── Glassnode Free Tier (requires free API key) ─────────────────────

GLASSNODE_API_KEY = ""  # Set if you have one, otherwise skip

def glassnode_active_addresses(asset: str = "BTC") -> Dict:
    """Get active addresses from Glassnode free tier (daily resolution)."""
    if not GLASSNODE_API_KEY:
        return {"active_addresses": 0, "signal": 0, "source": "unavailable"}

    cached = _cached(f"glassnode_aa_{asset}")
    if cached:
        return cached

    try:
        resp = requests.get(
            "https://api.glassnode.com/v1/metrics/addresses/active_count",
            params={"a": asset, "api_key": GLASSNODE_API_KEY, "i": "24h"},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            if data:
                latest = data[-1]
                value = latest.get("v", 0)
                # Compare to average
                avg = sum(d.get("v", 0) for d in data[-7:]) / max(len(data[-7:]), 1)
                signal = 0.5 if value > avg * 1.1 else (-0.3 if value < avg * 0.9 else 0.0)

                result = {"active_addresses": value, "signal": signal, "source": "glassnode"}
                _set_cache(f"glassnode_aa_{asset}", result)
                return result
        return {"active_addresses": 0, "signal": 0, "source": "error"}
    except Exception as e:
        log.debug(f"Glassnode failed: {e}")
        return {"active_addresses": 0, "signal": 0, "source": "error"}


# ── Combined On-Chain Signal ────────────────────────────────────────

def get_onchain_signal(symbol: str = "BTCUSDT") -> Dict:
    """
    Get combined on-chain signal from all free sources.
    Returns: {"signal": -1 to +1, "sources": {...}}
    """
    cached = _cached(f"onchain_{symbol}")
    if cached:
        return cached

    signals = []
    sources = {}

    # BTC-specific on-chain (works for all coins as BTC leads market)
    mempool = btc_mempool_stats()
    sources["mempool_activity"] = mempool.get("activity_signal", 0)
    signals.append(mempool.get("activity_signal", 0) * 0.3)

    fees = mempool_fees()
    sources["fee_signal"] = fees.get("fee_signal", 0)
    signals.append(fees.get("fee_signal", 0) * 0.3)

    blocks = mempool_blocks()
    sources["congestion"] = blocks.get("congestion_signal", 0)
    signals.append(blocks.get("congestion_signal", 0) * 0.2)

    hashrate = btc_hashrate()
    sources["hashrate"] = hashrate.get("signal", 0)
    signals.append(hashrate.get("signal", 0) * 0.2)

    # Glassnode if available
    if GLASSNODE_API_KEY:
        asset = symbol.replace("USDT", "")
        if asset in ["BTC", "ETH"]:
            aa = glassnode_active_addresses(asset)
            sources["active_addresses"] = aa.get("signal", 0)
            signals.append(aa.get("signal", 0) * 0.3)

    combined = sum(signals) / max(len(signals), 1)
    result = {
        "signal": round(max(-1, min(1, combined)), 4),
        "sources": sources,
    }
    _set_cache(f"onchain_{symbol}", result)
    return result


# ── Test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing on-chain data sources...\n")

    m = btc_mempool_stats()
    print(f"Mempool: {m['unconfirmed_tx']} unconfirmed, {m['btc_sent_24h']:.0f} BTC/24h, signal={m['activity_signal']}")

    f = mempool_fees()
    print(f"Fees: fastest={f['fastest_fee']} sat/vB, signal={f['fee_signal']}")

    b = mempool_blocks()
    print(f"Blocks: {b['tx_count']} pending tx, congestion={b['congestion_signal']}")

    h = btc_hashrate()
    print(f"Hashrate: {h['hashrate_gh']:.0f} GH/s")

    combined = get_onchain_signal("BTCUSDT")
    print(f"\nCombined signal: {combined['signal']:+.4f}")
    print(f"Sources: {combined['sources']}")
