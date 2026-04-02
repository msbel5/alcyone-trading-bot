#!/usr/bin/env python3
"""
On-Chain Data v2 — MVRV Z-Score, NVT Ratio, Exchange Flow.
All computed from free APIs (no paid keys). Extends onchain.py.

Research basis:
- MVRV Z-Score: realized cap vs market cap → >3.7 overbought, <1 oversold
- NVT Ratio: market cap / tx volume → >95 overbought, <45 oversold
- Exchange inflow spike → selling pressure
"""
import time
import logging
import numpy as np
import requests
from typing import Dict, Optional

log = logging.getLogger("ml.onchain_v2")

_cache: Dict[str, dict] = {}
_CACHE_TTL = 3600  # 1 hour (on-chain metrics are slower)


def _cached(key: str) -> Optional[dict]:
    if key in _cache and time.time() - _cache[key]["time"] < _CACHE_TTL:
        return _cache[key]["value"]
    return None


def _set_cache(key: str, value: dict):
    _cache[key] = {"value": value, "time": time.time()}


# ── MVRV Z-Score (approximated from price data) ───────────────────

def compute_mvrv_zscore(prices: np.ndarray, window: int = 365 * 24) -> float:
    """
    Approximate MVRV Z-Score from price data.

    Real MVRV = Market Cap / Realized Cap, but Realized Cap needs UTXO data.
    Approximation: Use rolling mean as proxy for "realized value" (cost basis).

    Z-Score = (Market Price - Realized Price) / StdDev(Market Price)
    - Z > 3.7 → extremely overbought (sell signal)
    - Z > 2.0 → overbought
    - Z < 1.0 → oversold (buy signal)
    - Z < 0 → deep undervalue (strong buy)
    """
    if len(prices) < 100:
        return 0.0

    # Use rolling mean as "realized price" proxy (avg acquisition cost)
    lookback = min(window, len(prices))
    realized_price = np.mean(prices[-lookback:])
    current_price = prices[-1]
    std_price = np.std(prices[-lookback:])

    if std_price == 0:
        return 0.0

    z_score = (current_price - realized_price) / std_price
    return float(np.clip(z_score, -4.0, 6.0))


def mvrv_signal(z_score: float) -> float:
    """Convert MVRV Z-Score to trading signal [-1, +1]."""
    if z_score > 3.7:
        return -1.0   # Extreme overbought — strong sell
    elif z_score > 2.0:
        return -0.5   # Overbought — moderate sell
    elif z_score > 0.5:
        return 0.0    # Neutral
    elif z_score > -0.5:
        return 0.3    # Slightly undervalued — mild buy
    elif z_score > -1.5:
        return 0.7    # Undervalued — buy
    else:
        return 1.0    # Deep undervalue — strong buy


# ── NVT Ratio (Network Value to Transactions) ─────────────────────

def compute_nvt_ratio(market_cap: float, tx_volume_24h: float) -> float:
    """
    NVT Ratio = Market Cap / Daily Transaction Volume.
    - NVT > 95: overvalued (sell)
    - NVT 45-95: fair value
    - NVT < 45: undervalued (buy)

    Uses blockchain.info for BTC tx volume.
    """
    if tx_volume_24h <= 0:
        return 50.0  # Neutral default
    return market_cap / tx_volume_24h


def nvt_signal(nvt_ratio: float) -> float:
    """Convert NVT Ratio to trading signal."""
    if nvt_ratio > 120:
        return -1.0   # Extremely overvalued
    elif nvt_ratio > 95:
        return -0.5   # Overvalued
    elif nvt_ratio > 45:
        return 0.0    # Fair value
    elif nvt_ratio > 20:
        return 0.5    # Undervalued
    else:
        return 1.0    # Very undervalued


def fetch_btc_nvt() -> Dict:
    """Fetch BTC NVT components from free APIs."""
    cached = _cached("btc_nvt")
    if cached:
        return cached

    try:
        # BTC market cap from blockchain.info
        resp_mc = requests.get("https://blockchain.info/q/marketcap", timeout=10)
        market_cap = float(resp_mc.text.strip())

        # BTC 24h transaction volume in USD
        resp_tv = requests.get("https://blockchain.info/q/24hrtransactioncount", timeout=10)
        tx_count = int(resp_tv.text.strip())

        resp_sent = requests.get("https://blockchain.info/q/24hrbtcsent", timeout=10)
        btc_sent = float(resp_sent.text.strip()) / 1e8  # satoshi → BTC

        # Estimate USD volume (using rough price)
        resp_price = requests.get("https://blockchain.info/q/24hrprice", timeout=10)
        btc_price = float(resp_price.text.strip())

        tx_volume_usd = btc_sent * btc_price

        nvt = compute_nvt_ratio(market_cap, tx_volume_usd)
        sig = nvt_signal(nvt)

        result = {
            "nvt_ratio": nvt,
            "market_cap": market_cap,
            "tx_volume_usd": tx_volume_usd,
            "tx_count_24h": tx_count,
            "signal": sig,
        }
        _set_cache("btc_nvt", result)
        return result
    except Exception as e:
        log.debug(f"BTC NVT fetch failed: {e}")
        return {"nvt_ratio": 50, "signal": 0, "error": str(e)}


# ── Exchange Flow (inflow/outflow signals) ─────────────────────────

def fetch_exchange_flow() -> Dict:
    """
    Approximate exchange flow from mempool data.
    High fee environment + high tx count = potential exchange deposits (selling).
    Low fees + low count = holding/accumulation.
    """
    cached = _cached("exchange_flow")
    if cached:
        return cached

    try:
        # Mempool fees
        resp_fees = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=10)
        fees = resp_fees.json()
        fastest_fee = fees.get("fastestFee", 10)

        # Mempool count
        resp_mem = requests.get("https://mempool.space/api/mempool", timeout=10)
        mem = resp_mem.json()
        tx_count = mem.get("count", 0)
        vsize = mem.get("vsize", 0)

        # Signal logic:
        # High fees + high count = exchange deposits = selling pressure
        # Low fees + low count = accumulation = buying opportunity
        fee_level = 1.0 if fastest_fee > 80 else (0.5 if fastest_fee > 30 else 0.0)
        count_level = 1.0 if tx_count > 60000 else (0.5 if tx_count > 30000 else 0.0)

        # Exchange inflow proxy: high = selling pressure (negative signal)
        inflow_signal = -(fee_level * 0.6 + count_level * 0.4)

        result = {
            "fastest_fee": fastest_fee,
            "mempool_count": tx_count,
            "mempool_vsize": vsize,
            "inflow_signal": round(inflow_signal, 4),
        }
        _set_cache("exchange_flow", result)
        return result
    except Exception as e:
        log.debug(f"Exchange flow fetch failed: {e}")
        return {"inflow_signal": 0, "error": str(e)}


# ── Compute On-Chain Features for ML ───────────────────────────────

def compute_onchain_features(prices: np.ndarray) -> Dict[str, float]:
    """
    Compute all on-chain features for ML feature engineering.
    Returns dict of feature names → values, suitable for adding to DataFrame.
    """
    # MVRV Z-Score
    mvrv_z = compute_mvrv_zscore(prices)
    mvrv_sig = mvrv_signal(mvrv_z)

    # NVT
    nvt_data = fetch_btc_nvt()
    nvt_sig = nvt_data.get("signal", 0)

    # Exchange flow
    flow_data = fetch_exchange_flow()
    flow_sig = flow_data.get("inflow_signal", 0)

    return {
        "mvrv_zscore": mvrv_z,
        "mvrv_signal": mvrv_sig,
        "nvt_ratio": nvt_data.get("nvt_ratio", 50),
        "nvt_signal": nvt_sig,
        "exchange_flow_signal": flow_sig,
    }


# ── Combined V2 On-Chain Signal ────────────────────────────────────

def get_onchain_signal_v2(symbol: str = "BTCUSDT", prices: Optional[np.ndarray] = None) -> Dict:
    """
    Enhanced on-chain signal combining MVRV + NVT + Exchange Flow + mempool.

    Weights (research-based):
    - MVRV Z-Score: 20% (strongest single on-chain predictor)
    - NVT Ratio: 15%
    - Exchange Flow: 10%
    - Funding Rate: 20% (crowding indicator, high alpha)
    - Long/Short Ratio: 15% (liquidation proxy)
    - Mempool Activity: 10%
    - Fee Signal: 10%
    """
    cached = _cached(f"onchain_v2_{symbol}")
    if cached:
        return cached

    signals = {}
    weights = {}

    # MVRV (needs price history)
    if prices is not None and len(prices) > 100:
        mvrv_z = compute_mvrv_zscore(prices)
        signals["mvrv"] = mvrv_signal(mvrv_z)
        weights["mvrv"] = 0.20

    # NVT
    nvt_data = fetch_btc_nvt()
    signals["nvt"] = nvt_data.get("signal", 0)
    weights["nvt"] = 0.15

    # Exchange flow
    flow = fetch_exchange_flow()
    signals["exchange_flow"] = flow.get("inflow_signal", 0)
    weights["exchange_flow"] = 0.10

    # Funding rate (FREE from Binance Futures API)
    funding = fetch_funding_rate(symbol)
    signals["funding_rate"] = funding.get("signal", 0)
    weights["funding_rate"] = 0.20

    # Long/Short ratio + liquidation proxy
    liq = fetch_liquidation_proxy(symbol)
    signals["long_short"] = liq.get("signal", 0)
    weights["long_short"] = 0.15

    # Mempool activity (from original onchain.py)
    try:
        from ml.onchain import btc_mempool_stats, mempool_fees

        mempool = btc_mempool_stats()
        signals["mempool"] = mempool.get("activity_signal", 0)
        weights["mempool"] = 0.10

        fees = mempool_fees()
        signals["fees"] = fees.get("fee_signal", 0)
        weights["fees"] = 0.10
    except Exception:
        pass

    # Weighted average
    if weights:
        total_weight = sum(weights.values())
        combined = sum(signals[k] * weights[k] for k in signals) / total_weight
    else:
        combined = 0.0

    result = {
        "signal": round(float(np.clip(combined, -1, 1)), 4),
        "signals": signals,
        "weights": weights,
    }
    _set_cache(f"onchain_v2_{symbol}", result)
    return result


# ── Funding Rate (FREE from Binance API) ───────────────────────────

def fetch_funding_rate(symbol: str = "BTCUSDT") -> Dict:
    """
    Fetch current funding rate from Binance Futures.
    Positive rate = longs pay shorts = crowded longs = bearish signal.
    Negative rate = shorts pay longs = crowded shorts = bullish signal.
    Extreme (>0.1%) = strong contrarian signal.
    """
    cached = _cached(f"funding_{symbol}")
    if cached:
        return cached

    try:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data:
            rate = float(data[0]["fundingRate"])
            # Signal: contrarian to funding rate
            if rate > 0.001:      # >0.1% — extreme long crowding
                signal = -0.8
            elif rate > 0.0005:   # >0.05% — moderate long crowding
                signal = -0.4
            elif rate < -0.001:   # <-0.1% — extreme short crowding
                signal = 0.8
            elif rate < -0.0005:  # <-0.05% — moderate short crowding
                signal = 0.4
            else:
                signal = 0.0

            result = {"funding_rate": rate, "signal": signal}
            _set_cache(f"funding_{symbol}", result)
            return result
    except Exception as e:
        log.debug(f"Funding rate fetch failed: {e}")

    return {"funding_rate": 0, "signal": 0}


def funding_rate_signal(symbol: str = "BTCUSDT") -> float:
    """Get funding rate as trading signal [-1, +1]."""
    return fetch_funding_rate(symbol).get("signal", 0)


# ── Liquidation Data (FREE from CoinGlass alternative) ─────────────

def fetch_liquidation_proxy(symbol: str = "BTCUSDT") -> Dict:
    """
    Approximate liquidation pressure from open interest changes.
    Large OI drops = liquidation cascades = volatility signal.
    Uses Binance Futures API (free, no key needed).
    """
    cached = _cached(f"liq_{symbol}")
    if cached:
        return cached

    try:
        url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        oi = float(data.get("openInterest", 0))

        # Also get long/short ratio
        url2 = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=2"
        resp2 = requests.get(url2, timeout=10)
        ls_data = resp2.json()

        if ls_data and len(ls_data) >= 2:
            current_ratio = float(ls_data[0].get("longShortRatio", 1))
            prev_ratio = float(ls_data[1].get("longShortRatio", 1))
            ratio_change = current_ratio - prev_ratio

            # Signal: extreme long/short ratio = contrarian
            if current_ratio > 2.0:    # Very long-heavy
                signal = -0.6
            elif current_ratio > 1.5:
                signal = -0.3
            elif current_ratio < 0.5:  # Very short-heavy
                signal = 0.6
            elif current_ratio < 0.7:
                signal = 0.3
            else:
                signal = 0.0
        else:
            current_ratio = 1.0
            signal = 0.0

        result = {
            "open_interest": oi,
            "long_short_ratio": current_ratio,
            "signal": signal,
        }
        _set_cache(f"liq_{symbol}", result)
        return result
    except Exception as e:
        log.debug(f"Liquidation proxy fetch failed: {e}")

    return {"open_interest": 0, "long_short_ratio": 1, "signal": 0}


# ── Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing On-Chain v2...\n")

    # Fake price data for testing
    fake_prices = np.random.normal(60000, 5000, 8760)  # 1 year of hourly
    fake_prices = np.cumsum(np.random.randn(8760) * 100) + 60000
    fake_prices = np.abs(fake_prices)

    z = compute_mvrv_zscore(fake_prices)
    print(f"MVRV Z-Score: {z:.3f} → signal: {mvrv_signal(z):+.2f}")

    nvt = fetch_btc_nvt()
    print(f"NVT Ratio: {nvt.get('nvt_ratio', 0):.1f} → signal: {nvt.get('signal', 0):+.2f}")

    flow = fetch_exchange_flow()
    print(f"Exchange Flow: signal={flow.get('inflow_signal', 0):+.4f}")

    combined = get_onchain_signal_v2("BTCUSDT", fake_prices)
    print(f"\nCombined V2 Signal: {combined['signal']:+.4f}")
    print(f"Components: {combined['signals']}")
