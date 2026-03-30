#!/usr/bin/env python3
"""
Download historical OHLCV data from Binance public API.
No auth needed — uses mainnet public endpoints.
Downloads 6+ months of 1h candles for all trading pairs.
"""
import os
import sys
import time
import json
import csv
import requests
from datetime import datetime, timedelta
from pathlib import Path

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]
INTERVAL = "1h"
MONTHS_BACK = 8  # 8 months of data
OUTPUT_DIR = Path("/home/msbel/.openclaw/workspace/trading/data/historical")
BASE_URL = "https://api.binance.com"  # Mainnet public (no auth needed for klines)

COLUMNS = ["open_time", "open", "high", "low", "close", "volume",
           "close_time", "quote_volume", "trades", "taker_buy_base",
           "taker_buy_quote", "ignore"]


def download_klines(symbol, interval, start_ms, end_ms, limit=1000):
    """Download klines from Binance public API."""
    url = f"{BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_symbol(symbol):
    """Download full history for one symbol."""
    output_file = OUTPUT_DIR / f"{symbol.lower()}_{INTERVAL}.csv"

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=MONTHS_BACK * 30)

    print(f"  {symbol}: {start_time.date()} → {end_time.date()}")

    all_klines = []
    current_start = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    while current_start < end_ms:
        try:
            batch = download_klines(symbol, INTERVAL, current_start, end_ms)
        except Exception as e:
            print(f"    Error at {current_start}: {e}")
            time.sleep(5)
            continue

        if not batch:
            break

        all_klines.extend(batch)
        # Move start to after last candle
        current_start = int(batch[-1][0]) + 1
        print(f"    Downloaded {len(all_klines)} candles so far...")
        time.sleep(0.5)  # Rate limit respect

    # Write CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for k in all_klines:
            writer.writerow(k)

    print(f"  {symbol}: {len(all_klines)} candles saved to {output_file}")
    return len(all_klines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MONTHS_BACK} months of {INTERVAL} data for {len(SYMBOLS)} symbols...")
    print(f"Output: {OUTPUT_DIR}\n")

    total = 0
    for symbol in SYMBOLS:
        count = download_symbol(symbol)
        total += count
        time.sleep(1)  # Pause between symbols

    print(f"\nDone! Total: {total} candles across {len(SYMBOLS)} symbols")


if __name__ == "__main__":
    main()
