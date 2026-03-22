#!/usr/bin/env python3
"""
Market Data Fetcher for Binance Testnet
Fetch historical and real-time candlestick data
"""
import sys
import json
import csv
import time
from pathlib import Path
from datetime import datetime

# Add binance_testnet to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'binance_testnet'))
from adapter_binance import BinanceTestnetAdapter


def load_config():
    """Load API credentials"""
    config_path = Path(__file__).parent.parent / 'config' / 'testnet_config.json'
    with open(config_path) as f:
        config = json.load(f)
    
    private_key_path = Path(config['private_key_path'])
    if not private_key_path.is_absolute():
        private_key_path = Path(__file__).parent.parent / private_key_path
    config['private_key_path'] = str(private_key_path)
    
    return config


def kline_to_dict(kline):
    """Convert kline array to dict"""
    return {
        'open_time': kline[0],
        'open': float(kline[1]),
        'high': float(kline[2]),
        'low': float(kline[3]),
        'close': float(kline[4]),
        'volume': float(kline[5]),
        'close_time': kline[6],
        'quote_volume': float(kline[7]),
        'trades': int(kline[8]),
        'taker_buy_base': float(kline[9]),
        'taker_buy_quote': float(kline[10]),
    }


def fetch_historical(symbol='BTCUSDT', interval='1h', limit=1000, output_file=None):
    """
    Fetch historical kline data
    
    Args:
        symbol: Trading pair (default: BTCUSDT)
        interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
        limit: Number of candles (max 1000)
        output_file: CSV output path (optional)
    
    Returns:
        List of kline dicts
    """
    print(f"📊 Fetching {limit} candles for {symbol} ({interval})...")
    
    config = load_config()
    client = BinanceTestnetAdapter(
        api_key=config['api_key'],
        private_key_path=config['private_key_path'],
        base_url=config.get('base_url', 'https://testnet.binance.vision')
    )
    
    # Fetch klines
    raw_klines = client.get_klines(symbol, interval, limit)
    
    if isinstance(raw_klines, dict) and 'error' in raw_klines:
        print(f"❌ Error: {raw_klines['error']}")
        return []
    
    klines = [kline_to_dict(k) for k in raw_klines]
    print(f"✅ Fetched {len(klines)} candles")
    
    # Save to CSV if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=klines[0].keys())
            writer.writeheader()
            writer.writerows(klines)
        
        print(f"💾 Saved to {output_path}")
        
        # Print summary stats
        closes = [k['close'] for k in klines]
        print(f"\n📈 Summary:")
        print(f"   First candle: {datetime.fromtimestamp(klines[0]['open_time']/1000)}")
        print(f"   Last candle:  {datetime.fromtimestamp(klines[-1]['open_time']/1000)}")
        print(f"   Price range:  ${min(closes):.2f} - ${max(closes):.2f}")
        print(f"   Last close:   ${closes[-1]:.2f}")
    
    return klines


def fetch_recent(symbol='BTCUSDT', interval='1h', limit=100):
    """
    Fetch recent kline data (for real-time updates)
    
    Args:
        symbol: Trading pair
        interval: Timeframe
        limit: Number of recent candles
    
    Returns:
        List of kline dicts
    """
    config = load_config()
    client = BinanceTestnetAdapter(
        api_key=config['api_key'],
        private_key_path=config['private_key_path'],
        base_url=config.get('base_url', 'https://testnet.binance.vision')
    )
    
    raw_klines = client.get_klines(symbol, interval, limit)
    
    if isinstance(raw_klines, dict) and 'error' in raw_klines:
        print(f"❌ Error: {raw_klines['error']}")
        return []
    
    return [kline_to_dict(k) for k in raw_klines]


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch Binance Testnet market data')
    parser.add_argument('symbol', nargs='?', default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    parser.add_argument('interval', nargs='?', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('limit', nargs='?', type=int, default=1000, help='Number of candles (default: 1000)')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Default output path if not specified
    if not args.output:
        data_dir = Path(__file__).parent.parent / 'data'
        args.output = data_dir / f"{args.symbol.lower()}_{args.interval}.csv"
    
    fetch_historical(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        output_file=args.output
    )
