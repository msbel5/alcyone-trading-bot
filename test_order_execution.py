#!/usr/bin/env python3
"""
Test live trader order execution (DRY RUN).
"""

import sys
sys.path.append('/home/msbel/.openclaw/workspace/trading')

from live_trader import LiveTrader
import json

# Load config
with open('/home/msbel/.openclaw/workspace/trading/config/testnet_config.json') as f:
    config = json.load(f)

trader = LiveTrader(
    api_key=config['api_key'],
    private_key_path=config['private_key_path'],
    dry_run=True
)

print("="*70)
print("TESTING ORDER EXECUTION (DRY RUN)")
print("="*70)

# Test buy
price = trader.get_current_price()
print(f"\nCurrent price: ${price:,.2f}")
print("Executing test BUY...")
trader.execute_buy(price)

# Test sell
print("\nExecuting test SELL...")
trader.execute_sell(price, reason='TEST')

print("\n" + "="*70)
print("✅ Order execution test complete")
print("Check logs/trades_*.jsonl for logged events")
