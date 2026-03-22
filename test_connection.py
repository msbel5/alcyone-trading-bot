#!/usr/bin/env python3
"""
Test Binance Testnet Connection (Ed25519)
Run: python test_connection.py
"""
import sys
import json
from pathlib import Path

# Add binance_testnet to path
sys.path.insert(0, str(Path(__file__).parent / 'binance_testnet'))
from adapter_binance import BinanceTestnetAdapter


def load_config():
    """Load API credentials from config file"""
    config_path = Path(__file__).parent / 'config' / 'testnet_config.json'
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("Please create config/testnet_config.json with your API keys")
        sys.exit(1)
    
    with open(config_path) as f:
        config = json.load(f)
    
    if 'api_key' not in config or config['api_key'] == 'YOUR_TESTNET_API_KEY_HERE':
        print("❌ API key not configured")
        print("Please edit config/testnet_config.json with your Binance Testnet API key")
        sys.exit(1)
    
    # Check for Ed25519 private key path
    if 'private_key_path' not in config:
        print("❌ private_key_path not configured")
        print("Please add 'private_key_path' to config/testnet_config.json")
        sys.exit(1)
    
    private_key_path = Path(config['private_key_path'])
    if not private_key_path.is_absolute():
        private_key_path = Path(__file__).parent / private_key_path
    
    if not private_key_path.exists():
        print(f"❌ Private key not found: {private_key_path}")
        sys.exit(1)
    
    config['private_key_path'] = str(private_key_path)
    return config


def test_connection():
    """Test Binance Testnet API connection"""
    print("🔌 Testing Binance Testnet Connection (Ed25519)...\n")
    
    # Load config
    config = load_config()
    client = BinanceTestnetAdapter(
        api_key=config['api_key'],
        private_key_path=config['private_key_path'],
        base_url=config.get('base_url', 'https://testnet.binance.vision')
    )
    
    # Test 1: Ping
    print("1️⃣ Testing connectivity (ping)...")
    result = client.ping()
    if 'error' in result:
        print(f"   ❌ Ping failed: {result['error']}")
        return False
    print("   ✅ Ping successful\n")
    
    # Test 2: Server time
    print("2️⃣ Getting server time...")
    result = client.server_time()
    if 'error' in result:
        print(f"   ❌ Failed: {result['error']}")
        return False
    print(f"   ✅ Server time: {result.get('serverTime', 'N/A')}\n")
    
    # Test 3: Account info
    print("3️⃣ Getting account info (Ed25519 signed)...")
    result = client.account_info()
    if 'error' in result:
        print(f"   ❌ Failed: {result['error']}")
        print("   Check your API key and Ed25519 private key")
        return False
    print(f"   ✅ Account type: {result.get('accountType', 'N/A')}")
    print(f"   ✅ Can trade: {result.get('canTrade', False)}")
    print(f"   ✅ Can withdraw: {result.get('canWithdraw', False)}\n")
    
    # Test 4: Account balance
    print("4️⃣ Getting account balance...")
    balances = client.account_balance()
    if not balances:
        print("   ⚠️  No balance found (this is normal for new testnet accounts)")
    else:
        print("   ✅ Balances:")
        for bal in balances:
            free = float(bal['free'])
            locked = float(bal['locked'])
            if free > 0 or locked > 0:
                print(f"      {bal['asset']}: {free:.8f} (free) + {locked:.8f} (locked)")
    print()
    
    # Test 5: Get BTCUSDT price
    print("5️⃣ Getting BTCUSDT price...")
    result = client.get_symbol_price('BTCUSDT')
    if 'error' in result:
        print(f"   ❌ Failed: {result['error']}")
        return False
    print(f"   ✅ BTCUSDT price: ${result.get('price', 'N/A')}\n")
    
    # Test 6: Get open orders
    print("6️⃣ Checking open orders...")
    orders = client.get_open_orders()
    if isinstance(orders, dict) and 'error' in orders:
        print(f"   ❌ Failed: {orders['error']}")
        return False
    print(f"   ✅ Open orders: {len(orders)}\n")
    
    print("=" * 50)
    print("✅ All tests passed! Ed25519 signing working.")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
