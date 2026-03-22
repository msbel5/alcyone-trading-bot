#!/usr/bin/env python3
"""
Binance Testnet API Adapter
REST + Ed25519 signing support
"""
import base64
import time
import json
import requests
from typing import Dict, List, Optional
from urllib.parse import urlencode
from pathlib import Path
from cryptography.hazmat.primitives.serialization import load_pem_private_key


class BinanceTestnetAdapter:
    """Binance Testnet API wrapper (REST with Ed25519 signing)"""
    
    def __init__(self, api_key: str, private_key_path: str, base_url: str = "https://testnet.binance.vision"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        
        # Load Ed25519 private key
        with open(private_key_path, 'rb') as f:
            self.private_key = load_pem_private_key(data=f.read(), password=None)
    
    def _sign_ed25519(self, payload: str) -> str:
        """Generate Ed25519 signature"""
        signature_bytes = self.private_key.sign(payload.encode('ASCII'))
        return base64.b64encode(signature_bytes).decode('ASCII')
    
    def _request(self, method: str, endpoint: str, signed: bool = False, **kwargs) -> Dict:
        """Send HTTP request to Binance Testnet API"""
        url = f"{self.base_url}{endpoint}"
        
        if signed:
            kwargs.setdefault('params', {})
            kwargs['params']['timestamp'] = int(time.time() * 1000)
            
            # Generate signature
            payload = urlencode(kwargs['params'])
            signature = self._sign_ed25519(payload)
            kwargs['params']['signature'] = signature
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status_code": getattr(e.response, 'status_code', None)}
    
    def ping(self) -> Dict:
        """Test connectivity"""
        return self._request('GET', '/api/v3/ping')
    
    def server_time(self) -> Dict:
        """Get server time"""
        return self._request('GET', '/api/v3/time')
    
    def account_info(self) -> Dict:
        """Get account information (requires authentication)"""
        return self._request('GET', '/api/v3/account', signed=True)
    
    def account_balance(self) -> List[Dict]:
        """Get account balance (non-zero only)"""
        info = self.account_info()
        if 'balances' in info:
            return [b for b in info['balances'] if float(b['free']) > 0 or float(b['locked']) > 0]
        return []
    
    def get_symbol_price(self, symbol: str = "BTCUSDT") -> Dict:
        """Get current price for a symbol"""
        return self._request('GET', '/api/v3/ticker/price', params={'symbol': symbol})
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> List:
        """
        Get candlestick data
        interval: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        """
        return self._request('GET', '/api/v3/klines', params={
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        })
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._request('GET', '/api/v3/openOrders', signed=True, params=params)
    
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, 
                    price: Optional[float] = None, time_in_force: str = "GTC") -> Dict:
        """
        Place a new order
        side: BUY or SELL
        order_type: LIMIT, MARKET, STOP_LOSS_LIMIT, etc.
        """
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity,
        }
        
        if order_type.upper() == 'LIMIT':
            params['price'] = price
            params['timeInForce'] = time_in_force
        
        return self._request('POST', '/api/v3/order', signed=True, params=params)
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an active order"""
        return self._request('DELETE', '/api/v3/order', signed=True, params={
            'symbol': symbol,
            'orderId': order_id
        })


if __name__ == "__main__":
    # Quick test (requires config)
    print("Binance Testnet Adapter (Ed25519) loaded. Use test_connection.py for testing.")
