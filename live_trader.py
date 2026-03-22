#!/usr/bin/env python3
"""
Live Trader
Connects to Binance Testnet and executes real orders based on strategy signals.
"""

import sys
import time
import json
import os
sys.path.append('/home/msbel/.openclaw/workspace/trading')

from binance_testnet.adapter_binance import BinanceTestnetAdapter
from strategies.ma_crossover import MACrossover
from risk_manager import RiskManager
from trade_logger import TradeLogger
from telegram_notifier import TelegramNotifier


class LiveTrader:
    def __init__(self, api_key, private_key_path, symbol='BTCUSDT', dry_run=True, telegram_enabled=False):
        """
        :param api_key: Binance API key
        :param private_key_path: Path to Ed25519 private key PEM
        :param symbol: Trading pair
        :param dry_run: If True, log orders but don't execute
        :param telegram_enabled: Enable Telegram notifications
        """
        self.adapter = BinanceTestnetAdapter(api_key, private_key_path)
        self.symbol = symbol
        self.dry_run = dry_run
        
        self.strategy = MACrossover(fast_period=12, slow_period=30)
        self.risk_manager = RiskManager()
        self.logger = TradeLogger()
        self.notifier = TelegramNotifier(enabled=telegram_enabled)
        
        self.position = 0  # BTC held
        self.entry_price = None
        self.is_active = True

    def get_balance(self):
        """Get USDT balance."""
        try:
            account = self.adapter.account_info()
            for balance in account.get('balances', []):
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
            return 0.0
        except Exception as e:
            self.logger.log_error(f'Failed to get balance: {e}')
            return None

    def get_current_price(self):
        """Get current BTC price."""
        try:
            ticker = self.adapter.get_symbol_price(self.symbol)
            return float(ticker['price'])
        except Exception as e:
            self.logger.log_error(f'Failed to get price: {e}')
            return None

    def place_market_buy(self, quantity):
        """
        Place market buy order.
        :param quantity: Amount of BTC to buy
        :return: Order response or None
        """
        if self.dry_run:
            self.logger.log_trade('DRY_RUN_BUY', {
                'symbol': self.symbol,
                'quantity': quantity,
                'type': 'MARKET'
            })
            print(f"[DRY RUN] Market BUY {quantity:.6f} {self.symbol}")
            return {'orderId': 'DRY_RUN', 'status': 'FILLED'}
        
        try:
            order = self.adapter.place_order(
                symbol=self.symbol,
                side='BUY',
                order_type='MARKET',
                quantity=quantity
            )
            self.logger.log_trade('ORDER_BUY', order)
            return order
        except Exception as e:
            self.logger.log_error(f'Buy order failed: {e}')
            return None

    def place_market_sell(self, quantity):
        """
        Place market sell order.
        :param quantity: Amount of BTC to sell
        :return: Order response or None
        """
        if self.dry_run:
            self.logger.log_trade('DRY_RUN_SELL', {
                'symbol': self.symbol,
                'quantity': quantity,
                'type': 'MARKET'
            })
            print(f"[DRY RUN] Market SELL {quantity:.6f} {self.symbol}")
            return {'orderId': 'DRY_RUN', 'status': 'FILLED'}
        
        try:
            order = self.adapter.place_order(
                symbol=self.symbol,
                side='SELL',
                order_type='MARKET',
                quantity=quantity
            )
            self.logger.log_trade('ORDER_SELL', order)
            return order
        except Exception as e:
            self.logger.log_error(f'Sell order failed: {e}')
            return None

    def execute_buy(self, price):
        """Execute a buy based on risk management."""
        balance = self.get_balance()
        if balance < 10:
            self.logger.log_error('Insufficient balance', {'balance': balance})
            print(f"❌ Insufficient balance: ${balance:.2f}")
            return False
        
        amount = self.risk_manager.calculate_position_size(balance, price)
        
        # Round to BTC precision (0.00001)
        amount = round(amount, 5)
        
        order = self.place_market_buy(amount)
        if order and order.get('status') == 'FILLED':
            self.position = amount
            self.entry_price = price
            self.logger.log_buy(price, amount, balance - (amount * price))
            print(f"✅ BUY executed: {amount:.5f} BTC @ ${price:,.2f}")
            return True
        
        return False

    def execute_sell(self, price, reason='SIGNAL'):
        """Execute a sell."""
        if self.position == 0:
            return False
        
        # Round to BTC precision
        amount = round(self.position, 5)
        
        order = self.place_market_sell(amount)
        if order and order.get('status') == 'FILLED':
            cost = self.position * self.entry_price
            revenue = self.position * price
            pnl = revenue - cost
            pnl_pct = (pnl / cost) * 100
            
            balance = self.get_balance()
            self.logger.log_sell(price, amount, pnl, pnl_pct, balance, reason)
            self.notifier.notify_trade('SELL', price, amount, pnl, pnl_pct, reason)
            
            emoji = "🟢" if pnl > 0 else "🔴"
            print(f"{emoji} SELL executed: {amount:.5f} BTC @ ${price:,.2f} | PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason}")
            
            self.position = 0
            self.entry_price = None
            return True
        
        return False

    def check_signal(self, candles):
        """
        Check if there's a trading signal.
        :param candles: Recent price candles (DataFrame)
        :return: signal value (-1, 0, 1)
        """
        df = self.strategy.calculate_signals(candles)
        return df['signal'].iloc[-1]

    def run_once(self):
        """Run one trading cycle."""
        if not self.is_active:
            print("⚠️ Trading paused")
            return
        
        # Get current price
        price = self.get_current_price()
        if not price:
            return
        
        # Check drawdown
        balance = self.get_balance()
        current_value = balance + (self.position * price if self.position > 0 else 0)
        if self.risk_manager.check_drawdown(current_value):
            self.is_active = False
            print(f"🛑 Max drawdown reached at ${current_value:.2f}")
            return
        
        # Check risk exit (SL/TP)
        if self.position > 0:
            should_exit, reason = self.risk_manager.should_exit(self.entry_price, price)
            if should_exit:
                self.execute_sell(price, reason)
                return
        
        # For now, just monitor
        # TODO: Implement signal checking with historical candles
        print(f"📊 Price: ${price:,.2f} | Position: {self.position:.5f} BTC | Balance: ${balance:.2f}")


if __name__ == '__main__':
    # Load config
    config_path = '/home/msbel/.openclaw/workspace/trading/config/testnet_config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"❌ Config not found: {config_path}")
        print("Create config/testnet_config.json with your API keys")
        sys.exit(1)
    
    api_key = config.get('api_key', '')
    private_key_path = config.get('private_key_path', '')
    
    if not api_key or not private_key_path:
        print("❌ API key or private_key_path not configured")
        sys.exit(1)
    
    # Load private key
    if not os.path.exists(private_key_path):
        print(f"❌ Private key not found: {private_key_path}")
        sys.exit(1)
    
    # Initialize trader (DRY RUN mode)
    trader = LiveTrader(
        api_key=api_key,
        private_key_path=private_key_path,
        dry_run=True  # Safety: always start in dry-run mode
    )
    
    print("="*70)
    print("LIVE TRADER (DRY RUN MODE)")
    print("="*70)
    print(f"Symbol: {trader.symbol}")
    print(f"Strategy: EMA 12/30 Crossover")
    print(f"Risk: 95% position size, 2% SL, 5% TP, 10% max drawdown")
    print()
    
    # Test connection
    print("Testing connection...")
    balance = trader.get_balance()
    price = trader.get_current_price()
    
    if balance is not None and price is not None:
        print(f"✅ Connected to Binance Testnet")
        print(f"   USDT Balance: ${balance:.2f}")
        print(f"   BTC Price: ${price:,.2f}")
    else:
        print("❌ Connection failed")
        sys.exit(1)
    
    # Run once
    print("\nRunning monitoring cycle...")
    trader.run_once()
    
    print("\n✅ Live trader initialized successfully")
    print("To enable real orders: set dry_run=False (⚠️ USE WITH CAUTION)")
