#!/usr/bin/env python3
"""
Telegram Notifier
Sends trading alerts via Telegram.
"""

import sys
sys.path.append('/home/msbel/.openclaw/workspace/trading')

from trade_logger import TradeLogger


class TelegramNotifier:
    def __init__(self, enabled=True, channel_id=None):
        """
        :param enabled: Enable/disable notifications
        :param channel_id: Telegram channel/chat ID (optional)
        """
        self.enabled = enabled
        self.channel_id = channel_id
        self.logger = TradeLogger()
    
    def send(self, message, priority='normal'):
        """
        Send a notification.
        :param message: Message text
        :param priority: 'low', 'normal', 'high', 'critical'
        """
        if not self.enabled:
            return
        
        # Log the notification
        self.logger.log_trade('TELEGRAM_NOTIFICATION', {
            'message': message,
            'priority': priority,
            'channel_id': self.channel_id
        })
        
        # Format with priority emoji
        emoji = {
            'low': 'ℹ️',
            'normal': '📊',
            'high': '⚠️',
            'critical': '🚨'
        }.get(priority, '📊')
        
        formatted = f"{emoji} {message}"
        
        # In production, use OpenClaw's message tool
        # For now, just print (will be integrated with OpenClaw)
        print(f"[TELEGRAM] {formatted}")
    
    def notify_trade(self, action, price, amount, pnl=None, pnl_pct=None, reason=None):
        """Send trade notification."""
        if action == 'BUY':
            msg = f"🟢 BUY executed\n"
            msg += f"Price: ${price:,.2f}\n"
            msg += f"Amount: {amount:.5f} BTC"
            self.send(msg, priority='normal')
        
        elif action == 'SELL':
            emoji = "🟢" if pnl and pnl > 0 else "🔴"
            msg = f"{emoji} SELL executed\n"
            msg += f"Price: ${price:,.2f}\n"
            msg += f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)"
            if reason:
                msg += f"\nReason: {reason}"
            
            priority = 'high' if abs(pnl_pct) > 3 else 'normal'
            self.send(msg, priority=priority)
    
    def notify_risk_event(self, event_type, data):
        """Send risk management alert."""
        if event_type == 'STOP_LOSS':
            msg = f"🛑 Stop-loss triggered\n"
            msg += f"Entry: ${data.get('entry_price', 0):,.2f}\n"
            msg += f"Exit: ${data.get('current_price', 0):,.2f}\n"
            msg += f"Loss: {data.get('loss_pct', 0):.2f}%"
            self.send(msg, priority='high')
        
        elif event_type == 'TAKE_PROFIT':
            msg = f"✅ Take-profit triggered\n"
            msg += f"Entry: ${data.get('entry_price', 0):,.2f}\n"
            msg += f"Exit: ${data.get('current_price', 0):,.2f}\n"
            msg += f"Profit: {data.get('profit_pct', 0):.2f}%"
            self.send(msg, priority='normal')
        
        elif event_type == 'MAX_DRAWDOWN':
            msg = f"🚨 MAX DRAWDOWN REACHED\n"
            msg += f"Trading paused at ${data.get('balance', 0):.2f}\n"
            msg += f"Threshold: {data.get('threshold', 0):.1f}%"
            self.send(msg, priority='critical')
    
    def notify_error(self, error_msg, context=None):
        """Send error notification."""
        msg = f"❌ ERROR: {error_msg}"
        if context:
            msg += f"\nContext: {context}"
        self.send(msg, priority='high')
    
    def notify_session_start(self, mode, balance, strategy):
        """Send session start notification."""
        msg = f"🤖 Trading bot started\n"
        msg += f"Mode: {mode}\n"
        msg += f"Balance: ${balance:.2f}\n"
        msg += f"Strategy: {strategy}"
        self.send(msg, priority='low')
    
    def notify_session_end(self, summary):
        """Send session end notification."""
        msg = f"⏹️ Trading session ended\n"
        msg += f"Trades: {summary.get('total_trades', 0)}\n"
        msg += f"PnL: ${summary.get('total_pnl', 0):+.2f}\n"
        msg += f"Win rate: {summary.get('win_rate', 0):.1f}%"
        self.send(msg, priority='low')


if __name__ == '__main__':
    # Test notifier
    notifier = TelegramNotifier(enabled=True)
    
    print("Testing Telegram Notifier...")
    print()
    
    notifier.notify_session_start(
        mode='DRY RUN',
        balance=10000,
        strategy='EMA 12/30'
    )
    
    notifier.notify_trade(
        action='BUY',
        price=70000,
        amount=0.14
    )
    
    notifier.notify_trade(
        action='SELL',
        price=71500,
        amount=0.14,
        pnl=210,
        pnl_pct=3.0,
        reason='TAKE_PROFIT'
    )
    
    notifier.notify_risk_event('MAX_DRAWDOWN', {
        'balance': 9000,
        'threshold': 10.0
    })
    
    notifier.notify_session_end({
        'total_trades': 5,
        'total_pnl': 125.50,
        'win_rate': 60.0
    })
    
    print("\n✅ Notification test complete")
