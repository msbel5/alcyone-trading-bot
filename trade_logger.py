#!/usr/bin/env python3
"""
Trade Logger
Structured logging for all trading activities.
"""

import json
import os
from datetime import datetime


class TradeLogger:
    def __init__(self, log_dir='/home/msbel/.openclaw/workspace/trading/logs'):
        """
        :param log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create daily log file
        today = datetime.now().strftime('%Y-%m-%d')
        self.log_file = os.path.join(log_dir, f'trades_{today}.jsonl')
        self.summary_file = os.path.join(log_dir, 'summary.json')

    def log_trade(self, event_type, data):
        """
        Log a trade event.
        :param event_type: Type of event (BUY, SELL, SIGNAL, ERROR, etc.)
        :param data: Event data (dict)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'data': data
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def log_buy(self, price, amount, balance, strategy=None):
        """Log a buy order."""
        self.log_trade('BUY', {
            'price': price,
            'amount': amount,
            'cost': price * amount,
            'balance_after': balance,
            'strategy': strategy
        })

    def log_sell(self, price, amount, pnl, pnl_pct, balance, reason=None):
        """Log a sell order."""
        self.log_trade('SELL', {
            'price': price,
            'amount': amount,
            'revenue': price * amount,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'balance_after': balance,
            'reason': reason
        })

    def log_signal(self, signal_type, price, ema_fast=None, ema_slow=None):
        """Log a strategy signal."""
        self.log_trade('SIGNAL', {
            'signal_type': signal_type,
            'price': price,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow
        })

    def log_error(self, error_msg, context=None):
        """Log an error."""
        self.log_trade('ERROR', {
            'error': error_msg,
            'context': context
        })

    def log_risk_event(self, event_type, data):
        """Log risk management events (stop-loss, drawdown, etc.)."""
        self.log_trade('RISK', {
            'event': event_type,
            'data': data
        })

    def get_daily_summary(self):
        """
        Generate summary from today's log file.
        :return: Dict with summary stats
        """
        if not os.path.exists(self.log_file):
            return {'total_trades': 0}
        
        trades = []
        errors = 0
        
        with open(self.log_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry['event_type'] == 'SELL':
                    trades.append(entry['data'])
                elif entry['event_type'] == 'ERROR':
                    errors += 1
        
        if not trades:
            return {
                'total_trades': 0,
                'errors': errors,
                'log_file': self.log_file
            }
        
        total_pnl = sum(t['pnl'] for t in trades)
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        
        return {
            'total_trades': len(trades),
            'total_pnl': round(total_pnl, 2),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round((len(wins) / len(trades)) * 100, 2) if trades else 0,
            'avg_win': round(sum(w['pnl'] for w in wins) / len(wins), 2) if wins else 0,
            'avg_loss': round(sum(l['pnl'] for l in losses) / len(losses), 2) if losses else 0,
            'errors': errors,
            'log_file': self.log_file
        }

    def print_summary(self):
        """Print daily summary to console."""
        summary = self.get_daily_summary()
        
        print("\n" + "="*60)
        print("DAILY TRADE SUMMARY")
        print("="*60)
        
        if summary['total_trades'] == 0:
            print("No trades today.")
            print(f"Log file: {summary['log_file']}")
            return
        
        print(f"Total Trades:    {summary['total_trades']}")
        print(f"Total PnL:       ${summary['total_pnl']:+.2f}")
        print(f"Wins / Losses:   {summary['wins']} / {summary['losses']}")
        print(f"Win Rate:        {summary['win_rate']:.2f}%")
        print(f"Avg Win:         ${summary['avg_win']:.2f}")
        print(f"Avg Loss:        ${summary['avg_loss']:.2f}")
        print(f"Errors:          {summary['errors']}")
        print(f"Log file:        {summary['log_file']}")
        print("="*60)


if __name__ == '__main__':
    # Test logger
    logger = TradeLogger()
    
    print("Testing Trade Logger...")
    
    # Simulate some trades
    logger.log_signal('BUY', price=70000, ema_fast=70100, ema_slow=69900)
    logger.log_buy(price=70000, amount=0.01, balance=300, strategy='MA_CROSS_7_25')
    
    logger.log_signal('SELL', price=71000, ema_fast=70900, ema_slow=71100)
    logger.log_sell(price=71000, amount=0.01, pnl=10, pnl_pct=1.43, balance=1010, reason='TP_HIT')
    
    logger.log_error('Connection timeout', context={'endpoint': '/api/v3/order'})
    
    logger.log_risk_event('DRAWDOWN_WARNING', {'current_drawdown': 0.08, 'threshold': 0.10})
    
    # Print summary
    logger.print_summary()
    
    print(f"\nLog file created: {logger.log_file}")
