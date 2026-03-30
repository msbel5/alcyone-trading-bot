"""
Balance cap wrapper — simulates trading with a smaller balance.
Wraps LiveTrader.get_balance() to return min(real_balance, cap).
This way we trade as if we only have $100 even on testnet $10K.
"""

def apply_balance_cap(trader, cap=100.0):
    """Monkey-patch trader to cap balance at given amount."""
    original_get_balance = trader.get_balance
    
    def capped_balance():
        real = original_get_balance()
        if real is None:
            return None
        return min(real, cap)
    
    trader.get_balance = capped_balance
    trader._balance_cap = cap
    return trader
