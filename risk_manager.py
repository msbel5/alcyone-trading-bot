#!/usr/bin/env python3
"""
Risk Manager
Handles position sizing, stop-loss, take-profit, and max drawdown limits.
"""


class RiskManager:
    def __init__(
        self,
        max_position_size=0.01,  # Use max 1% of balance per trade
        stop_loss_pct=2.0,       # Stop loss at -2%
        take_profit_pct=5.0,     # Take profit at +5%
        max_drawdown_pct=10.0    # Max portfolio drawdown before pause
    ):
        """
        :param max_position_size: Max % of balance to use per trade (0-1)
        :param stop_loss_pct: Stop loss percentage (e.g., 2.0 = -2%)
        :param take_profit_pct: Take profit percentage (e.g., 5.0 = +5%)
        :param max_drawdown_pct: Max allowed drawdown before halting (e.g., 10.0 = -10%)
        """
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct / 100
        self.take_profit_pct = take_profit_pct / 100
        self.max_drawdown_pct = max_drawdown_pct / 100
        
        self.peak_balance = None
        self.is_paused = False

    def calculate_position_size(self, balance, price):
        """
        Calculate how much to buy based on balance and max position size.
        :param balance: Current USDT balance
        :param price: Current asset price
        :return: Amount to buy (in BTC)
        """
        usdt_to_use = balance * self.max_position_size
        amount = usdt_to_use / price
        return amount

    def calculate_stop_loss(self, entry_price):
        """
        Calculate stop-loss price.
        :param entry_price: Entry price
        :return: Stop-loss price
        """
        return entry_price * (1 - self.stop_loss_pct)

    def calculate_take_profit(self, entry_price):
        """
        Calculate take-profit price.
        :param entry_price: Entry price
        :return: Take-profit price
        """
        return entry_price * (1 + self.take_profit_pct)

    def check_drawdown(self, current_balance):
        """
        Check if max drawdown limit is reached.
        :param current_balance: Current portfolio balance
        :return: True if drawdown limit reached (should pause trading)
        """
        if self.peak_balance is None:
            self.peak_balance = current_balance
            return False
        
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        if current_drawdown >= self.max_drawdown_pct:
            self.is_paused = True
            return True
        
        return False

    def should_exit(self, entry_price, current_price):
        """
        Check if stop-loss or take-profit is triggered.
        :param entry_price: Entry price
        :param current_price: Current price
        :return: tuple (should_exit: bool, reason: str)
        """
        sl_price = self.calculate_stop_loss(entry_price)
        tp_price = self.calculate_take_profit(entry_price)
        
        if current_price <= sl_price:
            return (True, f"STOP_LOSS (entry: ${entry_price:.2f}, current: ${current_price:.2f}, SL: ${sl_price:.2f})")
        
        if current_price >= tp_price:
            return (True, f"TAKE_PROFIT (entry: ${entry_price:.2f}, current: ${current_price:.2f}, TP: ${tp_price:.2f})")
        
        return (False, None)

    def get_status(self):
        """Get current risk manager status."""
        return {
            'max_position_size': f"{self.max_position_size * 100:.1f}%",
            'stop_loss': f"{self.stop_loss_pct * 100:.1f}%",
            'take_profit': f"{self.take_profit_pct * 100:.1f}%",
            'max_drawdown': f"{self.max_drawdown_pct * 100:.1f}%",
            'peak_balance': self.peak_balance,
            'is_paused': self.is_paused
        }


if __name__ == '__main__':
    # Test risk manager
    rm = RiskManager(
        max_position_size=0.01,  # Match production default
        stop_loss_pct=2.0,
        take_profit_pct=5.0,
        max_drawdown_pct=10.0
    )
    
    print("="*60)
    print("RISK MANAGER TEST")
    print("="*60)
    
    # Test position sizing
    balance = 1000
    price = 70000
    amount = rm.calculate_position_size(balance, price)
    print(f"\nPosition Sizing:")
    print(f"  Balance: ${balance}")
    print(f"  Price: ${price}")
    print(f"  Amount to buy: {amount:.6f} BTC (${amount * price:.2f})")
    
    # Test stop-loss / take-profit
    entry_price = 70000
    sl = rm.calculate_stop_loss(entry_price)
    tp = rm.calculate_take_profit(entry_price)
    print(f"\nRisk Levels:")
    print(f"  Entry: ${entry_price}")
    print(f"  Stop-Loss: ${sl:.2f} (-{rm.stop_loss_pct * 100:.1f}%)")
    print(f"  Take-Profit: ${tp:.2f} (+{rm.take_profit_pct * 100:.1f}%)")
    
    # Test exit conditions
    test_prices = [68500, 70000, 73500]
    print(f"\nExit Checks:")
    for test_price in test_prices:
        should_exit, reason = rm.should_exit(entry_price, test_price)
        status = f"EXIT: {reason}" if should_exit else "HOLD"
        print(f"  Price ${test_price}: {status}")
    
    # Test drawdown
    print(f"\nDrawdown Check:")
    print(f"  Initial balance: $1000")
    print(f"  Max drawdown: {rm.max_drawdown_pct * 100:.1f}%")
    
    test_balances = [1000, 950, 900, 850]
    for bal in test_balances:
        halted = rm.check_drawdown(bal)
        status = "🛑 PAUSED" if halted else "✅ OK"
        print(f"  Balance ${bal}: {status}")
    
    print("\n" + "="*60)
    print("Status:", rm.get_status())
