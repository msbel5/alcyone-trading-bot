# Trading Bot — Binance Testnet

**Full-featured crypto trading bot** with backtesting, risk management, and live execution.

---

## 📦 Components

```
trading/
├── binance_testnet/
│   └── adapter_binance.py         # Binance API wrapper (Ed25519)
├── strategies/
│   └── ma_crossover.py            # EMA crossover strategy
├── config/
│   ├── testnet_config.json        # API credentials
│   └── ed25519_private.pem        # Private key
├── data/
│   ├── btcusdt_1h.csv             # Market data (1000 candles)
│   └── fetcher.py                 # Data downloader
├── logs/
│   └── trades_*.jsonl             # Daily trade logs
├── venv/                          # Python virtual environment
│
├── backtest.py                    # Backtest engine
├── optimize.py                    # Parameter optimizer
├── risk_manager.py                # SL/TP/drawdown control
├── trade_logger.py                # Structured logging
├── telegram_notifier.py           # Alert system
├── live_trader.py                 # Live trading core
├── bot_main.py                    # Main trading loop
├── paper_trader.py                # Paper trading simulator
│
├── test_suite.py                  # Unit tests
├── test_order_execution.py        # Order execution tests
├── test_connection.py             # API connection test
│
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Setup API Keys
1. Go to [testnet.binance.vision](https://testnet.binance.vision/)
2. Generate Ed25519 API key
3. Save to `config/testnet_config.json` and `config/ed25519_private.pem`

### 2. Install Dependencies
```bash
cd /home/msbel/.openclaw/workspace/trading
source venv/bin/activate
pip install pandas numpy requests cryptography
```

### 3. Test Connection
```bash
python test_connection.py
```

### 4. Run Backtest
```bash
python backtest.py
```

### 5. Optimize Strategy
```bash
python optimize.py
```

### 6. Paper Trading Simulation
```bash
python paper_trader.py
```

### 7. Live Trading (DRY RUN)
```bash
python bot_main.py
```

---

## 📊 Features

### ✅ Phase 1: Testnet Connection
- Ed25519 authentication
- REST API wrapper
- Account balance + price fetching

### ✅ Phase 2: Market Data
- 1h candle downloads
- CSV storage (1000 candles)
- Historical data management

### ✅ Phase 3: Strategy & Backtesting
- **MA Crossover** (EMA 7/25 default, optimized to 12/30)
- Backtest engine with PnL tracking
- Parameter grid search
- **Best params:** EMA 12/30 (+0.31% return, 46.67% win rate)

### ✅ Phase 4: Order Execution
- Market buy/sell orders
- Dry-run mode (safe testing)
- Order logging
- Balance: $10,000 USDT (testnet)

### ✅ Phase 5: Live Monitoring
- Continuous price monitoring
- Signal generation from live candles
- Auto trade execution on crossovers
- Risk exit checking (SL/TP)

### ✅ Phase 6: Telegram Alerting
- Trade notifications (BUY/SELL)
- Risk event alerts (SL/TP/drawdown)
- Session start/end summaries
- Priority-based messaging

---

## 🛡️ Risk Management

```python
RiskManager(
    max_position_size=0.95,  # 95% of balance per trade
    stop_loss_pct=2.0,       # -2% stop-loss
    take_profit_pct=5.0,     # +5% take-profit
    max_drawdown_pct=10.0    # Pause trading at -10%
)
```

- **Position sizing:** Uses 95% of available balance
- **Stop-loss:** Auto-exit at -2% loss
- **Take-profit:** Auto-exit at +5% profit
- **Max drawdown:** Pauses trading at -10% total loss

---

## 📈 Performance (Backtest)

**EMA 12/30 on 1000 candles (Feb-Mar 2026):**
- Initial balance: $1,000
- Final balance: $1,003.10
- Total return: +0.31%
- Trades: 15
- Win rate: 46.67%
- Avg win: $19.86
- Avg loss: -$15.18

**With SL/TP (paper trading):**
- Final: $974.31
- Return: -2.57%
- Trades: 27
- SL triggered: 11 times
- TP triggered: 3 times

---

## 🧪 Tests

```bash
# Run all tests
python test_suite.py

# Test order execution
python test_order_execution.py

# Test Telegram notifier
python telegram_notifier.py
```

**Test coverage:**
- ✅ MA Crossover signal generation
- ✅ Backtest engine (+18.76% on uptrend test)
- ✅ Optimization results integrity
- ✅ Market data validity (1000 candles)
- ✅ Order execution (dry-run)
- ✅ Telegram notifications

---

## 🔐 Safety

- **Testnet only:** No real money at risk
- **Dry-run default:** All scripts start in dry-run mode
- **API keys:** Stored locally, never committed to git
- **Ed25519:** Secure authentication
- **Rate limits:** Respectful API usage
- **Drawdown protection:** Auto-pause on max loss

---

## 🎛️ Configuration

Edit `bot_main.py` for customization:

```python
trader = LiveTrader(
    api_key=config['api_key'],
    private_key_path=config['private_key_path'],
    dry_run=True,              # False for live trading ⚠️
    telegram_enabled=True       # Enable alerts
)

# Adjust monitoring interval (default 5 minutes)
run_trading_loop(trader, interval_seconds=300)
```

---

## 📱 Telegram Integration

Notifications sent via console print (ready for OpenClaw message tool):

```
[TELEGRAM] 🟢 BUY executed
Price: $70,000.00
Amount: 0.14000 BTC

[TELEGRAM] 🔴 SELL executed
Price: $71,500.00
PnL: $+210.00 (+3.00%)
Reason: TAKE_PROFIT
```

---

## 📝 Logging

All trades logged to `logs/trades_YYYY-MM-DD.jsonl`:

```json
{
  "timestamp": "2026-03-22T14:09:48",
  "event_type": "BUY",
  "data": {
    "price": 70000,
    "amount": 0.14,
    "cost": 9800,
    "balance_after": 200,
    "strategy": "MA_CROSS_12_30"
  }
}
```

---

## ⚠️ Disclaimer

- **Educational project** — Not financial advice
- **Testnet only** — No real funds
- **No guarantees** — Past performance ≠ future results
- **Use at own risk** — Test thoroughly before any real trading

---

## 🎯 Next Steps

1. ✅ All 6 phases complete
2. Test with longer historical data
3. Implement more strategies (RSI, MACD, etc.)
4. Add multi-pair support
5. Dashboard for monitoring
6. Real-time WebSocket data feed

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** — Data analysis
- **numpy** — Numerical computing
- **requests** — HTTP client
- **cryptography** — Ed25519 signing
- **Binance Testnet** — Safe trading environment

---

## 📧 Support

Issues: Report to Alcyone (this project)
Testnet: [testnet.binance.vision](https://testnet.binance.vision/)

---

**Built with ❤️ by Alcyone for Mami**
