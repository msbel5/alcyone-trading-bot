# Alcyone Trading Bot

Multi-asset crypto trading bot with 6-layer ML strategy. Runs 24/7 on Raspberry Pi 5.

## Architecture

```
Market Data (Binance Testnet, 7 coins)
    |
    v
ProStrategy (6 signal layers, weighted voting)
    |-- L1: Trend (EMA 12/26 + ADX strength)
    |-- L2: Momentum (RSI + MACD + Stochastic)
    |-- L3: Volatility (Bollinger Bands + ATR)
    |-- L4: Volume (OBV trend + spike detection)
    |-- L5: Sentiment (Fear & Greed Index)
    |-- L6: ML (XGBoost + GRU + CryptoBERT local)
    |
    v
Filters
    |-- Multi-timeframe (4h trend confirmation)
    |-- Correlation (max 3 correlated positions)
    |-- Grid mode (sideways market, ADX < 20)
    |
    v
Risk Management
    |-- Trailing stop (ATR-based, never decreases)
    |-- Kelly criterion (dynamic position sizing)
    |-- Circuit breaker (15% max drawdown)
    |
    v
Execution (Binance Testnet API)
    |
    v
Monitoring
    |-- Telegram notifications (real-time)
    |-- HTML dashboard
    |-- Weekly auto-retrain
```

## Coins
BTC, ETH, SOL, BNB, XRP, DOGE, AVAX

## Quick Start

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas scikit-learn xgboost transformers feedparser requests cryptography

# Download data + train models (one-time, ~30min)
python3 ml/download_historical.py
python3 ml/xgboost_model.py
python3 ml/gru_model.py

# Backtest
python3 backtester.py

# Run
python3 bot.py

# Or as systemd service
systemctl --user enable trading-bot
systemctl --user start trading-bot
```

## Files

```
bot.py                    Main 24/7 trading bot
backtester.py             Backtest framework + trailing stop
filters.py                Multi-timeframe + correlation + Kelly criterion
data_sources.py           Twitter + whale tracking + grid bot + dashboard + auto-retrain
live_trader.py            Binance API wrapper
risk_manager.py           Stop-loss / take-profit / drawdown
telegram_notifier.py      Telegram Bot API notifications
trade_logger.py           Trade logging to JSONL
strategies/
    pro_strategy.py       6-layer ProStrategy (main)
ml/
    data_pipeline.py      Feature engineering (15 indicators)
    download_historical.py Data downloader (Binance public API)
    gru_model.py          GRU neural network
    xgboost_model.py      XGBoost classifier
    local_sentiment.py    CryptoBERT on Pi (no API cost)
    hf_sentiment.py       HuggingFace API (backup)
    copilot_sentiment.py  GPT-4.1 via Copilot API (backup)
    news_parser.py        RSS feed parser
    ensemble.py           Model combiner
binance_testnet/
    adapter_binance.py    Binance REST + Ed25519 signing
config/                   API keys (gitignored)
data/                     Historical + features (gitignored)
logs/                     Trade logs (gitignored)
docs/
    PRD_ml_prediction_layer.md
    PRD_trading_improvements.md
```

## Backtest Results (8 months, $100)

| Coin | Trailing Stop | Win Rate | Sharpe |
|------|:---:|:---:|:---:|
| BTC | +18.7% | 52% | 1.59 |
| BNB | +10.4% | 53% | 0.79 |
| DOGE | +8.5% | 37% | 0.56 |
| SOL | -3.6% | 46% | -0.04 |
| ETH | -4.4% | 42% | -0.11 |
| XRP | -6.2% | 36% | -0.29 |
| AVAX | -24.7% | 29% | -1.45 |
| **Portfolio** | **-0.2%** | | |

## Security

- Repo is public, API keys are gitignored
- Private keys never in repo
- Testnet only — no real money without explicit code change
