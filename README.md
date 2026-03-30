# Alcyone Trading Bot

Multi-asset crypto trading bot with 6-layer ML strategy. Runs 24/7 on Raspberry Pi 5.

## Architecture

```
7 Coins (BTC/ETH/SOL/BNB/XRP/DOGE/AVAX)
    |
    v
ProStrategy (6 layers, weighted voting)
    |
    +-- Layer 1: Trend (EMA 12/26 + ADX)
    +-- Layer 2: Momentum (RSI + MACD + Stochastic)
    +-- Layer 3: Volatility (Bollinger Bands + ATR dynamic SL/TP)
    +-- Layer 4: Volume (OBV + spike detection)
    +-- Layer 5: Sentiment (Fear & Greed Index)
    +-- Layer 6: ML (XGBoost + GRU + CryptoBERT)
    |
    v
Risk Manager (dynamic SL/TP, circuit breaker)
    |
    v
Binance Testnet (real orders, fake money)
    |
    v
Telegram Notifications (trade alerts, daily reports)
```

## Quick Start

```bash
# Install dependencies
cd /path/to/trading
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # TODO: create this

# Download historical data (one-time)
python3 ml/download_historical.py

# Train ML models (one-time, ~30min on Pi)
python3 ml/xgboost_model.py
python3 ml/gru_model.py

# Run the bot
python3 multi_asset_bot.py

# Or install as systemd service (recommended)
cp trading-bot.service ~/.config/systemd/user/
systemctl --user enable trading-bot.service
systemctl --user start trading-bot.service
```

## Configuration

- Balance cap: $100 (configurable in multi_asset_bot.py)
- Per-coin allocation: ~$14.28 (7-way split)
- Position size: 90% of per-coin allocation
- Check interval: 5 minutes
- Risk: 2% SL, 5% TP, 15% max drawdown

## ML Models

| Model | Type | Runs On | Accuracy |
|-------|------|---------|----------|
| XGBoost | Classifier | Pi (local) | 46% avg |
| GRU | Neural Net | Pi (local) | Training... |
| CryptoBERT | Sentiment | Pi (local) | Bullish/Bearish/Neutral |
| Fear & Greed | Index API | Remote | 0-100 scale |
| News Keywords | Rule-based | Pi (local) | Keyword matching |

## Files

```
trading/
  multi_asset_bot.py     # Main 24/7 bot (systemd service)
  bot_service.py         # Single-asset bot (legacy)
  strategies/
    pro_strategy.py      # 6-layer ProStrategy
    enhanced_strategy.py # Intermediate strategy
    ma_crossover.py      # Original simple EMA
  ml/
    data_pipeline.py     # Feature engineering
    download_historical.py # Data downloader
    gru_model.py         # GRU training + inference
    xgboost_model.py     # XGBoost training + inference
    local_sentiment.py   # CryptoBERT on Pi
    hf_sentiment.py      # HuggingFace API (backup)
    copilot_sentiment.py # Copilot API sentiment (backup)
    news_parser.py       # RSS feed parser
    ensemble.py          # Model combiner
  risk_manager.py        # SL/TP/drawdown
  telegram_notifier.py   # Real Telegram messages
  trade_logger.py        # Trade logging
  balance_cap.py         # Simulated balance limit
  config/                # API keys (gitignored)
  data/                  # Historical + features (gitignored)
  logs/                  # Trade logs (gitignored)
```

## Security

- Repo is PRIVATE
- API keys in config/ (gitignored)
- Private keys never committed (removed from history)
- Testnet only — no real money without explicit flag

## License

Private project by msbel5 + Alcyone AI.
