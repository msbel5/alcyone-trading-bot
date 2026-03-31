# Alcyone Trading Bot

Multi-asset crypto trading bot with 6-layer ML strategy + GPT-4.1 AI analysis. Runs 24/7 on Raspberry Pi 5.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205-red.svg)](https://raspberrypi.com)
[![License](https://img.shields.io/badge/License-Private-gray.svg)]()

## Architecture

```
Market Data (Binance, 7 coins, ~20s tick)
    │
    ▼
┌─ ProStrategy (6 signal layers) ──────────────────┐
│  L1: Trend        EMA 12/26 + ADX strength       │
│  L2: Momentum     RSI + MACD + Stochastic         │
│  L3: Volatility   Bollinger Bands + ATR            │
│  L4: Volume       OBV trend + spike detection      │
│  L5: Sentiment    Fear & Greed Index               │
│  L6: ML           XGBoost + GRU + CryptoBERT      │
│  L7: AI           GPT-4.1 opinion (Copilot, free)  │
└──────────────────────────────────────────────────┘
    │
    ▼
┌─ Filters ────────────────────────────────────────┐
│  Multi-timeframe   4h trend must confirm 1h       │
│  Correlation       Max 3 correlated positions     │
│  Grid mode         Sideways market (ADX < 20)     │
└──────────────────────────────────────────────────┘
    │
    ▼
┌─ Risk Management ────────────────────────────────┐
│  Trailing stop     ATR-based, locks profit        │
│  Kelly criterion   Optimal position sizing        │
│  Circuit breaker   15% max drawdown → pause       │
│  Dynamic SL/TP     From ATR, not fixed %          │
└──────────────────────────────────────────────────┘
    │
    ▼
Binance API → Telegram Alerts → Web Dashboard
```

## Coins

BTC · ETH · SOL · BNB · XRP · DOGE · AVAX

## Quick Start

```bash
make full-setup    # Install + download data + train models + start bot
```

Or step by step:

```bash
make install       # Create venv, install PyTorch + ML packages
make download      # Download 8 months of 1h candles (7 coins)
make train         # Train XGBoost + GRU models (~30min on Pi)
make backtest      # Run backtest to verify strategy
make start         # Start bot + dashboard as systemd services
make status        # Check bot status + PnL + positions
make logs          # Tail live logs
```

## Dashboard

```
http://alcyone:8085        Web UI (auto-refresh 5min)
http://alcyone:8085/api/state  JSON API
```

Shows: Total PnL, open positions with trailing stops, equity curve, model status.

## Backtest Results (8 months, $100 portfolio)

| Coin | Return | Win Rate | Sharpe | Max Drawdown |
|------|:------:|:--------:|:------:|:------------:|
| BTC  | +18.7% | 52%      | 1.59   | 10.6%        |
| BNB  | +10.4% | 53%      | 0.79   | 10.1%        |
| DOGE | +8.5%  | 37%      | 0.56   | 17.7%        |
| SOL  | -3.6%  | 46%      | -0.04  | 19.0%        |
| ETH  | -4.4%  | 42%      | -0.11  | 20.1%        |
| XRP  | -6.2%  | 36%      | -0.29  | 23.6%        |
| AVAX | -24.7% | 29%      | -1.45  | 33.5%        |
| **Portfolio** | **-0.2%** | | | |

*Trailing stop vs fixed SL: trailing reduces max drawdown by ~40%*

## ML Models

| Model | Type | Runs On | Purpose | Accuracy |
|-------|------|---------|---------|:--------:|
| XGBoost | Classifier | Pi (CPU) | Price direction | 46% avg |
| GRU | Neural Net | Pi (CPU) | Temporal patterns | 47% avg |
| CryptoBERT | Transformer | Pi (CPU) | News sentiment | Bullish/Bearish |
| GPT-4.1 | LLM | Copilot API (free) | Trading opinion | Qualitative |
| Fear & Greed | Index | Remote API | Market mood | 0-100 |

## Data Sources

| Source | Type | Update Frequency |
|--------|------|:---:|
| Binance (testnet) | OHLCV + orders | Every tick (~20s) |
| Fear & Greed Index | Market sentiment | 15min cache |
| Nitter/Twitter RSS | Social sentiment | 15min cache |
| Blockchain.info | Whale movements | 30min cache |
| CoinDesk/CoinTelegraph RSS | News headlines | 15min cache |
| Copilot API (GPT-4.1) | AI analysis | Every tick |

## Files

```
bot.py                    Main trading bot (24/7 service)
backtester.py             Backtest framework + trailing stop
filters.py                Multi-timeframe + correlation + Kelly
data_sources.py           Twitter + whale + grid + retrain
dashboard.py              Web dashboard (port 8085)
live_trader.py            Binance API wrapper
risk_manager.py           Stop-loss / take-profit / drawdown
telegram_notifier.py      Telegram Bot API notifications
trade_logger.py           Trade logging
strategies/
    pro_strategy.py       7-layer ProStrategy
ml/
    data_pipeline.py      Feature engineering (15 indicators)
    download_historical.py Data downloader
    gru_model.py          GRU neural network
    xgboost_model.py      XGBoost classifier
    local_sentiment.py    CryptoBERT on Pi
    copilot_sentiment.py  GPT-4.1 via Copilot API
    hf_sentiment.py       HuggingFace API (backup)
    news_parser.py        RSS feed parser
    ensemble.py           Model combiner
binance_testnet/
    adapter_binance.py    REST + Ed25519 signing
Makefile                  Build/run/deploy commands
```

## Systemd Services

| Service | Port | Purpose |
|---------|:----:|---------|
| `trading-bot` | — | Main bot process |
| `copilot-api` | 4141 | GPT-4.1 free proxy |
| `openclaw-gateway` | — | Alcyone AI companion |

All auto-start on boot, auto-restart on crash.

## Security

- API keys in `config/` (gitignored, never committed)
- Testnet only — no real money without explicit code change
- Private keys rotated, old keys invalidated
- Repo is public, all secrets excluded

## Requirements

- Raspberry Pi 5 (4GB+ RAM) or any Linux
- Python 3.10+
- Node.js 18+ (for copilot-api)
- Binance Testnet account
- Telegram bot (for notifications)
- GitHub account (for Copilot API)
