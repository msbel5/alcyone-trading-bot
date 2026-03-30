# PRD: ML Price Prediction Layer (Layer 6)
**Project:** Alcyone Trading Bot
**Author:** Mami + Claude
**Date:** 2026-03-31
**Status:** Active

## 1. Purpose

Add a 6th prediction layer to the trading bot's ProStrategy. This layer uses multiple ML models to predict short-term price direction, providing a confidence-weighted signal that feeds into the existing 5-layer composite score.

## 2. Architecture

```
Historical OHLCV Data (7 coins, 1yr+)
        │
        ├──→ GRU Model (local, Pi) ──→ price_direction + confidence
        ├──→ Bi-LSTM Model (local, Pi) ──→ price_direction + confidence
        ├──→ XGBoost (local, Pi) ──→ price_direction + confidence
        │
        ├──→ CryptoBERT (HuggingFace API) ──→ sentiment_score
        ├──→ CryptoTrader-LM (HuggingFace API) ──→ buy/sell/hold
        │
        └──→ Ensemble Combiner ──→ ml_prediction_score (-1 to +1)
                    │
                    └──→ ProStrategy Layer 6 (weight: 0.15)
```

## 3. Models

### 3.1 Local Models (run on Pi 5, no API calls)

#### GRU (Gated Recurrent Unit)
- **Why:** Outperforms LSTM on crypto per 2026 research, lighter memory footprint
- **Input:** 24h lookback window of [close, volume, rsi, macd, atr, bb_pct, obv]
- **Output:** Next 1h price direction probability (up/down/flat)
- **Size:** <20MB ONNX, runs in <100ms on Pi
- **Training:** Offline on historical data, retrain weekly

#### Bi-LSTM (Bidirectional Long Short-Term Memory)
- **Why:** Captures both forward and backward temporal patterns
- **Input:** Same 24h window as GRU
- **Output:** Next 1h direction + 4h direction
- **Size:** <30MB ONNX
- **Training:** Same as GRU

#### XGBoost Classifier
- **Why:** Fast, interpretable, good at feature importance ranking
- **Input:** Engineered features: RSI, MACD_hist, BB_pct, volume_ratio, ADX, price_momentum_1h/4h/24h, hour_of_day, day_of_week
- **Output:** Classification: up/down/flat with probabilities
- **Size:** <5MB
- **Training:** Retrain daily with latest data

### 3.2 Remote Models (HuggingFace Inference API)

#### CryptoBERT (kk08/CryptoBERT)
- **Purpose:** Crypto-specific sentiment analysis
- **Input:** Latest crypto news headlines (from RSS)
- **Output:** Positive/Negative/Neutral sentiment score
- **Rate limit:** Free tier ~30 req/min, we need ~7 req/5min (1 per coin)
- **Fallback:** Cache last result, use cached if API fails

#### CryptoTrader-LM (agarkovv/CryptoTrader-LM)
- **Purpose:** News + price → trading decision
- **Input:** Recent news + current price data
- **Output:** Buy/Sell/Hold recommendation
- **Rate limit:** Same free tier
- **Fallback:** Skip if unavailable, use local models only

### 3.3 News Data Sources (for sentiment models)
- CoinDesk RSS: https://www.coindesk.com/arc/outboundfeeds/rss/
- CoinTelegraph RSS: https://cointelegraph.com/rss
- CryptoSlate RSS: https://cryptoslate.com/feed/
- Parse every 15 minutes, cache headlines
- Extract top 5 headlines per coin

## 4. Ensemble Strategy

```python
ml_score = (
    gru_signal * 0.30 +          # strongest temporal model
    bilstm_signal * 0.25 +       # bidirectional context
    xgboost_signal * 0.20 +      # feature-based classification
    cryptobert_sentiment * 0.15 + # news sentiment
    cryptotrader_signal * 0.10    # LLM trading opinion
)
```

All signals normalized to [-1, +1]. Missing signals (API down) → excluded from average, weights redistributed.

## 5. Data Requirements

### Historical Data (for training)
- **7 coins:** BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, DOGEUSDT, AVAXUSDT
- **Timeframe:** 1h candles, minimum 6 months (4380 candles per coin)
- **Source:** Binance public API (mainnet, no auth needed for klines)
- **Fields:** open_time, open, high, low, close, volume
- **Storage:** CSV per coin in `data/historical/`

### Live Data (for inference)
- Last 50 candles (already fetched by bot each tick)
- Fear & Greed Index (already integrated)
- News headlines (new: RSS parser needed)

## 6. Technical Constraints

- **Pi 5:** 4GB RAM, ARM64, no GPU
- **Model size:** Total <100MB for all local models
- **Inference time:** <2 seconds per coin per tick
- **Dependencies:** PyTorch (CPU), scikit-learn, xgboost — all have ARM64 wheels
- **HuggingFace:** Free tier, ~30 req/min, cache aggressively
- **Training:** Done offline (on Pi or on Windows), models saved as ONNX/pickle
- **No numpy heavy ops during inference** — pre-compute features

## 7. Implementation Sprints

### Sprint 1: Data Pipeline
- Download 6+ months historical data for 7 coins (Binance public API)
- Feature engineering: add RSI, MACD, BB, ATR, OBV, volume_ratio, momentum
- Label generation: next_1h_direction (up=1, flat=0, down=-1)
- Train/val/test split: 70/15/15 by time (no shuffle — time series!)
- Save as CSV + pickle

### Sprint 2: Local Models
- Train GRU model (PyTorch)
- Train Bi-LSTM model (PyTorch)
- Train XGBoost classifier (scikit-learn)
- Export to ONNX (GRU/LSTM) and pickle (XGBoost)
- Benchmark: accuracy, precision, recall, F1 per coin
- Target: >55% directional accuracy (random = 33% for 3-class)

### Sprint 3: Remote Models + News
- RSS parser for 3 crypto news sources
- CryptoBERT integration via HuggingFace Inference API
- CryptoTrader-LM integration
- Caching layer (15min TTL for news, 5min for sentiment)
- Fallback when API down

### Sprint 4: Ensemble + Integration
- Combine all models into EnsemblePredictor class
- Add as Layer 6 to ProStrategy
- Update bot to call ensemble each tick
- Weighted scoring with dynamic weight redistribution
- Add ml_prediction to signal breakdown logging

### Sprint 5: Testing + Deploy
- Unit tests for each model wrapper
- Integration test: full tick with ML
- Backtest: compare 5-layer vs 6-layer on historical data
- Deploy to Pi, restart bot
- Monitor for 24h
- Telegram: report ML predictions alongside trade decisions

## 8. Test Plan (TDD)

```
tests/
  test_ml_data_pipeline.py    — data download, feature eng, labeling
  test_ml_gru_model.py        — GRU train, predict, export
  test_ml_bilstm_model.py     — Bi-LSTM train, predict, export
  test_ml_xgboost_model.py    — XGBoost train, predict, export
  test_ml_news_parser.py      — RSS fetch, parse, cache
  test_ml_cryptobert.py       — HF API call, sentiment parse
  test_ml_ensemble.py         — Combined prediction, weight redistribution
  test_ml_integration.py      — Full strategy tick with ML layer
```

## 9. File Structure

```
trading/
  ml/
    __init__.py
    data_pipeline.py        — download, feature eng, labeling
    gru_model.py            — GRU train + predict
    bilstm_model.py         — Bi-LSTM train + predict
    xgboost_model.py        — XGBoost train + predict
    news_parser.py          — RSS feed parser + cache
    hf_sentiment.py         — CryptoBERT + CryptoTrader-LM via HF API
    ensemble.py             — Combine all models
    models/                 — Saved model files
      gru_btcusdt.onnx
      bilstm_btcusdt.onnx
      xgboost_btcusdt.pkl
      ...
  data/
    historical/             — Downloaded OHLCV CSVs
      btcusdt_1h.csv
      ethusdt_1h.csv
      ...
```

## 10. Acceptance Criteria

- AC-01: Historical data for 7 coins, 6+ months, saved as CSV
- AC-02: GRU model trained with >55% directional accuracy on test set
- AC-03: Bi-LSTM model trained with >55% accuracy
- AC-04: XGBoost model trained with >55% accuracy
- AC-05: CryptoBERT sentiment works via HF API with caching
- AC-06: RSS news parser fetches from 3 sources
- AC-07: Ensemble combines 5 sub-models with weighted average
- AC-08: ProStrategy Layer 6 produces ml_prediction_score
- AC-09: All inference <2 seconds per coin on Pi 5
- AC-10: Bot runs 24h with ML layer without crash
- AC-11: Backtest shows improvement vs 5-layer only
