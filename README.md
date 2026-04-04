# Alcyone Trading Bot v4

Scientific multi-asset crypto trading bot with 46 ML features, 9 signal layers, CNN-LSTM, CPCV, regime detection, and smart execution. Runs 24/7 on Raspberry Pi 5.

<img width="1594" height="930" alt="image" src="https://github.com/user-attachments/assets/9f27e299-1fef-4ad0-9ddd-013f4afa24ae" /> 

<img width="1597" height="888" alt="image" src="https://github.com/user-attachments/assets/5aba2fc5-c04e-4f79-8888-fd93f5dc815b" />


## Architecture

```
Market Data (Binance Testnet, 7 coins, ~5s tick)
    │
    ▼
┌─ 9-Layer Signal Engine ─────────────────────────────┐
│  L1: Trend        EMA 12/26 + ADX strength    (15%) │
│  L2: Momentum     RSI + MACD + Stochastic      (15%) │
│  L3: Volatility   BB + ATR + EWMA + VolCone    (10%) │
│  L4: Volume       OBV + spike + CMF            (8%)  │
│  L5: Sentiment    Fear & Greed Index            (10%) │
│  L6: ML Ensemble  Stacked v3 + CNN-LSTM + GRU  (18%) │
│  L7: Ichimoku     Cloud + Tenkan/Kijun cross    (10%) │
│  L8: Patterns     15 candlestick + confluence   (7%)  │
│  L9: Statistical  Hurst + Z-Score               (7%)  │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─ On-Chain + Microstructure ──────────────────────────┐
│  MVRV Z-Score      Market vs realized value          │
│  NVT Ratio         Network value / tx volume         │
│  Funding Rate      Crowded long/short detection      │
│  Long/Short Ratio  Liquidation proxy                 │
│  Exchange Flow     Mempool + fee analysis            │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─ Regime Detection ───────────────────────────────────┐
│  Hurst Exponent    H>0.5 trending, H<0.5 mean-revert│
│  Vol Cone          Percentile-based vol regime       │
│  ADX + BB-Keltner  Squeeze detection                 │
│  4 Regimes         TRENDING / SIDEWAYS / HIGH / EXTREME│
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─ Risk Management ────────────────────────────────────┐
│  Circuit Breaker   5% daily / 10% weekly / 20% max   │
│  Trailing Stop     ATR-based, regime-adaptive         │
│  Kelly Criterion   Half-Kelly position sizing         │
│  Vol-Adjusted      position = base * target/current   │
│  Viability Gate    SQN>1.7 AND PF>1.2 AND RoR<10%   │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌─ Smart Execution ────────────────────────────────────┐
│  TWAP / VWAP       Institutional order splitting      │
│  Slippage Model    Almgren-Chriss impact estimation   │
│  Smart Timing      Avoid hourly/daily close windows   │
│  Adaptive Entry    Scale in over multiple candles     │
└──────────────────────────────────────────────────────┘
    │
    ▼
Binance Testnet → Telegram → Dashboard (http://alcyone:8085)
```

## ML Pipeline (v4 — 46 Features)

| Category | Count | Features |
|----------|-------|----------|
| Technical (v3) | 25 | EMA, RSI, MACD, BB, ATR, ADX, Stoch, OBV, momentum, MVRV, NVT... |
| Advanced Indicators | 8 | Ichimoku, MFI, VWAP, Keltner squeeze, Williams %R, CMF, CCI, STC |
| Statistical Models | 5 | Hurst exponent, GARCH vol, Z-Score, Shannon entropy, frac diff |
| Candlestick Patterns | 3 | Pattern signal, confluence score, swing position |
| Volatility Engine | 5 | EWMA vol, vol cone percentile, vol ratio, BB-Keltner, term structure |

| Component | Method | Basis |
|-----------|--------|-------|
| Validation | CPCV (purge + embargo) | de Prado 2018 |
| Feature Selection | Boruta (shadow features) | 15-20 of 46 selected |
| Base Models | LightGBM + XGBoost + RF + ExtraTrees | Walk-forward proven |
| Neural | CNN-LSTM (Conv1D→LSTM→Dense) | Chen 2024 (82% lit.) |
| Meta-Learner | RidgeClassifier | Anti leakage |
| Retrain | Daily CPCV + PBO gate | Only deploy if PBO < 0.5 |

## Risk Analytics (12 Metrics)

| Metric | What It Measures |
|--------|-----------------|
| Sortino | Downside-only risk-adjusted return |
| Calmar | Return per max drawdown |
| Omega | Full distribution gains vs losses |
| SQN | System quality (Van Tharp scale) |
| VaR / CVaR | Value at Risk + Expected Shortfall |
| MAE / MFE | Optimal stop-loss / take-profit calibration |
| Risk of Ruin | Account destruction probability |

## Files

```
bot.py                    Main 24/7 trading bot (v4 integrated)
dashboard.py              Web dashboard v4 (risk metrics + regime)
backtester.py             Backtest framework + TrailingStop
backtest_v3.py            v3 vs v4 comparison
daily_retrain_v3.py       Nightly CPCV retrain (PBO-gated)
position_store.py         Disk persistence
risk_manager.py           SL / TP / drawdown
telegram_notifier.py      Telegram alerts
trade_logger.py           Trade audit log
data_sources.py           Twitter + whale + grid
filters.py                Multi-TF + correlation + Kelly

strategies/
    pro_strategy.py       9-layer ProStrategy

ml/
    data_pipeline.py      Base feature engineering
    data_pipeline_v4.py   V4 pipeline (46 features)
    ml_v3.py              ML pipeline (CPCV, Boruta, CNN-LSTM, Stacked)
    indicators_advanced.py  10 indicators (Ichimoku, MFI, VWAP, Keltner...)
    statistical_models.py   8 models (Hurst, GARCH, Z-Score, entropy...)
    candlestick_patterns.py 15 patterns + swing + confluence
    risk_metrics.py         12 risk metrics + viability gate
    volatility_engine.py    8 vol estimators + regime + sizing
    execution_engine.py     TWAP/VWAP + slippage + timing
    bot_v3_patch.py         Bot integration + CircuitBreaker
    onchain_v2.py           MVRV + NVT + funding + liquidation
    onchain.py              Mempool + fees + hashrate
    gru_model.py            GRU neural network
    local_sentiment.py      CryptoBERT on Pi
    copilot_sentiment.py    GPT-4.1 via Copilot API

binance_testnet/
    adapter_binance.py    REST + Ed25519 signing
```

## Systemd Services

| Service | Purpose |
|---------|---------|
| `trading-bot` | Main bot (24/7) |
| `trading-retrain.timer` | Daily retrain (03:00) |
| `copilot-api` | GPT-4.1 free proxy |

## Research Basis

| Book/Paper | What We Use |
|-----------|-------------|
| de Prado "Advances in Financial ML" (2018) | CPCV, fractional diff, PBO |
| Nison "Japanese Candlestick Charting" (1991) | 15 candlestick patterns |
| Chan "Quantitative Trading" (2008) | Hurst, mean reversion |
| Bollerslev (1986) | GARCH volatility |
| Almgren-Chriss (2001) | Execution optimization |
| Van Tharp "Trade Your Way" | SQN, risk of ruin |
| Hosoda (1968) | Ichimoku Cloud |
| Kelly (1956) | Position sizing |

## Security

- Testnet only — no real money
- API keys gitignored
- Circuit breaker halts on loss
- Ed25519 key signing
