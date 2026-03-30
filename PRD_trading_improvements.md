# PRD: Trading Bot 10 Improvements
**Project:** Alcyone Trading Bot
**Date:** 2026-03-31
**Status:** Active

---

## 1. Backtest Framework (CRITICAL)

### Problem
Strateji gerçekten kârlı mı bilmiyoruz. 8 aylık veri var ama hiç backtest yok.

### Solution
Historical data üzerinde ProStrategy'yi simüle et. Her trade'i kaydet, performans metrikleri hesapla.

### Metrics
- Total return (%), Annualized return
- Sharpe ratio (risk-adjusted return)
- Max drawdown (%)
- Win rate (%)
- Profit factor (gross profit / gross loss)
- Average trade duration
- Total trades count

### Implementation
```python
class Backtester:
    def __init__(self, strategy, initial_balance=100, commission_pct=0.1):
        ...
    def run(self, df: pd.DataFrame) -> BacktestResult:
        # Iterate candles, simulate trades, track equity curve
    def compare_strategies(self, strategies: list, df) -> ComparisonReport:
        # Run multiple strategies, compare side by side
```

### Test Plan
- test_backtest_deterministic: same data = same result
- test_backtest_no_trades_on_flat: no signal = no trade
- test_backtest_commission_deducted: commission reduces PnL
- test_backtest_drawdown_calculated: max DD correct
- test_backtest_sharpe_formula: Sharpe matches manual calc

### AC
- AC-01: Backtest runs on 8 months data for all 7 coins in <30 seconds
- AC-02: Produces equity curve, trade log, and summary metrics
- AC-03: Compare 5-layer vs 6-layer strategies

---

## 2. Trailing Stop Loss (HIGH)

### Problem
Fixed %2 SL and %5 TP. Price goes +4% then reverses → we get nothing. Trailing stop locks in profit as price rises.

### Solution
After entry, track highest price. SL = highest_price - ATR * multiplier. As price rises, SL rises. Never decreases.

### Implementation
```python
class TrailingStop:
    def __init__(self, atr_multiplier=2.0):
        self.highest_since_entry = None
        self.trailing_sl = None

    def update(self, current_price, current_atr):
        if self.highest_since_entry is None or current_price > self.highest_since_entry:
            self.highest_since_entry = current_price
            self.trailing_sl = current_price - current_atr * self.atr_multiplier

    def should_exit(self, current_price) -> bool:
        return current_price <= self.trailing_sl
```

### Test Plan
- test_trailing_sl_rises_with_price
- test_trailing_sl_never_decreases
- test_trailing_sl_triggers_on_reversal
- test_trailing_sl_preserves_profit

### AC
- AC-01: SL rises as price rises
- AC-02: SL never decreases
- AC-03: Backtest shows improved return vs fixed SL

---

## 3. Multi-Timeframe Analysis (HIGH)

### Problem
1h signal says BUY but 4h trend is bearish → false signal. Need higher timeframe confirmation.

### Solution
Fetch 4h candles alongside 1h. Only take 1h signals that align with 4h trend direction.

### Implementation
```python
def multi_timeframe_filter(signal_1h, trend_4h):
    # BUY only if 4h trend is bullish or neutral
    if signal_1h == 1 and trend_4h == -1:
        return 0  # Reject: 4h bearish
    # SELL only if 4h trend is bearish or neutral
    if signal_1h == -1 and trend_4h == 1:
        return 0  # Reject: 4h bullish
    return signal_1h
```

### Test Plan
- test_mtf_rejects_buy_in_bearish_4h
- test_mtf_allows_buy_in_bullish_4h
- test_mtf_neutral_4h_passes_through

### AC
- AC-01: 4h candles fetched alongside 1h
- AC-02: Signal filtered before execution
- AC-03: Backtest shows fewer false signals

---

## 4. Coin Correlation Filter (HIGH)

### Problem
BTC drops → all coins drop. If all 7 coins BUY at same time, we're making one big bet, not 7 diversified bets.

### Solution
Calculate rolling correlation between coins. Limit simultaneous positions in highly correlated pairs.

### Implementation
```python
def correlation_filter(positions: dict, new_signal: str, correlations: pd.DataFrame, max_correlated=3):
    open_coins = [s for s, p in positions.items() if p > 0]
    if len(open_coins) >= max_correlated:
        # Check if new coin is highly correlated with existing positions
        for open_coin in open_coins:
            corr = correlations.loc[new_signal, open_coin]
            if corr > 0.8:
                return False  # Reject: too correlated
    return True
```

### Test Plan
- test_corr_allows_uncorrelated_entry
- test_corr_blocks_highly_correlated
- test_corr_max_3_correlated_positions
- test_corr_matrix_calculated_from_returns

### AC
- AC-01: 30-day rolling correlation matrix updated daily
- AC-02: Max 3 positions in coins with >0.8 correlation
- AC-03: Reduces portfolio drawdown in backtest

---

## 5. Weekly Auto-Retrain (MEDIUM)

### Problem
Models trained once on historical data. Market changes, models go stale.

### Solution
Every Sunday at 03:00, download latest data, retrain XGBoost and GRU, swap models if accuracy improves.

### Implementation
```python
# Cron job or systemd timer
# 1. Download last 7 days of new data
# 2. Append to historical CSVs
# 3. Retrain models
# 4. Compare accuracy: new vs old
# 5. If new > old - 2%: swap models
# 6. Log results, notify via Telegram
```

### Test Plan
- test_retrain_downloads_new_data
- test_retrain_appends_not_overwrites
- test_retrain_only_swaps_if_better
- test_retrain_notifies_telegram

### AC
- AC-01: Runs automatically every Sunday 03:00
- AC-02: Only swaps model if accuracy >= old - 2%
- AC-03: Telegram notification with old vs new accuracy

---

## 6. Twitter/X Sentiment (MEDIUM)

### Problem
News RSS is slow (15min cache). Twitter/X is fastest sentiment source for crypto.

### Solution
Use Nitter (public Twitter scraper, no API key needed) or Twitter API v2 free tier (1500 tweets/month). Parse mentions of coin names, analyze sentiment.

### Implementation
```python
class TwitterSentiment:
    def fetch_tweets(self, query: str, count=10) -> list:
        # Use nitter.net RSS or Twitter API

    def analyze(self, tweets: list) -> float:
        # Local CryptoBERT on each tweet
        # Weight by follower count / engagement
```

### Data Sources
- Nitter RSS: `https://nitter.net/search/rss?f=tweets&q=bitcoin`
- Twitter API v2 free: 1500 tweets/month read (enough for hourly checks)
- Fallback: Google News RSS with crypto keywords

### Test Plan
- test_twitter_fetches_recent_tweets
- test_twitter_filters_by_coin
- test_twitter_sentiment_scored
- test_twitter_fallback_on_failure

### AC
- AC-01: Fetches tweets for each coin every 15 minutes
- AC-02: CryptoBERT sentiment on top 5 tweets per coin
- AC-03: Signal integrated into ensemble

---

## 7. On-Chain Whale Tracking (MEDIUM)

### Problem
Large wallet movements predict price moves. Whale deposits to exchange = incoming sell pressure.

### Solution
Use free blockchain APIs to track large transfers. Whale Alert style.

### Data Sources (free)
- Blockchain.com API: BTC large transactions
- Etherscan API (free tier): ETH large transfers
- Whale Alert Twitter/RSS: `@whale_alert`
- CryptoQuant free API: exchange inflow/outflow

### Implementation
```python
class WhaleTracker:
    def check_large_transfers(self, coin: str) -> dict:
        # Returns: {"inflow": amount, "outflow": amount, "net": amount, "signal": -1/0/+1}
        # Large exchange inflow = bearish (selling pressure)
        # Large exchange outflow = bullish (accumulation)
```

### Test Plan
- test_whale_detects_large_inflow
- test_whale_bearish_on_exchange_deposit
- test_whale_bullish_on_withdrawal
- test_whale_api_fallback

### AC
- AC-01: Tracks BTC and ETH whale movements
- AC-02: Signal: exchange inflow > threshold = bearish
- AC-03: Updates every 30 minutes

---

## 8. Performance Dashboard (LOW)

### Problem
Only Telegram text notifications. No visual equity curve, no at-a-glance portfolio view.

### Solution
Simple HTML dashboard served from Pi. Shows equity curve, open positions, daily PnL, model accuracy.

### Implementation
```python
# Flask/FastAPI lightweight dashboard on port 8080
# Endpoints:
# GET / → HTML dashboard with charts
# GET /api/equity → JSON equity curve
# GET /api/positions → JSON open positions
# GET /api/performance → JSON daily/weekly/monthly PnL

# Charts: matplotlib → base64 PNG embedded in HTML
# Auto-refresh every 5 minutes
```

### Test Plan
- test_dashboard_serves_html
- test_dashboard_equity_endpoint
- test_dashboard_positions_endpoint

### AC
- AC-01: Dashboard accessible at `http://alcyone:8080`
- AC-02: Equity curve chart
- AC-03: Open positions with PnL

---

## 9. Kelly Criterion Position Sizing (LOW)

### Problem
Fixed 90% of per-coin allocation per trade. Not mathematically optimal.

### Solution
Kelly formula: `f* = (bp - q) / b` where b=win/loss ratio, p=win probability, q=1-p. Tells optimal fraction of bankroll to bet.

### Implementation
```python
def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 0
    b = avg_win / abs(avg_loss)  # Win/loss ratio
    p = win_rate
    q = 1 - p
    kelly = (b * p - q) / b
    # Half-Kelly for safety (less aggressive)
    return max(0, min(kelly * 0.5, 0.25))  # Cap at 25%
```

### Test Plan
- test_kelly_zero_on_negative_edge
- test_kelly_positive_on_winning_strategy
- test_kelly_half_kelly_safer
- test_kelly_capped_at_25pct

### AC
- AC-01: Position size calculated from recent win rate + avg win/loss
- AC-02: Half-Kelly used (conservative)
- AC-03: Updates after every 20 trades

---

## 10. Grid Bot for Sideways Markets (LOW)

### Problem
EMA/trend strategies lose money in sideways (range-bound) markets. Grid bots profit from oscillation.

### Solution
When ADX < 20 (no trend), switch to grid mode: place buy orders at intervals below price, sell orders above. Profit from bouncing.

### Implementation
```python
class GridBot:
    def __init__(self, grid_size=10, grid_spacing_pct=0.5):
        # grid_size: number of grid levels above and below
        # grid_spacing_pct: distance between levels

    def calculate_grid(self, center_price: float) -> dict:
        # Returns buy_levels and sell_levels

    def should_activate(self, adx: float) -> bool:
        return adx < 20  # No clear trend
```

### Test Plan
- test_grid_activates_on_low_adx
- test_grid_deactivates_on_high_adx
- test_grid_places_symmetric_orders
- test_grid_profits_in_range_market
- test_grid_backtest_sideways_vs_trending

### AC
- AC-01: Activates when ADX < 20
- AC-02: Grid levels calculated from current price + ATR
- AC-03: Backtest shows profit in sideways market periods

---

## Implementation Order

```
Sprint 1 (Critical):  #1 Backtest + #2 Trailing Stop
Sprint 2 (Filters):   #3 Multi-timeframe + #4 Correlation
Sprint 3 (ML):        #5 Auto-retrain + #9 Kelly criterion
Sprint 4 (Data):      #6 Twitter + #7 Whale tracking
Sprint 5 (UX):        #8 Dashboard + #10 Grid bot
```

Each sprint: TDD → implement → backtest → deploy → verify
