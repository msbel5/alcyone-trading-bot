# Trading Bot Upgrade Plan
# Alcyone bu plani heartbeat lerinde adim adim implement etsin.
# Her adimi bitirince commit + push + Telegram bildir.

## Phase 1: Multi-Asset (7 coin minimum)
Sadece BTC degil, 7 farkli coin trade et:
- BTC/USDT (ana, en likit)
- ETH/USDT (ikinci buyuk)
- SOL/USDT (hizli, volatil)
- BNB/USDT (Binance native)
- XRP/USDT (yuksek hacim)
- DOGE/USDT (meme, volatil, kucuk fiyat)
- AVAX/USDT (DeFi)

Her coin icin ayri pozisyon takibi. Balance $100 / 7 = ~$14 per coin.
Position size: %15 of coin allocation = ~$2 per trade.

Dosya: multi_asset_trader.py
- Her coin icin ayri strategy instance
- Her coin icin ayri risk tracker
- Toplam portfolio PnL takibi

## Phase 2: RSI + Volume Filtresi
EMA crossover tek basina zayif. Ekle:
1. RSI(14) filtresi:
   - BUY sadece RSI 30-50 arasindaysa (oversold→recovery)
   - SELL sadece RSI 60-80 arasindaysa (overbought)
   - RSI < 25: STRONG BUY
   - RSI > 75: STRONG SELL
2. Volume confirmation:
   - Son 5 candle ortalama volume hesapla
   - Sinyal ancak volume > ortalama * 1.2 ise gecerli
   - Dusuk hacimde sinyal IGNORE

Dosya: strategies/enhanced_ema.py

## Phase 3: ATR-Based Dynamic SL/TP
Sabit %2/%5 yerine volatiliteye gore:
- ATR(14) hesapla
- Stop Loss = entry - (ATR * 1.5)
- Take Profit = entry + (ATR * 2.5)
- Trailing stop: fiyat yukselince SL da yukselsin

Dosya: risk_manager.py guncelle

## Phase 4: Fear & Greed Index
- API: https://api.alternative.me/fng/ (ucretsiz)
- Extreme Fear (0-25): agresif BUY sinyali
- Fear (25-45): normal BUY ok
- Neutral (45-55): sadece guclu sinyallerde trade
- Greed (55-75): sadece SELL sinyalleri
- Extreme Greed (75-100): agresif SELL, yeni BUY YASAK

Dosya: indicators/fear_greed.py

## Phase 5: Haber/Sentiment (ileri)
- CoinGecko RSS feed parse et
- Basit keyword sentiment: "hack","crash","ban" = negatif, "ETF","adoption","partnership" = pozitif
- Buyuk negatif haber → tum pozisyonlari kapat
- Buyuk pozitif haber → agresif BUY

Dosya: indicators/news_sentiment.py

## KURALLAR
- Her phase bitince: test yaz, calistir, commit, push
- Telegram ile Mami ye bildir
- Testnet te en az 24 saat calistir her phase sonrasi
- Gercek paraya gecmeden ONCE Mami onay vermeli
- Her phase basinda mevcut kodu oku ve anla
