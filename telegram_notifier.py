#!/usr/bin/env python3
"""
Telegram Notifier — sends real messages via Telegram Bot API.
"""
import json
import logging
import requests

log = logging.getLogger("trading_bot.telegram")

# Load config from OpenClaw
_OPENCLAW_CONFIG = "/home/msbel/.openclaw/openclaw.json"
_BOT_TOKEN = ""
_CHAT_ID = ""

try:
    with open(_OPENCLAW_CONFIG) as f:
        _cfg = json.load(f)
    _tg = _cfg.get("channels", {}).get("telegram", {})
    _BOT_TOKEN = _tg.get("botToken", "")
    _allowed = _tg.get("allowFrom", [])
    _CHAT_ID = str(_allowed[0]) if _allowed else ""
except Exception:
    pass


class TelegramNotifier:
    def __init__(self, enabled=True, channel_id=None):
        self.enabled = enabled
        self.bot_token = _BOT_TOKEN
        self.chat_id = channel_id or _CHAT_ID
        self._api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else ""

    def send(self, message, priority="normal"):
        if not self.enabled or not self._api_url or not self.chat_id:
            log.warning("Telegram not configured, skipping: %s", message[:60])
            return

        emoji = {"low": "ℹ️", "normal": "📊", "high": "⚠️", "critical": "🚨"}.get(priority, "📊")
        text = f"{emoji} {message}"

        try:
            resp = requests.post(self._api_url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }, timeout=10)
            if resp.status_code == 200:
                log.info("Telegram sent: %s", text[:80])
            else:
                log.warning("Telegram failed (%d): %s", resp.status_code, resp.text[:100])
        except Exception as e:
            log.warning("Telegram error: %s", e)

    def notify_trade(self, action, price, amount, pnl=None, pnl_pct=None, reason=None):
        if action == "BUY":
            msg = f"🟢 BUY executed\nPrice: ${price:,.2f}\nAmount: {amount:.5f} (${amount*price:.2f})"
            self.send(msg, priority="normal")
        elif action == "SELL":
            emoji = "🟢" if pnl and pnl > 0 else "🔴"
            msg = f"{emoji} SELL executed\nPrice: ${price:,.2f}\nPnL: ${pnl:+.2f} ({pnl_pct:+.2f}%)"
            if reason:
                msg += f"\nReason: {reason}"
            self.send(msg, priority="high" if abs(pnl_pct or 0) > 3 else "normal")

    def notify_risk_event(self, event_type, data):
        if event_type == "MAX_DRAWDOWN":
            self.send(f"🚨 MAX DRAWDOWN\nBalance: ${data.get(balance,0):.2f}", priority="critical")
        elif event_type == "ERROR":
            self.send(f"❌ Bot Error: {data.get(message,unknown)}", priority="high")
        else:
            self.send(f"⚠️ Risk: {event_type} — {data}", priority="high")

    def notify_session_start(self, mode, balance, strategy):
        self.send(f"🤖 Trading bot started\nMode: {mode}\nBalance: ${balance:.2f}\nStrategy: {strategy}", priority="low")

    def notify_session_end(self, summary):
        self.send(f"⏹️ Session ended\nTrades: {summary.get(total_trades,0)}\nPnL: ${summary.get(total_pnl,0):+.2f}\nWin rate: {summary.get(win_rate,0):.1f}%", priority="low")

    def notify_error(self, error_msg, context=None):
        self.send(f"❌ {error_msg}" + (f"\n{context}" if context else ""), priority="high")
