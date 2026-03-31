# Alcyone Trading Bot — Makefile
# Usage: make install && make train && make run

VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

.PHONY: install train backtest run stop status logs dashboard clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	python3 -m venv $(VENV)
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install pandas scikit-learn xgboost transformers feedparser requests cryptography
	@echo "\n✅ Dependencies installed"

download: ## Download 8 months historical data
	$(PYTHON) ml/download_historical.py
	@echo "\n✅ Historical data downloaded"

train: ## Train all ML models (XGBoost + GRU)
	$(PYTHON) ml/xgboost_model.py
	$(PYTHON) ml/gru_model.py
	@echo "\n✅ Models trained"

backtest: ## Run backtest on all 7 coins
	$(PYTHON) backtester.py

test: ## Run all component tests
	$(PYTHON) filters.py
	$(PYTHON) data_sources.py
	@echo "\n✅ All tests passed"

run: ## Start the trading bot (foreground)
	$(PYTHON) bot.py

service-install: ## Install as systemd user service (24/7)
	cp trading-bot.service ~/.config/systemd/user/
	cp copilot-api.service ~/.config/systemd/user/ 2>/dev/null || true
	systemctl --user daemon-reload
	systemctl --user enable trading-bot.service
	systemctl --user enable copilot-api.service 2>/dev/null || true
	@echo "\n✅ Services installed. Run: make start"

start: ## Start all services
	systemctl --user start copilot-api.service 2>/dev/null || true
	sleep 5
	systemctl --user start trading-bot.service
	@echo "\n✅ Bot started. Dashboard: http://localhost:8085"

stop: ## Stop the bot
	systemctl --user stop trading-bot.service
	@echo "\n⏹ Bot stopped"

restart: ## Restart the bot
	systemctl --user restart trading-bot.service
	@echo "\n🔄 Bot restarted"

status: ## Show bot status
	@echo "=== Services ==="
	@systemctl --user status trading-bot.service 2>&1 | head -5
	@systemctl --user status copilot-api.service 2>&1 | head -3
	@echo "\n=== Dashboard ==="
	@curl -s http://localhost:8085/api/state 2>/dev/null | python3 -c '\
		import json,sys; d=json.load(sys.stdin); \
		print(f"PnL: $${d.get(\"total_pnl\",0):+.2f}"); \
		print(f"Uptime: {d.get(\"uptime\",\"?\")}"); \
		open_count=sum(1 for v in d.get("positions",{}).values() if isinstance(v,dict) and v.get("amount",0)>0); \
		print(f"Positions: {open_count}/7")' 2>/dev/null || echo "Dashboard not available"

logs: ## Tail bot logs
	tail -f logs/bot_v2.log

dashboard: ## Open dashboard in browser
	@echo "Dashboard: http://localhost:8085"
	@echo "JSON API: http://localhost:8085/api/state"

clean: ## Remove logs and cached models
	rm -f logs/*.log logs/*.jsonl
	rm -f ml/models/*.pt ml/models/*.pkl
	@echo "\n🧹 Cleaned"

full-setup: install download train service-install start ## Full setup from scratch
	@echo "\n🚀 Full setup complete! Dashboard: http://localhost:8085"
