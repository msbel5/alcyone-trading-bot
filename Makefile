# Alcyone Trading Bot v4 — Makefile
# 9-layer strategy | 46 ML features | 6 scientific modules | Pi 5

VENV := .venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

.PHONY: help install download train train-v3 retrain backtest test test-models run start stop restart status logs dashboard clean full-setup

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install pandas numpy scikit-learn xgboost lightgbm scipy
	$(PIP) install transformers feedparser requests cryptography

download: ## Download 2 years historical data (7 coins)
	$(PYTHON) ml/download_historical.py

train: train-v3 ## Train all models (default: v3 scientific)

train-v3: ## Train v3: CPCV + CNN-LSTM + Stacked (~8h on Pi)
	$(PYTHON) ml/ml_v3.py

retrain: ## Run daily retrain (CPCV + PBO gating)
	$(PYTHON) daily_retrain_v3.py

backtest: ## Run v3 vs v4 backtest comparison
	$(PYTHON) backtest_v3.py

test: ## Test all 6 modules + v4 pipeline
	$(PYTHON) ml/indicators_advanced.py
	$(PYTHON) ml/statistical_models.py
	$(PYTHON) ml/candlestick_patterns.py
	$(PYTHON) ml/risk_metrics.py
	$(PYTHON) ml/volatility_engine.py
	$(PYTHON) ml/execution_engine.py
	$(PYTHON) ml/data_pipeline_v4.py

test-models: ## Test ML predictions for all 7 coins
	$(PYTHON) test_all_models.py

run: ## Start bot (foreground)
	$(PYTHON) bot.py

start: ## Start bot + copilot-api services
	systemctl --user start copilot-api.service 2>/dev/null || true
	sleep 3
	systemctl --user start trading-bot.service

stop: ## Stop bot service
	systemctl --user stop trading-bot.service

restart: ## Restart bot service
	systemctl --user restart trading-bot.service

status: ## Show bot status
	@systemctl --user status trading-bot.service 2>&1 | head -8

logs: ## Tail live bot logs
	tail -f logs/bot_v2.log

dashboard: ## Show dashboard URL
	@echo "Dashboard: http://alcyone:8085"
	@echo "JSON API:  http://alcyone:8085/api/state"

clean: ## Remove logs and models
	rm -f logs/*.log logs/*.json
	rm -f ml/models/*.pt ml/models/*.pkl

full-setup: install download train start ## Full setup from scratch
