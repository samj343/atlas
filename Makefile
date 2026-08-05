# Atlas - development tasks
#
# `make help` lists everything.

.DEFAULT_GOAL := help
PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PROVIDER ?= synthetic
START ?= 2008-01-01
END ?= 2018-12-31

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- setup ------------------------------------------------------------------

.PHONY: venv
venv: ## Create the virtual environment
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: venv ## Install Atlas with its development extras
	$(BIN)/pip install -e ".[dev,data,dashboard]"

.PHONY: install-all
install-all: venv ## Install everything, including the broker client
	$(BIN)/pip install -e ".[all,dev]"

.PHONY: hooks
hooks: ## Install the pre-commit hooks
	$(BIN)/pre-commit install

# --- quality ----------------------------------------------------------------

.PHONY: lint
lint: ## Run ruff
	$(BIN)/ruff check src tests scripts

.PHONY: format
format: ## Auto-format with ruff
	$(BIN)/ruff format src tests scripts
	$(BIN)/ruff check --fix src tests scripts

.PHONY: typecheck
typecheck: ## Run mypy
	$(BIN)/mypy src/atlas

.PHONY: test
test: ## Run the fast test suite
	$(BIN)/pytest -q -m "not slow"

.PHONY: test-all
test-all: ## Run every test, including the slow ones
	$(BIN)/pytest -q

.PHONY: coverage
coverage: ## Run the tests with a coverage report
	$(BIN)/pytest --cov=atlas --cov-report=term-missing --cov-report=xml -m "not slow"

.PHONY: check
check: lint typecheck test ## Lint, type-check and test

# --- research ---------------------------------------------------------------

.PHONY: data
data: ## Download and cache market data
	$(BIN)/atlas download-data --provider $(PROVIDER)

.PHONY: validate
validate: ## Validate the cached market data
	$(BIN)/atlas validate-data --provider $(PROVIDER)

.PHONY: backtest
backtest: ## Run a backtest with benchmarks, stress tests and a report
	$(BIN)/atlas backtest --provider $(PROVIDER) --start $(START) --end $(END)

.PHONY: walk-forward
walk-forward: ## Run walk-forward validation (slow)
	$(BIN)/atlas walk-forward --provider $(PROVIDER) --start $(START) --end $(END)

.PHONY: stability
stability: ## Run the parameter-stability sweep (slow)
	$(BIN)/atlas parameter-stability --provider $(PROVIDER) --start $(START) --end $(END)

.PHONY: experiment
experiment: ## Run the full baseline experiment end to end
	$(BIN)/python scripts/run_baseline_experiment.py --provider $(PROVIDER)

.PHONY: dashboard
dashboard: ## Launch the Streamlit dashboard
	$(BIN)/atlas dashboard

.PHONY: order-preview
order-preview: ## Show the orders Atlas would place today (nothing is sent)
	$(BIN)/atlas order-preview --provider $(PROVIDER) --mock

.PHONY: broker-check
broker-check: ## Connect to the broker and run the safety checks
	$(BIN)/atlas broker-check

# --- docker -----------------------------------------------------------------

.PHONY: docker-build
docker-build: ## Build the Docker image
	docker build -t atlas-trading-system:latest .

.PHONY: docker-test
docker-test: ## Run the test suite inside Docker
	docker run --rm atlas-trading-system:latest pytest -q -m "not slow"

.PHONY: docker-dashboard
docker-dashboard: ## Run the dashboard in Docker on port 8501
	docker compose up dashboard

# --- housekeeping -----------------------------------------------------------

.PHONY: clean
clean: ## Remove caches and build artefacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

.PHONY: clean-data
clean-data: ## Remove cached market data and the local database
	rm -rf data/raw/* data/processed/* data/database/*

.PHONY: clean-reports
clean-reports: ## Remove generated reports
	rm -rf reports/backtests/*
