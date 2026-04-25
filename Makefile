.PHONY: up down reset load bench bench-unrestricted bench-analyst clean logs help

# -------------------------------------------------------------------
# SAO Benchmark Harness — Makefile
# -------------------------------------------------------------------

COMPOSE := docker compose
PYTHON  := python3

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --- Infrastructure ---

up: ## Start the ClickHouse cluster + Prometheus
	$(COMPOSE) up -d
	@echo "Waiting for cluster to be healthy..."
	@sleep 10
	@$(COMPOSE) exec clickhouse-s1r1 clickhouse-client --query "SELECT 'Cluster ready'" 2>/dev/null || \
		(echo "Waiting longer..." && sleep 15)
	@echo "Cluster is up."

down: ## Stop all containers
	$(COMPOSE) down

reset: down ## Destroy volumes and restart fresh
	$(COMPOSE) down -v
	$(MAKE) up

# --- Data ---

install-deps: ## Install Python dependencies
	pip install -r requirements.txt

load: ## Generate and load synthetic data (50k customers, 2M transactions)
	$(PYTHON) generate_data.py --host localhost --port 8123

load-small: ## Load a smaller dataset for quick testing (5k customers, 200k transactions)
	$(PYTHON) generate_data.py --host localhost --port 8123 \
		--customers 5000 --products 500 --transactions 200000

# --- Benchmarks ---

bench: ## Run all scenarios as 'agent' (guarded profile)
	$(PYTHON) harness.py --profile agent --output reports/agent_results.json

bench-unrestricted: ## Run all scenarios as 'agent_unrestricted' (no guardrails — worst case)
	$(PYTHON) harness.py --profile unrestricted --output reports/unrestricted_results.json

bench-analyst: ## Run all scenarios as 'analyst' (read-only moderate limits)
	$(PYTHON) harness.py --profile analyst --output reports/analyst_results.json

bench-all: bench bench-unrestricted bench-analyst ## Run benchmarks for all profiles
	@echo "All profiles benchmarked. Compare reports in reports/"

bench-scenario: ## Run a specific scenario (usage: make bench-scenario SCENARIO=data_exfiltration)
	$(PYTHON) harness.py --profile agent --scenarios $(SCENARIO) \
		--output reports/$(SCENARIO)_results.json

# --- Compare profiles ---

compare: bench-all ## Run all profiles and print a comparison
	@echo ""
	@echo "===== PROFILE COMPARISON ====="
	@for f in reports/*_results.json; do \
		echo "--- $$(basename $$f) ---"; \
		$(PYTHON) -c "import json; r=json.load(open('$$f')); a=r['aggregate']; \
			print(f\"  Probes: {a['total_probes']}  Succeeded: {a['total_succeeded']}  Blocked: {a['total_blocked']}  Success rate: {a['overall_success_rate']*100:.1f}%\")"; \
	done

# --- Utilities ---

logs: ## Tail ClickHouse logs
	$(COMPOSE) logs -f clickhouse-s1r1

query-log: ## Show recent queries from the audit perspective
	$(COMPOSE) exec clickhouse-s1r1 clickhouse-client --query \
		"SELECT user, type, query, read_rows, memory_usage \
		 FROM system.query_log \
		 WHERE event_time > now() - INTERVAL 10 MINUTE \
		 ORDER BY event_time DESC LIMIT 30 FORMAT PrettyCompact"

clean: ## Remove report files
	rm -rf reports/*.json
