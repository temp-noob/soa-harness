.PHONY: up down reset load bench bench-unrestricted bench-analyst clean logs help middleware-build middleware-logs middleware-test

# -------------------------------------------------------------------
# SAO Benchmark Harness — Makefile
# -------------------------------------------------------------------

COMPOSE := docker compose
VENV := python -m venv .venv
PYTHON  := $(VENV)/bin/python

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
	$(PYTHON) -m pip install -r requirements.txt

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

# --- Middleware ---

middleware-build: ## Build the middleware container
	$(COMPOSE) build middleware

middleware-logs: ## Tail middleware logs
	$(COMPOSE) logs -f middleware

middleware-test: ## Test middleware health and explore endpoint
	@echo "=== Health Check ==="
	@curl -s http://localhost:8080/health && echo ""
	@echo ""
	@echo "=== Explore Endpoint (as agent) ==="
	@curl -s "http://localhost:8080/explore?agent_id=agent" && echo ""
	@echo ""
	@echo "=== Access Control: DENY PII row access ==="
	@curl -s -X POST http://localhost:8080/query \
		-H "Content-Type: application/json" \
		-d '{"sql": "SELECT email, full_name FROM sao.customers LIMIT 10", "agent_id": "agent"}' \
		&& echo ""
	@echo ""
	@echo "=== Access Control: ALLOW aggregation ==="
	@curl -s -X POST http://localhost:8080/query \
		-H "Content-Type: application/json" \
		-d '{"sql": "SELECT tier, COUNT(*) FROM sao.customers_distributed GROUP BY tier", "agent_id": "agent"}' \
		&& echo ""

bench-middleware: ## Run all scenarios through the middleware (port 8080)
	$(PYTHON) harness.py --profile agent --host localhost --port 8080 \
		--output reports/middleware_agent_results.json

clean: ## Remove report files
	rm -rf reports/*.json
