#!/usr/bin/env bash
# Experiment #5: Explore Endpoint Ablation
#
# Compares middleware mode WITH the /explore endpoint (schema context)
# against middleware mode WITHOUT it, to isolate the value of the explore
# endpoint vs. session learning alone.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "========================================"
echo "  Experiment 5: Explore Endpoint Ablation"
echo "========================================"

# ── Phase 1: Middleware WITH explore ──────────────────────────────────

echo ""
echo "--- Clearing ChromaDB sessions ---"
curl -s -X DELETE "http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database/collections/agent_sessions" > /dev/null 2>&1 || true
docker compose restart middleware
echo "Waiting for middleware to restart..."
sleep 5

echo ""
echo "--- Running middleware WITH explore (3 runs) ---"
python agent_bench.py --mode middleware --runs 3 -o reports/experiment_explore_on.json

# ── Phase 2: Middleware WITHOUT explore ───────────────────────────────

echo ""
echo "--- Clearing ChromaDB sessions ---"
curl -s -X DELETE "http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database/collections/agent_sessions" > /dev/null 2>&1 || true
docker compose restart middleware
echo "Waiting for middleware to restart..."
sleep 5

echo ""
echo "--- Running middleware WITHOUT explore (3 runs) ---"
python agent_bench.py --mode middleware --no-explore --runs 3 -o reports/experiment_explore_off.json

# ── Phase 3: Compare ─────────────────────────────────────────────────

echo ""
echo "--- Comparison ---"
python agent_bench.py --compare reports/experiment_explore_off.json reports/experiment_explore_on.json

echo ""
echo "Done. Reports saved in reports/"
