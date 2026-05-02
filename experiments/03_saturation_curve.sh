#!/usr/bin/env bash
# =============================================================================
# Experiment #3: Session Learning Saturation Curve
#
# Runs 10 agents sequentially on 3 tasks (revenue_by_category,
# category_refund_rate, product_category_aov) with feedback enabled.
#
# What to look for:
#   - similar_queries_received should increase across runs then plateau
#   - queries_executed should decrease then stabilize
#   - Plot: agent_run vs queries_executed and similar_queries_received
#   - The flattening point shows when session learning has converged
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${AGENT_LLM_API_URL:-}" ]; then
    echo "Error: AGENT_LLM_API_URL is not set."
    echo "  export AGENT_LLM_API_URL=https://api.openai.com/v1"
    exit 1
fi

echo "=== Experiment #3: Session Learning Saturation Curve ==="
echo "  Model: ${AGENT_LLM_MODEL:-gpt-4o-mini}"
echo "  Runs:  10 agents per task"
echo ""

# Step 1: Clean sessions
echo "Clearing ChromaDB sessions..."
curl -s -X DELETE "http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database/collections/agent_sessions" > /dev/null 2>&1 || true
docker compose restart middleware > /dev/null 2>&1
sleep 5
echo "Sessions cleared."
echo ""

# Step 2: Run 10 agents with feedback
echo "Running 10 agents per task..."
python agent_bench.py \
    --mode middleware \
    --runs 10 \
    --tasks revenue_by_category,category_refund_rate,product_category_aov \
    -o reports/experiment_saturation_curve.json

echo ""
echo "=== Done ==="
echo "Report: reports/experiment_saturation_curve.json"
echo ""
echo "To analyze: plot agent_run vs queries_executed and similar_queries_received"
echo "per task. The curve should show decreasing queries and increasing similar"
echo "queries, flattening after 4-6 runs."
