#!/usr/bin/env bash
# =============================================================================
# Experiment #4: Cross-Task Transfer
# =============================================================================
#
# Hypothesis:
#   When two tasks share an underlying table (revenue_daily), running the first
#   task (revenue_by_category) stores queries in the middleware's ChromaDB
#   session memory.  A subsequent task (category_refund_rate) that queries the
#   same table should retrieve those stored queries as "similar queries" and
#   benefit from them -- reducing discovery overhead and potentially improving
#   accuracy or efficiency.
#
# What to look for:
#   In the phase-2 report (experiment_cross_task_phase2.json), inspect each
#   run's "similar_queries_received" counter.  It should be > 0 because the
#   middleware returns relevant queries stored during phase 1.  If it is 0,
#   cross-task transfer is not working (embedding similarity threshold may be
#   too high, or sessions are not persisting correctly).
#
# Usage:
#   Activate your Python venv first, then:
#     bash experiments/04_cross_task_transfer.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$HARNESS_DIR"

# ── Pre-flight: required environment variables ──────────────────────────────
missing=()
[ -z "${AGENT_LLM_API_URL:-}" ]  && missing+=("AGENT_LLM_API_URL")
[ -z "${AGENT_LLM_MODEL:-}" ]    && missing+=("AGENT_LLM_MODEL")

if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: Required environment variables are not set:"
    for var in "${missing[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Example:"
    echo "  export AGENT_LLM_API_URL=http://localhost:11434/v1"
    echo "  export AGENT_LLM_MODEL=gpt-4o"
    exit 1
fi

echo "============================================="
echo "Experiment #4: Cross-Task Transfer"
echo "============================================="
echo "AGENT_LLM_API_URL = $AGENT_LLM_API_URL"
echo "AGENT_LLM_MODEL   = $AGENT_LLM_MODEL"
echo ""

# ── Step 1: Clear ChromaDB sessions ─────────────────────────────────────────
echo "[Step 1/4] Clearing ChromaDB session collection..."
curl -s -X DELETE \
    "http://localhost:8000/api/v2/tenants/default_tenant/databases/default_database/collections/agent_sessions" \
    || echo "(collection may not exist yet -- that is fine)"

echo ""
echo "[Step 1/4] Restarting middleware to pick up clean state..."
docker compose restart middleware
echo "Waiting 5 seconds for middleware to become ready..."
sleep 5
echo ""

# ── Step 2: Phase 1 -- revenue_by_category ───────────────────────────────────
PHASE1_OUTPUT="reports/experiment_cross_task_phase1.json"
echo "[Step 2/4] Phase 1: running revenue_by_category (3 runs)..."
echo "  Output -> $PHASE1_OUTPUT"
python agent_bench.py \
    --mode middleware \
    --runs 3 \
    --tasks revenue_by_category \
    -o "$PHASE1_OUTPUT"
echo ""

# ── Step 3: Phase 2 -- category_refund_rate (should benefit from phase 1) ───
PHASE2_OUTPUT="reports/experiment_cross_task_phase2.json"
echo "[Step 3/4] Phase 2: running category_refund_rate (3 runs)..."
echo "  Output -> $PHASE2_OUTPUT"
python agent_bench.py \
    --mode middleware \
    --runs 3 \
    --tasks category_refund_rate \
    -o "$PHASE2_OUTPUT"
echo ""

# ── Step 4: Summary ─────────────────────────────────────────────────────────
echo "============================================="
echo "Experiment #4 complete."
echo "============================================="
echo ""
echo "Results:"
echo "  Phase 1 (revenue_by_category):  $PHASE1_OUTPUT"
echo "  Phase 2 (category_refund_rate): $PHASE2_OUTPUT"
echo ""
echo "Next step:"
echo "  Compare 'similar_queries_received' in phase 2.  It should be > 0"
echo "  because phase 1 stored queries against the same revenue_daily table."
echo ""
echo "  Quick check:"
echo "    python -c \"import json; [print(f'  run {r[\\\"run\\\"]}: similar_queries_received = {r.get(\\\"similar_queries_received\\\", \\\"N/A\\\")}') for r in json.load(open('$PHASE2_OUTPUT')).get('results',[])]\" "
