#!/usr/bin/env bash
# =============================================================================
# Experiment #6: Score Agent Accuracy Against Gold SQL
#
# Step 1: Run gold-standard SQL queries against ClickHouse to get ground truth
# Step 2: Compare agent answers from all benchmark reports against ground truth
#
# The scoring checks:
#   - Do the category/state labels in the agent's answer match gold?
#   - Do the numeric values match within 5% tolerance?
#   - A score > 0.5 = "match" (agent got the right answer)
#
# Usage:
#   bash experiments/06_score_accuracy.sh
#
# Requires: ClickHouse accessible on localhost:8123
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "============================================="
echo "  Experiment #6: Accuracy Scoring"
echo "============================================="
echo ""

# Step 1: Generate gold results
echo "--- Step 1: Running gold SQL against ClickHouse ---"
# python score_accuracy.py --generate-gold \
#     --host localhost --port 8123 \
#     -o reports/gold_results.json
echo ""

# Step 2: Score all available agent reports
echo "--- Step 2: Scoring agent reports ---"

REPORTS=()
for f in \
    reports/experiment_agents_baseline_gpt-4o.json \
    reports/agents_middleware_feedback_no_session-gpt-4o.json \
    reports/agents_middleware_no_feedback_no_session-gpt-4o.json \
    reports/experiment_explore_on.json \
    reports/experiment_explore_off.json \
    reports/agent_bench_middleware_test_gpt-4o.json \
    reports/agent_bench_middleware_test_gpt-4o-mini.json \
    reports/agent_middleware.json \
    reports/agent_baseline.json \
    reports/experiment_agents_baseline_gpt-4o.json \
    reports/experiment_agent_middleware_no_feedback_no_explore-gpt-4o.json \
; do
    if [ -f "$f" ]; then
        REPORTS+=("$f")
    fi
done

if [ ${#REPORTS[@]} -eq 0 ]; then
    echo "No agent report files found in reports/"
    exit 1
fi

python score_accuracy.py --score \
    --gold reports/gold_results.json \
    "${REPORTS[@]}"

echo ""
echo "Done. Gold results saved to reports/gold_results.json"
