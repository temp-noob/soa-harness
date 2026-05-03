#!/usr/bin/env python3
"""
Score agent benchmark reports against gold-standard SQL results.

Step 1: Run gold SQL queries against ClickHouse, save ground truth.
Step 2: Compare agent answers from benchmark reports against ground truth.

Usage:
    # Step 1: Generate gold results (requires ClickHouse access)
    python score_accuracy.py --generate-gold \
        --host localhost --port 8123 \
        -o reports/gold_results.json

    # Step 2: Score one or more agent reports against gold
    python score_accuracy.py --score \
        --gold reports/gold_results.json \
        reports/experiment_explore_on.json \
        reports/experiment_agents_baseline_gpt-4o.json

    # Both steps at once
    python score_accuracy.py --generate-gold --score \
        --host localhost --port 8123 \
        -o reports/gold_results.json \
        reports/experiment_explore_on.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

from tasks import TASKS


# ── Step 1: Generate gold results ────────────────────────────────────


def generate_gold(host: str, port: int, output: str) -> dict:
    gold: dict[str, dict] = {}

    for task in TASKS:
        sql = task.gold_sql + " FORMAT TabSeparatedWithNames"
        try:
            resp = requests.post(
                f"http://{host}:{port}/",
                params={"user": "default"},
                data=sql,
                timeout=30,
            )
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                headers = lines[0].split("\t") if lines else []
                rows = [line.split("\t") for line in lines[1:]] if len(lines) > 1 else []
                gold[task.id] = {
                    "task_id": task.id,
                    "question": task.question,
                    "gold_sql": task.gold_sql,
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows),
                }
                print(f"  {task.id}: {len(rows)} rows")
            else:
                print(f"  {task.id}: ERROR {resp.status_code} — {resp.text[:100]}")
                gold[task.id] = {"task_id": task.id, "error": resp.text[:200]}
        except Exception as e:
            print(f"  {task.id}: FAILED — {e}")
            gold[task.id] = {"task_id": task.id, "error": str(e)}

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(gold, f, indent=2)
    print(f"\n  Gold results written to: {output}")
    return gold


# ── Step 2: Score agent reports ──────────────────────────────────────


def _extract_numbers(text: str) -> list[float]:
    """Extract all numbers from a text string."""
    return [float(x.replace(",", "")) for x in re.findall(r"[\d,]+\.?\d*", text)]


def _extract_top_values(text: str, n: int = 3) -> list[str]:
    """Extract the first N recognizable data values (numbers or labels)."""
    numbers = re.findall(r"\$?[\d,]+\.?\d*%?", text)
    return numbers[:n]


def _check_answer_against_gold(
    agent_answer: str,
    gold: dict,
    task_id: str,
) -> dict:
    """Compare an agent's final answer against gold results.

    Returns a scoring dict with match details.
    """
    if not agent_answer or not agent_answer.strip():
        return {"match": False, "reason": "empty_answer", "score": 0.0}

    if "policy restriction" in agent_answer.lower() or \
       "unable to" in agent_answer.lower() or \
       "access restriction" in agent_answer.lower():
        return {"match": False, "reason": "policy_blocked", "score": 0.0}

    gold_rows = gold.get("rows", [])
    if not gold_rows:
        return {"match": False, "reason": "no_gold_data", "score": 0.0}

    # Extract key values from gold
    gold_values = []
    for row in gold_rows:
        for cell in row:
            try:
                gold_values.append(round(float(cell), 2))
            except (ValueError, TypeError):
                gold_values.append(cell.strip().lower())

    # Extract numbers from agent answer
    agent_numbers = _extract_numbers(agent_answer)
    agent_text_lower = agent_answer.lower()

    # Check: do the gold row labels appear in the agent answer?
    label_matches = 0
    for row in gold_rows[:5]:
        label = row[0].strip().lower()
        if label in agent_text_lower:
            label_matches += 1

    # Check: do the gold numeric values appear in the agent answer?
    value_matches = 0
    for gv in gold_values:
        if isinstance(gv, float):
            for av in agent_numbers:
                if abs(av - gv) < max(abs(gv) * 0.05, 1.0):  # 5% tolerance
                    value_matches += 1
                    break

    total_gold_labels = min(len(gold_rows), 5)
    total_gold_values = sum(1 for v in gold_values if isinstance(v, float))

    label_score = label_matches / max(total_gold_labels, 1)
    value_score = value_matches / max(total_gold_values, 1)
    overall_score = (label_score + value_score) / 2

    return {
        "match": overall_score > 0.5,
        "reason": "value_comparison",
        "score": round(overall_score, 3),
        "label_matches": f"{label_matches}/{total_gold_labels}",
        "value_matches": f"{value_matches}/{total_gold_values}",
    }


def score_report(report_path: str, gold: dict) -> dict:
    with open(report_path) as f:
        report = json.load(f)

    mode = report["config"].get("mode", "unknown")
    model = report["config"].get("model", "unknown")
    results_by_task: dict[str, list] = {}

    for r in report["results"]:
        tid = r["task_id"]
        if tid not in results_by_task:
            results_by_task[tid] = []

        gold_data = gold.get(tid, {})
        scoring = _check_answer_against_gold(r.get("final_answer", ""), gold_data, tid)
        results_by_task[tid].append({
            "run": r["agent_run"],
            "queries": r["queries_executed"],
            "score": scoring["score"],
            "match": scoring["match"],
            "reason": scoring["reason"],
            "label_matches": scoring.get("label_matches", ""),
            "value_matches": scoring.get("value_matches", ""),
        })

    return {
        "report": Path(report_path).name,
        "mode": mode,
        "model": model,
        "tasks": results_by_task,
    }


def print_score_table(scored_reports: list[dict]) -> None:
    print()
    print("=" * 90)
    print("  ACCURACY SCORING: Agent Answers vs Gold SQL")
    print("=" * 90)

    all_task_ids = []
    for sr in scored_reports:
        for tid in sr["tasks"]:
            if tid not in all_task_ids:
                all_task_ids.append(tid)

    for tid in all_task_ids:
        print(f"\n  Task: {tid}")
        print(f"  {'Report':<50} {'Run':>4} {'Queries':>8} {'Score':>7} {'Match':>6} {'Labels':>8} {'Values':>8}")
        print(f"  {'':─<50} {'':─>4} {'':─>8} {'':─>7} {'':─>6} {'':─>8} {'':─>8}")

        for sr in scored_reports:
            runs = sr["tasks"].get(tid, [])
            for run_data in runs:
                match_str = "YES" if run_data["match"] else "no"
                print(
                    f"  {sr['report']:<50} {run_data['run']:>4} "
                    f"{run_data['queries']:>8} {run_data['score']:>7.3f} "
                    f"{match_str:>6} {run_data.get('label_matches',''):>8} "
                    f"{run_data.get('value_matches',''):>8}"
                )

    # Summary
    print(f"\n{'=' * 90}")
    print(f"  {'Report':<50} {'Avg Score':>10} {'Match Rate':>12} {'Avg Queries':>12}")
    print(f"  {'':─<50} {'':─>10} {'':─>12} {'':─>12}")

    for sr in scored_reports:
        all_scores = []
        all_matches = []
        all_queries = []
        for runs in sr["tasks"].values():
            for run_data in runs:
                all_scores.append(run_data["score"])
                all_matches.append(1 if run_data["match"] else 0)
                all_queries.append(run_data["queries"])

        avg_score = sum(all_scores) / max(len(all_scores), 1)
        match_rate = sum(all_matches) / max(len(all_matches), 1)
        avg_queries = sum(all_queries) / max(len(all_queries), 1)
        print(
            f"  {sr['report']:<50} {avg_score:>10.3f} "
            f"{match_rate * 100:>11.1f}% {avg_queries:>12.1f}"
        )
    print(f"{'=' * 90}\n")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Score agent benchmarks against gold SQL")
    parser.add_argument("--generate-gold", action="store_true",
                        help="Run gold SQL queries against ClickHouse")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--score", action="store_true",
                        help="Score agent reports against gold results")
    parser.add_argument("--gold", default="reports/gold_results.json",
                        help="Path to gold results JSON")
    parser.add_argument("-o", "--output", default="reports/gold_results.json")
    parser.add_argument("reports", nargs="*", help="Agent report JSON files to score")
    args = parser.parse_args()

    if args.generate_gold:
        print("Generating gold results from ClickHouse...")
        generate_gold(args.host, args.port, args.output)

    if args.score:
        if not Path(args.gold).exists():
            print(f"Error: gold results not found at {args.gold}")
            print("  Run with --generate-gold first")
            sys.exit(1)

        with open(args.gold) as f:
            gold = json.load(f)

        if not args.reports:
            print("Error: provide one or more agent report files to score")
            sys.exit(1)

        scored = []
        for report_path in args.reports:
            scored.append(score_report(report_path, gold))

        print_score_table(scored)


if __name__ == "__main__":
    main()
