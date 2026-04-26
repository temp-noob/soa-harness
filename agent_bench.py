#!/usr/bin/env python3
"""
Agent Efficiency Benchmark — LLM Agent vs. Middleware-Assisted Agent

Measures how many queries an LLM agent needs to solve analytical tasks,
comparing baseline (raw ClickHouse + information_schema) against
middleware-assisted (explore endpoint + session learning).

Usage:
    # Baseline mode — agent talks directly to ClickHouse
    python agent_bench.py --mode baseline --runs 3

    # Middleware mode — agent uses /explore and /session APIs
    python agent_bench.py --mode middleware --runs 3

    # Compare two result files
    python agent_bench.py --compare reports/agent_baseline.json reports/agent_middleware.json

Environment variables:
    AGENT_LLM_API_URL   OpenAI-compatible base URL (e.g. http://localhost:11434/v1)
    AGENT_LLM_API_KEY   API key (optional for Ollama)
    AGENT_LLM_MODEL     Model name (default: gpt-4o-mini)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx
import requests

from tasks import TASKS, AnalyticalTask

CLICKHOUSE_URL = "http://localhost:8123"
MIDDLEWARE_URL = "http://localhost:8080"

MAX_TURNS = 15


# ── LLM client ──────────────────────────────────────────────────────


def _llm_chat(messages: list[dict], api_url: str, api_key: str | None, model: str) -> tuple[str, int]:
    """Call an OpenAI-compatible chat completions endpoint.

    Returns (assistant_message, tokens_used).
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = httpx.post(
        f"{api_url.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.0,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("total_tokens", 0)
    return content, tokens


# ── Query execution ──────────────────────────────────────────────────


def _execute_sql_direct(sql: str, user: str = "agent", password: str = "agent_pass") -> tuple[str, bool]:
    """Execute SQL directly against ClickHouse. Returns (result_text, success)."""
    try:
        resp = requests.post(
            CLICKHOUSE_URL,
            params={"user": user, "password": password},
            data=sql + " FORMAT TabSeparatedWithNames",
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.text.strip(), True
        return f"Error: {resp.text.strip()}", False
    except Exception as e:
        return f"Error: {e}", False


def _execute_sql_middleware(sql: str, session_id: str) -> tuple[str, bool]:
    """Execute SQL through the middleware session. Returns (result_text, success)."""
    try:
        resp = requests.post(
            f"{MIDDLEWARE_URL}/session/{session_id}/query",
            json={"sql": sql + " FORMAT TabSeparatedWithNames"},
            timeout=30,
        )
        data = resp.json()
        if data.get("allowed") and data.get("status_code") == 200:
            return data["body"].strip(), True
        reason = data.get("reason", data.get("body", "Unknown error"))
        return f"Blocked: {reason}", False
    except Exception as e:
        return f"Error: {e}", False


# ── Result tracking ──────────────────────────────────────────────────


@dataclass
class AgentResult:
    task_id: str
    mode: str
    model: str
    agent_run: int
    queries_executed: int = 0
    exploration_queries: int = 0
    answer_queries: int = 0
    correct: bool = False
    total_latency_ms: float = 0.0
    llm_tokens_used: int = 0
    similar_queries_received: int = 0
    similar_queries_used: int = 0
    final_answer: str = ""
    error: str | None = None


# ── System prompts ───────────────────────────────────────────────────


BASELINE_SYSTEM = """You are a data analyst agent with access to a ClickHouse database.
You can execute SQL queries to answer analytical questions.

To execute a query, respond with exactly:
```sql
YOUR SQL QUERY HERE
```

To provide your final answer, respond with:
ANSWER: your answer here

You may execute multiple queries to explore the schema and find the answer.
Start by discovering the schema using information_schema or system.columns.
The database is named 'sao'."""

MIDDLEWARE_SYSTEM = """You are a data analyst agent with access to a ClickHouse database via a middleware.
You can execute SQL queries to answer analytical questions.

{schema_context}

{similar_queries_context}

To execute a query, respond with exactly:
```sql
YOUR SQL QUERY HERE
```

To provide your final answer, respond with:
ANSWER: your answer here

You may execute multiple queries. Use the schema information and sample queries above
to formulate efficient queries. The database is named 'sao'."""


def _build_schema_context(explore_data: dict) -> str:
    lines = ["## Available Schema\n"]
    for db in explore_data.get("databases", []):
        for table in db.get("tables", []):
            cols = ", ".join(
                f"{c['name']} ({c['type']}, {c.get('access', 'full')})"
                for c in table.get("columns", [])
            )
            lines.append(f"**{table['name']}** ({table.get('access', 'full')}): {cols}")
            if table.get("sample_queries"):
                for sq in table["sample_queries"][:2]:
                    sql_text = sq["sql"] if isinstance(sq, dict) else sq
                    lines.append(f"  Example: `{sql_text}`")
            if table.get("recommended_joins"):
                for j in table["recommended_joins"]:
                    lines.append(f"  Join → {j['target']} ON {j['key']} ({j['description']})")
            lines.append("")
    return "\n".join(lines)


def _build_similar_queries_context(similar: list[dict]) -> str:
    if not similar:
        return ""
    lines = ["## Queries from Similar Past Sessions\n"]
    for sq in similar[:5]:
        lines.append(f"- `{sq['sql']}` (notes: {sq.get('notes', 'none')})")
    return "\n".join(lines)


# ── Agent runners ────────────────────────────────────────────────────


def _extract_sql(text: str) -> str | None:
    match = re.search(r"```sql\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_answer(text: str) -> str | None:
    match = re.search(r"ANSWER:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def run_baseline_agent(
    task: AnalyticalTask,
    agent_run: int,
    api_url: str,
    api_key: str | None,
    model: str,
) -> AgentResult:
    result = AgentResult(
        task_id=task.id, mode="baseline", model=model, agent_run=agent_run,
    )
    t0 = time.monotonic()
    messages = [
        {"role": "system", "content": BASELINE_SYSTEM},
        {"role": "user", "content": task.question},
    ]

    for turn in range(MAX_TURNS):
        try:
            reply, tokens = _llm_chat(messages, api_url, api_key, model)
        except Exception as e:
            result.error = f"LLM error: {e}"
            break
        result.llm_tokens_used += tokens
        messages.append({"role": "assistant", "content": reply})

        answer = _extract_answer(reply)
        if answer:
            result.final_answer = answer
            break

        sql = _extract_sql(reply)
        if not sql:
            messages.append({"role": "user", "content": "Please provide a SQL query in ```sql ... ``` or your final ANSWER."})
            continue

        result.queries_executed += 1
        if any(kw in sql.lower() for kw in ["information_schema", "system.columns", "system.tables"]):
            result.exploration_queries += 1
        else:
            result.answer_queries += 1

        query_result, success = _execute_sql_direct(sql)
        if success:
            messages.append({"role": "user", "content": f"Query result:\n{query_result}"})
        else:
            messages.append({"role": "user", "content": f"Query failed: {query_result}"})

    result.total_latency_ms = (time.monotonic() - t0) * 1000
    return result


def run_middleware_agent(
    task: AnalyticalTask,
    agent_run: int,
    api_url: str,
    api_key: str | None,
    model: str,
) -> AgentResult:
    result = AgentResult(
        task_id=task.id, mode="middleware", model=model, agent_run=agent_run,
    )
    t0 = time.monotonic()

    # Step 1: Get schema from /explore
    try:
        explore_resp = requests.get(
            f"{MIDDLEWARE_URL}/explore", params={"agent_id": "agent"}, timeout=10,
        )
        explore_data = explore_resp.json()
        schema_ctx = _build_schema_context(explore_data)
    except Exception:
        schema_ctx = ""

    # Step 2: Start session
    session_id = None
    similar_ctx = ""
    try:
        sess_resp = requests.post(
            f"{MIDDLEWARE_URL}/session/start",
            json={"agent_id": f"bench-agent-{agent_run}", "description": task.question},
            timeout=10,
        )
        sess_data = sess_resp.json()
        session_id = sess_data.get("session_id")
        similar = sess_data.get("similar_queries", [])
        result.similar_queries_received = len(similar)
        similar_ctx = _build_similar_queries_context(similar)
    except Exception:
        pass

    system_prompt = MIDDLEWARE_SYSTEM.format(
        schema_context=schema_ctx,
        similar_queries_context=similar_ctx,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task.question},
    ]

    for turn in range(MAX_TURNS):
        try:
            reply, tokens = _llm_chat(messages, api_url, api_key, model)
        except Exception as e:
            result.error = f"LLM error: {e}"
            break
        result.llm_tokens_used += tokens
        messages.append({"role": "assistant", "content": reply})

        answer = _extract_answer(reply)
        if answer:
            result.final_answer = answer
            break

        sql = _extract_sql(reply)
        if not sql:
            messages.append({"role": "user", "content": "Please provide a SQL query in ```sql ... ``` or your final ANSWER."})
            continue

        result.queries_executed += 1
        if any(kw in sql.lower() for kw in ["information_schema", "system.columns", "system.tables"]):
            result.exploration_queries += 1
        else:
            result.answer_queries += 1

        if session_id:
            query_result, success = _execute_sql_middleware(sql, session_id)
        else:
            query_result, success = _execute_sql_direct(sql)

        # Submit feedback — if the query returned data, mark as relevant
        if session_id:
            try:
                requests.post(
                    f"{MIDDLEWARE_URL}/session/{session_id}/feedback",
                    json={"relevant": success, "notes": sql[:100]},
                    timeout=5,
                )
            except Exception:
                pass

        if success:
            messages.append({"role": "user", "content": f"Query result:\n{query_result}"})
        else:
            messages.append({"role": "user", "content": f"Query failed: {query_result}"})

    # End session
    if session_id:
        try:
            requests.post(f"{MIDDLEWARE_URL}/session/{session_id}/end", timeout=10)
        except Exception:
            pass

    result.total_latency_ms = (time.monotonic() - t0) * 1000
    return result


# ── Orchestrator ─────────────────────────────────────────────────────


def run_agent_bench(
    mode: str,
    runs: int,
    api_url: str,
    api_key: str | None,
    model: str,
    task_ids: list[str] | None,
    output_path: str,
) -> dict:
    selected_tasks = TASKS
    if task_ids:
        id_set = set(task_ids)
        selected_tasks = [t for t in TASKS if t.id in id_set]

    runner = run_baseline_agent if mode == "baseline" else run_middleware_agent

    print("=" * 70)
    print(f"  Agent Efficiency Benchmark — {mode.upper()} mode")
    print("=" * 70)
    print(f"  Model:  {model}")
    print(f"  Runs:   {runs} agents per task")
    print(f"  Tasks:  {len(selected_tasks)}")
    print("=" * 70)

    all_results: list[dict] = []

    for task in selected_tasks:
        print(f"\n{'─' * 70}")
        print(f"  Task: {task.id} ({task.difficulty})")
        print(f"  Q: {task.question[:80]}...")
        print(f"{'─' * 70}")

        for run_num in range(1, runs + 1):
            print(f"\n  Agent #{run_num}...", end=" ", flush=True)
            result = runner(task, run_num, api_url, api_key, model)

            status = "\033[92mCORRECT\033[0m" if result.correct else "\033[93mDONE\033[0m"
            if result.error:
                status = f"\033[91mERROR: {result.error[:40]}\033[0m"

            print(
                f"{status} | queries={result.queries_executed} "
                f"(explore={result.exploration_queries}, answer={result.answer_queries}) | "
                f"tokens={result.llm_tokens_used} | "
                f"time={result.total_latency_ms/1000:.1f}s"
                + (f" | similar_recv={result.similar_queries_received}" if mode == "middleware" else "")
            )

            all_results.append(asdict(result))

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")

    for task in selected_tasks:
        task_results = [r for r in all_results if r["task_id"] == task.id]
        avg_queries = sum(r["queries_executed"] for r in task_results) / max(len(task_results), 1)
        avg_explore = sum(r["exploration_queries"] for r in task_results) / max(len(task_results), 1)
        avg_tokens = sum(r["llm_tokens_used"] for r in task_results) / max(len(task_results), 1)
        print(f"  {task.id:<30} avg_queries={avg_queries:.1f}  avg_explore={avg_explore:.1f}  avg_tokens={avg_tokens:.0f}")

    report = {
        "benchmark": "agent-efficiency",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"mode": mode, "model": model, "runs": runs},
        "results": all_results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report written to: {output_path}")

    return report


# ── Compare ──────────────────────────────────────────────────────────


def compare_agent_reports(path_a: str, path_b: str) -> None:
    with open(path_a) as f:
        report_a = json.load(f)
    with open(path_b) as f:
        report_b = json.load(f)

    mode_a = report_a["config"]["mode"]
    mode_b = report_b["config"]["mode"]

    print(f"\n{'=' * 80}")
    print(f"  AGENT BENCHMARK COMPARISON: {mode_a.upper()} vs {mode_b.upper()}")
    print(f"{'=' * 80}\n")

    task_ids = list(dict.fromkeys(r["task_id"] for r in report_a["results"]))

    print(f"  {'Task':<28} | {mode_a.upper():^22} | {mode_b.upper():^22} | {'Delta':^10}")
    print(f"  {'':28} | {'Queries  Explore  Tok':^22} | {'Queries  Explore  Tok':^22} |")
    print(f"  {'':─<28}─┼─{'':─<22}─┼─{'':─<22}─┼─{'':─<10}")

    for tid in task_ids:
        ra = [r for r in report_a["results"] if r["task_id"] == tid]
        rb = [r for r in report_b["results"] if r["task_id"] == tid]

        def avg(lst, key):
            return sum(r[key] for r in lst) / max(len(lst), 1)

        qa, ea, ta = avg(ra, "queries_executed"), avg(ra, "exploration_queries"), avg(ra, "llm_tokens_used")
        qb, eb, tb = avg(rb, "queries_executed"), avg(rb, "exploration_queries"), avg(rb, "llm_tokens_used")

        delta = qb - qa
        sign = "+" if delta >= 0 else ""
        print(f"  {tid:<28} | {qa:>5.1f}   {ea:>5.1f}  {ta:>5.0f} | {qb:>5.1f}   {eb:>5.1f}  {tb:>5.0f} | {sign}{delta:.1f} q")

    print(f"{'=' * 80}\n")


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Agent Efficiency Benchmark")
    parser.add_argument("--mode", choices=["baseline", "middleware"], default="middleware")
    parser.add_argument("--runs", type=int, default=3, help="Number of agents per task")
    parser.add_argument("--tasks", type=lambda s: s.split(","), default=None, help="Comma-separated task IDs")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two agent reports")
    args = parser.parse_args()

    if args.compare:
        compare_agent_reports(args.compare[0], args.compare[1])
        return

    api_url = os.environ.get("AGENT_LLM_API_URL")
    if not api_url:
        print("Error: AGENT_LLM_API_URL environment variable is required.")
        print("  For Ollama: export AGENT_LLM_API_URL=http://localhost:11434/v1")
        print("  For OpenAI: export AGENT_LLM_API_URL=https://api.openai.com/v1")
        sys.exit(1)

    api_key = os.environ.get("AGENT_LLM_API_KEY")
    model = os.environ.get("AGENT_LLM_MODEL", "gpt-4o-mini")
    output = args.output or f"reports/agent_{args.mode}.json"

    run_agent_bench(
        mode=args.mode,
        runs=args.runs,
        api_url=api_url,
        api_key=api_key,
        model=model,
        task_ids=args.tasks,
        output_path=output,
    )


if __name__ == "__main__":
    main()
