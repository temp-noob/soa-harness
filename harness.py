#!/usr/bin/env python3
"""
SAO Benchmark Harness — Main Orchestrator

Runs attack scenarios against a distributed ClickHouse OLAP cluster
and produces structured reports for evaluating defensive approaches.

Supports two modes:
  - direct:     hit ClickHouse directly (baseline)
  - middleware:  hit the Agent-First Middleware (access control + session learning)

Usage:
    # Baseline — direct to ClickHouse
    python harness.py --mode direct --profile agent

    # Through middleware
    python harness.py --mode middleware --profile agent

    # Compare two reports side-by-side
    python harness.py --compare reports/direct_agent.json reports/middleware_agent.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from scenarios import ALL_SCENARIOS, ScenarioResult

PROFILES = {
    "agent": {"user": "agent", "password": "agent_pass"},
    "unrestricted": {"user": "agent_unrestricted", "password": "agent_unrestricted_pass"},
    "analyst": {"user": "analyst", "password": "analyst_pass"},
}

MODE_DEFAULTS = {
    "direct": {"port": 8123},
    "middleware": {"port": 8080},
}

MIDDLEWARE_API = "http://localhost:8080"


# ── Printing helpers ─────────────────────────────────────────────────


def print_probe_table(result: ScenarioResult) -> None:
    header = f"{'Probe':<35} {'Status':<10} {'Latency':<12} {'Rows':<10} {'Detail'}"
    print(header)
    print("-" * len(header))

    for p in result.probes:
        if p.succeeded:
            status = "\033[92mSUCCEED\033[0m"
        elif p.blocked:
            status = "\033[93mBLOCKED\033[0m"
        else:
            status = "\033[91mERROR\033[0m"

        detail = ""
        if p.error:
            detail = p.error[:60] + "..." if len(p.error) > 60 else p.error

        print(f"{p.name:<35} {status:<19} {p.latency_ms:>8.1f} ms {p.rows_returned:<10} {detail}")


def _grade(success_rate: float) -> tuple[str, str]:
    if success_rate == 0:
        return "A", "\033[92m"
    elif success_rate < 0.2:
        return "B", "\033[93m"
    elif success_rate < 0.5:
        return "C", "\033[93m"
    else:
        return "F", "\033[91m"


def print_summary(result: ScenarioResult) -> None:
    s = result.summary()
    grade_letter, color = _grade(s["success_rate"])

    print(f"\n  Total probes:  {s['total_probes']}")
    print(f"  Succeeded:     {s['succeeded']} ({s['success_rate']*100:.1f}%)")
    print(f"  Blocked:       {s['blocked']} ({s['block_rate']*100:.1f}%)")
    print(f"  Errored:       {s['errored']}")
    print(f"  Time:          {s['total_time_s']:.2f}s")
    print(f"  Defense grade: {color}{grade_letter} ({_grade_desc(grade_letter)})\033[0m")

    if result.metadata:
        print(f"\n  Scenario metadata:")
        for k, v in result.metadata.items():
            if isinstance(v, dict):
                print(f"    {k}:")
                for kk, vv in v.items():
                    print(f"      {kk}: {vv}")
            elif isinstance(v, list):
                print(f"    {k}: [{len(v)} items]")
            else:
                print(f"    {k}: {v}")


def _grade_desc(g: str) -> str:
    return {"A": "all attacks blocked", "B": "most attacks blocked",
            "C": "partial protection", "F": "defenses inadequate"}.get(g, "")


# ── Middleware session helpers ───────────────────────────────────────


def _middleware_session_start(profile: str) -> str | None:
    try:
        resp = requests.post(
            f"{MIDDLEWARE_API}/session/start",
            json={
                "agent_id": profile,
                "description": f"SAO benchmark run — {profile} profile",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("session_id")
    except Exception as e:
        print(f"  Warning: could not start middleware session: {e}")
    return None


def _middleware_session_end(session_id: str) -> None:
    try:
        requests.post(f"{MIDDLEWARE_API}/session/{session_id}/end", timeout=10)
    except Exception:
        pass


# ── Main harness ─────────────────────────────────────────────────────


def run_harness(
    host: str,
    port: int,
    profile: str,
    mode: str,
    scenario_names: list[str] | None,
    output_path: str | None,
    num_agents: int,
    flood_rounds: int,
) -> dict:
    creds = PROFILES[profile]

    scenarios_to_run = ALL_SCENARIOS
    if scenario_names:
        name_set = set(scenario_names)
        scenarios_to_run = [s for s in ALL_SCENARIOS if s.name in name_set]
        if not scenarios_to_run:
            print(f"Error: no matching scenarios for {scenario_names}")
            sys.exit(1)

    mode_label = "MIDDLEWARE" if mode == "middleware" else "DIRECT"

    print("=" * 70)
    print(f"  SAO Benchmark Harness — {mode_label} mode")
    print("=" * 70)
    print(f"  Target:   {host}:{port}")
    print(f"  Mode:     {mode}")
    print(f"  Profile:  {profile} (user={creds['user']})")
    print(f"  Scenarios: {len(scenarios_to_run)}")
    print(f"  Started:  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # Start middleware session if in middleware mode
    session_id = None
    if mode == "middleware":
        session_id = _middleware_session_start(profile)
        if session_id:
            print(f"\n  Middleware session: {session_id}")

    all_results: list[dict] = []
    overall_t0 = time.perf_counter()

    for scenario_cls in scenarios_to_run:
        kwargs = {
            "host": host,
            "port": port,
            "user": creds["user"],
            "password": creds["password"],
        }
        if scenario_cls.name == "concurrent_flooding":
            kwargs["num_agents"] = num_agents
            kwargs["rounds"] = flood_rounds

        scenario = scenario_cls(**kwargs)

        print(f"\n{'─' * 70}")
        print(f"  Scenario: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"{'─' * 70}\n")

        result = scenario.run()
        print_probe_table(result)
        print_summary(result)
        all_results.append(result.summary())

    overall_time = time.perf_counter() - overall_t0

    # End middleware session
    if session_id:
        _middleware_session_end(session_id)

    # Build report
    agg_total = sum(r["total_probes"] for r in all_results)
    agg_succeeded = sum(r["succeeded"] for r in all_results)
    agg_blocked = sum(r["blocked"] for r in all_results)

    report = {
        "harness": "sao-benchmark",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "host": host,
            "port": port,
            "profile": profile,
            "user": creds["user"],
            "mode": mode,
        },
        "overall_time_s": round(overall_time, 3),
        "scenarios": all_results,
        "aggregate": {
            "total_probes": agg_total,
            "total_succeeded": agg_succeeded,
            "total_blocked": agg_blocked,
            "overall_success_rate": round(agg_succeeded / max(agg_total, 1), 4),
            "overall_block_rate": round(agg_blocked / max(agg_total, 1), 4),
        },
    }

    agg = report["aggregate"]
    grade_letter, color = _grade(agg["overall_success_rate"])
    print(f"\n{'=' * 70}")
    print(f"  AGGREGATE RESULTS ({mode_label})")
    print(f"{'=' * 70}")
    print(f"  Total probes across all scenarios: {agg['total_probes']}")
    print(f"  Total succeeded:                   {agg['total_succeeded']} ({agg['overall_success_rate']*100:.1f}%)")
    print(f"  Total blocked:                     {agg['total_blocked']} ({agg['overall_block_rate']*100:.1f}%)")
    print(f"  Total time:                        {report['overall_time_s']:.2f}s")
    print(f"\n  {color}OVERALL GRADE: {grade_letter} — {_grade_desc(grade_letter)}\033[0m")
    print(f"{'=' * 70}\n")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report written to: {output_path}")

    return report


# ── Compare two reports ──────────────────────────────────────────────


def compare_reports(path_a: str, path_b: str) -> None:
    with open(path_a) as f:
        report_a = json.load(f)
    with open(path_b) as f:
        report_b = json.load(f)

    mode_a = report_a["config"].get("mode", "direct")
    mode_b = report_b["config"].get("mode", "middleware")
    label_a = mode_a.upper()
    label_b = mode_b.upper()

    scenarios_a = {s["scenario"]: s for s in report_a["scenarios"]}
    scenarios_b = {s["scenario"]: s for s in report_b["scenarios"]}
    all_names = list(dict.fromkeys(list(scenarios_a) + list(scenarios_b)))

    print()
    print("=" * 90)
    print(f"  BENCHMARK COMPARISON: {label_a} vs {label_b}")
    print("=" * 90)
    print()
    header = f"  {'Scenario':<26} | {label_a:^20} | {label_b:^20} | {'Delta':^14}"
    print(header)
    print(f"  {'':─<26}─┼─{'':─<20}─┼─{'':─<20}─┼─{'':─<14}")

    sub_header = f"  {'':26} | {'Grade Succ/Tot  Blk':^20} | {'Grade Succ/Tot  Blk':^20} | {'Blocked':^14}"
    print(sub_header)
    print(f"  {'':─<26}─┼─{'':─<20}─┼─{'':─<20}─┼─{'':─<14}")

    total_delta_blocked = 0

    for name in all_names:
        sa = scenarios_a.get(name)
        sb = scenarios_b.get(name)

        if sa:
            ga, _ = _grade(sa["success_rate"])
            col_a = f"  {ga}   {sa['succeeded']:>2}/{sa['total_probes']:<3}  {sa['blocked']:>3}"
        else:
            col_a = f"  {'N/A':^18}"

        if sb:
            gb, _ = _grade(sb["success_rate"])
            col_b = f"  {gb}   {sb['succeeded']:>2}/{sb['total_probes']:<3}  {sb['blocked']:>3}"
        else:
            col_b = f"  {'N/A':^18}"

        if sa and sb:
            delta = sb["blocked"] - sa["blocked"]
            total_delta_blocked += delta
            sign = "+" if delta >= 0 else ""
            col_d = f"  {sign}{delta} blocked"
        else:
            col_d = f"  {'—':^12}"

        print(f"  {name:<26} |{col_a} |{col_b} |{col_d}")

    # Totals
    agg_a = report_a["aggregate"]
    agg_b = report_b["aggregate"]
    print(f"  {'':─<26}─┼─{'':─<20}─┼─{'':─<20}─┼─{'':─<14}")

    ga_all, _ = _grade(agg_a["overall_success_rate"])
    gb_all, _ = _grade(agg_b["overall_success_rate"])
    sign = "+" if total_delta_blocked >= 0 else ""

    col_a = f"  {ga_all}   {agg_a['total_succeeded']:>2}/{agg_a['total_probes']:<3}  {agg_a['total_blocked']:>3}"
    col_b = f"  {gb_all}   {agg_b['total_succeeded']:>2}/{agg_b['total_probes']:<3}  {agg_b['total_blocked']:>3}"
    col_d = f"  {sign}{total_delta_blocked} blocked"
    print(f"  {'OVERALL':<26} |{col_a} |{col_b} |{col_d}")

    print()
    print(f"  {label_a} time: {report_a['overall_time_s']:.2f}s")
    print(f"  {label_b} time: {report_b['overall_time_s']:.2f}s")
    print("=" * 90)
    print()


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SAO Benchmark Harness — AI Agent OLAP Abuse Testing"
    )
    parser.add_argument("--host", default="localhost", help="Target host")
    parser.add_argument("--port", type=int, default=None, help="Target port (default: 8123 for direct, 8080 for middleware)")
    parser.add_argument(
        "--mode",
        choices=["direct", "middleware"],
        default="direct",
        help="direct = raw ClickHouse, middleware = through Agent-First Middleware",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="agent",
        help="User profile to test",
    )
    parser.add_argument(
        "--scenarios",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated list of scenarios to run (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path for JSON report output",
    )
    parser.add_argument("--num-agents", type=int, default=50, help="Agent count for flooding scenario")
    parser.add_argument("--flood-rounds", type=int, default=3, help="Rounds for flooding scenario")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("REPORT_A", "REPORT_B"),
        help="Compare two JSON reports side-by-side",
    )
    args = parser.parse_args()

    if args.compare:
        compare_reports(args.compare[0], args.compare[1])
        return

    port = args.port or MODE_DEFAULTS[args.mode]["port"]
    output = args.output or f"reports/{args.mode}_{args.profile}.json"

    run_harness(
        host=args.host,
        port=port,
        profile=args.profile,
        mode=args.mode,
        scenario_names=args.scenarios,
        output_path=output,
        num_agents=args.num_agents,
        flood_rounds=args.flood_rounds,
    )


if __name__ == "__main__":
    main()
