#!/usr/bin/env python3
"""
SAO Benchmark Harness — Main Orchestrator

Runs attack scenarios against a distributed ClickHouse OLAP cluster
and produces structured reports for evaluating defensive approaches.

Usage:
    # Run all scenarios against a cluster with default guardrails
    python harness.py --profile agent

    # Run against an unrestricted cluster (worst-case baseline)
    python harness.py --profile unrestricted

    # Run specific scenarios
    python harness.py --scenarios resource_exhaustion,data_exfiltration

    # Output JSON report
    python harness.py --output reports/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scenarios import ALL_SCENARIOS, ScenarioResult

# Profile presets mapping to ClickHouse user accounts
PROFILES = {
    "agent": {"user": "agent", "password": "agent_pass"},
    "unrestricted": {"user": "agent_unrestricted", "password": "agent_unrestricted_pass"},
    "analyst": {"user": "analyst", "password": "analyst_pass"},
}


def print_probe_table(result: ScenarioResult) -> None:
    """Print a formatted table of probe results."""
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
            # Truncate long error messages
            detail = p.error[:60] + "..." if len(p.error) > 60 else p.error

        print(f"{p.name:<35} {status:<19} {p.latency_ms:>8.1f} ms {p.rows_returned:<10} {detail}")


def print_summary(result: ScenarioResult) -> None:
    """Print scenario summary with security assessment."""
    s = result.summary()
    total = s["total_probes"]
    succeeded = s["succeeded"]
    blocked = s["blocked"]

    # Security grade: how well did defenses hold?
    if succeeded == 0:
        grade = "A (all attacks blocked)"
        color = "\033[92m"
    elif s["success_rate"] < 0.2:
        grade = "B (most attacks blocked)"
        color = "\033[93m"
    elif s["success_rate"] < 0.5:
        grade = "C (partial protection)"
        color = "\033[93m"
    else:
        grade = "F (defenses inadequate)"
        color = "\033[91m"

    print(f"\n  Total probes:  {total}")
    print(f"  Succeeded:     {succeeded} ({s['success_rate']*100:.1f}%)")
    print(f"  Blocked:       {blocked} ({s['block_rate']*100:.1f}%)")
    print(f"  Errored:       {s['errored']}")
    print(f"  Time:          {s['total_time_s']:.2f}s")
    print(f"  Defense grade: {color}{grade}\033[0m")

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


def run_harness(
    host: str,
    port: int,
    profile: str,
    scenario_names: list[str] | None,
    output_path: str | None,
    num_agents: int,
    flood_rounds: int,
) -> dict:
    """Run the full benchmark harness."""
    creds = PROFILES[profile]

    # Filter scenarios if specific ones requested
    scenarios_to_run = ALL_SCENARIOS
    if scenario_names:
        name_set = set(scenario_names)
        scenarios_to_run = [s for s in ALL_SCENARIOS if s.name in name_set]
        if not scenarios_to_run:
            print(f"Error: no matching scenarios for {scenario_names}")
            sys.exit(1)

    print("=" * 70)
    print("  SAO Benchmark Harness — AI Agent OLAP Abuse Testing")
    print("=" * 70)
    print(f"  Target:   {host}:{port}")
    print(f"  Profile:  {profile} (user={creds['user']})")
    print(f"  Scenarios: {len(scenarios_to_run)}")
    print(f"  Started:  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

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

    # Final report
    report = {
        "harness": "sao-benchmark",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "host": host,
            "port": port,
            "profile": profile,
            "user": creds["user"],
        },
        "overall_time_s": round(overall_time, 3),
        "scenarios": all_results,
        "aggregate": {
            "total_probes": sum(r["total_probes"] for r in all_results),
            "total_succeeded": sum(r["succeeded"] for r in all_results),
            "total_blocked": sum(r["blocked"] for r in all_results),
            "overall_success_rate": round(
                sum(r["succeeded"] for r in all_results)
                / max(sum(r["total_probes"] for r in all_results), 1),
                4,
            ),
            "overall_block_rate": round(
                sum(r["blocked"] for r in all_results)
                / max(sum(r["total_probes"] for r in all_results), 1),
                4,
            ),
        },
    }

    # Print final aggregate
    agg = report["aggregate"]
    print(f"\n{'=' * 70}")
    print("  AGGREGATE RESULTS")
    print(f"{'=' * 70}")
    print(f"  Total probes across all scenarios: {agg['total_probes']}")
    print(f"  Total succeeded:                   {agg['total_succeeded']} ({agg['overall_success_rate']*100:.1f}%)")
    print(f"  Total blocked:                     {agg['total_blocked']} ({agg['overall_block_rate']*100:.1f}%)")
    print(f"  Total time:                        {report['overall_time_s']:.2f}s")

    if agg["overall_success_rate"] == 0:
        print(f"\n  \033[92mOVERALL GRADE: A — All attacks blocked\033[0m")
    elif agg["overall_success_rate"] < 0.3:
        print(f"\n  \033[93mOVERALL GRADE: B — Most attacks blocked\033[0m")
    else:
        print(f"\n  \033[91mOVERALL GRADE: F — Defenses need work\033[0m")
    print(f"{'=' * 70}\n")

    # Write JSON report
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Report written to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="SAO Benchmark Harness — AI Agent OLAP Abuse Testing"
    )
    parser.add_argument("--host", default="localhost", help="ClickHouse host")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port")
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="agent",
        help="User profile to test (agent=guarded, unrestricted=no limits, analyst=read-only)",
    )
    parser.add_argument(
        "--scenarios",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated list of scenarios to run (default: all)",
    )
    parser.add_argument(
        "--output", "-o",
        default="reports/harness_results.json",
        help="Path for JSON report output",
    )
    parser.add_argument("--num-agents", type=int, default=50, help="Agent count for flooding scenario")
    parser.add_argument("--flood-rounds", type=int, default=3, help="Rounds for flooding scenario")
    args = parser.parse_args()

    run_harness(
        host=args.host,
        port=args.port,
        profile=args.profile,
        scenario_names=args.scenarios,
        output_path=args.output,
        num_agents=args.num_agents,
        flood_rounds=args.flood_rounds,
    )


if __name__ == "__main__":
    main()
