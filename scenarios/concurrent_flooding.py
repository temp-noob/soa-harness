"""
Scenario 2: Concurrent Agent Flooding

Simulates a swarm of AI agents hitting the OLAP cluster simultaneously,
testing correctness under concurrency and cluster stability.  In a real
deployment, hundreds of agents might independently decide to run complex
analytics at the same time with no coordination.

Pressure point from SAO paper: "correctness under concurrency" and
"low-to-zero trust actors."
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scenarios.base import BaseScenario, ScenarioResult, AttackProbe


# Workload mix: an agent swarm running varied queries simultaneously
AGENT_QUERIES = [
    ("analytical_heavy", """
        SELECT category, toStartOfWeek(created_at) AS week,
               count(), sum(amount_cents), avg(fraud_score)
        FROM sao.transactions_distributed
        GROUP BY category, week
        ORDER BY week DESC, category
    """),
    ("full_customer_scan", """
        SELECT state, tier, count(), avg(credit_score)
        FROM sao.customers_distributed
        GROUP BY state, tier
    """),
    ("join_heavy", """
        SELECT c.tier, p.category, count(), sum(t.amount_cents)
        FROM sao.transactions_distributed AS t
        INNER JOIN sao.customers_distributed AS c ON t.customer_id = c.customer_id
        INNER JOIN sao.products_distributed AS p ON t.product_id = p.product_id
        GROUP BY c.tier, p.category
    """),
    ("distinct_probe", """
        SELECT DISTINCT customer_id, ip_address
        FROM sao.transactions_distributed
        LIMIT 50000
    """),
    ("window_function", """
        SELECT customer_id, created_at, amount_cents,
               row_number() OVER (PARTITION BY customer_id ORDER BY created_at DESC) AS rn
        FROM sao.transactions_distributed
        LIMIT 100000
    """),
]


class ConcurrentFloodingScenario(BaseScenario):
    name = "concurrent_flooding"
    description = (
        "Swarm of concurrent agents flooding the cluster with analytical "
        "queries to test concurrency limits, queueing, and stability."
    )

    def __init__(self, *args, num_agents: int = 50, rounds: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_agents = num_agents
        self.rounds = rounds

    def _run_single_agent(self, agent_id: int, query_name: str, query: str) -> AttackProbe:
        """Simulate a single agent running a query."""
        probe = AttackProbe(
            name=f"agent_{agent_id}_{query_name}",
            query=query,
            metadata={"agent_id": agent_id},
        )
        try:
            client = self.get_client()
            self.execute_probe(client, probe)
        except Exception as e:
            probe.error = str(e)
            probe.blocked = True
        return probe

    def run(self) -> ScenarioResult:
        t0 = time.perf_counter()
        all_probes: list[AttackProbe] = []

        for round_num in range(self.rounds):
            futures = []
            with ThreadPoolExecutor(max_workers=self.num_agents) as pool:
                for agent_id in range(self.num_agents):
                    qname, qtext = AGENT_QUERIES[agent_id % len(AGENT_QUERIES)]
                    futures.append(pool.submit(
                        self._run_single_agent, agent_id, qname, qtext
                    ))

                for future in as_completed(futures):
                    all_probes.append(future.result())

        # --- Phase: Rate-limit enforcement probes ---
        # Demonstrate the per-agent middleware rate limiter: a single agent
        # flooding above its quota gets 429 / TOO_MANY (blocked=True), while
        # a separate agent retains its own full independent quota.

        raw_rl = os.environ.get("RATE_LIMIT_REQUESTS", "20")
        try:
            rl_limit = int(raw_rl)
        except ValueError:
            raise ValueError(
                f"RATE_LIMIT_REQUESTS must be a non-negative integer, got: {raw_rl!r}"
            )
        burst_total = rl_limit + 5  # enough requests to exceed the window quota

        burst_client = self.get_client(username="rl_burst_agent")
        burst_probes: list[AttackProbe] = []
        for i in range(burst_total):
            expected = "allowed" if i < rl_limit else "blocked"
            probe = AttackProbe(
                name=f"rate_limit_burst_{i:02d}",
                query="SELECT 1",
                metadata={
                    "rate_limit_phase": "burst",
                    "index": i,
                    "expected": expected,
                },
            )
            self.execute_probe(burst_client, probe)
            actual = "blocked" if probe.blocked else "allowed"
            probe.metadata["actual"] = actual
            probe.metadata["expectation_met"] = actual == expected
            burst_probes.append(probe)
            all_probes.append(probe)

        # Per-agent isolation: a different agent must not be affected by the
        # burst above — its own quota is still fully available.
        isolation_client = self.get_client(username="rl_isolation_agent")
        isolation_probes: list[AttackProbe] = []
        for i in range(3):
            probe = AttackProbe(
                name=f"rate_limit_isolation_{i:02d}",
                query="SELECT 1",
                metadata={"rate_limit_phase": "isolation", "index": i, "expected": "allowed"},
            )
            self.execute_probe(isolation_client, probe)
            actual = "blocked" if probe.blocked else "allowed"
            probe.metadata["actual"] = actual
            probe.metadata["expectation_met"] = not probe.blocked
            isolation_probes.append(probe)
            all_probes.append(probe)

        elapsed = time.perf_counter() - t0

        # Compute concurrency-specific metrics
        latencies = [p.latency_ms for p in all_probes if p.succeeded]
        meta = {
            "num_agents": self.num_agents,
            "rounds": self.rounds,
            "total_queries": len(all_probes),
        }
        if latencies:
            latencies.sort()
            meta.update({
                "p50_latency_ms": round(latencies[len(latencies) // 2], 2),
                "p95_latency_ms": round(latencies[int(len(latencies) * 0.95)], 2),
                "p99_latency_ms": round(latencies[int(len(latencies) * 0.99)], 2),
                "max_latency_ms": round(latencies[-1], 2),
            })

        if burst_probes:
            meta.update({
                "rate_limit_burst_total": len(burst_probes),
                "rate_limit_burst_allowed": sum(1 for p in burst_probes if p.succeeded),
                "rate_limit_burst_blocked": sum(1 for p in burst_probes if p.blocked),
                "rate_limit_burst_expectations_met": sum(
                    1 for p in burst_probes if p.metadata.get("expectation_met")
                ),
                "rate_limit_isolation_blocked": sum(1 for p in isolation_probes if p.blocked),
                "rate_limit_isolation_expectations_met": sum(
                    1 for p in isolation_probes if p.metadata.get("expectation_met")
                ),
            })

        return ScenarioResult(
            scenario_name=self.name,
            description=self.description,
            probes=all_probes,
            total_time_s=elapsed,
            metadata=meta,
        )
