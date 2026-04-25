"""
Scenario 1: Resource Exhaustion

Simulates an AI agent generating queries designed to consume excessive
CPU, memory, or I/O on the OLAP cluster.  Real-world examples include
agents generating cartesian joins, unbounded GROUP BYs, or massive
full-table scans because the LLM has no awareness of query cost.

Pressure point from SAO paper: "predictable performance under new
access patterns" — agents don't self-regulate resource usage.
"""

from __future__ import annotations

import time
from scenarios.base import BaseScenario, ScenarioResult, AttackProbe


class ResourceExhaustionScenario(BaseScenario):
    name = "resource_exhaustion"
    description = (
        "Agent generates expensive queries that exhaust cluster resources: "
        "cartesian joins, memory bombs, full scans, and recursive CTEs."
    )

    def run(self) -> ScenarioResult:
        client = self.get_client()
        t0 = time.perf_counter()
        probes: list[AttackProbe] = []

        # --- Probe 1: Cartesian join (exploding row count) ---
        probes.append(self.execute_probe(client, AttackProbe(
            name="cartesian_join",
            query="""
                SELECT count()
                FROM sao.transactions_distributed AS a, sao.transactions_distributed AS b
                WHERE a.amount_cents > 0
                LIMIT 1
            """,
        )))

        # --- Probe 2: Unbounded aggregation with high cardinality ---
        probes.append(self.execute_probe(client, AttackProbe(
            name="high_cardinality_groupby",
            query="""
                SELECT ip_address, user_agent, card_last4,
                       count(), avg(amount_cents), uniqExact(customer_id)
                FROM sao.transactions_distributed
                GROUP BY ip_address, user_agent, card_last4
            """,
        )))

        # --- Probe 3: Memory bomb via large array materialization ---
        probes.append(self.execute_probe(client, AttackProbe(
            name="array_memory_bomb",
            query="""
                SELECT arrayJoin(range(100000000)) AS x, count()
                FROM numbers(1)
                GROUP BY x
            """,
        )))

        # --- Probe 4: Full scan with regexp on every row ---
        probes.append(self.execute_probe(client, AttackProbe(
            name="full_scan_regexp",
            query="""
                SELECT count()
                FROM sao.transactions_distributed
                WHERE match(user_agent, '.*[a-z]{3,}.*[0-9]{2,}.*')
                  AND match(ip_address, '^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$')
            """,
        )))

        # --- Probe 5: Nested subqueries creating O(n^2) work ---
        probes.append(self.execute_probe(client, AttackProbe(
            name="nested_subquery_explosion",
            query="""
                SELECT customer_id, (
                    SELECT count() FROM sao.transactions_distributed AS t2
                    WHERE t2.customer_id = t1.customer_id
                ) AS tx_count
                FROM sao.customers_distributed AS t1
                LIMIT 10000
            """,
        )))

        # --- Probe 6: ORDER BY on non-indexed large column ---
        probes.append(self.execute_probe(client, AttackProbe(
            name="expensive_sort",
            query="""
                SELECT *
                FROM sao.transactions_distributed
                ORDER BY sipHash64(ip_address, user_agent, card_last4)
                LIMIT 100000
            """,
        )))

        # --- Probe 7: UNION ALL stacking to multiply work ---
        probes.append(self.execute_probe(client, AttackProbe(
            name="union_stacking",
            query="""
                SELECT count(), 'a' FROM sao.transactions_distributed
                UNION ALL
                SELECT count(), 'b' FROM sao.transactions_distributed
                UNION ALL
                SELECT count(), 'c' FROM sao.transactions_distributed
                UNION ALL
                SELECT count(), 'd' FROM sao.transactions_distributed
                UNION ALL
                SELECT count(), 'e' FROM sao.transactions_distributed
            """,
        )))

        return ScenarioResult(
            scenario_name=self.name,
            description=self.description,
            probes=probes,
            total_time_s=time.perf_counter() - t0,
        )
