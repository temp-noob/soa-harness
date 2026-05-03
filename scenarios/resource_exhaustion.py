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

        # --- Advanced: probes caught by middleware detect_resource_exhaustion() ---
        # These demonstrate the value of SQL-level resource-exhaustion detection
        # in the middleware access control layer.  In direct-to-ClickHouse mode
        # each query reaches the database and may OOM or time out.  Through the
        # middleware they are BLOCKED before touching ClickHouse, protecting
        # cluster stability regardless of per-user memory/CPU quotas.

        # Category 1: Cartesian join — explicit CROSS JOIN keyword
        probes.append(self.execute_probe(client, AttackProbe(
            name="explicit_cross_join",
            query="""
                SELECT count()
                FROM sao.transactions_distributed AS a
                CROSS JOIN sao.customers_distributed AS b
            """,
            metadata={
                "caught_by_middleware": True,
                "detection_category": "cartesian_join",
            },
        )))

        # Category 1: Cartesian join — JOIN with no ON/USING condition
        probes.append(self.execute_probe(client, AttackProbe(
            name="join_without_condition",
            query="""
                SELECT a.customer_id, b.email
                FROM sao.transactions_distributed AS a
                JOIN sao.customers_distributed AS b
                LIMIT 100
            """,
            metadata={
                "caught_by_middleware": True,
                "detection_category": "cartesian_join",
            },
        )))

        # Category 2: Memory bomb — numbers() table function with huge N
        probes.append(self.execute_probe(client, AttackProbe(
            name="numbers_table_bomb",
            query="""
                SELECT number, count()
                FROM numbers(50000000)
                GROUP BY number % 1000
            """,
            metadata={
                "caught_by_middleware": True,
                "detection_category": "memory_bomb",
            },
        )))

        # Category 3: Regexp full scan — LIKE with leading wildcard on large table
        probes.append(self.execute_probe(client, AttackProbe(
            name="leading_wildcard_like",
            query="""
                SELECT count()
                FROM sao.transactions_distributed
                WHERE ip_address LIKE '%192.168%'
                  AND user_agent LIKE '%Chrome%'
            """,
            metadata={
                "caught_by_middleware": True,
                "detection_category": "regexp_full_scan",
            },
        )))

        # Category 4: Unbounded high-cardinality GROUP BY — email column
        probes.append(self.execute_probe(client, AttackProbe(
            name="groupby_email_unbounded",
            query="""
                SELECT email, count(), sum(credit_score)
                FROM sao.customers_distributed
                GROUP BY email
            """,
            metadata={
                "caught_by_middleware": True,
                "detection_category": "unbounded_high_cardinality_groupby",
            },
        )))

        # Category 4: Unbounded high-cardinality GROUP BY — tx_id column
        probes.append(self.execute_probe(client, AttackProbe(
            name="groupby_tx_id_unbounded",
            query="""
                SELECT tx_id, sum(amount_cents)
                FROM sao.transactions_distributed
                GROUP BY tx_id
            """,
            metadata={
                "caught_by_middleware": True,
                "detection_category": "unbounded_high_cardinality_groupby",
            },
        )))

        middleware_caught = sum(
            1 for p in probes
            if p.blocked and p.metadata.get("caught_by_middleware")
        )
        middleware_probes_total = sum(
            1 for p in probes if p.metadata.get("caught_by_middleware")
        )

        return ScenarioResult(
            scenario_name=self.name,
            description=self.description,
            probes=probes,
            total_time_s=time.perf_counter() - t0,
            metadata={
                "middleware_caught": middleware_caught,
                "middleware_probes_total": middleware_probes_total,
            },
        )
