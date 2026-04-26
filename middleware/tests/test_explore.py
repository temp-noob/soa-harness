"""Tests for the Explore endpoint (unit tests — no ClickHouse needed)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from access_control import AccessControlService


class TestExploreAccessFiltering:
    """Test that the access control correctly classifies tables/columns
    so that the explore endpoint can filter them."""

    def test_system_table_blocked(self):
        ac = AccessControlService()
        assert ac.registry.is_system_table("system.columns")
        assert ac.registry.is_system_table("system.query_log")
        assert ac.registry.is_system_table("system.processes")

    def test_audit_log_is_system(self):
        ac = AccessControlService()
        assert ac.registry.get_table_classification("agent_audit_log") == "SYSTEM"

    def test_revenue_daily_is_safe(self):
        ac = AccessControlService()
        assert ac.registry.get_table_classification("revenue_daily") == "SAFE_AGGREGATE"

    def test_customers_is_pii(self):
        ac = AccessControlService()
        assert ac.registry.get_table_classification("customers") == "PII"

    def test_email_is_pii(self):
        ac = AccessControlService()
        assert ac.registry.get_column_classification("customers", "email") == "PII"

    def test_city_is_safe(self):
        ac = AccessControlService()
        assert ac.registry.get_column_classification("customers", "city") == "SAFE"

    def test_credit_score_is_financial(self):
        ac = AccessControlService()
        assert ac.registry.get_column_classification("customers", "credit_score") == "FINANCIAL"

    def test_cost_cents_is_secret(self):
        ac = AccessControlService()
        assert ac.registry.get_column_classification("products", "cost_cents") == "BUSINESS_SECRET"

    def test_fraud_score_is_secret(self):
        ac = AccessControlService()
        assert ac.registry.get_column_classification("transactions", "fraud_score") == "BUSINESS_SECRET"

    def test_distributed_table_lookup(self):
        ac = AccessControlService()
        assert ac.registry.get_table_classification("customers_distributed") == "PII"
        assert ac.registry.get_table_classification("sao.customers_distributed") == "PII"

    def test_sensitivity_ordering(self):
        ac = AccessControlService()
        pii = ac.registry.get_sensitivity_level("PII")
        financial = ac.registry.get_sensitivity_level("FINANCIAL")
        safe = ac.registry.get_sensitivity_level("SAFE")
        assert pii > financial > safe


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
