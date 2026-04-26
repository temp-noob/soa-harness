"""
Integration tests for ChromaDB-backed session learning.

These tests talk to a **real** ChromaDB instance at localhost:8000
(the ``vector-db`` service from docker-compose.yml).  They exercise
the full session-learning loop:

  1. Create a collection
  2. Start a session with a description
  3. Record queries with feedback
  4. End the session (persists to ChromaDB)
  5. Start a NEW session with a similar description
  6. Verify the new session receives relevant queries from the first session
  7. Clean up (delete the test collection)
"""

import sys
from pathlib import Path

import httpx
import pytest

# Ensure the middleware package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from session_manager import (
    EMBEDDING_DIM,
    SessionManager,
    SimilarQuery,
    _ChromaHTTP,
    _text_to_embedding,
)

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# Use a dedicated collection name so tests never collide with production data.
TEST_COLLECTION = "integration_test_sessions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chroma_reachable() -> bool:
    """Return True when the ChromaDB server answers on localhost:8000."""
    try:
        r = httpx.get(
            f"http://{CHROMA_HOST}:{CHROMA_PORT}/api/v2/version", timeout=3.0,
        )
        return r.status_code == 200
    except httpx.ConnectError:
        return False


skip_if_no_chroma = pytest.mark.skipif(
    not _chroma_reachable(),
    reason="ChromaDB not reachable at localhost:8000",
)


def _cleanup_collection() -> None:
    """Delete the test collection if it exists (best-effort)."""
    try:
        httpx.delete(
            f"http://{CHROMA_HOST}:{CHROMA_PORT}"
            f"/api/v2/tenants/default_tenant/databases/default_database"
            f"/collections/{TEST_COLLECTION}",
            timeout=5.0,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Embedding function tests
# ---------------------------------------------------------------------------


class TestEmbeddingFunction:
    """Verify properties of the lightweight embedding helper."""

    def test_output_dimension(self):
        vec = _text_to_embedding("hello world")
        assert len(vec) == EMBEDDING_DIM

    def test_deterministic(self):
        a = _text_to_embedding("quarterly revenue analysis")
        b = _text_to_embedding("quarterly revenue analysis")
        assert a == b

    def test_unit_norm(self):
        import math

        vec = _text_to_embedding("some arbitrary text about sales metrics")
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-6

    def test_similar_texts_closer_than_different(self):
        """Cosine similarity between similar texts should be higher."""
        import math

        def cosine(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0

        v1 = _text_to_embedding("quarterly revenue by product category")
        v2 = _text_to_embedding("revenue breakdown by product type each quarter")
        v3 = _text_to_embedding("instructions for assembling garden furniture")

        sim_close = cosine(v1, v2)
        sim_far = cosine(v1, v3)
        assert sim_close > sim_far, (
            f"Similar texts should have higher cosine similarity: "
            f"{sim_close:.4f} vs {sim_far:.4f}"
        )


# ---------------------------------------------------------------------------
# _ChromaHTTP low-level tests
# ---------------------------------------------------------------------------


@skip_if_no_chroma
class TestChromaHTTP:
    """Low-level tests against the ChromaDB v2 REST API."""

    def setup_method(self):
        _cleanup_collection()

    def teardown_method(self):
        _cleanup_collection()

    def test_create_and_count(self):
        ch = _ChromaHTTP(CHROMA_HOST, CHROMA_PORT)
        cid = ch.get_or_create_collection(TEST_COLLECTION, {"hnsw:space": "cosine"})
        assert cid  # non-empty UUID string
        assert ch.count() == 0

    def test_add_and_query(self):
        ch = _ChromaHTTP(CHROMA_HOST, CHROMA_PORT)
        ch.get_or_create_collection(TEST_COLLECTION, {"hnsw:space": "cosine"})

        emb = _text_to_embedding("sales analysis")
        ch.add(
            ids=["doc-1"],
            documents=["sales analysis session"],
            metadatas=[{"agent": "a1"}],
            embeddings=[emb],
        )
        assert ch.count() == 1

        results = ch.query(query_embeddings=[emb], n_results=1)
        assert results["ids"][0] == ["doc-1"]
        assert results["documents"][0] == ["sales analysis session"]
        assert results["metadatas"][0][0]["agent"] == "a1"

    def test_idempotent_get_or_create(self):
        ch = _ChromaHTTP(CHROMA_HOST, CHROMA_PORT)
        id1 = ch.get_or_create_collection(TEST_COLLECTION)
        id2 = ch.get_or_create_collection(TEST_COLLECTION)
        assert id1 == id2


# ---------------------------------------------------------------------------
# Full session-learning loop
# ---------------------------------------------------------------------------


@skip_if_no_chroma
class TestSessionLearningLoop:
    """End-to-end test: session 1 stores queries, session 2 retrieves them."""

    def setup_method(self):
        _cleanup_collection()

    def teardown_method(self):
        _cleanup_collection()

    def _make_manager(self) -> SessionManager:
        sm = SessionManager(chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT)
        # Override collection name to use the test-specific one.
        sm.COLLECTION_NAME = TEST_COLLECTION
        sm._chroma.get_or_create_collection(
            name=TEST_COLLECTION, metadata={"hnsw:space": "cosine"},
        )
        return sm

    # -- core loop ---------------------------------------------------------

    def test_full_learning_loop(self):
        sm = self._make_manager()
        assert sm._available, "SessionManager should connect to ChromaDB"

        # -- Session 1: record queries and persist -------------------------
        sid1, similar1 = sm.start_session(
            agent_id="analyst-agent",
            description="quarterly revenue analysis by product category",
        )
        assert sid1
        assert similar1 == [], "No prior data, so no similar queries expected"

        sm.record_query(
            sid1,
            "SELECT category, SUM(amount) FROM sao.revenue_daily GROUP BY category",
        )
        sm.submit_feedback(sid1, relevant=True, notes="perfect breakdown")

        sm.record_query(sid1, "SELECT * FROM sao.revenue_daily LIMIT 10")
        sm.submit_feedback(sid1, relevant=False, notes="too broad")

        sm.record_query(
            sid1,
            "SELECT category, quarter, SUM(amount) "
            "FROM sao.revenue_daily GROUP BY category, quarter",
        )
        sm.submit_feedback(sid1, relevant=True, notes="great quarterly view")

        ok = sm.end_session(sid1)
        assert ok, "end_session should succeed"

        # -- Session 2: similar description should surface session-1 queries
        sid2, similar2 = sm.start_session(
            agent_id="reporting-agent",
            description="revenue breakdown by product type each quarter",
        )
        assert sid2
        assert sid2 != sid1
        assert len(similar2) > 0, (
            "Session 2 should find queries from session 1 via similarity"
        )

        # Only positively-reviewed queries should appear.
        for sq in similar2:
            assert isinstance(sq, SimilarQuery)
            assert sq.feedback is True
            assert sq.similarity_score > 0.0
            assert sq.session_description  # non-empty

        # The SQL from session-1's approved queries should be present.
        returned_sqls = {sq.sql for sq in similar2}
        assert (
            "SELECT category, SUM(amount) FROM sao.revenue_daily GROUP BY category"
            in returned_sqls
        ), f"Expected approved query not found in {returned_sqls}"

        # The rejected query should NOT be in the results.
        assert (
            "SELECT * FROM sao.revenue_daily LIMIT 10" not in returned_sqls
        ), "Rejected query should not appear"

        sm.end_session(sid2)

    # -- edge cases --------------------------------------------------------

    def test_session_no_queries_does_not_persist(self):
        """A session with zero queries should not add a document."""
        sm = self._make_manager()
        sid, _ = sm.start_session("agent-x", "empty session test")
        sm.end_session(sid)
        assert sm._chroma.count() == 0

    def test_multiple_sessions_accumulate(self):
        """Multiple ended sessions should all be retrievable."""
        sm = self._make_manager()

        # Session A: about customer churn
        sid_a, _ = sm.start_session("agent-a", "customer churn analysis")
        sm.record_query(sid_a, "SELECT customer_id, churn_flag FROM customers")
        sm.submit_feedback(sid_a, relevant=True)
        sm.end_session(sid_a)

        # Session B: about product inventory
        sid_b, _ = sm.start_session("agent-b", "product inventory tracking")
        sm.record_query(sid_b, "SELECT sku, stock_level FROM inventory")
        sm.submit_feedback(sid_b, relevant=True)
        sm.end_session(sid_b)

        assert sm._chroma.count() == 2

        # Session C: asking about churn should find session A, not B
        sid_c, similar_c = sm.start_session("agent-c", "predicting customer churn")
        churn_sqls = {sq.sql for sq in similar_c}
        assert "SELECT customer_id, churn_flag FROM customers" in churn_sqls
        sm.end_session(sid_c)

    def test_similarity_score_ordering(self):
        """Results should be ordered by descending similarity score."""
        sm = self._make_manager()

        sid, _ = sm.start_session("agent", "daily sales summary by region")
        sm.record_query(sid, "SELECT region, SUM(sales) FROM orders GROUP BY region")
        sm.submit_feedback(sid, relevant=True)
        sm.end_session(sid)

        sid2, similar = sm.start_session("agent", "sales totals grouped by region")
        if len(similar) >= 2:
            scores = [sq.similarity_score for sq in similar]
            assert scores == sorted(scores, reverse=True)
        sm.end_session(sid2)

    def test_feedback_notes_preserved(self):
        """Notes attached to feedback should be retrievable in similar queries."""
        sm = self._make_manager()

        sid1, _ = sm.start_session("agent", "monthly active users report")
        sm.record_query(sid1, "SELECT month, COUNT(DISTINCT user_id) FROM events GROUP BY month")
        sm.submit_feedback(sid1, relevant=True, notes="perfect MAU query")
        sm.end_session(sid1)

        sid2, similar = sm.start_session("agent", "report on monthly active users")
        assert len(similar) > 0
        assert any(sq.notes == "perfect MAU query" for sq in similar)
        sm.end_session(sid2)


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
