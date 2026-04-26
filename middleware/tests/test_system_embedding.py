"""
System tests for real embedding API integration.

These tests require a live OpenAI-compatible embeddings endpoint.
They are skipped automatically when ``EMBEDDING_API_URL`` is not set
in the environment.

Run selectively with::

    pytest -m system_embedding

Environment variables:
  EMBEDDING_API_URL   -- base URL (e.g. http://localhost:11434/v1)
  EMBEDDING_API_KEY   -- optional Bearer token
  EMBEDDING_MODEL     -- model name (default: text-embedding-3-small)
"""

import math
import os
import sys
from pathlib import Path

import httpx
import pytest

# Ensure the middleware package is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from session_manager import (
    SessionManager,
    SimilarQuery,
    _ChromaHTTP,
    _api_embedding,
    get_embedding,
)

# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

_EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "")

skip_if_no_embedding_api = pytest.mark.skipif(
    not _EMBEDDING_API_URL,
    reason="EMBEDDING_API_URL not set -- skipping real embedding tests",
)

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
TEST_COLLECTION = "system_embedding_test_sessions"


def _chroma_reachable() -> bool:
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
    try:
        httpx.delete(
            f"http://{CHROMA_HOST}:{CHROMA_PORT}"
            f"/api/v2/tenants/default_tenant/databases/default_database"
            f"/collections/{TEST_COLLECTION}",
            timeout=5.0,
        )
    except Exception:
        pass


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Embedding API tests
# ---------------------------------------------------------------------------


@pytest.mark.system_embedding
@skip_if_no_embedding_api
class TestEmbeddingAPI:
    """Verify that the real embedding API returns well-formed vectors."""

    def test_response_structure(self):
        """The API should return a list of floats."""
        emb = _api_embedding("hello world")
        assert isinstance(emb, list), "Embedding should be a list"
        assert len(emb) > 0, "Embedding should be non-empty"
        assert all(isinstance(v, (int, float)) for v in emb), (
            "Every element should be numeric"
        )

    def test_deterministic_or_near(self):
        """Two calls with the same text should produce very similar vectors."""
        a = _api_embedding("quarterly revenue analysis by region")
        b = _api_embedding("quarterly revenue analysis by region")
        sim = _cosine(a, b)
        assert sim > 0.99, (
            f"Same text should yield near-identical embeddings, got cosine={sim:.4f}"
        )

    def test_similar_texts_higher_cosine(self):
        """Semantically similar texts should have higher cosine similarity."""
        v1 = _api_embedding("quarterly revenue by product category")
        v2 = _api_embedding("revenue breakdown by product type each quarter")
        v3 = _api_embedding("instructions for assembling garden furniture")

        sim_close = _cosine(v1, v2)
        sim_far = _cosine(v1, v3)
        assert sim_close > sim_far, (
            f"Similar texts should be closer: "
            f"similar={sim_close:.4f}  vs  dissimilar={sim_far:.4f}"
        )

    def test_get_embedding_dispatches_to_api(self):
        """get_embedding() should use the API when EMBEDDING_API_URL is set."""
        # Since EMBEDDING_API_URL is set (we only run when it is),
        # get_embedding should produce the same result as _api_embedding.
        api_emb = _api_embedding("test dispatch routing")
        ge_emb = get_embedding("test dispatch routing")
        sim = _cosine(api_emb, ge_emb)
        assert sim > 0.99, (
            f"get_embedding should route to API, got cosine={sim:.4f}"
        )


# ---------------------------------------------------------------------------
# Full session learning loop with real embeddings
# ---------------------------------------------------------------------------


@pytest.mark.system_embedding
@skip_if_no_embedding_api
@skip_if_no_chroma
class TestSessionLearningLoopRealEmbeddings:
    """End-to-end session learning using a real embedding API + ChromaDB."""

    def setup_method(self):
        _cleanup_collection()

    def teardown_method(self):
        _cleanup_collection()

    def _make_manager(self) -> SessionManager:
        sm = SessionManager(chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT)
        sm.COLLECTION_NAME = TEST_COLLECTION
        sm._chroma.get_or_create_collection(
            name=TEST_COLLECTION, metadata={"hnsw:space": "cosine"},
        )
        return sm

    def test_full_learning_loop(self):
        """Session 1 stores queries; session 2 retrieves them via similarity."""
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
            assert sq.session_description

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

    def test_dissimilar_session_not_surfaced(self):
        """A session about a completely different topic should score low."""
        sm = self._make_manager()

        # Session about customer churn
        sid1, _ = sm.start_session("agent-a", "customer churn analysis")
        sm.record_query(sid1, "SELECT customer_id, churn_flag FROM customers")
        sm.submit_feedback(sid1, relevant=True)
        sm.end_session(sid1)

        # Session asking about garden furniture (unrelated)
        sid2, similar = sm.start_session(
            "agent-b", "instructions for assembling garden furniture",
        )
        # Either no results, or results with low similarity
        if similar:
            for sq in similar:
                assert sq.similarity_score < 0.8, (
                    f"Unrelated topic should not match strongly, "
                    f"got score={sq.similarity_score:.4f}"
                )
        sm.end_session(sid2)
