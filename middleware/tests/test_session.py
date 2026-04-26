"""Tests for the Session Manager (offline mode — no ChromaDB)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_manager import SessionManager


def make_manager():
    return SessionManager(chroma_host="nonexistent", chroma_port=9999)


class TestSessionLifecycle:
    def test_start_session(self):
        sm = make_manager()
        sid, similar = sm.start_session("agent-1", "analyze customer churn")
        assert sid
        assert isinstance(similar, list)
        assert len(similar) == 0

    def test_record_and_feedback(self):
        sm = make_manager()
        sid, _ = sm.start_session("agent-1", "test session")

        ok = sm.record_query(sid, "SELECT COUNT(*) FROM sao.customers")
        assert ok

        ok = sm.submit_feedback(sid, relevant=True, notes="useful count")
        assert ok

        session = sm.get_session(sid)
        assert len(session.queries) == 1
        assert session.queries[0].feedback is True
        assert session.queries[0].notes == "useful count"

    def test_end_session(self):
        sm = make_manager()
        sid, _ = sm.start_session("agent-1", "test session")
        sm.record_query(sid, "SELECT 1")
        ok = sm.end_session(sid)
        assert ok
        assert sm.get_session(sid) is None

    def test_invalid_session(self):
        sm = make_manager()
        ok = sm.record_query("nonexistent", "SELECT 1")
        assert not ok

        ok = sm.submit_feedback("nonexistent", relevant=True)
        assert not ok

        ok = sm.end_session("nonexistent")
        assert not ok

    def test_multiple_queries_in_session(self):
        sm = make_manager()
        sid, _ = sm.start_session("agent-1", "revenue analysis")

        sm.record_query(sid, "SELECT * FROM sao.revenue_daily")
        sm.submit_feedback(sid, relevant=True)

        sm.record_query(sid, "SELECT category FROM sao.products")
        sm.submit_feedback(sid, relevant=False, notes="wrong table")

        sm.record_query(sid, "SELECT category, SUM(total_amount) FROM sao.revenue_daily GROUP BY category")
        sm.submit_feedback(sid, relevant=True, notes="this is what I needed")

        session = sm.get_session(sid)
        assert len(session.queries) == 3
        assert session.queries[0].feedback is True
        assert session.queries[1].feedback is False
        assert session.queries[2].feedback is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
