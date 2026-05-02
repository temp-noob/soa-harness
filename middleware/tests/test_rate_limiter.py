"""Tests for the per-agent rate limiter."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rate_limiter import RateLimiter


class TestRateLimiter:

    def test_allows_within_limit(self):
        rl = RateLimiter(max_requests=5, window_s=60)
        for _ in range(5):
            allowed, retry = rl.check("agent-1")
            assert allowed
            assert retry == 0.0

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=3, window_s=60)
        for _ in range(3):
            rl.check("agent-1")
        allowed, retry = rl.check("agent-1")
        assert not allowed
        assert retry > 0

    def test_retry_after_is_positive(self):
        rl = RateLimiter(max_requests=2, window_s=10)
        rl.check("agent-1")
        rl.check("agent-1")
        allowed, retry = rl.check("agent-1")
        assert not allowed
        assert 0 < retry <= 10

    def test_per_agent_isolation(self):
        rl = RateLimiter(max_requests=2, window_s=60)
        rl.check("agent-1")
        rl.check("agent-1")
        allowed_1, _ = rl.check("agent-1")
        assert not allowed_1

        allowed_2, _ = rl.check("agent-2")
        assert allowed_2

    def test_window_expiry(self):
        rl = RateLimiter(max_requests=2, window_s=0.1)
        rl.check("agent-1")
        rl.check("agent-1")
        allowed, _ = rl.check("agent-1")
        assert not allowed

        time.sleep(0.15)
        allowed, _ = rl.check("agent-1")
        assert allowed

    def test_enabled_property(self):
        rl = RateLimiter(max_requests=10, window_s=60)
        assert rl.enabled

        rl_disabled = RateLimiter(max_requests=0, window_s=60)
        assert not rl_disabled.enabled


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
