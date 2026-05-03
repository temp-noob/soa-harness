"""
Per-agent token-bucket rate limiter for the Agent-First Middleware.

Limits the number of queries each agent can execute within a time window.
When an agent exceeds its quota, the middleware returns HTTP 429 with a
``Retry-After`` header indicating how many seconds the agent should wait.

Configuration is via environment variables:
    RATE_LIMIT_REQUESTS   Max requests per window (default: 20)
    RATE_LIMIT_WINDOW_S   Window size in seconds  (default: 60)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DEFAULT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "20"))
DEFAULT_WINDOW_S = float(os.environ.get("RATE_LIMIT_WINDOW_S", "60"))


@dataclass
class _Bucket:
    timestamps: list[float] = field(default_factory=list)


class RateLimiter:
    """Sliding-window rate limiter keyed by agent ID."""

    def __init__(
        self,
        max_requests: int = DEFAULT_REQUESTS,
        window_s: float = DEFAULT_WINDOW_S,
    ):
        self.max_requests = max_requests
        self.window_s = window_s
        self._buckets: dict[str, _Bucket] = {}

    def _prune(self, bucket: _Bucket, now: float) -> None:
        cutoff = now - self.window_s
        bucket.timestamps = [t for t in bucket.timestamps if t > cutoff]

    def check(self, agent_id: str) -> tuple[bool, float]:
        """Check whether *agent_id* may proceed.

        Returns ``(allowed, retry_after_s)``.
        - If allowed: ``(True, 0.0)``
        - If rate-limited: ``(False, seconds_until_a_slot_opens)``
        """
        now = time.monotonic()
        bucket = self._buckets.setdefault(agent_id, _Bucket())
        self._prune(bucket, now)

        if len(bucket.timestamps) < self.max_requests:
            bucket.timestamps.append(now)
            return True, 0.0

        oldest = bucket.timestamps[0]
        retry_after = (oldest + self.window_s) - now
        retry_after = max(retry_after, 0.1)

        logger.info(
            "RATE_LIMITED agent=%s (%d/%d in %.0fs window, retry in %.1fs)",
            agent_id, len(bucket.timestamps), self.max_requests,
            self.window_s, retry_after,
        )
        return False, retry_after

    @property
    def enabled(self) -> bool:
        return self.max_requests > 0
