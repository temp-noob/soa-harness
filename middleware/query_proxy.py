"""
Query Proxy for the Agent-First Middleware.

Sits between agents and ClickHouse:
1. Receives SQL from the agent
2. Runs it through the Access Control Service
3. If allowed, forwards to ClickHouse and returns results
4. Records the query in the session manager
5. Logs to the agent audit log
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from access_control import AccessControlService, AccessResult, Decision

logger = logging.getLogger(__name__)


@dataclass
class ProxyResult:
    allowed: bool
    status_code: int
    body: str
    access_result: AccessResult
    execution_time_ms: float
    rows_read: int = 0
    bytes_read: int = 0


class QueryProxy:
    """Proxies SQL queries to ClickHouse with access control."""

    def __init__(
        self,
        access_control: AccessControlService,
        clickhouse_host: str = "localhost",
        clickhouse_port: int = 8123,
        clickhouse_user: str = "default",
        clickhouse_password: str = "",
    ):
        self.ac = access_control
        self.ch_host = clickhouse_host
        self.ch_port = clickhouse_port
        self.ch_user = clickhouse_user
        self.ch_password = clickhouse_password

    async def execute(
        self,
        agent_id: str,
        sql: str,
        session_id: Optional[str] = None,
    ) -> ProxyResult:
        start = time.monotonic()

        access = self.ac.check(agent_id, sql)

        if access.decision == Decision.DENY:
            elapsed = (time.monotonic() - start) * 1000
            return ProxyResult(
                allowed=False,
                status_code=403,
                body=f"Access denied: {access.reason}",
                access_result=access,
                execution_time_ms=elapsed,
            )

        url = f"http://{self.ch_host}:{self.ch_port}/"
        params = {"user": self.ch_user}
        if self.ch_password:
            params["password"] = self.ch_password

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    url,
                    content=sql,
                    params=params,
                    timeout=30.0,
                )
            elapsed = (time.monotonic() - start) * 1000

            return ProxyResult(
                allowed=True,
                status_code=resp.status_code,
                body=resp.text,
                access_result=access,
                execution_time_ms=elapsed,
            )

        except httpx.TimeoutException:
            elapsed = (time.monotonic() - start) * 1000
            return ProxyResult(
                allowed=True,
                status_code=504,
                body="Query timed out",
                access_result=access,
                execution_time_ms=elapsed,
            )
        except httpx.ConnectError as e:
            elapsed = (time.monotonic() - start) * 1000
            return ProxyResult(
                allowed=True,
                status_code=502,
                body=f"Cannot connect to ClickHouse: {e}",
                access_result=access,
                execution_time_ms=elapsed,
            )

    async def execute_raw(
        self,
        body: bytes,
        query_params: dict[str, str],
    ) -> httpx.Response:
        """Forward a raw request to ClickHouse and return the raw response.

        Used by the ClickHouse-compatible endpoint to transparently proxy
        requests from clickhouse_connect without parsing or reformatting.
        """
        url = f"http://{self.ch_host}:{self.ch_port}/"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                content=body,
                params=query_params,
                timeout=60.0,
            )
        return resp

    async def log_to_audit(
        self,
        agent_id: str,
        sql: str,
        result: ProxyResult,
    ) -> None:
        query_hash = int(hashlib.sha256(sql.encode()).hexdigest()[:16], 16)
        was_blocked = 0 if result.allowed else 1
        block_reason = "" if result.allowed else result.access_result.reason

        tables_str = "','".join(result.access_result.tables_accessed)
        columns_str = "','".join(result.access_result.columns_accessed)

        escaped_sql = sql.replace("'", "\\'").replace("\\", "\\\\")

        audit_sql = f"""
        INSERT INTO sao.agent_audit_log
        (agent_id, query_text, query_hash, tables_accessed, columns_accessed,
         rows_read, bytes_read, execution_time_ms, was_blocked, block_reason)
        VALUES (
            '{agent_id}',
            '{escaped_sql}',
            {query_hash},
            ['{tables_str}'],
            ['{columns_str}'],
            {result.rows_read},
            {result.bytes_read},
            {int(result.execution_time_ms)},
            {was_blocked},
            {'NULL' if not block_reason else f"'{block_reason}'"}
        )
        """

        try:
            url = f"http://{self.ch_host}:{self.ch_port}/"
            async with httpx.AsyncClient() as client:
                await client.post(
                    url,
                    content=audit_sql,
                    params={"user": "default"},
                    timeout=5.0,
                )
        except Exception as e:
            logger.warning("Failed to write audit log: %s", e)
