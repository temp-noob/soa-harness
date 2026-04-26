"""
Agent-First OLAP Middleware — FastAPI Application

Sits between AI agents and a ClickHouse OLAP cluster, providing:
1. Session-based query learning via vector DB
2. Intent-based access control (beyond RBAC)
3. Curated schema exploration with semantic metadata

Endpoints:
  POST /session/start          — Start a session, get similar past queries
  POST /session/{id}/query     — Execute a query with access control
  POST /session/{id}/feedback  — Submit feedback on the last query
  POST /session/{id}/end       — End session, persist learnings
  GET  /explore                — Get access-controlled schema metadata
  POST /query                  — Stateless query execution (no session)
  GET  /health                 — Health check
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel

from access_control import AccessControlService, Decision
from explore import ExploreService
from query_proxy import QueryProxy
from session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("middleware")

CH_HOST = os.environ.get("CLICKHOUSE_HOST", "clickhouse-s1r1")
CH_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8123"))
VECTOR_DB_HOST = os.environ.get("VECTOR_DB_HOST", "vector-db")
VECTOR_DB_PORT = int(os.environ.get("VECTOR_DB_PORT", "8000"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting middleware — ClickHouse at %s:%d", CH_HOST, CH_PORT)
    ac = AccessControlService()
    app.state.ac = ac
    app.state.proxy = QueryProxy(ac, CH_HOST, CH_PORT)
    app.state.sessions = SessionManager(VECTOR_DB_HOST, VECTOR_DB_PORT)
    app.state.explore = ExploreService(ac, CH_HOST, CH_PORT)
    yield
    logger.info("Middleware shutting down")


app = FastAPI(
    title="Agent-First OLAP Middleware",
    description="Middleware layer providing session learning, access control, and schema exploration for AI agents interacting with OLAP systems.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Request/Response Models ──────────────────────────────────────────


class SessionStartRequest(BaseModel):
    agent_id: str
    description: str
    top_k: int = 6


class SessionStartResponse(BaseModel):
    session_id: str
    similar_queries: list[dict]


class QueryRequest(BaseModel):
    sql: str
    agent_id: str = "agent"


class QueryResponse(BaseModel):
    allowed: bool
    status_code: int
    body: str
    query_type: str
    reason: str
    execution_time_ms: float
    tables_accessed: list[str]
    columns_accessed: list[str]


class FeedbackRequest(BaseModel):
    relevant: bool
    notes: Optional[str] = None


# ── ClickHouse-Compatible Endpoint ───────────────────────────────────


@app.api_route("/", methods=["GET", "POST"])
async def clickhouse_compat(request: Request):
    """ClickHouse-compatible HTTP endpoint.

    Accepts raw SQL as the POST body with query params like
    ``?user=agent&password=agent_pass`` — the same protocol that
    ``clickhouse_connect`` and other ClickHouse HTTP clients use.

    Runs the SQL through access control before proxying to ClickHouse.
    Denied queries return HTTP 403 with ``ACCESS_DENIED`` in the body
    so the SAO harness correctly marks them as blocked.
    """
    proxy: QueryProxy = app.state.proxy
    ac: AccessControlService = app.state.ac

    raw_body = await request.body()
    body_text = raw_body.decode("utf-8", errors="replace")

    # GET requests may carry the query in ?query= param
    if request.method == "GET":
        body_text = request.query_params.get("query", body_text)

    agent_id = request.query_params.get("user", "agent")

    # Strip FORMAT clause for access control (keep original for forwarding)
    sql_for_check = re.sub(
        r"\s+FORMAT\s+\w+\s*$", "", body_text, flags=re.IGNORECASE,
    ).strip()

    # Allow client initialization queries that clickhouse_connect sends
    # automatically (system.settings, system.functions, etc.).  These are
    # non-sensitive server metadata, unlike system.query_log or
    # system.processes which reveal other users' activity.
    INIT_TABLES = {"system.settings", "system.functions", "system.time_zones",
                   "system.formats", "system.table_functions", "system.data_type_families"}
    sql_upper = sql_for_check.upper()
    if sql_upper.startswith("SELECT") and any(t in sql_for_check.lower() for t in INIT_TABLES):
        params = dict(request.query_params)
        resp = await proxy.execute_raw(raw_body, params)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/plain"),
        )

    if not sql_for_check:
        # Empty body or ping — forward directly to ClickHouse
        params = dict(request.query_params)
        resp = await proxy.execute_raw(raw_body, params)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/plain"),
        )

    access = ac.check(agent_id, sql_for_check)

    if access.decision == Decision.DENY:
        logger.info("BLOCKED [%s]: %s — %s", agent_id, sql_for_check[:80], access.reason)
        return Response(
            content=f"Code: 497. DB::Exception: ACCESS_DENIED: {access.reason}",
            status_code=403,
            media_type="text/plain",
        )

    # Forward the original request to ClickHouse transparently
    params = dict(request.query_params)
    resp = await proxy.execute_raw(raw_body, params)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "text/plain"),
    )


# ── Routes ───────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "service": "agent-first-middleware"}


@app.post("/session/start", response_model=SessionStartResponse)
async def session_start(req: SessionStartRequest):
    sessions: SessionManager = app.state.sessions
    session_id, similar = sessions.start_session(
        agent_id=req.agent_id,
        description=req.description,
        top_k=req.top_k,
    )
    return SessionStartResponse(
        session_id=session_id,
        similar_queries=[
            {
                "sql": q.sql,
                "feedback": q.feedback,
                "notes": q.notes,
                "session_description": q.session_description,
                "similarity_score": q.similarity_score,
            }
            for q in similar
        ],
    )


@app.post("/session/{session_id}/query", response_model=QueryResponse)
async def session_query(session_id: str, req: QueryRequest):
    sessions: SessionManager = app.state.sessions
    proxy: QueryProxy = app.state.proxy

    session = sessions.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = await proxy.execute(
        agent_id=session.agent_id,
        sql=req.sql,
        session_id=session_id,
    )

    sessions.record_query(session_id, req.sql)

    await proxy.log_to_audit(session.agent_id, req.sql, result)

    return QueryResponse(
        allowed=result.allowed,
        status_code=result.status_code,
        body=result.body,
        query_type=result.access_result.query_type.value,
        reason=result.access_result.reason,
        execution_time_ms=result.execution_time_ms,
        tables_accessed=result.access_result.tables_accessed,
        columns_accessed=result.access_result.columns_accessed,
    )


@app.post("/session/{session_id}/feedback")
async def session_feedback(session_id: str, req: FeedbackRequest):
    sessions: SessionManager = app.state.sessions
    ok = sessions.submit_feedback(session_id, req.relevant, req.notes)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found or no queries recorded")
    return {"status": "ok"}


@app.post("/session/{session_id}/end")
async def session_end(session_id: str):
    sessions: SessionManager = app.state.sessions
    ok = sessions.end_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "ok", "message": "Session ended and learnings persisted"}


@app.post("/query", response_model=QueryResponse)
async def stateless_query(req: QueryRequest):
    """Execute a query without a session — access control still applies."""
    proxy: QueryProxy = app.state.proxy

    result = await proxy.execute(agent_id=req.agent_id, sql=req.sql)
    await proxy.log_to_audit(req.agent_id, req.sql, result)

    return QueryResponse(
        allowed=result.allowed,
        status_code=result.status_code,
        body=result.body,
        query_type=result.access_result.query_type.value,
        reason=result.access_result.reason,
        execution_time_ms=result.execution_time_ms,
        tables_accessed=result.access_result.tables_accessed,
        columns_accessed=result.access_result.columns_accessed,
    )


@app.get("/explore")
async def explore(
    agent_id: str = Query(default="agent", description="Agent identifier for access control"),
    database: Optional[str] = Query(default=None, description="Filter to specific database"),
    table: Optional[str] = Query(default=None, description="Filter to specific table"),
):
    explore_svc: ExploreService = app.state.explore
    return await explore_svc.explore(
        agent_id=agent_id,
        database=database,
        table=table,
    )


@app.post("/explore/refresh")
async def refresh_metadata():
    """Force a refresh of the metadata cache from ClickHouse."""
    explore_svc: ExploreService = app.state.explore
    await explore_svc.refresh_cache()
    return {"status": "ok", "message": "Metadata cache refreshed"}
