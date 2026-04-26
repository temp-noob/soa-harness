"""
Session Manager for the Agent-First Middleware.

Manages agent sessions with vector-DB-backed query learning:
- On session start: embed description, find similar past sessions, return relevant queries
- During session: record queries with agent feedback
- On session end: store session embedding + query tuples for future retrieval

Uses raw httpx calls against the ChromaDB v2 REST API to avoid
version-mismatch issues between the chromadb-client library and the
ChromaDB server (the server runs v1.0.0 which only exposes /api/v2).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight text embedding (fallback)
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 64


def _text_to_embedding(text: str) -> list[float]:
    """Deterministic, lightweight text embedding.

    Produces a unit-norm vector of dimension ``EMBEDDING_DIM`` by hashing
    overlapping character trigrams into buckets and L2-normalising.  This is
    intentionally simple -- all we need is a *stable* projection that
    preserves rough lexical similarity so that cosine-distance retrieval in
    ChromaDB surfaces sessions with similar descriptions.
    """
    vec = [0.0] * EMBEDDING_DIM
    text_lower = text.lower()
    for i in range(max(1, len(text_lower) - 2)):
        trigram = text_lower[i : i + 3]
        h = int(hashlib.sha256(trigram.encode()).hexdigest(), 16)
        bucket = h % EMBEDDING_DIM
        # Use a secondary hash bit to decide sign so that vectors are not
        # all-positive (improves cosine discrimination).
        sign = 1.0 if (h // EMBEDDING_DIM) % 2 == 0 else -1.0
        vec[bucket] += sign
    # L2 normalise
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# ---------------------------------------------------------------------------
# API-backed embedding (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _extract_openai_embedding(data: object) -> Optional[list[float]]:
    """Return embedding from OpenAI-compatible response shape."""
    if not isinstance(data, dict):
        return None

    maybe_data = data.get("data")
    if isinstance(maybe_data, list) and maybe_data:
        first = maybe_data[0]
        if isinstance(first, dict):
            emb = first.get("embedding")
            if isinstance(emb, list):
                return emb
    return None


def _extract_ollama_embedding(data: object) -> Optional[list[float]]:
    """Return embedding from Ollama response shapes."""
    if not isinstance(data, dict):
        return None

    # /api/embeddings shape: {"embedding": [...]} 
    emb = data.get("embedding")
    if isinstance(emb, list):
        return emb

    # /api/embed shape: {"embeddings": [[...]]}
    embeddings = data.get("embeddings")
    if isinstance(embeddings, list) and embeddings:
        first = embeddings[0]
        if isinstance(first, list):
            return first
        if all(isinstance(v, (int, float)) for v in embeddings):
            return embeddings

    return None


def _is_openai_style_base_url(base_url: str) -> bool:
    """Treat URLs ending in /v1 as OpenAI-compatible."""
    return base_url.rstrip("/").endswith("/v1")


def _openai_style_embedding(
    base_url: str,
    text: str,
    model: str,
    headers: dict[str, str],
) -> list[float]:
    """Call OpenAI-compatible embeddings endpoint."""
    resp = httpx.post(
        f"{base_url}/embeddings",
        headers=headers,
        json={"input": text, "model": model},
        timeout=30.0,
    )
    resp.raise_for_status()

    emb = _extract_openai_embedding(resp.json())
    if emb is None:
        raise ValueError("OpenAI embedding response missing data[0].embedding")
    return emb


def _ollama_style_embedding(
    base_url: str,
    text: str,
    model: str,
    headers: dict[str, str],
) -> list[float]:
    """Call Ollama embeddings endpoint(s)."""
    root_url = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url

    # Prefer the classic Ollama endpoint first.
    resp = httpx.post(
        f"{root_url}/api/embeddings",
        headers=headers,
        json={"model": model, "prompt": text},
        timeout=30.0,
    )

    if resp.status_code == 404:
        # Newer endpoint variant.
        resp = httpx.post(
            f"{root_url}/api/embed",
            headers=headers,
            json={"model": model, "input": text},
            timeout=30.0,
        )

    resp.raise_for_status()

    emb = _extract_ollama_embedding(resp.json())
    if emb is None:
        raise ValueError("Ollama embedding response missing embedding payload")
    return emb


def _api_embedding(text: str) -> list[float]:
    """Call embeddings API and return a vector of floats.

        Reads configuration from environment variables:
            - ``EMBEDDING_API_URL``  -- base URL, e.g. ``http://localhost:11434`` or ``https://api.openai.com/v1``
            - ``EMBEDDING_API_KEY``  -- optional Bearer token
            - ``EMBEDDING_MODEL``    -- model name (default: ``text-embedding-3-small``)

        Provider selection is based on ``EMBEDDING_API_URL``:
            - URLs ending with ``/v1`` are treated as OpenAI-compatible
            - all other URLs are treated as Ollama-style
        """
    base_url = os.environ["EMBEDDING_API_URL"].rstrip("/")
    model = os.environ.get("EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL)
    api_key = os.environ.get("EMBEDDING_API_KEY")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if _is_openai_style_base_url(base_url):
        return _openai_style_embedding(base_url, text, model, headers)
    return _ollama_style_embedding(base_url, text, model, headers)


def get_embedding(text: str) -> list[float]:
    """Return an embedding vector for *text*.

    When ``EMBEDDING_API_URL`` is set in the environment, delegates to a real
    OpenAI-compatible embeddings endpoint (see :func:`_api_embedding`).
    Otherwise falls back to the local trigram-hash function
    :func:`_text_to_embedding`.
    """
    if os.environ.get("EMBEDDING_API_URL"):
        return _api_embedding(text)
    return _text_to_embedding(text)


# ---------------------------------------------------------------------------
# ChromaDB HTTP helper
# ---------------------------------------------------------------------------


class _ChromaHTTP:
    """Thin wrapper around the ChromaDB v2 REST API."""

    _BASE = "/api/v2/tenants/default_tenant/databases/default_database"

    def __init__(self, host: str, port: int, timeout: float = 10.0):
        self._client = httpx.Client(
            base_url=f"http://{host}:{port}",
            timeout=timeout,
        )
        self._collection_id: Optional[str] = None

    # -- collection operations ---------------------------------------------

    def get_or_create_collection(
        self, name: str, metadata: Optional[dict] = None,
    ) -> str:
        """Return the collection UUID, creating it if necessary."""
        body: dict = {"name": name, "get_or_create": True}
        if metadata:
            body["metadata"] = metadata
        resp = self._client.post(
            f"{self._BASE}/collections",
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        self._collection_id = data["id"]
        return self._collection_id

    def delete_collection(self, name: str) -> None:
        resp = self._client.delete(f"{self._BASE}/collections/{name}")
        resp.raise_for_status()

    # -- document operations -----------------------------------------------

    def _col_url(self, suffix: str = "") -> str:
        assert self._collection_id, "collection not initialised"
        return f"{self._BASE}/collections/{self._collection_id}{suffix}"

    def count(self) -> int:
        resp = self._client.get(self._col_url("/count"))
        resp.raise_for_status()
        return int(resp.text)

    def add(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        resp = self._client.post(
            self._col_url("/add"),
            json={
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "embeddings": embeddings,
            },
        )
        resp.raise_for_status()

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = 10,
        include: Optional[list[str]] = None,
    ) -> dict:
        payload: dict = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include or ["documents", "metadatas", "distances"],
        }
        resp = self._client.post(self._col_url("/query"), json=payload)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class QueryRecord:
    sql: str
    feedback: Optional[bool] = None
    notes: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    result_summary: Optional[str] = None


@dataclass
class SessionInfo:
    session_id: str
    agent_id: str
    description: str
    queries: list[QueryRecord] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    similar_sessions: list[dict] = field(default_factory=list)


@dataclass
class SimilarQuery:
    sql: str
    feedback: bool
    notes: Optional[str]
    session_description: str
    similarity_score: float


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------


class SessionManager:
    """Manages agent sessions with vector-DB-backed collective learning."""

    COLLECTION_NAME = "agent_sessions"

    def __init__(self, chroma_host: str = "localhost", chroma_port: int = 8000):
        self._sessions: dict[str, SessionInfo] = {}

        try:
            self._chroma = _ChromaHTTP(chroma_host, chroma_port)
            self._chroma.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.info("Connected to ChromaDB at %s:%d", chroma_host, chroma_port)
        except Exception as e:
            logger.warning(
                "ChromaDB unavailable (%s), running without session persistence", e,
            )
            self._available = False
            self._chroma = None  # type: ignore[assignment]

    # -- public API --------------------------------------------------------

    def start_session(
        self,
        agent_id: str,
        description: str,
        top_k: int = 6,
    ) -> tuple[str, list[SimilarQuery]]:
        session_id = str(uuid.uuid4())

        similar_queries: list[SimilarQuery] = []
        if self._available and self._chroma.count() > 0:
            try:
                emb = get_embedding(description)
                results = self._chroma.query(
                    query_embeddings=[emb],
                    n_results=min(top_k, self._chroma.count()),
                )

                if results and results.get("documents"):
                    for i, _doc in enumerate(results["documents"][0]):
                        meta = (
                            results["metadatas"][0][i]
                            if results.get("metadatas")
                            else {}
                        )
                        distance = (
                            results["distances"][0][i]
                            if results.get("distances")
                            else 1.0
                        )
                        similarity = 1.0 - distance

                        queries_str = meta.get("queries_json", "[]")
                        try:
                            stored_queries = json.loads(queries_str)
                        except json.JSONDecodeError:
                            stored_queries = []

                        for sq in stored_queries:
                            if sq.get("feedback") is True:
                                similar_queries.append(
                                    SimilarQuery(
                                        sql=sq["sql"],
                                        feedback=True,
                                        notes=sq.get("notes"),
                                        session_description=meta.get(
                                            "description", ""
                                        ),
                                        similarity_score=similarity,
                                    )
                                )
            except Exception as e:
                logger.warning("Error querying ChromaDB: %s", e)

        similar_queries.sort(key=lambda q: q.similarity_score, reverse=True)
        similar_queries = similar_queries[:top_k]

        session = SessionInfo(
            session_id=session_id,
            agent_id=agent_id,
            description=description,
        )
        self._sessions[session_id] = session

        logger.info(
            "Session %s started for agent=%s, found %d similar queries",
            session_id,
            agent_id,
            len(similar_queries),
        )
        return session_id, similar_queries

    def record_query(
        self,
        session_id: str,
        sql: str,
        result_summary: Optional[str] = None,
    ) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False

        record = QueryRecord(sql=sql, result_summary=result_summary)
        session.queries.append(record)
        return True

    def submit_feedback(
        self,
        session_id: str,
        relevant: bool,
        notes: Optional[str] = None,
    ) -> bool:
        session = self._sessions.get(session_id)
        if not session or not session.queries:
            return False

        last_query = session.queries[-1]
        last_query.feedback = relevant
        last_query.notes = notes
        return True

    def end_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if not session:
            return False

        if not self._available or not session.queries:
            return True

        queries_data = []
        for q in session.queries:
            queries_data.append(
                {
                    "sql": q.sql,
                    "feedback": q.feedback,
                    "notes": q.notes,
                }
            )

        try:
            emb = get_embedding(session.description)
            self._chroma.add(
                ids=[session.session_id],
                documents=[session.description],
                metadatas=[
                    {
                        "agent_id": session.agent_id,
                        "description": session.description,
                        "query_count": str(len(session.queries)),
                        "queries_json": json.dumps(queries_data),
                        "created_at": str(session.created_at),
                    }
                ],
                embeddings=[emb],
            )
            logger.info(
                "Session %s persisted with %d queries",
                session.session_id,
                len(session.queries),
            )
        except Exception as e:
            logger.warning("Failed to persist session %s: %s", session_id, e)

        return True

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        return self._sessions.get(session_id)
