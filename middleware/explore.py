"""
Explore Endpoint for the Agent-First Middleware.

Provides agents with a curated, access-controlled view of database metadata:
- Fetches schema from ClickHouse system tables
- Enriches with semantic descriptions from metadata.yaml
- Filters tables/columns by agent access level
- Returns statistics and sample queries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml

from access_control import AccessControlService, Decision, QueryType

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "config"


@dataclass
class ColumnMetadata:
    name: str
    type: str
    description: str = ""
    access: str = "full"
    reason: str = ""
    classification: str = "SAFE"
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableMetadata:
    name: str
    description: str = ""
    access: str = "full"
    row_count: int = 0
    columns: list[ColumnMetadata] = field(default_factory=list)
    sample_queries: list[dict[str, str]] = field(default_factory=list)
    partition_key: str = ""
    primary_key: str = ""
    recommended_joins: list[dict] = field(default_factory=list)


@dataclass
class DatabaseMetadata:
    name: str
    description: str = ""
    tables: list[TableMetadata] = field(default_factory=list)


class ExploreService:
    """Builds access-controlled schema metadata for agents."""

    def __init__(
        self,
        access_control: AccessControlService,
        clickhouse_host: str = "localhost",
        clickhouse_port: int = 8123,
    ):
        self.ac = access_control
        self.ch_host = clickhouse_host
        self.ch_port = clickhouse_port
        self._metadata_config = self._load_metadata_config()
        self._cache: Optional[dict[str, DatabaseMetadata]] = None

    def _load_metadata_config(self) -> dict:
        path = CONFIG_DIR / "metadata.yaml"
        with open(path) as f:
            return yaml.safe_load(f)

    async def _query_clickhouse(self, sql: str) -> list[dict]:
        url = f"http://{self.ch_host}:{self.ch_port}/"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                content=sql + " FORMAT JSON",
                params={"user": "default"},
                timeout=10.0,
            )
            if resp.status_code != 200:
                logger.error("ClickHouse query failed: %s", resp.text[:200])
                return []
            data = resp.json()
            return data.get("data", [])

    async def _fetch_tables(self, database: str) -> list[dict]:
        sql = f"""
        SELECT name, engine, total_rows, total_bytes
        FROM system.tables
        WHERE database = '{database}'
          AND name NOT LIKE '%_distributed'
          AND engine NOT IN ('Distributed')
        ORDER BY name
        """
        return await self._query_clickhouse(sql)

    async def _fetch_columns(self, database: str, table: str) -> list[dict]:
        sql = f"""
        SELECT name, type
        FROM system.columns
        WHERE database = '{database}' AND table = '{table}'
        ORDER BY position
        """
        return await self._query_clickhouse(sql)

    async def _fetch_column_stats(
        self, database: str, table: str, column: str, col_type: str
    ) -> dict[str, Any]:
        if any(t in col_type for t in ("String", "Array", "Enum", "Nullable")):
            try:
                sql = f"SELECT count() as cnt FROM {database}.{table}"
                rows = await self._query_clickhouse(sql)
                if rows:
                    return {"count": int(rows[0]["cnt"])}
            except Exception:
                pass
            return {}

        try:
            sql = f"""
            SELECT
                min({column}) as min_val,
                max({column}) as max_val,
                avg({column}) as avg_val
            FROM {database}.{table}
            """
            rows = await self._query_clickhouse(sql)
            if rows:
                return {
                    "min": rows[0].get("min_val"),
                    "max": rows[0].get("max_val"),
                    "avg": rows[0].get("avg_val"),
                }
        except Exception:
            pass
        return {}

    async def refresh_cache(self) -> None:
        """Fetch metadata from ClickHouse and build the cache."""
        databases: dict[str, DatabaseMetadata] = {}
        db_config = self._metadata_config.get("databases", {})

        for db_name, db_info in db_config.items():
            db_meta = DatabaseMetadata(
                name=db_name,
                description=db_info.get("description", ""),
            )

            ch_tables = await self._fetch_tables(db_name)
            table_configs = db_info.get("tables", {})

            for t in ch_tables:
                table_name = t["name"]
                table_config = table_configs.get(table_name, {})

                table_meta = TableMetadata(
                    name=table_name,
                    description=table_config.get("description", ""),
                    row_count=int(t.get("total_rows") or 0),
                    sample_queries=table_config.get("sample_queries", []),
                    recommended_joins=table_config.get("recommended_joins", []),
                )

                ch_columns = await self._fetch_columns(db_name, table_name)
                col_configs = {}
                reg_tables = self.ac.registry._data.get("databases", {}).get(db_name, {}).get("tables", {})
                if table_name in reg_tables:
                    col_configs = reg_tables[table_name].get("columns", {})

                for c in ch_columns:
                    col_name = c["name"]
                    col_config = col_configs.get(col_name, {})
                    classification = col_config.get("classification", "SAFE")

                    col_meta = ColumnMetadata(
                        name=col_name,
                        type=c["type"],
                        description=col_config.get("description", ""),
                        classification=classification,
                    )
                    table_meta.columns.append(col_meta)

                db_meta.tables.append(table_meta)

            databases[db_name] = db_meta

        self._cache = databases
        logger.info("Metadata cache refreshed: %d databases", len(databases))

    async def explore(
        self,
        agent_id: str,
        database: Optional[str] = None,
        table: Optional[str] = None,
    ) -> dict:
        if not self._cache:
            await self.refresh_cache()

        result = {"databases": []}

        for db_name, db_meta in self._cache.items():
            if database and db_name != database:
                continue

            db_output = {
                "name": db_meta.name,
                "description": db_meta.description,
                "tables": [],
            }

            for table_meta in db_meta.tables:
                if table and table_meta.name != table:
                    continue

                table_class = self.ac.registry.get_table_classification(table_meta.name)
                if table_class == "SYSTEM":
                    continue

                table_output = {
                    "name": table_meta.name,
                    "description": table_meta.description,
                    "row_count": table_meta.row_count,
                    "columns": [],
                    "sample_queries": [],
                    "recommended_joins": table_meta.recommended_joins,
                }

                has_restricted = False
                for col in table_meta.columns:
                    sensitivity = self.ac.registry.get_sensitivity_level(col.classification)

                    if sensitivity >= 7:
                        col_output = {
                            "name": col.name,
                            "type": col.type,
                            "access": "denied",
                            "reason": "PII",
                            "description": col.description,
                        }
                        has_restricted = True
                    elif sensitivity >= 5:
                        col_output = {
                            "name": col.name,
                            "type": col.type,
                            "access": "aggregation_only",
                            "description": col.description,
                        }
                        if col.stats:
                            col_output["stats"] = col.stats
                        has_restricted = True
                    else:
                        col_output = {
                            "name": col.name,
                            "type": col.type,
                            "access": "full",
                            "description": col.description,
                        }
                        if col.stats:
                            col_output["stats"] = col.stats

                    table_output["columns"].append(col_output)

                if has_restricted:
                    table_output["access"] = "aggregation_only"
                else:
                    table_output["access"] = "full"

                for sq in table_meta.sample_queries:
                    test_sql = sq["sql"] if isinstance(sq, dict) else sq
                    check = self.ac.check(agent_id, test_sql)
                    if check.decision == Decision.ALLOW:
                        table_output["sample_queries"].append(sq)

                db_output["tables"].append(table_output)

            if db_output["tables"]:
                result["databases"].append(db_output)

        return result
