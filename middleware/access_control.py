"""
Access Control Service for the Agent-First Middleware.

Parses SQL queries into ASTs, extracts referenced tables/columns,
classifies the query type (row-level SELECT vs aggregation vs DDL/DML),
and evaluates against the policy engine to produce ALLOW/DENY decisions.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import sqlglot
import sqlglot.expressions as exp
import yaml

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "config"


class Decision(enum.Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class QueryType(enum.Enum):
    SELECT_ROW = "SELECT_ROW"
    AGGREGATE = "AGGREGATE"
    DDL = "DDL"
    DML = "DML"
    SYSTEM_CMD = "SYSTEM_CMD"
    UNKNOWN = "UNKNOWN"


@dataclass
class AccessResult:
    decision: Decision
    reason: str
    query_type: QueryType
    tables_accessed: list[str] = field(default_factory=list)
    columns_accessed: list[str] = field(default_factory=list)
    denied_columns: list[str] = field(default_factory=list)


@dataclass
class ColumnInfo:
    table: str
    column: str
    classification: str
    full_name: str


class ResourceRegistry:
    """Loads and queries the resource classification registry."""

    def __init__(self, path: Path | None = None):
        path = path or CONFIG_DIR / "resource_registry.yaml"
        with open(path) as f:
            self._data = yaml.safe_load(f)

        self._table_classifications: dict[str, str] = {}
        self._column_classifications: dict[str, str] = {}
        self._sensitivity: dict[str, int] = {}

        for level_name, level_info in self._data.get("sensitivity_levels", {}).items():
            self._sensitivity[level_name] = level_info["level"]

        for db_name, db_info in self._data.get("databases", {}).items():
            for table_name, table_info in db_info.get("tables", {}).items():
                self._table_classifications[table_name] = table_info["classification"]
                for col_name, col_info in table_info.get("columns", {}).items():
                    key = f"{table_name}.{col_name}"
                    self._column_classifications[key] = col_info["classification"]

    def get_table_classification(self, table: str) -> str:
        base = table.split(".")[-1].replace("_distributed", "")
        return self._table_classifications.get(base, "SAFE")

    def get_column_classification(self, table: str, column: str) -> str:
        base = table.split(".")[-1].replace("_distributed", "")
        key = f"{base}.{column}"
        return self._column_classifications.get(key, "SAFE")

    def get_sensitivity_level(self, classification: str) -> int:
        return self._sensitivity.get(classification, 0)

    def is_system_table(self, table: str) -> bool:
        parts = table.split(".")
        if len(parts) >= 2 and parts[0] in ("system", "information_schema"):
            return True
        base = table.split(".")[-1].replace("_distributed", "")
        return self._table_classifications.get(base) == "SYSTEM"


class PolicyEngine:
    """Evaluates queries against role-based policies."""

    def __init__(self, registry: ResourceRegistry, path: Path | None = None):
        self.registry = registry
        path = path or CONFIG_DIR / "policies.yaml"
        with open(path) as f:
            self._data = yaml.safe_load(f)
        self._roles = self._data.get("roles", {})
        self._default_role = self._data.get("default_role", "agent_basic")

    def get_role(self, agent_id: str) -> str:
        if agent_id in self._roles:
            return agent_id
        if agent_id == "agent_unrestricted":
            return "admin"
        return self._default_role

    def evaluate(
        self,
        agent_id: str,
        query_type: QueryType,
        tables: list[str],
        columns: list[ColumnInfo],
    ) -> AccessResult:
        role_name = self.get_role(agent_id)
        role = self._roles.get(role_name)
        if not role:
            return AccessResult(
                decision=Decision.DENY,
                reason=f"Unknown role: {role_name}",
                query_type=query_type,
                tables_accessed=tables,
            )

        if role_name == "admin":
            return AccessResult(
                decision=Decision.ALLOW,
                reason="Admin has unrestricted access",
                query_type=query_type,
                tables_accessed=tables,
                columns_accessed=[c.full_name for c in columns],
            )

        if query_type in (QueryType.DDL, QueryType.DML, QueryType.SYSTEM_CMD):
            return AccessResult(
                decision=Decision.DENY,
                reason=f"Write operations ({query_type.value}) are prohibited for role {role_name}",
                query_type=query_type,
                tables_accessed=tables,
            )

        for table in tables:
            if self.registry.is_system_table(table):
                return AccessResult(
                    decision=Decision.DENY,
                    reason=f"Access to system table '{table}' is prohibited",
                    query_type=query_type,
                    tables_accessed=tables,
                )

        denied_columns = []
        for col in columns:
            classification = col.classification
            sensitivity = self.registry.get_sensitivity_level(classification)

            if sensitivity >= 5 and query_type == QueryType.SELECT_ROW:
                denied_columns.append(col.full_name)

        if denied_columns:
            return AccessResult(
                decision=Decision.DENY,
                reason=f"Row-level access to sensitive columns is prohibited: {', '.join(denied_columns)}",
                query_type=query_type,
                tables_accessed=tables,
                columns_accessed=[c.full_name for c in columns],
                denied_columns=denied_columns,
            )

        return AccessResult(
            decision=Decision.ALLOW,
            reason="Query permitted by policy",
            query_type=query_type,
            tables_accessed=tables,
            columns_accessed=[c.full_name for c in columns],
        )


class QueryAnalyzer:
    """Parses SQL and extracts tables, columns, and query type."""

    DDL_KEYWORDS = {"CREATE", "DROP", "ALTER", "TRUNCATE", "RENAME"}
    DML_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "MERGE"}
    SYSTEM_KEYWORDS = {"SYSTEM", "GRANT", "REVOKE", "SET", "KILL"}

    AGGREGATE_FUNCTIONS = {
        "count", "sum", "avg", "min", "max", "any", "anyLast",
        "groupArray", "groupUniqArray", "uniq", "uniqExact",
        "quantile", "median", "stddevPop", "stddevSamp",
        "varPop", "varSamp", "covarPop", "covarSamp",
        "argMin", "argMax", "topK", "histogram",
    }

    def __init__(self, registry: ResourceRegistry):
        self.registry = registry

    def classify_query(self, sql: str) -> QueryType:
        normalized = sql.strip().upper()

        for kw in self.DDL_KEYWORDS:
            if normalized.startswith(kw):
                return QueryType.DDL

        for kw in self.DML_KEYWORDS:
            if normalized.startswith(kw):
                return QueryType.DML

        for kw in self.SYSTEM_KEYWORDS:
            if normalized.startswith(kw):
                return QueryType.SYSTEM_CMD

        if not normalized.startswith("SELECT"):
            return QueryType.UNKNOWN

        try:
            parsed = sqlglot.parse_one(sql, dialect="clickhouse")
        except sqlglot.errors.ParseError:
            return QueryType.UNKNOWN

        if self._has_aggregation(parsed):
            return QueryType.AGGREGATE

        return QueryType.SELECT_ROW

    def _has_aggregation(self, tree: exp.Expression) -> bool:
        if tree.find(exp.Group):
            return True

        for func in tree.find_all(exp.Anonymous):
            if func.name.lower() in self.AGGREGATE_FUNCTIONS:
                return True

        agg_types = (
            exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max,
            exp.Stddev, exp.Variance,
        )
        for agg_type in agg_types:
            if tree.find(agg_type):
                return True

        return False

    def extract_tables(self, sql: str) -> list[str]:
        try:
            parsed = sqlglot.parse_one(sql, dialect="clickhouse")
        except sqlglot.errors.ParseError:
            return self._extract_tables_regex(sql)

        tables = []
        for table in parsed.find_all(exp.Table):
            parts = []
            if table.catalog:
                parts.append(table.catalog)
            if table.db:
                parts.append(table.db)
            parts.append(table.name)
            tables.append(".".join(parts))

        return list(set(tables))

    def _extract_tables_regex(self, sql: str) -> list[str]:
        import re
        pattern = r'\b(?:FROM|JOIN|INTO|UPDATE|TABLE)\s+([a-zA-Z_][\w.]*)'
        return list(set(re.findall(pattern, sql, re.IGNORECASE)))

    def extract_columns(self, sql: str, tables: list[str]) -> list[ColumnInfo]:
        try:
            parsed = sqlglot.parse_one(sql, dialect="clickhouse")
        except sqlglot.errors.ParseError:
            return []

        columns = []
        seen = set()

        for col in parsed.find_all(exp.Column):
            col_name = col.name
            table_name = col.table or (tables[0] if len(tables) == 1 else "")

            if not table_name and tables:
                table_name = tables[0]

            full_name = f"{table_name}.{col_name}" if table_name else col_name

            if full_name in seen:
                continue
            seen.add(full_name)

            classification = self.registry.get_column_classification(table_name, col_name)
            columns.append(ColumnInfo(
                table=table_name,
                column=col_name,
                classification=classification,
                full_name=full_name,
            ))

        return columns

    def has_dangerous_functions(self, sql: str) -> Optional[str]:
        dangerous = ["url(", "file(", "remote(", "remoteSecure(", "mysql(", "postgresql("]
        lower = sql.lower()
        for func in dangerous:
            if func in lower:
                return func.rstrip("(")
        return None


class AccessControlService:
    """Main entry point: checks whether an agent can execute a given query."""

    def __init__(self):
        self.registry = ResourceRegistry()
        self.policy = PolicyEngine(self.registry)
        self.analyzer = QueryAnalyzer(self.registry)

    def check(self, agent_id: str, sql: str) -> AccessResult:
        dangerous = self.analyzer.has_dangerous_functions(sql)
        if dangerous:
            return AccessResult(
                decision=Decision.DENY,
                reason=f"Dangerous function '{dangerous}' is prohibited",
                query_type=QueryType.UNKNOWN,
            )

        query_type = self.analyzer.classify_query(sql)
        tables = self.analyzer.extract_tables(sql)
        columns = self.analyzer.extract_columns(sql, tables)

        result = self.policy.evaluate(agent_id, query_type, tables, columns)
        if result.decision == Decision.DENY:
            logger.warning(
                "DENIED query from agent=%s: %s | reason=%s",
                agent_id, sql[:100], result.reason,
            )
        return result
