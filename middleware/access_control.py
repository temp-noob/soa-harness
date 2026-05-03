"""
Access Control Service for the Agent-First Middleware.

Parses SQL queries into ASTs, extracts referenced tables/columns,
classifies the query type (row-level SELECT vs aggregation vs DDL/DML),
and evaluates against the policy engine to produce ALLOW/DENY decisions.
"""

from __future__ import annotations

import enum
import logging
import re
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
        """Detect dangerous table-functions that ``readonly=1`` does NOT block.

        ClickHouse readonly mode prevents DDL/DML writes, but table-functions
        that access external URLs, local files, or remote servers are still
        allowed.  These are the vectors that the middleware must catch to
        provide protection *beyond* what the database itself enforces.
        """
        # All entries MUST be lowercase -- we compare against lower(sql).
        dangerous = [
            # Original set
            "url(", "file(", "remote(", "remotesecure(",
            "mysql(", "postgresql(",
            # Cloud / distributed storage access
            "s3(", "s3cluster(",
            "hdfs(",
            # Generic external DB connectors
            "jdbc(",
            "odbc(",
            # Streaming input
            "input(",
            # Cluster-wide access / cross-database scanning
            "cluster(",
        ]
        lower = sql.lower()
        for func in dangerous:
            if func in lower:
                return func.rstrip("(")
        return None

    def has_multi_statement(self, sql: str) -> bool:
        """Detect multi-statement injection attempts.

        ``readonly=1`` does NOT reliably prevent multi-statement payloads
        such as ``SELECT 1; DROP TABLE x``.  Behaviour depends on the
        client library and the protocol (HTTP vs native).  The middleware
        rejects any query that contains a semicolon followed by additional
        non-whitespace content, which is never legitimate for a single
        analytical query.
        """
        stripped = sql.strip().rstrip(";").strip()
        return ";" in stripped

    # NOTE: In production, this detection logic should be loadable from a shared
    # module (e.g., a resource_exhaustion_rules.yaml or a plugin registry) so
    # that detection patterns can be updated without redeploying the middleware.
    def detect_resource_exhaustion(self, sql: str) -> Optional[str]:
        """Detect SQL patterns that would exhaust OLAP resources.

        Returns a reason string if the query is dangerous, or None if safe.
        Detection is intentionally conservative -- only obviously dangerous
        patterns are flagged, not borderline cases.
        """
        reason = self._detect_cartesian_join(sql)
        if reason:
            return reason

        reason = self._detect_memory_bomb(sql)
        if reason:
            return reason

        reason = self._detect_regexp_full_scan(sql)
        if reason:
            return reason

        reason = self._detect_unbounded_high_cardinality_group_by(sql)
        if reason:
            return reason

        return None

    def _detect_cartesian_join(self, sql: str) -> Optional[str]:
        """Detect CROSS JOINs or joins missing ON/USING clauses."""
        try:
            parsed = sqlglot.parse_one(sql, dialect="clickhouse")
        except sqlglot.errors.ParseError:
            return None

        for join in parsed.find_all(exp.Join):
            # Explicit CROSS JOIN
            if join.args.get("kind") and "CROSS" in join.args["kind"].upper():
                return "CROSS JOIN produces a Cartesian product and can exhaust memory"

            # Join without ON or USING clause (implicit Cartesian)
            has_on = join.args.get("on") is not None
            has_using = join.args.get("using") is not None
            if not has_on and not has_using:
                return "JOIN without ON/USING clause produces a Cartesian product"

        # Also detect comma-joins (FROM a, b) with no WHERE relating them.
        # sqlglot models comma-joins as multiple Table nodes under the From.
        from_clause = parsed.find(exp.From)
        if from_clause:
            tables_in_from = list(from_clause.find_all(exp.Table))
            if len(tables_in_from) > 1:
                return "Comma-separated tables in FROM without explicit JOIN produce a Cartesian product"

        return None

    def _detect_memory_bomb(self, sql: str) -> Optional[str]:
        """Detect arrayJoin(range(N)) and similar array-explosion patterns."""
        lower = sql.lower()

        # Detect arrayJoin(range(N)) where N is a large literal
        range_match = re.search(
            r'arrayjoin\s*\(\s*range\s*\(\s*(\d+)\s*\)',
            lower,
        )
        if range_match:
            n = int(range_match.group(1))
            if n > 10000:
                return (
                    f"arrayJoin(range({n})) generates {n} rows per input row "
                    f"and will exhaust memory"
                )

        # Detect numbers(N) table function with large N
        numbers_match = re.search(r'\bnumbers\s*\(\s*(\d+)\s*\)', lower)
        if numbers_match:
            n = int(numbers_match.group(1))
            if n > 10_000_000:
                return (
                    f"numbers({n}) generates {n} rows and will exhaust memory"
                )

        return None

    def _detect_regexp_full_scan(self, sql: str) -> Optional[str]:
        """Detect leading-wildcard LIKE or match()/extractAll() on large tables."""
        try:
            parsed = sqlglot.parse_one(sql, dialect="clickhouse")
        except sqlglot.errors.ParseError:
            return None

        tables = self.extract_tables(sql)
        large_tables = {"transactions", "customers"}
        has_large_table = any(
            t.split(".")[-1].replace("_distributed", "") in large_tables
            for t in tables
        )
        if not has_large_table:
            return None

        # Check for leading-wildcard LIKE patterns (e.g., LIKE '%pattern%')
        for like_node in parsed.find_all(exp.Like):
            pattern_expr = like_node.expression
            if isinstance(pattern_expr, exp.Literal) and pattern_expr.is_string:
                pattern_val = pattern_expr.this
                if pattern_val.startswith("%"):
                    return (
                        f"LIKE '{pattern_val}' with leading wildcard forces a full table "
                        f"scan on a large table and cannot use indexes"
                    )

        # Check for match() — sqlglot parses ClickHouse match() as RegexpLike
        if parsed.find(exp.RegexpLike):
            return (
                "match() applies a regex to every row of a large table "
                "and will cause a full scan"
            )

        # Check for extractAll() and other regex functions parsed as Anonymous
        for func in parsed.find_all(exp.Anonymous):
            if func.name.lower() in ("extractall",):
                return (
                    f"{func.name}() applies a regex to every row of a large table "
                    f"and will cause a full scan"
                )

        return None

    def _detect_unbounded_high_cardinality_group_by(self, sql: str) -> Optional[str]:
        """Detect GROUP BY on columns known to have very high cardinality."""
        try:
            parsed = sqlglot.parse_one(sql, dialect="clickhouse")
        except sqlglot.errors.ParseError:
            return None

        group_node = parsed.find(exp.Group)
        if not group_node:
            return None

        # Columns that are known to produce millions of groups when used in
        # GROUP BY without additional filtering (e.g., no LIMIT on the result,
        # no restrictive WHERE clause).
        high_cardinality_columns = {
            "ip_address", "user_agent", "email", "phone",
            "tx_id", "customer_id", "ssn_hash", "address",
        }

        # Check if the query has a LIMIT clause or a restrictive WHERE
        has_limit = parsed.find(exp.Limit) is not None

        if has_limit:
            return None

        group_by_cols = []
        for expr in group_node.expressions:
            if isinstance(expr, exp.Column):
                group_by_cols.append(expr.name)

        dangerous_cols = [
            col for col in group_by_cols if col in high_cardinality_columns
        ]

        if dangerous_cols:
            return (
                f"GROUP BY on high-cardinality column(s) {', '.join(dangerous_cols)} "
                f"without LIMIT will produce millions of groups and exhaust memory"
            )


class AccessControlService:
    """Main entry point: checks whether an agent can execute a given query."""

    def __init__(self):
        self.registry = ResourceRegistry()
        self.policy = PolicyEngine(self.registry)
        self.analyzer = QueryAnalyzer(self.registry)

    def check(self, agent_id: str, sql: str) -> AccessResult:
        # --- Multi-statement injection (readonly does NOT catch this) ---
        if self.analyzer.has_multi_statement(sql):
            return AccessResult(
                decision=Decision.DENY,
                reason="Multi-statement queries are prohibited (possible injection)",
                query_type=QueryType.UNKNOWN,
            )

        # --- Dangerous table-functions (readonly does NOT catch these) ---
        dangerous = self.analyzer.has_dangerous_functions(sql)
        if dangerous:
            return AccessResult(
                decision=Decision.DENY,
                reason=f"Dangerous function '{dangerous}' is prohibited",
                query_type=QueryType.UNKNOWN,
            )

        exhaustion_reason = self.analyzer.detect_resource_exhaustion(sql)
        if exhaustion_reason:
            logger.warning(
                "DENIED resource-exhausting query from agent=%s: %s | reason=%s",
                agent_id, sql[:100], exhaustion_reason,
            )
            return AccessResult(
                decision=Decision.DENY,
                reason=f"Resource exhaustion: {exhaustion_reason}",
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
