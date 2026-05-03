"""
Analytical tasks for the Agent Efficiency Benchmark.

Each task represents a realistic analytical question that an LLM agent
must answer by querying the SAO e-commerce database.  Tasks have a known
gold-standard SQL query and a validation function that checks whether
the agent's answer is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class AnalyticalTask:
    id: str
    question: str
    difficulty: str  # easy, medium, hard
    gold_sql: str
    validate: Callable[[list[list[Any]]], bool]
    description: str = ""


def _has_rows(result: list[list[Any]]) -> bool:
    return len(result) > 0


def _has_n_rows(n: int) -> Callable:
    def check(result: list[list[Any]]) -> bool:
        return len(result) == n
    return check


def _has_numeric_col(col: int) -> Callable:
    def check(result: list[list[Any]]) -> bool:
        return len(result) > 0 and all(
            isinstance(r[col], (int, float)) for r in result
        )
    return check


TASKS: list[AnalyticalTask] = [
    AnalyticalTask(
        id="revenue_by_category",
        question="What is the total revenue (in USD) by product category for completed transactions? Return category and total revenue, ordered by revenue descending.",
        difficulty="easy",
        gold_sql=(
            "SELECT category, SUM(amount_cents) / 100 AS revenue_usd "
            "FROM sao.transactions_distributed "
            "WHERE status = 'completed' "
            "GROUP BY category "
            "ORDER BY revenue_usd DESC"
        ),
        validate=lambda r: len(r) > 0 and all(isinstance(row[1], (int, float)) for row in r),
        description="Tests basic aggregation on the transactions table.",
    ),
    AnalyticalTask(
        id="top_5_states",
        question="Which are the top 5 states by number of enterprise-tier customers? Return state and count.",
        difficulty="easy",
        gold_sql=(
            "SELECT state, COUNT(*) AS cnt "
            "FROM sao.customers_distributed "
            "WHERE tier = 'enterprise' "
            "GROUP BY state "
            "ORDER BY cnt DESC "
            "LIMIT 5"
        ),
        validate=_has_n_rows(5),
        description="Tests filtering + aggregation on customers table.",
    ),
    AnalyticalTask(
        id="daily_revenue_trend",
        question="What is the daily total revenue (in USD) for the most recent 30 days in the revenue_daily table? Return date and total revenue.",
        difficulty="easy",
        gold_sql=(
            "SELECT dt, SUM(total_amount) / 100 AS revenue_usd "
            "FROM sao.revenue_daily_distributed "
            "GROUP BY dt "
            "ORDER BY dt DESC "
            "LIMIT 30"
        ),
        validate=lambda r: len(r) > 0 and len(r) <= 30,
        description="Tests the pre-aggregated revenue_daily table.",
    ),
    AnalyticalTask(
        id="category_refund_rate",
        question="What is the refund rate (refund_count / tx_count) by product category across all time? Return category and refund rate, ordered by refund rate descending.",
        difficulty="medium",
        gold_sql=(
            "SELECT category, "
            "SUM(refund_count) / SUM(tx_count) AS refund_rate "
            "FROM sao.revenue_daily_distributed "
            "GROUP BY category "
            "ORDER BY refund_rate DESC"
        ),
        validate=lambda r: len(r) > 0 and all(0 <= row[1] <= 1 for row in r),
        description="Tests ratio computation across aggregated data.",
    ),
    AnalyticalTask(
        id="payment_method_breakdown",
        question="What is the distribution of payment methods for completed transactions? Return payment method and percentage of total transactions, ordered by percentage descending.",
        difficulty="medium",
        gold_sql=(
            "SELECT payment_method, "
            "COUNT(*) * 100.0 / (SELECT COUNT(*) FROM sao.transactions_distributed WHERE status = 'completed') AS pct "
            "FROM sao.transactions_distributed "
            "WHERE status = 'completed' "
            "GROUP BY payment_method "
            "ORDER BY pct DESC"
        ),
        validate=lambda r: len(r) > 0 and abs(sum(row[1] for row in r) - 100.0) < 1.0,
        description="Tests subquery + percentage computation.",
    ),
    AnalyticalTask(
        id="customer_tier_revenue",
        question="What is the average transaction amount (in USD) by customer tier? Join the customers and transactions tables. Return tier and average amount.",
        difficulty="medium",
        gold_sql=(
            "SELECT c.tier, AVG(t.amount_cents) / 100 AS avg_usd "
            "FROM sao.transactions t "
            "JOIN sao.customers c ON t.customer_id = c.customer_id "
            "WHERE t.status = 'completed' "
            "GROUP BY c.tier "
            "ORDER BY avg_usd DESC"
        ),
        validate=lambda r: len(r) == 3,
        description="Tests JOIN between transactions and customers.",
    ),
    AnalyticalTask(
        id="monthly_growth",
        question="What is the month-over-month revenue growth rate? Using revenue_daily, compute total monthly revenue and the percentage change from the previous month. Return month, revenue, and growth rate.",
        difficulty="hard",
        gold_sql=(
            "SELECT month, revenue_usd, "
            "round((revenue_usd - prev_revenue) / prev_revenue * 100, 2) AS growth_pct "
            "FROM ("
            "  SELECT toStartOfMonth(dt) AS month, "
            "  SUM(total_amount) / 100 AS revenue_usd, "
            "  lagInFrame(SUM(total_amount) / 100) OVER (ORDER BY toStartOfMonth(dt)) AS prev_revenue "
            "  FROM sao.revenue_daily_distributed "
            "  GROUP BY month "
            "  ORDER BY month"
            ") "
            "WHERE prev_revenue > 0"
        ),
        validate=lambda r: len(r) > 0,
        description="Tests window functions and month-over-month computation.",
    ),
    AnalyticalTask(
        id="product_category_aov",
        question="What is the average order value (AOV) in USD by product category, and how many distinct products are in each category? Return category, AOV, and product count.",
        difficulty="hard",
        gold_sql=(
            "SELECT t.category, "
            "AVG(t.amount_cents) / 100 AS aov_usd, "
            "COUNT(DISTINCT t.product_id) AS product_count "
            "FROM sao.transactions_distributed t "
            "WHERE t.status = 'completed' "
            "GROUP BY t.category "
            "ORDER BY aov_usd DESC"
        ),
        validate=lambda r: len(r) > 0 and all(len(row) >= 3 for row in r),
        description="Tests multi-metric aggregation.",
    ),
]
