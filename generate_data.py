#!/usr/bin/env python3
"""
SAO Benchmark — Synthetic Data Generator

Populates the ClickHouse cluster with realistic e-commerce analytics
data including PII, financial records, and transaction logs.  The data
is designed so that attack scenarios have meaningful targets to probe.
"""

import argparse
import hashlib
import random
import string
import time
from datetime import datetime, timedelta

import clickhouse_connect

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CATEGORIES = [
    "electronics", "clothing", "home_garden", "sports", "books",
    "toys", "automotive", "health", "grocery", "jewelry",
]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "crypto", "bank_transfer"]
CURRENCIES = ["USD", "EUR", "GBP"]
STATES = [
    "CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
]
TIERS = ["free", "premium", "enterprise"]
FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
]


def fake_email(first: str, last: str, idx: int) -> str:
    domains = ["gmail.com", "yahoo.com", "outlook.com", "company.com"]
    return f"{first.lower()}.{last.lower()}{idx}@{random.choice(domains)}"


def fake_phone() -> str:
    return f"+1{random.randint(2000000000, 9999999999)}"


def fake_ssn_hash() -> str:
    ssn = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
    return hashlib.sha256(ssn.encode()).hexdigest()[:16]


def fake_ip() -> str:
    return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def fake_card_last4() -> str:
    return f"{random.randint(1000, 9999)}"


def fake_address() -> str:
    return f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Pine', 'Elm', 'Cedar', 'Maple'])} {random.choice(['St', 'Ave', 'Blvd', 'Dr', 'Ln'])}"


def random_dt(start: datetime, end: datetime) -> datetime:
    delta = end - start
    secs = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=secs)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_customers(n: int) -> list[dict]:
    now = datetime.now()
    rows = []
    for i in range(1, n + 1):
        first = random.choice(FIRST_NAMES)
        last = random.choice(LAST_NAMES)
        created = random_dt(now - timedelta(days=730), now - timedelta(days=30))
        rows.append({
            "customer_id": i,
            "email": fake_email(first, last, i),
            "full_name": f"{first} {last}",
            "phone": fake_phone(),
            "ssn_hash": fake_ssn_hash(),
            "credit_score": random.randint(300, 850),
            "address": fake_address(),
            "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
            "state": random.choice(STATES),
            "zip_code": f"{random.randint(10000, 99999)}",
            "country": "US",
            "tier": random.choice(TIERS),
            "created_at": created,
            "updated_at": random_dt(created, now),
        })
    return rows


def generate_products(n: int) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        price = random.randint(500, 200000)
        rows.append({
            "product_id": i,
            "name": f"Product-{''.join(random.choices(string.ascii_uppercase, k=4))}-{i}",
            "category": random.choice(CATEGORIES),
            "price_cents": price,
            "cost_cents": int(price * random.uniform(0.2, 0.7)),
            "supplier_id": random.randint(1, 200),
            "is_active": random.choices([1, 0], weights=[9, 1])[0],
            "created_at": random_dt(
                datetime.now() - timedelta(days=365),
                datetime.now() - timedelta(days=10),
            ),
        })
    return rows


def generate_transactions(n: int, num_customers: int, num_products: int) -> list[dict]:
    now = datetime.now()
    rows = []
    for i in range(1, n + 1):
        created = random_dt(now - timedelta(days=365), now)
        status = random.choices(
            ["completed", "pending", "refunded", "fraud"],
            weights=[85, 5, 8, 2],
        )[0]
        rows.append({
            "tx_id": i,
            "customer_id": random.randint(1, num_customers),
            "product_id": random.randint(1, num_products),
            "category": random.choice(CATEGORIES),
            "amount_cents": random.randint(100, 500000),
            "currency": random.choice(CURRENCIES),
            "payment_method": random.choice(PAYMENT_METHODS),
            "card_last4": fake_card_last4(),
            "ip_address": fake_ip(),
            "user_agent": random.choice([
                "Mozilla/5.0 (Agent/v1)", "Agent-SDK/2.1", "DataBot/0.9",
                "Mozilla/5.0 (Windows)", "curl/8.0", "python-requests/2.31",
            ]),
            "status": status,
            "fraud_score": round(random.uniform(0.0, 1.0), 4),
            "created_at": created,
            "processed_at": created + timedelta(seconds=random.randint(1, 300)) if status != "pending" else None,
        })
    return rows


def generate_revenue_daily(transactions: list[dict]) -> list[dict]:
    """Aggregate transactions into a daily revenue summary."""
    agg: dict[tuple, dict] = {}
    for tx in transactions:
        dt = tx["created_at"].date()
        cat = tx["category"]
        key = (dt, cat)
        if key not in agg:
            agg[key] = {"total": 0, "count": 0, "refunds": 0}
        agg[key]["total"] += tx["amount_cents"]
        agg[key]["count"] += 1
        if tx["status"] == "refunded":
            agg[key]["refunds"] += 1

    rows = []
    for (dt, cat), v in agg.items():
        rows.append({
            "dt": dt,
            "category": cat,
            "total_amount": v["total"],
            "tx_count": v["count"],
            "avg_amount": round(v["total"] / v["count"], 2),
            "refund_count": v["refunds"],
        })
    return rows


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def insert_batch(client, table: str, rows: list[dict], batch_size: int = 10000):
    if not rows:
        return
    columns = list(rows[0].keys())
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        data = [[row[c] for c in columns] for row in batch]
        client.insert(table, data, column_names=columns)
    print(f"  Inserted {len(rows)} rows into {table}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAO Benchmark Data Generator")
    parser.add_argument("--host", default="localhost", help="ClickHouse host")
    parser.add_argument("--port", type=int, default=8123, help="ClickHouse HTTP port")
    parser.add_argument("--customers", type=int, default=50000, help="Number of customers")
    parser.add_argument("--products", type=int, default=5000, help="Number of products")
    parser.add_argument("--transactions", type=int, default=2000000, help="Number of transactions")
    args = parser.parse_args()

    print(f"Connecting to ClickHouse at {args.host}:{args.port} ...")
    client = clickhouse_connect.get_client(
        host=args.host, port=args.port, username="default", password=""
    )

    # Wait for schema to be ready
    for attempt in range(30):
        try:
            client.query("SELECT 1 FROM sao.customers LIMIT 0")
            break
        except Exception:
            print(f"  Waiting for schema (attempt {attempt + 1}/30)...")
            time.sleep(2)
    else:
        raise RuntimeError("Schema not ready after 60 seconds")

    print(f"Generating {args.customers} customers...")
    customers = generate_customers(args.customers)
    insert_batch(client, "sao.customers", customers)

    print(f"Generating {args.products} products...")
    products = generate_products(args.products)
    insert_batch(client, "sao.products", products)

    print(f"Generating {args.transactions} transactions...")
    transactions = generate_transactions(args.transactions, args.customers, args.products)
    insert_batch(client, "sao.transactions", transactions)

    print("Generating daily revenue summaries...")
    revenue = generate_revenue_daily(transactions)
    insert_batch(client, "sao.revenue_daily", revenue)

    print("Data generation complete.")


if __name__ == "__main__":
    main()
