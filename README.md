# SAO Benchmark Harness

**A benchmark for evaluating OLAP system resilience against malicious/abusive AI agent workloads.**

Built for the [SAO Workshop (Supporting Our AI Overlords)](https://bauplanlabs.github.io/SAO-workshop/) — studying what changes when OLAP systems meet AI agents.

## Motivation

Today's OLAP systems were designed for a small number of careful human operators. When AI agents become first-class users, three system-level pressure points emerge (from the [SAO vision paper](https://arxiv.org/abs/2509.00997)):

1. **Correctness under concurrency** — agent swarms with no coordination
2. **Semantic probing** — agents systematically extracting data through analytical queries
3. **Unpredictable access patterns** — agents generating expensive queries with no cost awareness

This harness provides a **reproducible, Docker-based benchmark** that simulates these abuse patterns against a distributed ClickHouse cluster, so researchers can quantitatively evaluate their defensive approaches.

## Architecture

```
┌─────────────────────────────────────────────┐
│              SAO Harness (Python)            │
│  ┌──────────┬──────────┬──────────────────┐  │
│  │ Resource │Concurrent│ Data Exfiltration│  │
│  │Exhaustion│ Flooding │ & Side-Channel   │  │
│  │          │          │ & SQL Injection  │  │
│  └────┬─────┴────┬─────┴────────┬─────────┘  │
│       │          │              │             │
│       ▼          ▼              ▼             │
│  ┌─────────────────────────────────────────┐  │
│  │        ClickHouse Cluster (Docker)      │  │
│  │  ┌──────────┐        ┌──────────┐       │  │
│  │  │ Shard 1  │        │ Shard 2  │       │  │
│  │  │ R1 / R2  │        │ R1 / R2  │       │  │
│  │  └────┬─────┘        └────┬─────┘       │  │
│  │       └──────┬────────────┘             │  │
│  │              ▼                          │  │
│  │         ZooKeeper                       │  │
│  └─────────────────────────────────────────┘  │
│       │                                      │
│       ▼                                      │
│  Prometheus (metrics) + JSON Reports         │
└─────────────────────────────────────────────┘
```

- **2 shards x 2 replicas** — real distributed OLAP, not a toy single-node setup
- **ZooKeeper** — coordination, just like production deployments
- **4 user profiles** — admin, analyst, agent (guarded), agent (unrestricted)
- **Prometheus** — cluster-level metrics during attack runs

## Attack Scenarios

| # | Scenario | What it tests | SAO Pressure Point |
|---|----------|---------------|-------------------|
| 1 | **Resource Exhaustion** | Cartesian joins, memory bombs, full scans, regexp abuse | Predictable performance |
| 2 | **Concurrent Flooding** | 50+ agents running heavy analytics simultaneously | Correctness under concurrency |
| 3 | **Data Exfiltration** | PII access, schema recon, aggregation inference, system snooping | Semantic probing, zero-trust |
| 4 | **Query Injection** | DDL, DML, privilege escalation, remote functions, multi-statement | Agent safety |
| 5 | **Timing Side-Channel** | Credit score inference, existence oracles, binary search via timing | Semantic probing |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+

### Setup & Run

```bash
# 1. Install Python dependencies
make install-deps

# 2. Start the cluster (4 ClickHouse nodes + ZooKeeper + Prometheus)
make up

# 3. Load synthetic data (50k customers, 2M transactions)
make load

# 4. Run the benchmark against the guarded agent profile
make bench

# 5. Run against the unrestricted profile (worst-case baseline)
make bench-unrestricted

# 6. Compare all profiles side-by-side
make compare
```

If `make load` or `make load-small` fails with `TABLE_IS_READ_ONLY` mentioning missing ZooKeeper metadata, your existing ClickHouse volumes are out of sync with ZooKeeper state from an earlier restart. Run `make reset` once to recreate the cluster cleanly. After that, ordinary `make down` / `make up` cycles should keep working because ZooKeeper state is now persisted too.

### Run a specific scenario

```bash
make bench-scenario SCENARIO=data_exfiltration
```

### Custom runs

```bash
python harness.py \
    --host localhost \
    --port 8123 \
    --profile agent \
    --scenarios resource_exhaustion,side_channel \
    --num-agents 100 \
    --output reports/custom_run.json
```

## User Profiles

| Profile | User | Limits | Purpose |
|---------|------|--------|---------|
| `agent` | `agent` | 2GB RAM, 30s timeout, 100k result rows, read-only | **Default test target** — what should defenses look like? |
| `unrestricted` | `agent_unrestricted` | 10GB RAM, 300s timeout, full access | **Worst-case baseline** — no guardrails at all |
| `analyst` | `analyst` | 4GB RAM, 60s timeout, read-only | **Human baseline** — typical analyst constraints |

## Output

The harness produces:
- **Console output** with color-coded probe results and defense grades (A/B/C/F)
- **JSON reports** in `reports/` with full probe details and aggregate metrics
- **Prometheus metrics** at `localhost:9090` for cluster-level observability

### Defense Grading

| Grade | Meaning |
|-------|---------|
| **A** | All attack probes blocked |
| **B** | >80% of attacks blocked |
| **C** | 50-80% blocked |
| **F** | <50% blocked — defenses need work |

## Testing Your Defenses

The benchmark is designed to evaluate approaches like:

1. **Query guardrails** — cost estimation, query rewriting, allow-lists
2. **Access control policies** — column-level security, row-level filtering
3. **Rate limiting** — per-agent query quotas, concurrent query limits
4. **Audit & observability** — detecting probe patterns, anomaly detection
5. **Query sandboxing** — resource isolation, execution plan analysis

To test a defense:

1. Implement your defense (e.g., as a ClickHouse proxy, custom user settings, or middleware)
2. Run `make bench` to get the baseline with default guardrails
3. Run `make bench-unrestricted` to see the worst case
4. Apply your defense and re-run — compare the JSON reports

## Schema

The benchmark uses an e-commerce analytics warehouse with intentionally sensitive data:

- **`sao.customers`** — PII (email, phone, SSN hash, credit score, address)
- **`sao.transactions`** — financial data (card info, IP, fraud scores)
- **`sao.products`** — business data (cost/margins, supplier info)
- **`sao.revenue_daily`** — aggregated view (safe for agents to query)
- **`sao.agent_audit_log`** — tracks agent query activity

## Cleanup

```bash
make down        # Stop containers
make reset       # Destroy volumes and restart fresh
make clean       # Remove report files
```

## Citation

If you use this benchmark in your research:

```bibtex
@misc{sao-benchmark,
  title={SAO Benchmark: Evaluating OLAP Resilience Against AI Agent Abuse},
  year={2026},
  note={Built for the SAO Workshop (Supporting Our AI Overlords): AI Agents and Data Systems}
}
```
