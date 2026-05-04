#!/usr/bin/env python3
"""
Plot the session learning saturation curve from Experiment #3.

Produces a 2-row x 3-column figure:
  Top row:    similar_queries_received vs agent_run (per task)
  Bottom row: queries_executed vs agent_run (per task)

Usage:
    python tables/plot_saturation_curve.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path(__file__).parent.parent / "reports" / "experiment_saturation_curve.json"
OUTPUT = Path(__file__).parent / "saturation_curve.png"

with open(REPORT) as f:
    data = json.load(f)

tasks_order = ["revenue_by_category", "category_refund_rate", "product_category_aov"]
task_labels = {
    "revenue_by_category": "Revenue by Category",
    "category_refund_rate": "Category Refund Rate",
    "product_category_aov": "Product Category AOV",
}

by_task: dict[str, dict] = {}
for r in data["results"]:
    tid = r["task_id"]
    if tid not in by_task:
        by_task[tid] = {"runs": [], "queries": [], "similar": [], "tokens": []}
    by_task[tid]["runs"].append(r["agent_run"])
    by_task[tid]["queries"].append(r["queries_executed"])
    by_task[tid]["similar"].append(r["similar_queries_received"])
    by_task[tid]["tokens"].append(r["llm_tokens_used"])

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
fig.suptitle("Experiment #3: Session Learning Saturation Curve (gpt-4o)", fontsize=14, fontweight="bold")

colors = {"similar": "#2196F3", "queries": "#FF5722", "tokens": "#4CAF50"}

for col, tid in enumerate(tasks_order):
    td = by_task[tid]
    runs = td["runs"]

    ax_sim = axes[0][col]
    ax_sim.plot(runs, td["similar"], "o-", color=colors["similar"], linewidth=2, markersize=6)
    ax_sim.set_ylabel("Similar Queries Received" if col == 0 else "")
    ax_sim.set_title(task_labels[tid])
    ax_sim.set_ylim(-0.5, max(td["similar"]) + 1)
    ax_sim.axhline(y=6, color=colors["similar"], linestyle="--", alpha=0.3, label="top_k cap (6)")
    ax_sim.grid(True, alpha=0.3)
    if col == 0:
        ax_sim.legend(fontsize=8)

    ax_q = axes[1][col]
    ax_q.plot(runs, td["queries"], "s-", color=colors["queries"], linewidth=2, markersize=6)
    ax_q.set_xlabel("Agent Run")
    ax_q.set_ylabel("Queries Executed" if col == 0 else "")
    ax_q.set_ylim(0, max(td["queries"]) + 1)
    ax_q.set_xticks(runs)
    ax_q.grid(True, alpha=0.3)

    final_q = td["queries"][-1]
    ax_q.axhline(y=final_q, color=colors["queries"], linestyle="--", alpha=0.3)
    ax_q.annotate(f"converges to {final_q}", xy=(runs[-1], final_q),
                  xytext=(-60, 15), textcoords="offset points", fontsize=8,
                  arrowprops=dict(arrowstyle="->", color="gray"),
                  color=colors["queries"])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {OUTPUT}")
