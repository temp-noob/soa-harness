"""Generate the AgentGate architecture diagram for the paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_box(ax, xy, w, h, label, color, fontsize=9, bold=False, sublabel=None):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="black",
        linewidth=1.2,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    text_y = y + h / 2 if sublabel is None else y + h * 0.62
    ax.text(
        x + w / 2, text_y, label,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight,
    )
    if sublabel:
        ax.text(
            x + w / 2, y + h * 0.30, sublabel,
            ha="center", va="center",
            fontsize=6.5, fontstyle="italic", color="#444444",
        )

def draw_arrow(ax, start, end, label=None, color="black"):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color=color,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.06, label, ha="center", va="bottom", fontsize=7, color="#333333")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.set_xlim(-0.1, 7.6)
ax.set_ylim(-0.1, 4.7)
ax.set_aspect("equal")
ax.axis("off")

# --- Column 1: LLM Agents ---
agent_x = 0.0
agent_w = 1.3
agent_h = 0.55
draw_box(ax, (agent_x, 3.6), agent_w, agent_h, "LLM Agent 1", "#D5E8D4")
draw_box(ax, (agent_x, 2.8), agent_w, agent_h, "LLM Agent 2", "#D5E8D4")
draw_box(ax, (agent_x, 2.0), agent_w, agent_h, "LLM Agent N", "#D5E8D4")
ax.text(agent_x + agent_w / 2, 2.65, "...", ha="center", va="center", fontsize=12, color="#666666")

# label below agents
ax.text(agent_x + agent_w / 2, 1.65, "Standard\nDB Client", ha="center", va="top", fontsize=6.5, fontstyle="italic", color="#666666")

# --- Column 2: HTTP Proxy ---
proxy_x = 2.0
proxy_w = 1.1
proxy_h = 2.85
proxy_box = FancyBboxPatch(
    (proxy_x, 1.3), proxy_w, proxy_h,
    boxstyle="round,pad=0.03",
    facecolor="#FFF2CC",
    edgecolor="black",
    linewidth=1.5,
)
ax.add_patch(proxy_box)
ax.text(proxy_x + proxy_w / 2, 3.95, "HTTP Proxy", ha="center", va="center", fontsize=9, fontweight="bold")
ax.text(proxy_x + proxy_w / 2, 3.65, "(OLAP-compatible)", ha="center", va="center", fontsize=6.5, fontstyle="italic", color="#555555")

# sub-items inside proxy
ax.text(proxy_x + proxy_w / 2, 3.10, "SQL\nInterception", ha="center", va="center", fontsize=7.5)
ax.text(proxy_x + proxy_w / 2, 2.20, "Request\nRouting", ha="center", va="center", fontsize=7.5)

# --- Column 3: Middleware services (AgentGate core) ---
mw_x = 3.8
mw_w = 1.6
mw_h = 0.55
mw_pad = 0.15

# bounding box for middleware (taller to fit 4 services)
mw_outer = FancyBboxPatch(
    (mw_x - 0.1, 1.1), mw_w + 0.2, 3.4,
    boxstyle="round,pad=0.05",
    facecolor="#F0F0F0",
    edgecolor="#888888",
    linewidth=1.0,
    linestyle="--",
)
ax.add_patch(mw_outer)
ax.text(mw_x + mw_w / 2, 4.35, "AgentGate Middleware", ha="center", va="center", fontsize=9, fontweight="bold", color="#333333")

# Four middleware services (top to bottom)
draw_box(ax, (mw_x, 3.65), mw_w, mw_h, "Intent-Based", "#DAE8FC", sublabel="Access Control")
draw_box(ax, (mw_x, 2.85), mw_w, mw_h, "Session-Based", "#E1D5E7", sublabel="Query Learning")
draw_box(ax, (mw_x, 2.05), mw_w, mw_h, "Curated Explore", "#FFD6D6", sublabel="Schema + Stats + Joins")
draw_box(ax, (mw_x, 1.25), mw_w, mw_h, "Agent Resource", "#D5F5E3", sublabel="Governance / Rate Limiter")

# --- Column 4: OLAP Database ---
ch_x = 6.3
ch_w = 1.1
ch_h = 1.0
draw_box(ax, (ch_x, 2.5), ch_w, ch_h, "OLAP", "#F8CECC", fontsize=9, bold=True, sublabel="Database")

# --- Column 3 side: ChromaDB ---
cr_x = 3.8
cr_w = 0.8
cr_h = 0.45
draw_box(ax, (cr_x + 0.4, 0.35), cr_w, cr_h, "VectorStore", "#E6E6E6", fontsize=7.5, sublabel="Embeddings")

# ---- Arrows ----

# Agents -> Proxy
for y_base in [3.6, 2.8, 2.0]:
    draw_arrow(ax, (agent_x + agent_w, y_base + agent_h / 2), (proxy_x, y_base + agent_h / 2))

# Proxy -> middleware services
draw_arrow(ax, (proxy_x + proxy_w, 3.92), (mw_x, 3.92))
draw_arrow(ax, (proxy_x + proxy_w, 3.12), (mw_x, 3.12))
draw_arrow(ax, (proxy_x + proxy_w, 2.32), (mw_x, 2.32))
draw_arrow(ax, (proxy_x + proxy_w, 1.52), (mw_x, 1.52))

# Middleware -> OLAP Database (arrows fan out to different heights on the OLAP box)
# Access Control -> OLAP (top)
draw_arrow(ax, (mw_x + mw_w, 3.92), (ch_x, 3.40))
# Explore -> OLAP (middle)
draw_arrow(ax, (mw_x + mw_w, 2.32), (ch_x, 3.00))
# Resource Governance -> OLAP (bottom)
draw_arrow(ax, (mw_x + mw_w, 1.52), (ch_x, 2.65), color="#27AE60")

# Annotation labels positioned above each arrow's midpoint
ax.text(5.85, 3.80, "allow / deny", ha="center", va="bottom", fontsize=6, color="#555555")
ax.text(5.85, 2.82, "metadata", ha="center", va="bottom", fontsize=6, color="#555555")
ax.text(5.85, 2.12, "throttle / quota", ha="center", va="bottom", fontsize=6, color="#27AE60")

# Query Learning -> ChromaDB
draw_arrow(ax, (mw_x + mw_w / 2, 2.85), (mw_x + mw_w / 2, 0.80), color="#7B68AE")
ax.text(mw_x + mw_w / 2 + 0.08, 1.05, "embed / retrieve", ha="left", va="center", fontsize=6, color="#7B68AE")

# Protocol labels on arrows
ax.text(1.65, 3.95, "HTTP", ha="center", va="bottom", fontsize=7, color="#888888")

plt.tight_layout(pad=0.2)
plt.savefig("/workarea/code-analysis-mcp/soa-harness/paper/figures/architecture.pdf", bbox_inches="tight", dpi=300)
plt.savefig("/workarea/code-analysis-mcp/soa-harness/paper/figures/architecture.png", bbox_inches="tight", dpi=300)
print("Saved architecture diagram to paper/figures/architecture.pdf and .png")
