import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pathlib import Path

# =========================
# LOAD GRAPH
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

graph_path = BASE_DIR / "data" / "graph.txt"

with open(graph_path, "r") as f:
    lines = f.readlines()

V, E = map(int, lines[0].split())

G = nx.DiGraph()

edges = []

for line in lines[1:]:
    u, v, w = map(int, line.split())

    G.add_edge(u, v, weight=w)

    edges.append((u, v, w))

# =========================
# LOAD STEPS
# =========================

steps_path = BASE_DIR / "results" / "steps.txt"

steps = []

with open(steps_path, "r") as f:
    for line in f:
        iteration, u, v, oldDist, newDist, updated = map(int, line.split())

        steps.append({
            "iteration": iteration,
            "u": u,
            "v": v,
            "oldDist": oldDist,
            "newDist": newDist,
            "updated": updated
        })

# =========================
# GRAPH LAYOUT
# =========================

pos = nx.spring_layout(G, seed=42)

fig, ax = plt.subplots(figsize=(8, 6))

# =========================
# ANIMATION FUNCTION
# =========================

def update(frame):

    ax.clear()

    step = steps[frame]

    u = step["u"]
    v = step["v"]

    updated = step["updated"]

    # Default edge colors
    edge_colors = []

    for edge in G.edges():

        if edge == (u, v):

            if updated:
                edge_colors.append("red")
            else:
                edge_colors.append("orange")

        else:
            edge_colors.append("gray")

    # Node colors
    node_colors = []

    for node in G.nodes():

        if node == v and updated:
            node_colors.append("yellow")
        else:
            node_colors.append("skyblue")

    # Draw graph
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        node_size=1200,
        font_size=12,
        arrows=True
    )

    # Edge labels
    edge_labels = {
        (a, b): w
        for a, b, w in edges
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        ax=ax
    )

    ax.set_title(
        f"Iteration {step['iteration']} | "
        f"Relax Edge {u} → {v}"
    )

# =========================
# RUN ANIMATION
# =========================

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(steps),
    interval=1000,
    repeat=False
)

plt.show()