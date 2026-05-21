import random

# =========================
# GRAPH SIZE
# =========================

V = 100000
E = 500000
# =========================
# OUTPUT FILE
# =========================

output_file = "data/graph.txt"

# =========================
# GENERATE GRAPH
# =========================

edges = set()

while len(edges) < E:

    u = random.randint(0, V - 1)

    v = random.randint(0, V - 1)

    if u == v:
        continue

    w = random.randint(1, 20)

    edges.add((u, v, w))

# =========================
# SAVE GRAPH
# =========================

with open(output_file, "w") as f:

    f.write(f"{V} {E}\n")

    for (u, v, w) in edges:
        f.write(f"{u} {v} {w}\n")

print("Graph generated successfully!")

print(f"Vertices: {V}")
print(f"Edges: {E}")