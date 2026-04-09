from __future__ import annotations

import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

def wrap_label(text: str, max_chars: int = 20) -> str:
    """
    Wrap a label into multiple lines by inserting newlines every ~max_chars.
    Simple word-based wrapping.
    """
    if not text:
        return ""

    words = text.split()
    lines = []
    current = ""

    for w in words:
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= max_chars:
            current += " " + w
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)

    return "\n".join(lines)

def draw_hairball(
    nodes_path="data/kg_nodes.pkl",
    edges_path="data/kg_edges.pkl",
    output_path="kg_hairball.jpg",
    figsize=(14, 14),
):
    with open(nodes_path, "rb") as f:
        nodes = pickle.load(f)

    with open(edges_path, "rb") as f:
        edges = pickle.load(f)

    G = nx.Graph()

    # add nodes with label attr
    for node_id, node in nodes.items():
        G.add_node(node_id, label=node.name, type=node.type)

    for edge in edges:
        G.add_edge(edge.source_id, edge.target_id)

    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # spring layout – tight cluster for "furball" look
    pos = nx.spring_layout(
        G,
        k=0.05,          # smaller => tighter
        iterations=200,
        seed=42,
    )

    plt.figure(figsize=figsize, dpi=400)
    plt.axis("off")
    plt.tight_layout()

    # edges: light red, semi-transparent
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="red",
        alpha=0.28,
        width=0.7,
    )

    # nodes: small black points
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2,
        node_color="black",
    )

    # ---- NEW: labels for high-degree nodes only ----
    degrees = dict(G.degree())
    # label nodes with degree >= 4 (you can tweak this threshold)
    label_nodes = {
        n: wrap_label(data["label"], max_chars=18)
        for n, data in G.nodes(data=True)
        if degrees.get(n, 0) >= 1
    }

    # nx.draw_networkx_labels(
    #     G,
    #     pos,
    #     labels=label_nodes,
    #     font_size=4,      # tiny, so it looks like the paper
    #     font_color="black",
    #     alpha=0.9,
    # )

    ax = plt.gca()
    for n, label in label_nodes.items():
        x, y = pos[n]
        ax.text(
            x,
            y,
            label,
            fontsize=4,
            ha="center",
            va="center",
            zorder=5,  # above nodes/edges
        )

    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Saved hairball visualization to {output_path}")

def draw_zoomed_subgraph(
    nodes_path="data/kg_nodes.pkl",
    edges_path="data/kg_edges.pkl",
    output_path="kg_zoom.jpg",
    figsize=(6, 6),
    top_k=1,         # number of highest-degree nodes to include
):
    # Load nodes/edges
    with open(nodes_path, "rb") as f:
        nodes = pickle.load(f)

    with open(edges_path, "rb") as f:
        edges = pickle.load(f)

    # Build full graph
    G = nx.Graph()
    for node_id, node in nodes.items():
        G.add_node(node_id, label=node.name, type=node.type)

    for edge in edges:
        G.add_edge(edge.source_id, edge.target_id)

    print(f"[Zoom] Full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ---- Select top-K highest-degree nodes ----
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, d in sorted_nodes[:top_k]]

    print(f"[Zoom] Selected {len(top_nodes)} top-degree nodes")

    # ---- Build a zoom subgraph = hubs + neighbors ----
    zoom_nodes = set(top_nodes)
    for n in top_nodes:
        zoom_nodes.update(G.neighbors(n))

    H = G.subgraph(zoom_nodes).copy()
    print(f"[Zoom] Subgraph: {H.number_of_nodes()} nodes, {H.number_of_edges()} edges")

    # ---- Layout ----
    pos = nx.spring_layout(
        H,
        k=0.05,            # more spacing than main hairball
        iterations=300,
        seed=42,
    )

    # ---- Plot ----
    plt.figure(figsize=figsize, dpi=400)
    plt.axis("off")

    nx.draw_networkx_edges(
        H,
        pos,
        edge_color="red",
        alpha=0.28,
        width=0.7,
    )

    nx.draw_networkx_nodes(
        H,
        pos,
        node_size=4,
        node_color="black",
    )

    # Labels: bigger, readable
    labels = {n: data["label"] for n, data in H.nodes(data=True)}
    # nx.draw_networkx_labels(
    #     H,
    #     pos,
    #     labels=labels,
    #     font_size=8,
    #     font_color="black",
    #     alpha=0.9,
    # )

    ax = plt.gca()
    for n, data in H.nodes(data=True):
        x, y = pos[n]
        label = wrap_label(data["label"], max_chars=22)
        ax.text(
            x,
            y,
            label,
            fontsize=8,
            ha="center",
            va="center",
            color="black",
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="none",
                alpha=0.4,
            ),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"[Zoom] Saved zoomed subgraph to {output_path}")


if __name__ == "__main__":
    draw_hairball()
    draw_zoomed_subgraph()
