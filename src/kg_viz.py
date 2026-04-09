from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

from pyvis.network import Network

from .graph_build import KGNode, KGEdge


def build_kg_visualization(
    nodes_path: str = "data/kg_nodes.pkl",
    edges_path: str = "data/kg_edges.pkl",
    output_html: str = "graph.html",
    max_nodes: int = 1000,
):
    """
    Build an interactive KG visualization using pyvis.
    max_nodes: limit number of nodes for readability; if too big, simple subsample.
    """
    nodes_file = Path(nodes_path)
    edges_file = Path(edges_path)

    if not nodes_file.exists() or not edges_file.exists():
        raise FileNotFoundError(
            f"KG node/edge files not found. Expected {nodes_file} and {edges_file}. "
            "Run `python -m src.graph_build` first."
        )

    with nodes_file.open("rb") as f:
        nodes: Dict[str, KGNode] = pickle.load(f)
    with edges_file.open("rb") as f:
        edges: List[KGEdge] = pickle.load(f)

    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges for visualization")

    if len(nodes) > max_nodes:
        keep_ids = set(list(nodes.keys())[:max_nodes])
        nodes = {nid: n for nid, n in nodes.items() if nid in keep_ids}
        edges = [e for e in edges if e.source_id in keep_ids and e.target_id in keep_ids]
        print(f"Subsampled to {len(nodes)} nodes and {len(edges)} edges")

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#111111",
        font_color="white",
        notebook=False,
        directed=True,
    )

    # Cyber/SMB entity type color mapping
    type_colors = {
        "Organization": "#4E79A7",
        "GovernmentBody": "#F28E2B",
        "PolicyOrRegulation": "#E15759",
        "ThreatOrAttack": "#76B7B2",
        "Vulnerability": "#59A14F",
        "ControlOrMeasure": "#EDC948",
        "AssetOrData": "#B07AA1",
        "ImpactOrOutcome": "#FF9DA7",
        "Other": "#9C9C9C",
    }

    # Add nodes
    for node in nodes.values():
        color = type_colors.get(node.type, "#9C9C9C")
        net.add_node(
            node.id,
            label=node.name,
            title=f"Type: {node.type}",
            color=color,
        )

    # Add edges
    for edge in edges:
        if edge.source_id not in nodes or edge.target_id not in nodes:
            continue
        net.add_edge(
            edge.source_id,
            edge.target_id,
            title=f"{edge.relation} (doc: {edge.source_doc}, chunk: {edge.chunk_id})",
            label=edge.relation,
            physics=True,
        )

    net.toggle_physics(True)
    net.write_html(output_html)
    print(f"Saved KG visualization to {output_html}")


if __name__ == "__main__":
    build_kg_visualization()
