from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from neo4j import GraphDatabase

from .config_loader import load_config
from .graph_build import (
    ALLOWED_ENTITY_TYPES,
    ALLOWED_REL_TYPES,
    KGNode,
    KGEdge,
)


def normalize_entity(name: str, ent_type: str) -> Tuple[str, str, str]:
    """
    Normalize entity name & type, and produce a node_id.
    """
    name = (name or "").strip()
    if not name:
        name = "UNKNOWN"

    simple_name = name.lower().strip()
    ent_type = (ent_type or "Other").strip()
    if ent_type not in ALLOWED_ENTITY_TYPES:
        ent_type = "Other"

    node_id = f"{simple_name}|{ent_type}"
    return node_id, name, ent_type


def import_triples(
    cfg_path: str = "config/vector_qwen.yaml",
    triples_path: str = "triples.json",
):
    """
    Import triples from a JSON file (NotebookLM output) into Neo4j and
    build kg_nodes.pkl and kg_edges.pkl.
    Expected JSON structure:

    {
      "triples": [
        {
          "head": "...",
          "head_type": "...",
          "relation": "...",
          "tail": "...",
          "tail_type": "...",
          "supporting_sentence": "...",   # optional
          "source_document": "..."        # optional
        },
        ...
      ]
    }
    """
    cfg = load_config(cfg_path)

    triples_file = Path(triples_path)
    if not triples_file.exists():
        raise FileNotFoundError(
            f"Triples file not found at {triples_file.resolve()}. "
            "Make sure triples.json is in the project root."
        )

    print(f"Loading triples from {triples_file.resolve()}")
    with triples_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_triples = data.get("triples", [])
    if not isinstance(raw_triples, list):
        raise ValueError("Expected 'triples' to be a list in triples.json")

    # Neo4j connection
    graph_cfg = cfg["graph"]
    driver = GraphDatabase.driver(
        graph_cfg["neo4j_uri"],
        auth=(graph_cfg["neo4j_user"], graph_cfg["neo4j_password"]),
    )

    nodes: Dict[str, KGNode] = {}
    edges: List[KGEdge] = []

    with driver.session() as session:
        for i, t in enumerate(raw_triples):
            head = (t.get("head") or "").strip()
            tail = (t.get("tail") or "").strip()
            rel = (t.get("relation") or "").strip().upper()
            head_type = (t.get("head_type") or "Other").strip()
            tail_type = (t.get("tail_type") or "Other").strip()
            source_doc = (t.get("source_document") or "unknown_source").strip()

            if not head or not tail:
                continue

            if rel not in ALLOWED_REL_TYPES:
                rel = "RELATED_TO"

            head_id, head_name, head_type_norm = normalize_entity(head, head_type)
            tail_id, tail_name, tail_type_norm = normalize_entity(tail, tail_type)

            if head_id not in nodes:
                nodes[head_id] = KGNode(id=head_id, name=head_name, type=head_type_norm)
            if tail_id not in nodes:
                nodes[tail_id] = KGNode(id=tail_id, name=tail_name, type=tail_type_norm)

            edge = KGEdge(
                source_id=head_id,
                target_id=tail_id,
                relation=rel,
                source_doc=source_doc,
                chunk_id=f"manual_triple_{i}",
            )
            edges.append(edge)

            # Write to Neo4j
            session.run(
                """
                MERGE (h:Entity {name: $head_name})
                ON CREATE SET h.type = $head_type
                ON MATCH SET h.type = coalesce(h.type, $head_type)
                """,
                head_name=head_name,
                head_type=head_type_norm,
            )
            session.run(
                """
                MERGE (t:Entity {name: $tail_name})
                ON CREATE SET t.type = $tail_type
                ON MATCH SET t.type = coalesce(t.type, $tail_type)
                """,
                tail_name=tail_name,
                tail_type=tail_type_norm,
            )
            session.run(
                """
                MATCH (h:Entity {name: $head_name})
                MATCH (t:Entity {name: $tail_name})
                MERGE (h)-[r:RELATION {type: $rel_type, source_doc: $source_doc, chunk_id: $chunk_id}]->(t)
                """,
                head_name=head_name,
                tail_name=tail_name,
                rel_type=rel,
                source_doc=source_doc,
                chunk_id=f"manual_triple_{i}",
            )

    print(f"Imported triples into Neo4j. Nodes: {len(nodes)}, edges: {len(edges)}")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    with (data_dir / "kg_nodes.pkl").open("wb") as f:
        pickle.dump(nodes, f)
    with (data_dir / "kg_edges.pkl").open("wb") as f:
        pickle.dump(edges, f)

    print("Saved KG nodes/edges to data/kg_nodes.pkl and data/kg_edges.pkl")


if __name__ == "__main__":
    import_triples()
