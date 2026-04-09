from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from neo4j import GraphDatabase

from .config_loader import load_config
from .llm_backend import LLMBackend, LLMConfig
from .ingest import Chunk

# --------- Cyber/SMB schema config ---------
# Based on your NotebookLM summary

ALLOWED_ENTITY_TYPES = {
    "Organization",
    "GovernmentBody",
    "PolicyOrRegulation",
    "ThreatOrAttack",
    "Vulnerability",
    "ControlOrMeasure",
    "AssetOrData",
    "ImpactOrOutcome",
    "Other",
}

ALLOWED_REL_TYPES = {
    "TARGETS",
    "MITIGATES",
    "CAUSES",
    "LEADS_TO",
    "REQUIRES",
    "APPLIES_TO",
    "RESPONSIBLE_FOR",
    "LOCATED_IN",   # useful even if not emphasized in prompt 1
    "RELATED_TO",
}


@dataclass
class KGNode:
    id: str    # internal ID, e.g. "phishing|ThreatOrAttack"
    name: str
    type: str


@dataclass
class KGEdge:
    source_id: str
    target_id: str
    relation: str
    source_doc: str
    chunk_id: str


# --------- Helper: parse JSON from LLM output ---------


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try to extract the first JSON object from the LLM output.
    Handles cases with ```json ...``` or extra text around it.
    """
    code_block_match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        candidate = code_block_match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


# --------- KG Builder ---------


class KGBuilder:
    def __init__(self, cfg_path: str = "config/vector_qwen.yaml"):
        """
        Uses Qwen config by default for triple extraction.
        """
        self.cfg = load_config(cfg_path)

        llm_cfg = LLMConfig(**self.cfg["llm"])
        self.llm = LLMBackend(llm_cfg)

        # Neo4j
        graph_cfg = self.cfg["graph"]
        self.driver = GraphDatabase.driver(
            graph_cfg["neo4j_uri"],
            auth=(graph_cfg["neo4j_user"], graph_cfg["neo4j_password"]),
        )

        # Corpus
        self.corpus_cache = Path(self.cfg["data"]["corpus_cache"])
        if not self.corpus_cache.exists():
            raise FileNotFoundError(
                f"Corpus cache not found at {self.corpus_cache}. "
                "Run `python -m src.ingest` first."
            )

        with self.corpus_cache.open("rb") as f:
            self.chunks: List[Chunk] = pickle.load(f)

        # In-memory KG for visualization
        self.nodes: Dict[str, KGNode] = {}     # key: node_id
        self.edges: List[KGEdge] = []

    # ------------- Triple extraction prompt -------------

    def build_triple_prompt(self, text: str, doc_name: str, chunk_id: str) -> str:
        """
        Prompt for structured triple extraction with a constrained cyber/SMB schema.
        """
        text = text.strip()
        if len(text) > 3000:
            text = text[:3000]

        # Note: this is aligned with your NotebookLM schema
        prompt = f"""You are an information extraction system.

From the following text about cybersecurity and small businesses (Australian context, policies, threats, controls, surveys, reports), extract up to 5 knowledge triples.

Each triple has:
- head: the subject entity
- head_type: one of {sorted(ALLOWED_ENTITY_TYPES)}
- relation: one of {sorted(ALLOWED_REL_TYPES)}
- tail: the object entity
- tail_type: one of {sorted(ALLOWED_ENTITY_TYPES)}

Entity type guidance:
- Organization: businesses, small & medium enterprises, industry bodies, universities, insurers, etc.
- GovernmentBody: government agencies and regulators (e.g. ACSC, ASD, OAIC, AFP, Department of Home Affairs).
- PolicyOrRegulation: laws, strategies, frameworks, guidelines (e.g. Privacy Act 1988, NIST CSF, Australian Cyber Security Strategy).
- ThreatOrAttack: phishing, ransomware, business email compromise, scams, malware, data breaches, etc.
- Vulnerability: lack of funds, weak passwords, unpatched software, lack of in-house skills, shared accounts, underestimation of cyber risk, etc.
- ControlOrMeasure: multi-factor authentication, backups, training, patching, cyber insurance, policies, checklists, etc.
- AssetOrData: customer data, financial records, IT systems, cloud services, operating systems, personal information, etc.
- ImpactOrOutcome: financial loss, downtime, reputational damage, insolvency, compliance risk, identity theft, etc.
- Other: anything important that does not clearly fit above.

Relation guidance:
- TARGETS: a ThreatOrAttack or actor targets an Organization or AssetOrData.
- MITIGATES: a ControlOrMeasure reduces the likelihood or impact of a ThreatOrAttack, Vulnerability, or ImpactOrOutcome.
- CAUSES: a Vulnerability, constraint, or ThreatOrAttack directly causes a negative ImpactOrOutcome.
- LEADS_TO: an event, condition, or threat leads to an ImpactOrOutcome or consequence.
- REQUIRES: a PolicyOrRegulation or contract requires a ControlOrMeasure, reporting, or other action.
- APPLIES_TO: a PolicyOrRegulation, guideline, or framework applies to a particular Organization, system, or scope.
- RESPONSIBLE_FOR: an Organization or role is responsible for an action, duty, or outcome.
- LOCATED_IN: an Organization or GovernmentBody is located in or scoped to a region (e.g. Australia).
- RELATED_TO: a general conceptual connection when none of the above fits.

Rules:
- Use ONLY the allowed entity types and relation types above.
- If you are unsure about an entity type, use "Other".
- If you are unsure about the relation type, use "RELATED_TO".
- Do NOT invent information that is not stated or strongly implied.
- If there are no meaningful triples, return an empty list.

Return ONLY valid JSON in the following format (no extra text):

{{
  "triples": [
    {{
      "head": "...",
      "head_type": "...",
      "relation": "...",
      "tail": "...",
      "tail_type": "..."
    }}
  ]
}}

TEXT (from document: {doc_name}, chunk id: {chunk_id}):
{text}
"""
        return prompt

    def _normalize_entity(self, name: str, ent_type: str) -> Tuple[str, str, str]:
        """
        Normalize entity name & type, and produce a node_id.
        """
        name = name.strip()
        if not name:
            name = "UNKNOWN"

        simple_name = name.lower().strip()
        ent_type = ent_type.strip() if ent_type else "Other"
        if ent_type not in ALLOWED_ENTITY_TYPES:
            ent_type = "Other"

        node_id = f"{simple_name}|{ent_type}"
        return node_id, name, ent_type

    # ------------- Neo4j write helpers -------------

    def _merge_node(self, session, node: KGNode):
        """
        MERGE an :Entity node in Neo4j.
        """
        session.run(
            """
            MERGE (e:Entity {name: $name})
            ON CREATE SET e.type = $type
            ON MATCH SET e.type = coalesce(e.type, $type)
            """,
            name=node.name,
            type=node.type,
        )

    def _merge_edge(
        self,
        session,
        source: KGNode,
        target: KGNode,
        relation: str,
        source_doc: str,
        chunk_id: str,
    ):
        """
        MERGE a :RELATION edge with 'type' property.
        """
        relation = relation if relation in ALLOWED_REL_TYPES else "RELATED_TO"
        session.run(
            """
            MATCH (h:Entity {name: $head_name})
            MATCH (t:Entity {name: $tail_name})
            MERGE (h)-[r:RELATION {type: $rel_type, source_doc: $source_doc, chunk_id: $chunk_id}]->(t)
            """,
            head_name=source.name,
            tail_name=target.name,
            rel_type=relation,
            source_doc=source_doc,
            chunk_id=chunk_id,
        )

    # ------------- Main build function -------------

    def build_graph(self, max_chunks: Optional[int] = None):
        """
        Iterate over chunks, extract triples with LLM, populate Neo4j and in-memory KG.
        If max_chunks is provided, process only that many chunks (for faster testing).
        """
        total_chunks = len(self.chunks)
        if max_chunks is not None:
            total_chunks = min(total_chunks, max_chunks)

        print(f"Building KG from {total_chunks} chunks (out of {len(self.chunks)})")

        with self.driver.session() as session:
            for i, chunk in enumerate(self.chunks[:total_chunks]):
                if i % 10 == 0:
                    print(f"Processing chunk {i+1}/{total_chunks} (doc: {chunk.doc_name})")

                prompt = self.build_triple_prompt(chunk.text, chunk.doc_name, chunk.id)
                raw_output = self.llm.generate(prompt, max_new_tokens=512)

                data = extract_json(raw_output)
                if not data or "triples" not in data:
                    continue

                triples = data.get("triples", [])
                if not isinstance(triples, list):
                    continue

                for t in triples:
                    head = t.get("head", "").strip()
                    tail = t.get("tail", "").strip()
                    rel = t.get("relation", "").strip().upper()
                    head_type = t.get("head_type", "Other").strip()
                    tail_type = t.get("tail_type", "Other").strip()

                    if not head or not tail:
                        continue

                    if rel not in ALLOWED_REL_TYPES:
                        rel = "RELATED_TO"

                    head_id, head_name, head_type_norm = self._normalize_entity(head, head_type)
                    tail_id, tail_name, tail_type_norm = self._normalize_entity(tail, tail_type)

                    if head_id not in self.nodes:
                        self.nodes[head_id] = KGNode(id=head_id, name=head_name, type=head_type_norm)
                    if tail_id not in self.nodes:
                        self.nodes[tail_id] = KGNode(id=tail_id, name=tail_name, type=tail_type_norm)

                    edge = KGEdge(
                        source_id=head_id,
                        target_id=tail_id,
                        relation=rel,
                        source_doc=chunk.doc_name,
                        chunk_id=chunk.id,
                    )
                    self.edges.append(edge)

                    self._merge_node(session, self.nodes[head_id])
                    self._merge_node(session, self.nodes[tail_id])
                    self._merge_edge(
                        session,
                        self.nodes[head_id],
                        self.nodes[tail_id],
                        rel,
                        chunk.doc_name,
                        chunk.id,
                    )

        print(f"KG build complete. Nodes: {len(self.nodes)}, edges: {len(self.edges)}")

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        with (data_dir / "kg_nodes.pkl").open("wb") as f:
            pickle.dump(self.nodes, f)
        with (data_dir / "kg_edges.pkl").open("wb") as f:
            pickle.dump(self.edges, f)

        print("Saved KG nodes/edges to data/kg_nodes.pkl and data/kg_edges.pkl")


if __name__ == "__main__":
    # Start with a subset; increase or remove max_chunks once you're happy
    builder = KGBuilder("config/vector_llama.yaml")
    builder.build_graph(max_chunks=50)
