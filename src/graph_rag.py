from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase

from .config_loader import load_config
from .llm_backend import LLMBackend, LLMConfig


# Allowed schema types — must match what you used when building the KG
ALLOWED_ENTITY_TYPES = [
    "Organization",
    "GovernmentBody",
    "PolicyOrRegulation",
    "ThreatOrAttack",
    "Vulnerability",
    "ControlOrMeasure",
    "AssetOrData",
    "ImpactOrOutcome",
    "Other",
]

ALLOWED_REL_TYPES = [
    "TARGETS",
    "MITIGATES",
    "CAUSES",
    "LEADS_TO",
    "REQUIRES",
    "APPLIES_TO",
    "RESPONSIBLE_FOR",
    "LOCATED_IN",
    "RELATED_TO",
]


@dataclass
class GraphFact:
    head: str
    relation: str
    tail: str


@dataclass
class GraphIntent:
    head_type: Optional[str]
    tail_type: Optional[str]
    relations: List[str]
    head_keywords: List[str]
    tail_keywords: List[str]


def clean_answer_text(text: str, max_sentences: int = 4) -> str:
    """
    Post-process LLM output:
    - strip whitespace
    - keep only the first N sentences
    """
    text = text.strip()
    if not text:
        return text

    # crude sentence split: punctuation + whitespace
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return ""

    clipped = " ".join(sentences[:max_sentences])
    return clipped.strip()


class GraphRAG:
    """
    GraphRAG using:
      - LLM ONLY for semantic intent extraction (JSON)
      - deterministic Cypher templates (no NL->Cypher)
    """

    def __init__(self, cfg_path: str):
        self.cfg = load_config(cfg_path)

        llm_cfg = LLMConfig(**self.cfg["llm"])
        self.llm = LLMBackend(llm_cfg)

        graph_cfg = self.cfg["graph"]
        self.driver = GraphDatabase.driver(
            graph_cfg["neo4j_uri"],
            auth=(graph_cfg["neo4j_user"], graph_cfg["neo4j_password"]),
        )

    # ---------- INTENT EXTRACTION (LLM) ----------

    def _intent_prompt(self, question: str) -> str:
        """
        Ask the LLM to map the question to:
        - head_type, tail_type
        - relations
        - head_keywords, tail_keywords
        """
        return f"""You are helping to query a small Neo4j knowledge graph about
cybersecurity and Australian small businesses.

GRAPH SCHEMA:

- Nodes use label: Entity
- Entity.type is one of:
  {", ".join(ALLOWED_ENTITY_TYPES)}

- Relationships use type: RELATION
- RELATION.type is one of:
  {", ".join(ALLOWED_REL_TYPES)}

Given the user's QUESTION, identify:

1) The most likely head entity type (head_type)
2) The most likely tail entity type (tail_type)
3) Which relation types are relevant (relations)
4) 0–3 keywords that should appear in head entity names (head_keywords)
5) 0–3 keywords that should appear in tail entity names (tail_keywords)

QUESTION:
\"\"\"{question}\"\"\"

Return a single JSON object with this exact structure:

{{
  "head_type": "ControlOrMeasure",
  "tail_type": "ThreatOrAttack",
  "relations": ["MITIGATES"],
  "head_keywords": [],
  "tail_keywords": ["phishing", "scam"]
}}

Rules:
- head_type and tail_type must be one of the allowed types above, or null if unsure.
- relations must be a list of allowed relation types, or an empty list.
- All keywords should be lowercase phrases.
- If unsure, use null for head_type/tail_type and [] for relations/keywords.
- Output ONLY the JSON, with no explanation.
"""

    def _augment_relations_from_question(self, question: str, rels: List[str]) -> List[str]:
        """
        Heuristically add relation hints from the question text.
        """
        q = question.lower()
        rels_set = set(rels)

        if "mitigat" in q or "reduce" in q or "prevent" in q:
            rels_set.add("MITIGATES")
        if "cause" in q or "driver" in q or "reason" in q:
            rels_set.add("CAUSES")
        if "lead to" in q or "leads to" in q or "result" in q or "impact" in q:
            rels_set.add("LEADS_TO")
        if "require" in q or "must" in q or "oblig" in q or "need to" in q:
            rels_set.add("REQUIRES")

        # keep only valid ones
        return [r for r in rels_set if r in ALLOWED_REL_TYPES]

    def _heuristic_intent(self, question: str) -> GraphIntent:
        """
        Fallback intent if JSON parsing fails completely.
        """
        q = question.lower()
        head_type: Optional[str] = None
        tail_type: Optional[str] = None
        rels: List[str] = []

        if "control" in q or "mitigat" in q:
            head_type = "ControlOrMeasure"
            tail_type = "ThreatOrAttack"
            rels.append("MITIGATES")
        elif "impact" in q or "consequence" in q or "effect" in q:
            head_type = "ThreatOrAttack"
            tail_type = "ImpactOrOutcome"
            rels.append("LEADS_TO")

        rels = self._augment_relations_from_question(question, rels)

        # simple keyword extraction: words >=4 chars
        tokens = re.findall(r"[A-Za-z]{4,}", q)
        stop = {
            "which", "what", "when", "where", "who",
            "help", "helps", "helping",
            "mitigate", "mitigates", "mitigating",
            "small", "business", "businesses",
            "australian", "australia",
            "cyber", "security", "threat", "threats",
            "related", "about", "that", "with", "from",
            "controls", "control",
        }
        keywords = [t for t in tokens if t not in stop]

        # heuristically treat first few as tail keywords
        tail_kws = keywords[:3]
        head_kws: List[str] = []

        return GraphIntent(
            head_type=head_type,
            tail_type=tail_type,
            relations=rels,
            head_keywords=head_kws,
            tail_keywords=tail_kws,
        )

    def _parse_intent_json(self, raw: str, question: str) -> GraphIntent:
        """
        Parse the LLM JSON; if anything goes wrong, fall back to heuristics.
        """
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return self._heuristic_intent(question)

        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return self._heuristic_intent(question)

        head_type = data.get("head_type")
        tail_type = data.get("tail_type")
        if head_type not in ALLOWED_ENTITY_TYPES:
            head_type = None
        if tail_type not in ALLOWED_ENTITY_TYPES:
            tail_type = None

        rels = data.get("relations") or []
        if not isinstance(rels, list):
            rels = []
        rels = [r for r in rels if isinstance(r, str) and r in ALLOWED_REL_TYPES]

        head_kws = data.get("head_keywords") or []
        tail_kws = data.get("tail_keywords") or []
        if not isinstance(head_kws, list):
            head_kws = []
        if not isinstance(tail_kws, list):
            tail_kws = []

        head_kws = [str(k).lower().strip() for k in head_kws if str(k).strip()]
        tail_kws = [str(k).lower().strip() for k in tail_kws if str(k).strip()]

        rels = self._augment_relations_from_question(question, rels)

        return GraphIntent(
            head_type=head_type,
            tail_type=tail_type,
            relations=rels,
            head_keywords=head_kws,
            tail_keywords=tail_kws,
        )

    def extract_intent(self, question: str) -> GraphIntent:
        prompt = self._intent_prompt(question)
        raw = self.llm.generate(prompt, max_new_tokens=128)
        return self._parse_intent_json(raw, question)

    # ---------- CYPHER BUILDING (DETERMINISTIC) ----------

    def build_cypher(self, intent: GraphIntent, limit: int = 25) -> tuple[str, Dict[str, Any]]:
        """
        Build a Cypher query + parameters from a GraphIntent.
        """
        params: Dict[str, Any] = {
            "head_type": intent.head_type,
            "tail_type": intent.tail_type,
            "relations": intent.relations or [],
            "head_keywords": intent.head_keywords or [],
            "tail_keywords": intent.tail_keywords or [],
            "limit": limit,
        }

        cypher = """
MATCH (h:Entity)-[r:RELATION]->(t:Entity)
WHERE ($head_type IS NULL OR h.type = $head_type)
  AND ($tail_type IS NULL OR t.type = $tail_type)
  AND (size($relations) = 0 OR r.type IN $relations)
  AND (
        size($head_keywords) = 0
        OR ANY(kw IN $head_keywords WHERE toLower(h.name) CONTAINS toLower(kw))
      )
  AND (
        size($tail_keywords) = 0
        OR ANY(kw IN $tail_keywords WHERE toLower(t.name) CONTAINS toLower(kw))
      )
RETURN h.name AS head, r.type AS relation, t.name AS tail
LIMIT $limit
""".strip()

        return cypher, params

    # ---------- RUN CYPHER ----------

    def run_cypher(self, cypher: str, params: Dict[str, Any]) -> List[GraphFact]:
        facts: List[GraphFact] = []
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            for record in result:
                head = record.get("head")
                relation = record.get("relation")
                tail = record.get("tail")
                if head and tail:
                    facts.append(
                        GraphFact(
                            head=str(head),
                            relation=str(relation) if relation else "",
                            tail=str(tail),
                        )
                    )
        return facts

    # ---------- ANSWER GENERATION ----------

    def build_answer_prompt(self, question: str, facts: List[GraphFact]) -> str:
        if facts:
            facts_text = "\n".join(
                f"- {f.head} --{f.relation}--> {f.tail}" for f in facts
            )
        else:
            facts_text = "(no relevant facts were retrieved from the knowledge graph)"

        return f"""You are answering a question using ONLY the following facts from a
cybersecurity knowledge graph about Australian small businesses.

FACTS:
{facts_text}

INSTRUCTIONS:
- Use ONLY these facts. Do NOT add any outside knowledge or assumptions.
- If the facts are insufficient, say exactly:
  "I do not have enough information from the knowledge graph to answer this question."
- Write a single concise paragraph (Answer in at most 3 sentences).
- Do NOT include notes, explanations of your reasoning, or self-corrections.
- Do NOT mention chunks, documents, or context.
- Do NOT apologize or self-correct.
- Do NOT repeat the question.
- Do NOT output anything after the answer paragraph.
- Do NOT say "note:", "chunk", "data source", or "let me know".
- Do NOT explain your reasoning.
Your entire reply must be just the final answer.

Question: {question}

Answer (one concise paragraph):
"""

    def answer(self, question: str, max_facts: int = 25) -> Dict[str, Any]:
        """
        Full GraphRAG pipeline with deterministic Cypher:
        - LLM: extract intent (types + keywords)
        - Template: build Cypher
        - Neo4j: fetch facts
        - LLM: answer from facts
        """
        # Step 1: intent
        intent = self.extract_intent(question)

        # Step 2: build & run Cypher
        cypher, params = self.build_cypher(intent, limit=max_facts)
        facts = self.run_cypher(cypher, params)

        # Step 3: if no facts, relax types/relations (keep keywords)
        if not facts:
            print("[INFO] No facts with intent filter. Falling back to type-agnostic keyword search.")
            relaxed_intent = GraphIntent(
                head_type=None,
                tail_type=None,
                relations=[],
                head_keywords=intent.head_keywords,
                tail_keywords=intent.tail_keywords,
            )
            cypher, params = self.build_cypher(relaxed_intent, limit=max_facts)
            facts = self.run_cypher(cypher, params)

        # Step 4: if still nothing, generic graph sample
        if not facts:
            print("[INFO] No facts found even after relaxing. Using generic graph sample.")
            cypher = """
MATCH (h:Entity)-[r:RELATION]->(t:Entity)
RETURN h.name AS head, r.type AS relation, t.name AS tail
LIMIT $limit
""".strip()
            params = {"limit": max_facts}
            facts = self.run_cypher(cypher, params)

        if len(facts) > max_facts:
            facts = facts[:max_facts]

        # Step 5: answer from facts
        answer_prompt = self.build_answer_prompt(question, facts)
        answer_text = self.llm.generate(answer_prompt, max_new_tokens=128)
        cleaned = clean_answer_text(answer_text, max_sentences=4)

        return {
            "question": question,
            "intent": intent,
            "cypher": cypher,
            "facts": facts,
            "answer": cleaned,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deterministic GraphRAG demo")
    parser.add_argument(
        "--config",
        type=str,
        default="config/vector_qwen.yaml",
        help="Path to YAML config (must contain llm + graph sections)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Which government bodies are responsible for cyber security guidance for Australian SMEs?",
        help="Test question to ask via GraphRAG",
    )

    args = parser.parse_args()

    rag = GraphRAG(args.config)
    result = rag.answer(args.question, max_facts=25)

    print("\n=== GRAPH RAG RESULT ===")
    print("Q:", result["question"])
    print("\nIntent:", result["intent"])
    print("\nCypher used:\n", result["cypher"])
    print("\nFacts used:")
    for f in result["facts"]:
        print(f"  - {f.head} --{f.relation}--> {f.tail}")
    print("\nAnswer:\n", result["answer"])






#for llama
# python -m src.graph_rag --config config/vector_llama.yaml \
#   --question "Which cyber security controls help Australian small businesses mitigate phishing attacks?"

# for mistral
# python -m src.graph_rag --config config/vector_mistral.yaml \
#   --question "What impacts do ransomware incidents have on small businesses in Australia?"

# for qwen
# python -m src.graph_rag --config config/vector_qwen.yaml \
#   --question "Which government bodies are responsible for cyber security guidance for Australian SMEs?"



















# from __future__ import annotations

# import re
# from dataclasses import dataclass
# from typing import List, Dict, Any, Optional

# from neo4j import GraphDatabase
# from neo4j.exceptions import CypherSyntaxError, Neo4jError

# from .config_loader import load_config
# from .llm_backend import LLMBackend, LLMConfig


# def clean_answer_text(text: str, max_sentences: int = 4) -> str:
#     """
#     Post-process LLM output:
#     - strip whitespace
#     - keep only the first N sentences
#     - drop trailing meta-instructions / repetition
#     """
#     text = text.strip()
#     if not text:
#         return text

#     # crude sentence split: split on . ! ? followed by whitespace
#     sentences = re.split(r"(?<=[.!?])\s+", text)
#     sentences = [s.strip() for s in sentences if s.strip()]

#     if not sentences:
#         return ""

#     clipped = " ".join(sentences[:max_sentences])
#     return clipped.strip()



# @dataclass
# class GraphFact:
#     head: str
#     relation: str
#     tail: str


# class GraphRAG:
#     """
#     Simple GraphRAG:
#     - LLM generates Cypher from a natural language question
#     - Cypher is executed on the Neo4j Aura KG
#     - Results are turned into facts and fed back to the LLM to answer
#     """

#     def __init__(self, cfg_path: str):
#         self.cfg = load_config(cfg_path)

#         llm_cfg = LLMConfig(**self.cfg["llm"])
#         self.llm = LLMBackend(llm_cfg)

#         graph_cfg = self.cfg["graph"]
#         self.driver = GraphDatabase.driver(
#             graph_cfg["neo4j_uri"],
#             auth=(graph_cfg["neo4j_user"], graph_cfg["neo4j_password"]),
#         )

#     # --------- 1. NL -> Cypher generation ---------

#     def build_cypher_prompt(self, question: str) -> str:
#         """
#         Prompt the LLM to generate a single Cypher query for our schema.
#         Schema reminder:
#           - Nodes: (:Entity {name, type})
#           - Relationships: (:Entity)-[:RELATION {type, source_doc, chunk_id}]->(:Entity)
#         We always want: RETURN head, relation, tail
#         """
#         prompt = f"""You are an expert Cypher query generator for a Neo4j knowledge graph
# about cybersecurity and small businesses in Australia.

# GRAPH SCHEMA:
# - Nodes have label :Entity with properties:
#   - name: string (entity name, e.g. "Small businesses", "Phishing", "ACSC")
#   - type: string (one of: Organization, GovernmentBody, PolicyOrRegulation,
#     ThreatOrAttack, Vulnerability, ControlOrMeasure, AssetOrData,
#     ImpactOrOutcome, Other)

# - Relationships have type :RELATION with properties:
#   - type: string (semantic relation, one of:
#     TARGETS, MITIGATES, CAUSES, LEADS_TO, REQUIRES, APPLIES_TO,
#     RESPONSIBLE_FOR, LOCATED_IN, RELATED_TO)
#   - source_doc: string (source document)
#   - chunk_id: string (source text chunk id)

# PATTERN:
# - Queries typically look like:
#   MATCH (h:Entity)-[r:RELATION]->(t:Entity)
#   WHERE ...
#   RETURN h.name AS head, r.type AS relation, t.name AS tail
#   LIMIT 25

# TASK:
# Given the user's question, write ONE Cypher query that:
# - Uses the above schema
# - Focuses on relevant entities and relations
# - Returns columns: head, relation, tail
# - Includes a LIMIT (e.g. 25)

# RULES:
# - Output ONLY the Cypher query text, no explanation, no English sentences.
# - Do NOT include multiple queries; output exactly one query.
# - Do NOT prefix with "Cypher:" or wrap in triple backticks.

# EXAMPLES:

# Q: "Which controls help small businesses mitigate phishing attacks?"
# Cypher:
# MATCH (h:Entity)-[r:RELATION]->(t:Entity)
# WHERE h.type = "ControlOrMeasure"
#   AND t.type = "ThreatOrAttack"
#   AND toLower(t.name) CONTAINS "phishing"
#   AND r.type = "MITIGATES"
# RETURN h.name AS head, r.type AS relation, t.name AS tail
# LIMIT 25

# Q: "What impacts do ransomware incidents have on Australian small businesses?"
# Cypher:
# MATCH (h:Entity)-[r:RELATION]->(t:Entity)
# WHERE h.type = "ThreatOrAttack"
#   AND toLower(h.name) CONTAINS "ransomware"
#   AND t.type = "ImpactOrOutcome"
# RETURN h.name AS head, r.type AS relation, t.name AS tail
# LIMIT 25

# Q: "Which government bodies are responsible for cyber security guidance?"
# Cypher:
# MATCH (h:Entity)-[r:RELATION]->(t:Entity)
# WHERE h.type = "GovernmentBody"
#   AND r.type = "RESPONSIBLE_FOR"
# RETURN h.name AS head, r.type AS relation, t.name AS tail
# LIMIT 25

# Now generate a Cypher query for this question:

# Q: "{question}"
# Cypher:
# """
#         return prompt

#     def _extract_cypher(self, raw: str) -> str:
#         """
#         Extract a clean Cypher query from the LLM output.

#         - If there's a ``` ``` block, use its contents.
#         - Otherwise, take only lines that look like Cypher
#           (MATCH/WHERE/RETURN/etc.) and stop when we see plain English.
#         """
#         # Strip code fences if present
#         code_block = re.search(
#             r"```(?:cypher)?(.*?)```",
#             raw,
#             re.DOTALL | re.IGNORECASE,
#         )
#         if code_block:
#             candidate = code_block.group(1).strip()
#         else:
#             candidate = raw.strip()

#         cypher_lines = []
#         started = False

#         for line in candidate.splitlines():
#             stripped = line.strip()
#             if not stripped:
#                 continue

#             lower = stripped.lower()
#             is_cypher_line = lower.startswith(
#                 (
#                     "match ",
#                     "optional match",
#                     "where ",
#                     "with ",
#                     "return ",
#                     "limit ",
#                     "call ",
#                     "unwind ",
#                     "merge ",
#                     "create ",
#                     "delete ",
#                 )
#             )

#             if is_cypher_line:
#                 cypher_lines.append(stripped)
#                 started = True
#             else:
#                 # after we've started collecting cypher lines,
#                 # treat the first non-cypher-looking line as the end
#                 if started:
#                     break

#         cypher = "\n".join(cypher_lines).strip()
#         return cypher


#     def generate_cypher(self, question: str) -> str:
#         prompt = self.build_cypher_prompt(question)
#         raw = self.llm.generate(prompt, max_new_tokens=256)
#         cypher = self._extract_cypher(raw)
#         return cypher

#     # --------- 2. Run Cypher and collect facts ---------

#     def run_cypher(self, cypher: str) -> List[GraphFact]:
#         facts: List[GraphFact] = []
#         with self.driver.session() as session:
#             try:
#                 result = session.run(cypher)
#             except (CypherSyntaxError, Neo4jError) as e:
#                 raise RuntimeError(f"Cypher execution error: {e}") from e

#             for record in result:
#                 # Expect aliases: head, relation, tail
#                 head = record.get("head")
#                 relation = record.get("relation")
#                 tail = record.get("tail")

#                 if head and tail:
#                     facts.append(
#                         GraphFact(
#                             head=str(head),
#                             relation=str(relation) if relation else "",
#                             tail=str(tail),
#                         )
#                     )
#         return facts

#     # --------- 3. Answer question using graph facts ---------

#     def build_answer_prompt(self, question: str, facts: List[GraphFact]) -> str:
#         if facts:
#             facts_text = "\n".join(
#                 f"- {f.head} --{f.relation}--> {f.tail}" for f in facts
#             )
#         else:
#             facts_text = "(no relevant facts were retrieved from the knowledge graph)"

#         prompt = f"""You are answering a question using ONLY the following facts from a
# cybersecurity knowledge graph about Australian small businesses.

# FACTS:
# {facts_text}

# INSTRUCTIONS:
# - Use ONLY these facts. Do NOT add any outside knowledge or assumptions.
# - If the facts do not mention a specific control, do NOT invent it.
# - If the facts are insufficient to answer, say exactly:
#   "I do not have enough information from the knowledge graph to answer this question."
# - Write a single concise paragraph (3–5 sentences).
# - Do NOT include notes, explanations of your reasoning, or self-corrections.
# - Do NOT repeat the question.
# - Do NOT output anything after the answer paragraph.

# Question: {question}

# Answer (one concise paragraph):
# """
#         return prompt



#     def build_keyword_cypher(self, question: str, limit: int = 25) -> str:
#         """
#         Simple deterministic fallback:
#         - Extract a few keywords from the question
#         - Match any triples where head or tail name contains those keywords
#         """
#         # very simple keyword filter: words >=4 chars, not in stoplist
#         tokens = re.findall(r"[A-Za-z]{4,}", question.lower())
#         stop = {
#             "which", "what", "when", "where", "help", "helps", "helping",
#             "mitigate", "mitigates", "mitigating", "small", "business",
#             "businesses", "australian", "australia", "cyber", "security",
#             "threat", "threats", "related", "about", "that", "with", "from",
#         }
#         keywords = [t for t in tokens if t not in stop]
#         keywords = keywords[:3]  # keep top 3 to keep query reasonable

#         if keywords:
#             where_parts = []
#             for kw in keywords:
#                 where_parts.append(
#                     f"toLower(h.name) CONTAINS '{kw}'"
#                 )
#                 where_parts.append(
#                     f"toLower(t.name) CONTAINS '{kw}'"
#                 )
#             where_clause = " OR ".join(where_parts)
#         else:
#             where_clause = "true"

#         cypher = f"""
# MATCH (h:Entity)-[r:RELATION]->(t:Entity)
# WHERE {where_clause}
# RETURN h.name AS head, r.type AS relation, t.name AS tail
# LIMIT {limit}
# """.strip()
#         return cypher


#     def answer(
#         self,
#         question: str,
#         max_facts: int = 25,
#         retry_on_error: bool = True,
#     ) -> Dict[str, Any]:
#         """
#         Full GraphRAG pipeline, with robust fallbacks:
#         - Try LLM -> Cypher
#         - If Cypher looks bad or fails, use keyword-based Cypher
#         - If that returns no facts, use generic fallback
#         """
#         # ---- Step 1: try LLM-generated Cypher ----
#         cypher = self.generate_cypher(question)
#         print("Generated Cypher:\n", cypher)

#         def looks_incomplete(c: str) -> bool:
#             low = c.lower()
#             return ("return " not in low) or ("-[" not in c and ":RELATION" not in c)

#         # If the LLM output looks obviously incomplete, skip straight to keyword-based
#         if not cypher or looks_incomplete(cypher):
#             print("[INFO] LLM Cypher looks incomplete, using keyword-based Cypher.")
#             cypher = self.build_keyword_cypher(question)

#         # ---- Step 2: run primary Cypher (LLM or keyword-based) ----
#         try:
#             facts = self.run_cypher(cypher)
#         except RuntimeError as e:
#             print("[WARN] Primary Cypher failed:", e)
#             # If primary fails, fall back to keyword-based query once
#             if "build_keyword" not in cypher:
#                 print("[INFO] Falling back to keyword-based Cypher after error.")
#                 cypher = self.build_keyword_cypher(question)
#                 try:
#                     facts = self.run_cypher(cypher)
#                 except RuntimeError as e2:
#                     print("[WARN] Keyword-based Cypher also failed:", e2)
#                     facts = []
#             else:
#                 facts = []

#         # ---- Step 3: if still no facts, generic fallback query ----
#         if not facts:
#             print("[INFO] Primary/keyword queries returned no facts. Using generic fallback Cypher.")
#             fallback_cypher = """
# MATCH (h:Entity)-[r:RELATION]->(t:Entity)
# RETURN h.name AS head, r.type AS relation, t.name AS tail
# LIMIT 25
# """.strip()
#             try:
#                 facts = self.run_cypher(fallback_cypher)
#                 cypher = fallback_cypher
#             except RuntimeError as e:
#                 print("[WARN] Generic fallback Cypher also failed:", e)
#                 facts = []

#         # ---- Truncate and answer ----
#         if len(facts) > max_facts:
#             facts = facts[:max_facts]

#         answer_prompt = self.build_answer_prompt(question, facts)
#         answer_text = self.llm.generate(answer_prompt, max_new_tokens=256)
#         answer_text = clean_answer_text(answer_text, max_sentences=4)

#         # Keep only the first paragraph (before first blank line)
#         # answer_text = answer_text.strip()
#         # parts = answer_text.split("\n\n")
#         # answer_text = parts[0].strip()


#         return {
#             "question": question,
#             "cypher": cypher,
#             "facts": facts,
#             "answer": answer_text,
#         }




# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Simple GraphRAG demo")
#     parser.add_argument(
#         "--config",
#         type=str,
#         default="config/vector_llama.yaml",
#         help="Path to YAML config (must contain llm + graph sections)",
#     )
#     parser.add_argument(
#         "--question",
#         type=str,
#         default="Which controls help Australian small businesses mitigate phishing or scam-related cyber threats?",
#         help="Test question to ask via GraphRAG",
#     )

#     args = parser.parse_args()

#     rag = GraphRAG(args.config)
#     result = rag.answer(args.question, max_facts=25)

#     print("\n=== GRAPH RAG RESULT ===")
#     print("Q:", result["question"])
#     print("\nCypher used:\n", result["cypher"])
#     print("\nFacts used:")
#     for f in result["facts"]:
#         print(f"  - {f.head} --{f.relation}--> {f.tail}")
#     print("\nAnswer:\n", result["answer"])


