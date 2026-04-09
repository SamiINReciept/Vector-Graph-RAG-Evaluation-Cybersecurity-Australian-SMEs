from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

from .vector_rag import VectorRAG
from .graph_rag import GraphRAG
from .llm_backend import EmbeddingBackend, EmbeddingConfig
from .eval_metrics import compute_all_metrics
from .text_utils import clean_for_eval


# =========================
# CONFIG
# =========================

TEST_CSV = "data/test_qas.csv"
RESULTS_DIR = "results"

# Retrieval sizes
VECTOR_TOP_K = 5       # chunks for vector RAG
GRAPH_MAX_FACTS = 10   # facts for graph RAG (keep it modest)

@dataclass
class PipelineSpec:
    name: str
    kind: str          # "vector" or "graph"
    config_path: str


PIPELINES: List[PipelineSpec] = [
    PipelineSpec("vector_llama",   "vector", "config/vector_llama.yaml"),
    PipelineSpec("vector_mistral", "vector", "config/vector_mistral.yaml"),
    PipelineSpec("vector_qwen",    "vector", "config/vector_qwen.yaml"),
    PipelineSpec("graph_llama",    "graph",  "config/vector_llama.yaml"),
    PipelineSpec("graph_mistral",  "graph",  "config/vector_mistral.yaml"),
    PipelineSpec("graph_qwen",     "graph",  "config/vector_qwen.yaml"),
]

# If you want to do a small/quick run, you can temporarily comment out some
# pipelines above or later slice the list, e.g. PIPELINES = PIPELINES[:2]


# =========================
# Helpers
# =========================

def load_test_set(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Test set CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Expect at least: question, answer
    required = {"question", "answer"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Test set must contain columns {required}, found: {df.columns}"
        )

    if "id" not in df.columns:
        df["id"] = [f"Q{i+1}" for i in range(len(df))]

    return df[["id", "question", "answer"]].copy()


def init_rag(spec: PipelineSpec):
    if spec.kind == "vector":
        return VectorRAG(spec.config_path)
    elif spec.kind == "graph":
        return GraphRAG(spec.config_path)
    else:
        raise ValueError(f"Unknown pipeline kind: {spec.kind}")


def serialize_contexts_vector(contexts) -> str:
    """
    contexts: list[RetrievedChunk] from VectorRAG.answer
    We serialize only the text, joined with ' ||| ' so RAGAS can split later.
    """
    if not contexts:
        return ""
    texts = [c.text for c in contexts]
    return " ||| ".join(str(t) for t in texts)


def serialize_contexts_graph(facts) -> str:
    """
    facts: list[GraphFact] from GraphRAG.answer
    We serialize as 'head --REL--> tail', joined with ' ||| '.
    """
    if not facts:
        return ""
    texts = [f"{f.head} --{f.relation}--> {f.tail}" for f in facts]
    return " ||| ".join(str(t) for t in texts)


# =========================
# Per-pipeline evaluation
# =========================

def evaluate_pipeline(
    spec: PipelineSpec,
    df_test: pd.DataFrame,
    embedder: EmbeddingBackend,
) -> Dict[str, Any]:
    """
    Run one pipeline (vector or graph) over all test questions,
    save per-pipeline CSV, and return aggregate metrics.
    """
    rag = init_rag(spec)

    rows: List[Dict[str, Any]] = []

    print(f"\n=== Evaluating pipeline: {spec.name} ({spec.kind}) ===")
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc=spec.name):
        qid = row["id"]
        question = row["question"]
        gold = row["answer"]

        t0 = time.time()
        try:
            if spec.kind == "vector":
                out = rag.answer(question, top_k=VECTOR_TOP_K)
                raw_answer = str(out.get("answer", "")).strip()
                ctx_ser = serialize_contexts_vector(out.get("contexts", []))
            else:  # graph
                out = rag.answer(question, max_facts=GRAPH_MAX_FACTS)
                raw_answer = str(out.get("answer", "")).strip()
                ctx_ser = serialize_contexts_graph(out.get("facts", []))

            error_flag = ""
        except Exception as e:
            raw_answer = f"[ERROR: {e}]"
            ctx_ser = ""
            error_flag = "error"
        t1 = time.time()

        clean_answer = clean_for_eval(raw_answer)
        latency = t1 - t0

        rows.append(
            {
                "pipeline": spec.name,
                "id": qid,
                "question": question,
                "gold_answer": gold,
                "model_answer_raw": raw_answer,
                "model_answer_clean": clean_answer,
                "contexts": ctx_ser,
                "latency_sec": latency,
                "status": error_flag or "ok",
            }
        )

    df = pd.DataFrame(rows)

    # Save detailed results for this pipeline
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{spec.name}_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved per-question results to {out_path}")

    # Compute metrics using CLEAN answers
    refs = df["gold_answer"].tolist()
    hyps = df["model_answer_clean"].tolist()
    metrics = compute_all_metrics(refs, hyps, embedder=embedder)

    avg_latency = float(df["latency_sec"].mean())

    summary = {
        "pipeline": spec.name,
        "meteor": metrics.meteor,
        "bertscore_f1": metrics.bert_f1,
        "cosine_sim": metrics.cosine,
        "avg_latency_sec": avg_latency,
    }

    print(
        f"METRICS for {spec.name} -> "
        f"METEOR: {metrics.meteor:.4f}, "
        f"BERTScore F1: {metrics.bert_f1:.4f}, "
        f"Cosine: {metrics.cosine:.4f}, "
        f"Avg latency: {avg_latency:.2f}s"
    )

    return summary


# =========================
# Main
# =========================

def main():
    df_test = load_test_set(TEST_CSV)
    print(f"Loaded test set with {len(df_test)} questions from {TEST_CSV}")

    # Shared embedder for cosine similarity
    embedder = EmbeddingBackend(EmbeddingConfig())

    all_summaries: List[Dict[str, Any]] = []
    for spec in PIPELINES:
        summary = evaluate_pipeline(spec, df_test, embedder=embedder)
        all_summaries.append(summary)

    df_summary = pd.DataFrame(all_summaries)
    df_summary = df_summary[
        ["pipeline", "meteor", "bertscore_f1", "cosine_sim", "avg_latency_sec"]
    ]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "summary_metrics.csv")
    df_summary.to_csv(summary_path, index=False)

    print("\n=== Summary over all pipelines ===")
    print(df_summary.to_string(index=False))
    print(f"\nSaved summary metrics to {summary_path}")


if __name__ == "__main__":
    main()
