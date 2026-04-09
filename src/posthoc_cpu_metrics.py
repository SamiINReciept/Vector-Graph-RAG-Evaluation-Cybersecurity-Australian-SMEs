# src/posthoc_cpu_metrics.py

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from sentence_transformers import CrossEncoder
from rouge_score import rouge_scorer

from nltk.translate.meteor_score import meteor_score
import nltk


# ---------- NLTK setup ----------

def ensure_nltk():
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")


def clean_text_for_eval(s: str) -> str:
    """
    Light text cleaning:
    - strip,
    - collapse whitespace,
    - optionally truncate at common ramble markers.
    """
    if not isinstance(s, str):
        return ""

    s = s.strip()

    bad_markers = [
        "I do not have enough information to answer this question.",
        "I do not have enough information from the knowledge graph to answer this question.",
        "Let me know if this is acceptable.",
        "Here is the final answer:",
        "Here is the revised answer:",
    ]
    for marker in bad_markers:
        idx = s.find(marker)
        if idx != -1:
            s = s[: idx + len(marker)]
            break

    s = " ".join(s.split())
    return s


def build_context_string(contexts_cell) -> str:
    """
    contexts column is stored as 'chunk1 ||| chunk2 ||| ...' or maybe NaN.
    """
    if not isinstance(contexts_cell, str):
        return ""
    parts = [p.strip() for p in contexts_cell.split("|||") if p.strip()]
    return " ".join(parts)


def meteor_safe(ref: str, hyp: str) -> float:
    ref = (ref or "").strip()
    hyp = (hyp or "").strip()
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    try:
        return float(meteor_score([ref], hyp))
    except Exception:
        return 0.0


# ---------- CPU RAG evaluator ----------

class CPURAGEvaluator:
    def __init__(self):
        print("Loading CPU models... (first time may take a minute)")

        # NLI model for hallucination / faithfulness
        self.nli_model = CrossEncoder(
            "cross-encoder/nli-deberta-v3-small",
            device="cpu",
        )

        # Relevance scorer
        self.rel_model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
        )

        # ROUGE-L for completeness
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def hallucination_from_context(
        self,
        contexts: List[str],
        answers: List[str],
        batch_size: int = 16,
    ) -> Tuple[List[float], List[float]]:
        """
        Returns (faithfulness_list, hallucination_risk_list)
        context = premise, answer = hypothesis
        logits order: [contradiction, entailment, neutral]
        """
        pairs = [(c or "", a or "") for c, a in zip(contexts, answers)]
        logits = self.nli_model.predict(pairs, batch_size=batch_size)
        logits = np.array(logits)
        probs = self._softmax(logits)
        contradiction = probs[:, 0]
        entailment = probs[:, 1]
        return entailment.tolist(), contradiction.tolist()

    def relevance(
        self,
        queries: List[str],
        texts: List[str],
        batch_size: int = 16,
    ) -> List[float]:
        pairs = [(q or "", t or "") for q, t in zip(queries, texts)]
        scores = self.rel_model.predict(pairs, batch_size=batch_size)
        scores = np.array(scores, dtype=float)
        norm = 1 / (1 + np.exp(-scores))  # sigmoid to ~[0,1]
        return norm.tolist()

    def completeness_rougeL(
        self,
        golds: List[str],
        answers: List[str],
    ) -> List[float]:
        out = []
        for g, a in zip(golds, answers):
            g = g or ""
            a = a or ""
            if not g.strip() and not a.strip():
                out.append(1.0)
                continue
            if not g.strip() or not a.strip():
                out.append(0.0)
                continue
            scores = self.rouge.score(g, a)
            out.append(scores["rougeL"].recall)
        return out


# ---------- Per-pipeline processing ----------

def process_pipeline(csv_path: Path, evaluator: CPURAGEvaluator) -> Tuple[pd.DataFrame, Dict]:
    print(f"\n=== Processing {csv_path.name} ===")
    df = pd.read_csv(csv_path)

    # choose answer column: prefer cleaned if available
    answer_col = "model_answer_clean" if "model_answer_clean" in df.columns else "model_answer"
    if answer_col not in df.columns:
        raise ValueError(f"{csv_path} missing both 'model_answer_clean' and 'model_answer'")

    for col in ["question", "gold_answer"]:
        if col not in df.columns:
            raise ValueError(f"{csv_path} missing column '{col}'")

    df["gold_clean"] = df["gold_answer"].astype(str).apply(clean_text_for_eval)
    df["answer_clean"] = df[answer_col].astype(str).apply(clean_text_for_eval)

    if "contexts" in df.columns:
        df["contexts_joined"] = df["contexts"].apply(build_context_string)
    else:
        df["contexts_joined"] = ""

    # METEOR
    print("  -> METEOR (CPU)...")
    df["meteor"] = [
        meteor_safe(ref, hyp)
        for ref, hyp in zip(df["gold_clean"], df["answer_clean"])
    ]

    # Hallucination / faithfulness
    print("  -> hallucination / faithfulness (NLI cross-encoder)...")
    faithfulness, hall_risk = evaluator.hallucination_from_context(
        df["contexts_joined"].tolist(),
        df["answer_clean"].tolist(),
    )
    df["faithfulness"] = faithfulness
    df["hallucination_risk"] = hall_risk

    # Relevance (query vs answer)
    print("  -> answer relevance (query vs answer)...")
    df["answer_relevance"] = evaluator.relevance(
        df["question"].astype(str).tolist(),
        df["answer_clean"].tolist(),
    )

    # Context relevance (query vs contexts)
    print("  -> context relevance (query vs retrieved contexts)...")
    df["context_relevance"] = evaluator.relevance(
        df["question"].astype(str).tolist(),
        df["contexts_joined"].tolist(),
    )

    # Completeness via ROUGE-L recall
    print("  -> completeness (ROUGE-L recall gold vs answer)...")
    df["completeness_rougeL_recall"] = evaluator.completeness_rougeL(
        df["gold_clean"].tolist(),
        df["answer_clean"].tolist(),
    )

    # ---- Explicit student-facing metrics ----
    # hallucination: higher = more hallucination (worse)
    df["hallucination"] = df["hallucination_risk"]

    # irrelevance: higher = more irrelevant (worse)
    df["irrelevance"] = 1.0 - df["answer_relevance"]

    # completeness: higher = more complete (better)
    df["completeness"] = df["completeness_rougeL_recall"]

    # Prepare summary
    pipeline_name = df["pipeline"].iloc[0] if "pipeline" in df.columns else csv_path.stem

    metrics_to_avg = [
        "meteor",
        "faithfulness",
        "hallucination_risk",
        "answer_relevance",
        "context_relevance",
        "completeness_rougeL_recall",
        "hallucination",
        "irrelevance",
        "completeness",
        "latency_sec",
    ]

    summary = {"pipeline": pipeline_name}
    for m in metrics_to_avg:
        if m in df.columns:
            summary[m] = float(df[m].mean())

    return df, summary


# ---------- Main ----------

def main():
    ensure_nltk()

    results_dir = Path("results")
    if not results_dir.exists():
        raise FileNotFoundError("results/ directory not found. Run evaluate_pipelines first.")

    evaluator = CPURAGEvaluator()

    expected = [
        "vector_llama_raw.csv",
        "vector_mistral_raw.csv",
        "vector_qwen_raw.csv",
        "graph_llama_raw.csv",
        "graph_mistral_raw.csv",
        "graph_qwen_raw.csv",
    ]

    summaries = []

    for fname in expected:
        csv_path = results_dir / fname
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping.")
            continue

        df, summary = process_pipeline(csv_path, evaluator)

        out_path = results_dir / f"{summary['pipeline']}_with_cpu_metrics.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> saved enriched CSV to {out_path}")

        summaries.append(summary)

    if summaries:
        summary_df = pd.DataFrame(summaries)
        cols = ["pipeline"] + [c for c in summary_df.columns if c != "pipeline"]
        summary_df = summary_df[cols]

        summary_out = results_dir / "summary_cpu_metrics.csv"
        summary_df.to_csv(summary_out, index=False)
        print(f"\nSaved summary metrics to {summary_out}")
        print(summary_df)
    else:
        print("No pipelines processed. Check that *_raw.csv files exist in results/.")


if __name__ == "__main__":
    main()
