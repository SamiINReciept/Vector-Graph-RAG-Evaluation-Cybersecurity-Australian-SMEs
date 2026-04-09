from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
)

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# =========================
# CONFIG
# =========================

RESULTS_DIR = "results"

# How many Qs per pipeline to use for RAGAS.
# You can lower this (e.g. 10) to save credits/time.
N_QUESTIONS = 30

@dataclass
class PipelineSpec:
    name: str
    csv_path: str


PIPELINES: List[PipelineSpec] = [
    PipelineSpec("vector_llama",   os.path.join(RESULTS_DIR, "vector_llama_raw.csv")),
    PipelineSpec("vector_mistral", os.path.join(RESULTS_DIR, "vector_mistral_raw.csv")),
    PipelineSpec("vector_qwen",    os.path.join(RESULTS_DIR, "vector_qwen_raw.csv")),
    PipelineSpec("graph_llama",    os.path.join(RESULTS_DIR, "vector_llama_raw.csv")),
    PipelineSpec("graph_mistral",  os.path.join(RESULTS_DIR, "vector_mistral_raw.csv")),
    PipelineSpec("graph_qwen",     os.path.join(RESULTS_DIR, "vector_qwen_raw.csv")),
]

# If you want to be extra conservative with credit:
# PIPELINES = [
#     PipelineSpec("vector_mistral", os.path.join(RESULTS_DIR, "vector_mistral_raw.csv")),
#     PipelineSpec("graph_mistral",  os.path.join(RESULTS_DIR, "graph_mistral_raw.csv")),
# ]


# =========================
# RAGAS judge + embeddings
# =========================

def build_ragas_models():
    """
    Build the judge LLM + embeddings used by RAGAS.
    This is completely separate from your RAG pipelines.
    """
    judge_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    print(f"Loading RAGAS judge model: {judge_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(judge_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,    # judge doesn't need long outputs
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)

    emb_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading RAGAS embedding model: {emb_model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)

    return llm, embeddings


# =========================
# Data loading
# =========================

def load_pipeline_results(spec: PipelineSpec, limit: int | None = None) -> pd.DataFrame:
    if not os.path.exists(spec.csv_path):
        raise FileNotFoundError(f"Per-pipeline CSV not found: {spec.csv_path}")

    df = pd.read_csv(spec.csv_path)

    required = {"question", "gold_answer", "model_answer_clean", "contexts"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{spec.csv_path} is missing required columns {required}, has: {df.columns}"
        )

    if limit is not None:
        df = df.iloc[:limit].copy()

    # Normalize columns to what RAGAS needs
    df = df[["question", "gold_answer", "model_answer_clean", "contexts"]].copy()
    df.rename(columns={"gold_answer": "ground_truth", "model_answer_clean": "answer"}, inplace=True)

    # Convert serialized contexts ("text1 ||| text2") to list[str]
    def split_contexts(val: Any) -> List[str]:
        if not isinstance(val, str) or not val.strip():
            return []
        parts = [p.strip() for p in val.split("|||") if p.strip()]
        return parts

    df["contexts"] = df["contexts"].apply(split_contexts)
    return df


# =========================
# RAGAS evaluation for one pipeline
# =========================

def evaluate_pipeline_with_ragas(
    spec: PipelineSpec,
    llm,
    embeddings,
    limit: int,
) -> Dict[str, float]:
    """
    For a given pipeline:
      - load its per-question CSV
      - build a RAGAS Dataset
      - run faithfulness / answer_relevancy / context_recall
    """
    print(f"\n→ Loading results for {spec.name} from {spec.csv_path}")
    df = load_pipeline_results(spec, limit=limit)
    print(f"Using {len(df)} samples for {spec.name}")

    # RAGAS expects: question, answer, contexts, ground_truth
    samples: List[Dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=spec.name):
        samples.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
        )

    dataset = Dataset.from_list(samples)

    print(f"Running RAGAS metrics for {spec.name} ...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    scores = {
        "pipeline": spec.name,
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_recall": float(result["context_recall"]),
    }
    return scores


# =========================
# Main
# =========================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    llm, embeddings = build_ragas_models()

    all_scores: List[Dict[str, Any]] = []
    for spec in tqdm(PIPELINES, desc="Pipelines (RAGAS)"):
        scores = evaluate_pipeline_with_ragas(
            spec,
            llm=llm,
            embeddings=embeddings,
            limit=N_QUESTIONS,
        )
        all_scores.append(scores)

    df_scores = pd.DataFrame(all_scores)
    df_scores = df_scores[["pipeline", "faithfulness", "answer_relevancy", "context_recall"]]

    out_path = os.path.join(RESULTS_DIR, "ragas_metrics_light.csv")
    df_scores.to_csv(out_path, index=False)

    print("\n=== RAGAS metrics (light) across pipelines ===")
    print(df_scores.to_string(index=False))
    print(f"\nSaved RAGAS scores to {out_path}")


if __name__ == "__main__":
    main()
