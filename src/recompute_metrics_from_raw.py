# src/recompute_metrics_from_raw.py

from __future__ import annotations

import glob
import os
from typing import List, Dict, Any

import pandas as pd

from .llm_backend import EmbeddingBackend, EmbeddingConfig
from .eval_metrics import compute_all_metrics
from .text_utils import clean_for_eval


def main():
    results_dir = "results"
    pattern = os.path.join(results_dir, "*_raw.csv")
    raw_files = sorted(glob.glob(pattern))

    if not raw_files:
        raise FileNotFoundError(f"No *_raw.csv files found in {results_dir}")

    print("Found raw result files:")
    for f in raw_files:
        print("  -", f)

    all_summaries: List[Dict[str, Any]] = []

    # Shared embedder for cosine similarity
    embedder = EmbeddingBackend(EmbeddingConfig())

    for path in raw_files:
        df = pd.read_csv(path)
        if not {"gold_answer", "model_answer"}.issubset(df.columns):
            print(f"Skipping {path}, missing required columns.")
            continue

        # Clean model answers for eval
        df["model_answer_clean"] = df["model_answer"].apply(clean_for_eval)

        refs = df["gold_answer"].tolist()
        hyps = df["model_answer_clean"].tolist()

        metrics = compute_all_metrics(refs, hyps, embedder=embedder)
        avg_latency = float(df["latency_sec"].mean()) if "latency_sec" in df.columns else None

        # infer pipeline name from first row or filename
        if "pipeline" in df.columns:
            pipeline_name = df["pipeline"].iloc[0]
        else:
            base = os.path.basename(path)
            pipeline_name = base.replace("_raw.csv", "")

        summary = {
            "pipeline": pipeline_name,
            "meteor": metrics.meteor,
            "bertscore_f1": metrics.bert_f1,
            "cosine_sim": metrics.cosine,
        }
        if avg_latency is not None:
            summary["avg_latency_sec"] = avg_latency

        all_summaries.append(summary)

        # overwrite the CSV with cleaned answers included (optional)
        out_path = path.replace("_raw.csv", "_raw_clean.csv")
        df.to_csv(out_path, index=False)
        print(f"Processed {pipeline_name}, saved cleaned file to {out_path}")

    if not all_summaries:
        print("No summaries computed.")
        return

    df_summary = pd.DataFrame(all_summaries)
    cols = ["pipeline", "meteor", "bertscore_f1", "cosine_sim"]
    if "avg_latency_sec" in df_summary.columns:
        cols.append("avg_latency_sec")
    df_summary = df_summary[cols]

    out_summary = os.path.join(results_dir, "summary_metrics.csv")
    df_summary.to_csv(out_summary, index=False)
    print("\n=== Summary metrics ===")
    print(df_summary.to_string(index=False))
    print(f"\nSaved summary to {out_summary}")


if __name__ == "__main__":
    main()
