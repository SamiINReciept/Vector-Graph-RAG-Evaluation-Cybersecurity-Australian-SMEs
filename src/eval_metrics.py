# src/eval_metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score


@dataclass
class EvalMetrics:
    meteor: float
    bert_f1: float
    cosine: float


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Simple cosine similarity between two 1D vectors."""
    if v1.ndim > 1:
        v1 = v1.flatten()
    if v2.ndim > 1:
        v2 = v2.flatten()
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def compute_all_metrics(
    refs: List[str],
    hyps: List[str],
    embedder,
) -> EvalMetrics:
    """
    Compute:
      - METEOR
      - BERTScore-F1
      - Cosine similarity (using the provided embedder)
    `embedder` is expected to have a .embed(List[str]) -> List[list] method
    (your EmbeddingBackend from llm_backend.py).
    """
    if not refs:
        return EvalMetrics(meteor=0.0, bert_f1=0.0, cosine=0.0)

    # ----- METEOR -----
    meteor_scores = []
    for r, h in zip(refs, hyps):
        try:
            meteor_scores.append(meteor_score([r], h))
        except Exception:
            meteor_scores.append(0.0)
    meteor_avg = float(np.mean(meteor_scores))

    # ----- BERTScore -----
    P, R, F1 = bert_score(hyps, refs, lang="en", verbose=False)
    bert_f1_avg = float(F1.mean())

    # ----- Cosine similarity (via embedder) -----
    ref_vecs = np.array(embedder.embed(refs))
    hyp_vecs = np.array(embedder.embed(hyps))

    cos_scores = [
        _cosine_similarity(r_vec, h_vec)
        for r_vec, h_vec in zip(ref_vecs, hyp_vecs)
    ]
    cosine_avg = float(np.mean(cos_scores))

    return EvalMetrics(
        meteor=meteor_avg,
        bert_f1=bert_f1_avg,
        cosine=cosine_avg,
    )
