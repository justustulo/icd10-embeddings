"""Fraud anomaly scoring (minimal stub).

Fraud was explicitly a minor consideration, so this is a deliberately simple
starting point, not a finished detector. The idea: a member whose embedding sits
far from the population is "unusual" in the same code-co-occurrence space the
model learned, which is a reasonable first anomaly signal.

A natural next step (left as a TODO) is a cross-type mismatch score -- e.g. a
procedure or drug token with no supporting diagnosis -- which is the classic
fraud pattern and is exactly what the shared, type-aware embedding can surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def anomaly_scores(member_vectors: pd.DataFrame) -> pd.DataFrame:
    """Score each member by how far their embedding is from the population center.

    Score = 1 - cosine_similarity(member_vector, mean_member_vector). Higher means
    more unusual. This is an unsupervised outlier signal, not a fraud label.

    Args:
        member_vectors: Output of extract_member_vectors (member_id + vector).

    Returns:
        DataFrame with columns member_id, anomaly_score, sorted most-anomalous first.
    """
    if member_vectors.empty:
        raise ValueError("member_vectors is empty; nothing to score.")

    vectors = np.asarray(member_vectors["vector"].tolist(), dtype=np.float32)
    center = vectors.mean(axis=0)

    center_norm = np.linalg.norm(center)
    vector_norms = np.linalg.norm(vectors, axis=1)
    # Guard against zero-length vectors so we never divide by zero.
    safe_denominator = np.where(vector_norms == 0, 1.0, vector_norms) * (
        center_norm if center_norm != 0 else 1.0
    )
    cosine_similarity = (vectors @ center) / safe_denominator

    scored = member_vectors[["member_id"]].copy()
    scored["anomaly_score"] = 1.0 - cosine_similarity

    # TODO: add a cross-type mismatch score (procedure/drug token with no
    # supporting diagnosis) using the type-aware code embeddings.
    return scored.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
