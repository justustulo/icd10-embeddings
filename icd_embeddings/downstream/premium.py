"""Aggregate member embeddings up to the client level for premium modeling.

This module ONLY builds features. It deliberately does not predict cost or
premium at the member level -- a member-level cost model on these embeddings is
too noisy. Instead we summarize the *distribution* of each client's member
embeddings (a client is a portfolio of members, and the tail matters as much as
the average), add exposure (member-months) and demographic mix, and hand the
result to your own client-level premium model.

Why distribution-aware rather than a plain mean: two clients with the same
average member can need very different premiums if one carries a few very
high-risk members. Percentiles and spread capture that; a mean alone hides it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from icd_embeddings.config import SEX_TO_ID, UNKNOWN_AGE_ID


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Column-wise weighted mean of a (n_members, dim) array.

    Args:
        values: Member vectors stacked as rows.
        weights: Non-negative per-member weights (e.g. member-months).

    Returns:
        A (dim,) weighted mean. Falls back to an unweighted mean if all weights
        are zero (so a client with no exposure info still produces features).
    """
    total_weight = weights.sum()
    if total_weight <= 0:
        return values.mean(axis=0)
    return (values * weights[:, None]).sum(axis=0) / total_weight


def _demographic_features(member_block: pd.DataFrame) -> dict:
    """Summarize age and sex mix for one client's members.

    Args:
        member_block: The member_vectors rows for a single client.

    Returns:
        Dict of demographic summary features.
    """
    known_age = member_block.loc[member_block["age_id"] != UNKNOWN_AGE_ID, "age_id"]
    mean_age = float(known_age.mean()) if len(known_age) > 0 else float("nan")

    n_members = len(member_block)
    sex_counts = member_block["sex_id"].value_counts()
    fraction_male = float(sex_counts.get(SEX_TO_ID["M"], 0)) / n_members
    fraction_female = float(sex_counts.get(SEX_TO_ID["F"], 0)) / n_members

    return {
        "mean_age": mean_age,
        "fraction_male": fraction_male,
        "fraction_female": fraction_female,
    }


def build_client_features(
    member_vectors: pd.DataFrame,
    exposure: pd.Series | None = None,
    include_percentiles: bool = True,
) -> pd.DataFrame:
    """Aggregate member embeddings into one feature row per client.

    For each client we emit, per embedding dimension, the exposure-weighted mean
    and the standard deviation (and optionally the 10th/90th percentiles to
    capture the tail), plus member count, total exposure, and demographic mix.

    Args:
        member_vectors: Output of extract_member_vectors (member_id, client_id,
            age_id, sex_id, vector).
        exposure: Optional per-member weight indexed by member_id (e.g.
            member-months). When given, the mean is exposure-weighted and the
            total is reported; when None, a simple mean is used.
        include_percentiles: If True, also emit per-dimension p10 and p90.

    Returns:
        DataFrame with one row per client_id and the aggregated feature columns.
    """
    if member_vectors.empty:
        raise ValueError("member_vectors is empty; nothing to aggregate.")

    embedding_dim = len(member_vectors["vector"].iloc[0])

    client_feature_rows = []
    for client_id, member_block in member_vectors.groupby("client_id"):
        vectors = np.asarray(member_block["vector"].tolist(), dtype=np.float32)

        if exposure is not None:
            weights = (
                exposure.reindex(member_block["member_id"]).fillna(0.0).to_numpy(dtype=np.float32)
            )
        else:
            weights = np.ones(len(member_block), dtype=np.float32)

        mean_vector = _weighted_mean(vectors, weights)
        std_vector = vectors.std(axis=0)

        features = {"client_id": client_id, "member_count": len(member_block)}
        if exposure is not None:
            features["total_member_months"] = float(weights.sum())

        for dim_index in range(embedding_dim):
            features[f"emb_mean_{dim_index}"] = float(mean_vector[dim_index])
            features[f"emb_std_{dim_index}"] = float(std_vector[dim_index])

        if include_percentiles:
            p10_vector = np.percentile(vectors, 10, axis=0)
            p90_vector = np.percentile(vectors, 90, axis=0)
            for dim_index in range(embedding_dim):
                features[f"emb_p10_{dim_index}"] = float(p10_vector[dim_index])
                features[f"emb_p90_{dim_index}"] = float(p90_vector[dim_index])

        features.update(_demographic_features(member_block))
        client_feature_rows.append(features)

    return pd.DataFrame(client_feature_rows)
