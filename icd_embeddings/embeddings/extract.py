"""Extract the reusable embeddings from a trained model.

Two artifacts come out of here, and they are the actual deliverable of the
project:

  * code vectors  - one vector per vocabulary token (from the code embedding
    table), plus a cosine nearest-neighbor helper used for HCC-suspect candidate
    generation and the fraud "what's normal together" check.
  * member vectors - one vector per member (the CLS hidden state over the
    member's real, unmasked sequence), the feature fed to downstream models.

Both are written as parquet with the vector stored as a list column named
"vector".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from icd_embeddings.config import CODE_TYPES, Config
from icd_embeddings.model.dataset import MaskedCodeCollator, MemberSequenceDataset, load_sequences
from icd_embeddings.model.transformer import MaskedCodeTransformer


def extract_code_vectors(
    config: Config, model: MaskedCodeTransformer, vocab: pd.DataFrame
) -> pd.DataFrame:
    """Pull the per-token code vectors out of the model and save them.

    Args:
        config: The run configuration (uses config.code_vectors_path).
        model: A trained model.
        vocab: The vocabulary DataFrame (token, token_id, code_type, member_count).

    Returns:
        DataFrame with columns token, token_id, code_type, member_count, vector.
    """
    weight_matrix = model.code_embedding.weight.detach().cpu().numpy()
    if len(vocab) != weight_matrix.shape[0]:
        raise ValueError(
            f"Vocab size ({len(vocab)}) does not match the code embedding table "
            f"({weight_matrix.shape[0]}). Vocab and model are out of sync."
        )

    code_vectors = vocab.copy()
    # Order vocab by token_id so row i of the matrix lines up with token_id i.
    code_vectors = code_vectors.sort_values("token_id").reset_index(drop=True)
    code_vectors["vector"] = [row.tolist() for row in weight_matrix]

    code_vectors.to_parquet(config.code_vectors_path, index=False)
    print(f"[extract] {len(code_vectors)} code vectors written to {config.code_vectors_path}")
    return code_vectors


def _vectors_as_matrix(vectors_df: pd.DataFrame) -> np.ndarray:
    """Stack a 'vector' list column into a 2-D float32 numpy array."""
    return np.asarray(vectors_df["vector"].tolist(), dtype=np.float32)


def nearest_neighbors(
    code_vectors: pd.DataFrame,
    query_token: str,
    query_code_type: str,
    k: int = 10,
) -> pd.DataFrame:
    """Return the k most cosine-similar real-code tokens to a query token.

    Useful as a clinical sanity check (e.g. diabetes codes should neighbor each
    other) and as HCC-suspect / fraud candidate generation.

    Args:
        code_vectors: Output of extract_code_vectors.
        query_token: The token string to search around (e.g. "E11.9").
        query_code_type: The code type of the query ("dx"/"proc"/"rx").
        k: Number of neighbors to return (excluding the query itself).

    Returns:
        DataFrame of the top-k neighbors with columns token, code_type, similarity,
        sorted by descending cosine similarity.
    """
    real_codes = code_vectors[code_vectors["code_type"].isin(CODE_TYPES)].reset_index(
        drop=True
    )
    query_mask = (real_codes["token"] == query_token) & (
        real_codes["code_type"] == query_code_type
    )
    if not query_mask.any():
        raise ValueError(
            f"Query token ({query_code_type}, '{query_token}') is not in the vocabulary."
        )

    matrix = _vectors_as_matrix(real_codes)
    # Cosine similarity = dot product of L2-normalized vectors.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against a zero vector
    normalized = matrix / norms

    query_index = int(query_mask.to_numpy().nonzero()[0][0])
    similarities = normalized @ normalized[query_index]

    result = real_codes[["token", "code_type"]].copy()
    result["similarity"] = similarities
    result = result.drop(index=query_index)  # don't return the query itself
    return result.sort_values("similarity", ascending=False).head(k).reset_index(drop=True)


@torch.no_grad()
def extract_member_vectors(
    config: Config, model: MaskedCodeTransformer
) -> pd.DataFrame:
    """Compute one CLS embedding per member and save it.

    Runs the model over each member's real (unmasked) sequence. Member-level
    fields (client_id, age_id, sex_id) are carried through so downstream code can
    aggregate to the client and add demographics.

    Args:
        config: The run configuration (uses config.member_vectors_path).
        model: A trained model in eval mode.

    Returns:
        DataFrame with columns member_id, client_id, age_id, sex_id, vector.
    """
    sequences = load_sequences(config)
    vocab_size = model.vocab_size
    collator = MaskedCodeCollator(config=config, vocab_size=vocab_size)

    # shuffle=False so the output order matches the sequences row order, letting us
    # line member vectors back up with member_id below.
    loader = DataLoader(
        MemberSequenceDataset(sequences),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator.inference_batch,
    )

    model.eval()
    member_embeddings = []
    for batch in loader:
        batch = {key: value.to(config.device) for key, value in batch.items()}
        hidden_states = model.encode(
            token_ids=batch["token_ids"],
            type_ids=batch["type_ids"],
            recency_ids=batch["recency_ids"],
            age_ids=batch["age_ids"],
            sex_ids=batch["sex_ids"],
            attention_mask=batch["attention_mask"],
        )
        cls_vectors = hidden_states[:, 0, :].cpu().numpy()
        member_embeddings.append(cls_vectors)

    all_vectors = np.concatenate(member_embeddings, axis=0)

    member_vectors = sequences[["member_id", "client_id", "age_id", "sex_id"]].copy()
    member_vectors["vector"] = [row.tolist() for row in all_vectors]

    member_vectors.to_parquet(config.member_vectors_path, index=False)
    print(
        f"[extract] {len(member_vectors)} member vectors written to "
        f"{config.member_vectors_path}"
    )
    return member_vectors
