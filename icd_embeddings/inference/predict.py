"""Interactive inference helpers for the code prediction visualization app.

Three-step calling convention:
  1. Load once per session: read the vocab parquet and build the model from a
     checkpoint. Store both in st.session_state to avoid reloading.
  2. Prepare input: call build_sequence_tensors() whenever the code list changes.
  3. Predict: call predict_single_masked() or predict_chained() on each run.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from icd_embeddings.config import (
    CLS_TOKEN_ID,
    MASK_TOKEN_ID,
    MAX_AGE,
    SEX_TO_ID,
    SPECIAL_TYPE_ID,
    TYPE_TO_ID,
    UNKNOWN_AGE_ID,
    UNK_TOKEN_ID,
)
from icd_embeddings.data.build_vocab import build_token_lookup, map_codes_to_tokens
from icd_embeddings.model.transformer import MaskedCodeTransformer


@dataclass
class ArchConfig:
    """Minimal architecture settings for inference — no data paths or training parameters.

    Use this to reconstruct a trained model from a checkpoint without needing a
    full Config (which requires claims_path and observation dates). All defaults
    match the Config class defaults.

    Attributes:
        embedding_dim: Size of each code/member embedding vector.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads. Must evenly divide embedding_dim.
        feedforward_dim: Width of the transformer feed-forward sublayer.
        dropout: Dropout probability. Not applied during eval, but must match
            the saved weights' architecture.
        recency_bucket_day_edges: Ascending day thresholds for recency buckets,
            measured backwards from the observation anchor. Must match training.
        max_sequence_length: Maximum number of code tokens per member (not counting CLS).
        rollup_rare_dx_to_3char: Whether to try the 3-char parent rollup for unknown
            diagnosis codes. Must match the setting used when building the vocab.
    """

    embedding_dim: int = 128
    n_layers: int = 4
    n_heads: int = 8
    feedforward_dim: int = 512
    dropout: float = 0.1
    recency_bucket_day_edges: tuple[int, ...] = (30, 90, 180, 365, 730)
    max_sequence_length: int = 256
    rollup_rare_dx_to_3char: bool = True

    @property
    def n_recency_buckets(self) -> int:
        """Number of recency buckets, including the final open-ended bucket."""
        return len(self.recency_bucket_day_edges) + 1


def _days_to_recency_id(days_ago: int, bucket_day_edges: tuple[int, ...]) -> int:
    """Map a count of days to a recency bucket index.

    Mirrors _recency_bucket_id in build_sequences.py. Replicated here so the
    inference module has no dependency on a private function in another subpackage.

    Bucket boundaries are half-open intervals measured backwards from the
    observation anchor. For edges (30, 90, 180): days 0-30 map to bucket 0,
    31-90 to bucket 1, 91-180 to bucket 2, and >180 to the final open-ended
    bucket 3.

    Args:
        days_ago: Whole days between the code's date and the reference point.
            Values less than 0 are treated as 0 (most recent bucket).
        bucket_day_edges: Ascending tuple of day thresholds, e.g. (30, 90, 180, 365, 730).

    Returns:
        Integer bucket id in [0, len(bucket_day_edges)].

    Example:
        >>> _days_to_recency_id(45, (30, 90, 180, 365))
        1
    """
    days_ago = max(0, days_ago)
    for bucket_index, edge in enumerate(bucket_day_edges):
        if days_ago <= edge:
            return bucket_index
    return len(bucket_day_edges)


def _sex_string_to_id(sex: str) -> int:
    """Convert a sex string to its integer id, defaulting unrecognized values to 0.

    Mirrors _sex_id in build_sequences.py. Only "M" and "F" (after stripping
    whitespace and uppercasing) map to non-zero ids; everything else is unknown (0).

    Args:
        sex: A sex label string, e.g. "M", "F", "male", "unknown", or "".

    Returns:
        0 for unknown, 1 for M, 2 for F.

    Example:
        >>> _sex_string_to_id("F")
        2
        >>> _sex_string_to_id("unknown")
        0
    """
    return SEX_TO_ID.get(sex.strip().upper(), 0)


def build_sequence_tensors(
    codes: list[str],
    code_types: list[str],
    days_ago: list[int],
    vocab: pd.DataFrame,
    arch: ArchConfig,
    age: int = UNKNOWN_AGE_ID,
    sex: str = "unknown",
) -> dict[str, torch.Tensor]:
    """Convert a list of raw code strings into the 6-tensor dict the model expects.

    Replicates what the training pipeline does in build_sequences.py +
    MaskedCodeCollator._build_padded_batch, but for a single user-supplied
    sequence instead of a batch of members from disk.

    All returned tensors have a batch dimension of 1. The sequence layout is:
        [CLS, code_0, code_1, ..., code_{n-1}]
    CLS is always at position 0.

    Codes not in the vocabulary are silently mapped to UNK. Call
    validate_codes_against_vocab() first to surface a warning to the user.

    If the code list exceeds arch.max_sequence_length, the most recent codes
    (lowest days_ago values) are kept, matching build_sequences.py truncation.

    Args:
        codes: Raw code strings, e.g. ["E11.9", "99213", "ANTIDIABETIC"].
        code_types: Code type for each code, one of "dx", "proc", or "rx".
            Must be the same length as codes.
        days_ago: How many days ago each code occurred (>= 0). Must be the same
            length as codes.
        vocab: The vocabulary DataFrame (columns: token, token_id, code_type,
            member_count).
        arch: Architecture config used for recency bucketing and truncation.
        age: Member's age in whole years (0-100). Pass UNKNOWN_AGE_ID (101)
            when unknown. Values outside [0, MAX_AGE] become UNKNOWN_AGE_ID.
        sex: Member's sex: "M", "F", or anything else for unknown.

    Returns:
        Dict with keys matching model.forward() signature:
            token_ids:      LongTensor shape (1, seq_len)
            type_ids:       LongTensor shape (1, seq_len)
            recency_ids:    LongTensor shape (1, seq_len)
            age_ids:        LongTensor shape (1,)
            sex_ids:        LongTensor shape (1,)
            attention_mask: LongTensor shape (1, seq_len)
        where seq_len = min(len(codes), max_sequence_length) + 1.

    Raises:
        ValueError: If codes, code_types, and days_ago are not the same length.
        ValueError: If codes is empty.
        ValueError: If any code_type is not "dx", "proc", or "rx".
        ValueError: If any days_ago value is negative.

    Example:
        >>> arch = ArchConfig()
        >>> tensors = build_sequence_tensors(
        ...     codes=["E11.9", "I10"],
        ...     code_types=["dx", "dx"],
        ...     days_ago=[30, 90],
        ...     vocab=vocab_df,
        ...     arch=arch,
        ...     age=55,
        ...     sex="M",
        ... )
        >>> tensors["token_ids"].shape
        torch.Size([1, 3])
    """
    if len(codes) == 0:
        raise ValueError("codes must contain at least one entry.")
    if len(codes) != len(code_types) or len(codes) != len(days_ago):
        raise ValueError(
            f"codes, code_types, and days_ago must be the same length. "
            f"Got lengths {len(codes)}, {len(code_types)}, {len(days_ago)}."
        )
    invalid_types = [t for t in code_types if t not in TYPE_TO_ID]
    if invalid_types:
        raise ValueError(
            f"Unrecognized code_types: {invalid_types}. Must be 'dx', 'proc', or 'rx'."
        )
    negative_days = [d for d in days_ago if d < 0]
    if negative_days:
        raise ValueError(
            f"days_ago values must be >= 0, got: {negative_days}"
        )

    # Build a frame to make truncation easy — sort by recency and keep the most recent.
    sequence_df = pd.DataFrame({
        "code": codes,
        "code_type": code_types,
        "days_ago": days_ago,
    })
    if len(sequence_df) > arch.max_sequence_length:
        sequence_df = (
            sequence_df.sort_values("days_ago")
            .head(arch.max_sequence_length)
            .reset_index(drop=True)
        )

    n_codes = len(sequence_df)
    seq_len = n_codes + 1  # +1 for CLS at position 0

    # Map raw code strings to vocabulary token ids.
    token_lookup = build_token_lookup(vocab)
    raw_token_ids = map_codes_to_tokens(
        code_type_series=sequence_df["code_type"],
        code_series=sequence_df["code"],
        token_lookup=token_lookup,
        rollup_rare_dx_to_3char=arch.rollup_rare_dx_to_3char,
        unk_token_id=UNK_TOKEN_ID,
    ).tolist()

    type_id_list = [TYPE_TO_ID[t] for t in sequence_df["code_type"]]
    recency_id_list = [
        _days_to_recency_id(int(d), arch.recency_bucket_day_edges)
        for d in sequence_df["days_ago"]
    ]

    # CLS and PAD use a dedicated recency slot just past the real buckets.
    special_recency_id = arch.n_recency_buckets

    token_ids = torch.full((1, seq_len), 0, dtype=torch.long)
    type_ids = torch.full((1, seq_len), SPECIAL_TYPE_ID, dtype=torch.long)
    recency_ids = torch.full((1, seq_len), special_recency_id, dtype=torch.long)
    attention_mask = torch.zeros((1, seq_len), dtype=torch.long)

    # Position 0: CLS token (type and recency defaults are already set by .full())
    token_ids[0, 0] = CLS_TOKEN_ID
    attention_mask[0, 0] = 1

    # Positions 1..n_codes: the user's codes
    token_ids[0, 1:seq_len] = torch.tensor(raw_token_ids, dtype=torch.long)
    type_ids[0, 1:seq_len] = torch.tensor(type_id_list, dtype=torch.long)
    recency_ids[0, 1:seq_len] = torch.tensor(recency_id_list, dtype=torch.long)
    attention_mask[0, 1:seq_len] = 1

    # Clamp age: anything outside [0, MAX_AGE] maps to UNKNOWN_AGE_ID.
    age_id = age if 0 <= age <= MAX_AGE else UNKNOWN_AGE_ID
    sex_id = _sex_string_to_id(sex)

    return {
        "token_ids": token_ids,
        "type_ids": type_ids,
        "recency_ids": recency_ids,
        "age_ids": torch.tensor([age_id], dtype=torch.long),
        "sex_ids": torch.tensor([sex_id], dtype=torch.long),
        "attention_mask": attention_mask,
    }


def validate_codes_against_vocab(
    codes: list[str],
    code_types: list[str],
    vocab: pd.DataFrame,
    rollup_rare_dx_to_3char: bool,
) -> list[str]:
    """Check which codes in the user's input are unknown to the vocabulary.

    A code is unknown if it maps to UNK_TOKEN_ID after applying the same lookup
    rule used in build_sequence_tensors. Returns a list of "code (type)" strings
    so the UI can show a warning. Does not raise errors.

    Args:
        codes: Raw code strings to check.
        code_types: Code type for each code ("dx", "proc", or "rx").
        vocab: The vocabulary DataFrame.
        rollup_rare_dx_to_3char: Whether to try the 3-char parent rollup for dx.

    Returns:
        List of "code (type)" labels for codes that resolve to UNK. Empty if all
        codes are recognized.

    Example:
        >>> validate_codes_against_vocab(["E11.9", "Z99.99"], ["dx", "dx"], vocab, True)
        ["Z99.99 (dx)"]
    """
    token_lookup = build_token_lookup(vocab)
    token_ids = map_codes_to_tokens(
        code_type_series=pd.Series(code_types),
        code_series=pd.Series(codes),
        token_lookup=token_lookup,
        rollup_rare_dx_to_3char=rollup_rare_dx_to_3char,
        unk_token_id=UNK_TOKEN_ID,
    )
    unknown_labels = []
    for code, code_type, token_id in zip(codes, code_types, token_ids):
        if token_id == UNK_TOKEN_ID:
            unknown_labels.append(f"{code} ({code_type})")
    return unknown_labels


def _run_model_at_position(
    model: MaskedCodeTransformer,
    tensors: dict[str, torch.Tensor],
    sequence_position: int,
) -> torch.Tensor:
    """Replace one sequence position with MASK, run the model, return logits there.

    Shared core of predict_single_masked and predict_chained. Does NOT mutate
    the input tensors — a copy of token_ids is made before masking.

    Args:
        model: The trained model. Called in eval mode with no_grad.
        tensors: The 6-tensor dict from build_sequence_tensors.
        sequence_position: Absolute position index (0-based) in the full sequence,
            including the CLS offset. To mask user code at index i, pass i + 1.

    Returns:
        1-D FloatTensor of shape (vocab_size,) — raw logits at sequence_position.
    """
    device = next(model.parameters()).device

    masked_token_ids = tensors["token_ids"].clone().to(device)
    masked_token_ids[0, sequence_position] = MASK_TOKEN_ID

    model.eval()
    with torch.no_grad():
        outputs = model(
            token_ids=masked_token_ids,
            type_ids=tensors["type_ids"].to(device),
            recency_ids=tensors["recency_ids"].to(device),
            age_ids=tensors["age_ids"].to(device),
            sex_ids=tensors["sex_ids"].to(device),
            attention_mask=tensors["attention_mask"].to(device),
        )

    # outputs["logits"] shape: (1, seq_len, vocab_size)
    return outputs["logits"][0, sequence_position, :]


def _logits_to_top_k_df(
    logits: torch.Tensor,
    vocab: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Convert raw logits to a ranked top-K DataFrame of real-code predictions.

    Applies softmax over the full vocabulary (so probabilities reflect the true
    distribution), then filters to real code tokens (code_type in dx/proc/rx)
    before selecting the top-K by probability.

    Args:
        logits: 1-D FloatTensor of shape (vocab_size,).
        vocab: The vocabulary DataFrame.
        top_k: How many top predictions to return.

    Returns:
        DataFrame with columns rank, token, token_id, code_type, member_count,
        probability, logit. Sorted descending by probability (rank 1 = highest).
    """
    probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
    logit_values = logits.detach().cpu().numpy()

    real_codes = vocab[vocab["code_type"].isin(("dx", "proc", "rx"))].copy()
    real_codes["probability"] = probs[real_codes["token_id"].values]
    real_codes["logit"] = logit_values[real_codes["token_id"].values]

    n_return = min(top_k, len(real_codes))
    top_results = real_codes.nlargest(n_return, "probability").reset_index(drop=True)
    top_results.insert(0, "rank", range(1, n_return + 1))

    return top_results[
        ["rank", "token", "token_id", "code_type", "member_count", "probability", "logit"]
    ]


def predict_single_masked(
    model: MaskedCodeTransformer,
    tensors: dict[str, torch.Tensor],
    mask_position: int,
    vocab: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """Run one forward pass with one position masked and return the top-K predictions.

    mask_position is 0-based into the user's code list (not counting CLS). The
    CLS offset is applied internally — passing 0 masks the first code entered.

    Softmax is computed over the full vocabulary. Special tokens (PAD, MASK, CLS,
    UNK) and the "special" code_type entries are excluded from the output, but
    their probability mass is included in the softmax denominator, so probabilities
    in the returned DataFrame may not sum to 1.

    Args:
        model: The trained MaskedCodeTransformer.
        tensors: The 6-tensor dict from build_sequence_tensors.
        mask_position: Which code to mask, 0-based into the user's code list.
        vocab: The vocabulary DataFrame (token, token_id, code_type, member_count).
        top_k: Number of top predictions to return.

    Returns:
        DataFrame with columns rank, token, token_id, code_type, member_count,
        probability, logit. Rank 1 is the highest-probability prediction.

    Raises:
        ValueError: If mask_position is out of range for the provided tensors.

    Example:
        >>> results = predict_single_masked(model, tensors, mask_position=0, vocab=vocab_df)
        >>> results["token"].iloc[0]
        "E11.9"
    """
    seq_len = tensors["token_ids"].shape[1]
    n_codes = seq_len - 1  # subtract CLS
    if mask_position < 0 or mask_position >= n_codes:
        raise ValueError(
            f"mask_position {mask_position} is out of range for a sequence with "
            f"{n_codes} code(s). Valid range: 0 to {n_codes - 1}."
        )

    sequence_position = mask_position + 1  # CLS is at position 0
    logits = _run_model_at_position(model, tensors, sequence_position)
    return _logits_to_top_k_df(logits, vocab, top_k)


def predict_chained(
    model: MaskedCodeTransformer,
    tensors: dict[str, torch.Tensor],
    first_mask_position: int,
    second_mask_position: int,
    vocab: pd.DataFrame,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run chained masked prediction: predict first position, inject top-1, predict second.

    Step 1 predicts first_mask_position with original context (all other codes
    visible). Step 2 substitutes the top-1 predicted token from Step 1 into the
    sequence at first_mask_position, then predicts second_mask_position in that
    updated context.

    This shows how comorbidity context shapes predictions. For example, if E11.9
    (type 2 diabetes) is the top-1 prediction in Step 1 and is injected as context,
    Step 2 may assign higher probability to I10 (hypertension) because the two
    commonly co-occur.

    Args:
        model: The trained MaskedCodeTransformer.
        tensors: The 6-tensor dict from build_sequence_tensors.
        first_mask_position: 0-based index of the first code to predict.
        second_mask_position: 0-based index of the second code to predict. Must
            differ from first_mask_position.
        vocab: The vocabulary DataFrame.
        top_k: Number of top predictions per step.

    Returns:
        Tuple (step1_predictions, step2_predictions). Each DataFrame has columns
        rank, token, token_id, code_type, member_count, probability, logit.

    Raises:
        ValueError: If first_mask_position == second_mask_position.
        ValueError: If either position is out of range.

    Example:
        >>> step1, step2 = predict_chained(model, tensors, 0, 1, vocab)
        >>> step1["token"].iloc[0]
        "E11.9"
    """
    seq_len = tensors["token_ids"].shape[1]
    n_codes = seq_len - 1
    for pos, label in [
        (first_mask_position, "first_mask_position"),
        (second_mask_position, "second_mask_position"),
    ]:
        if pos < 0 or pos >= n_codes:
            raise ValueError(
                f"{label} {pos} is out of range for a sequence with {n_codes} "
                f"code(s). Valid range: 0 to {n_codes - 1}."
            )
    if first_mask_position == second_mask_position:
        raise ValueError(
            f"first_mask_position and second_mask_position must differ. "
            f"Got {first_mask_position} for both."
        )

    # Step 1: predict the first position with all other codes visible.
    first_seq_pos = first_mask_position + 1
    step1_logits = _run_model_at_position(model, tensors, first_seq_pos)
    step1_df = _logits_to_top_k_df(step1_logits, vocab, top_k)

    # Identify the top-1 real-code token id from Step 1.
    top1_token = step1_df["token"].iloc[0]
    top1_code_type = step1_df["code_type"].iloc[0]
    top1_match = vocab[
        (vocab["token"] == top1_token) & (vocab["code_type"] == top1_code_type)
    ]
    top1_token_id = int(top1_match["token_id"].iloc[0])

    # Inject the top-1 prediction and predict the second position.
    modified_tensors = {key: val.clone() for key, val in tensors.items()}
    modified_tensors["token_ids"][0, first_seq_pos] = top1_token_id

    second_seq_pos = second_mask_position + 1
    step2_logits = _run_model_at_position(model, modified_tensors, second_seq_pos)
    step2_df = _logits_to_top_k_df(step2_logits, vocab, top_k)

    return step1_df, step2_df
