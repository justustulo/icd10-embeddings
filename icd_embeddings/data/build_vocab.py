"""Build the code vocabulary for one LOB.

A "token" is one entry in the model's vocabulary. We keep a token for any code
that appears in enough distinct members (the per-type frequency floor). Rare
diagnoses are optionally rolled up to their 3-character ICD-10 parent so we keep
some signal instead of throwing them away; anything still too rare becomes <UNK>.

The same raw-code -> token rule is needed when building member sequences, so it
lives here as `map_codes_to_tokens` and is imported by build_sequences.py. That
guarantees the vocabulary and the sequences always agree.
"""

from __future__ import annotations

import pandas as pd

from icd_embeddings.config import (
    CODE_TYPES,
    SPECIAL_TOKENS,
    Config,
)
from icd_embeddings.data.load import load_claims


def _three_char_parent(dx_code: str) -> str:
    """Return the 3-character ICD-10 category for a diagnosis code.

    Example: "E11.621" -> "E11". ICD-10-CM codes are always at least 3 characters,
    and the first three characters are the category (e.g. E11 = type 2 diabetes).

    Args:
        dx_code: A diagnosis code string, with or without the decimal point.

    Returns:
        The first three characters, uppercased and with any leading/trailing
        whitespace removed.
    """
    return dx_code.strip().upper()[:3]


def _member_counts_by_code(claims: pd.DataFrame) -> pd.DataFrame:
    """Count the number of distinct members each (code_type, code) appears in.

    We count distinct members rather than rows so a single member with many
    repeats of the same code does not inflate the code's apparent prevalence.

    Args:
        claims: Validated claims with columns member_id, code_type, code.

    Returns:
        DataFrame with columns [code_type, code, member_count].
    """
    distinct = claims[["member_id", "code_type", "code"]].drop_duplicates()
    counts = (
        distinct.groupby(["code_type", "code"])["member_id"]
        .count()
        .reset_index(name="member_count")
    )
    return counts


def _kept_tokens_for_type(
    counts: pd.DataFrame,
    claims: pd.DataFrame,
    code_type: str,
    min_count: int,
    rollup_rare_dx_to_3char: bool,
) -> pd.DataFrame:
    """Decide which tokens to keep for one code type, applying the dx rollup.

    Args:
        counts: Output of `_member_counts_by_code`.
        claims: Validated claims (needed to recount rolled-up dx parents by member).
        code_type: One of "dx", "proc", "rx".
        min_count: Frequency floor (distinct members) for this type.
        rollup_rare_dx_to_3char: Whether to roll rare diagnoses up to 3-char parents.

    Returns:
        DataFrame with columns [code_type, token, member_count] for the kept tokens.
    """
    type_counts = counts[counts["code_type"] == code_type]

    # Full codes that already clear the floor are kept as-is.
    kept_full = type_counts[type_counts["member_count"] >= min_count].copy()
    kept_full = kept_full.rename(columns={"code": "token"})[
        ["code_type", "token", "member_count"]
    ]

    # Rollup only applies to diagnoses. Procedures and pharmacy have no hierarchy
    # we attempt to exploit here, so their rare codes simply fall out (-> <UNK>).
    if code_type != "dx" or not rollup_rare_dx_to_3char:
        return kept_full

    kept_full_codes = set(kept_full["token"])
    rare_dx_codes = set(
        type_counts[type_counts["member_count"] < min_count]["code"]
    )

    # Take the actual dx claim lines whose full code did NOT clear the floor,
    # map each to its 3-char parent, and recount distinct members per parent.
    rare_dx_claims = claims[
        (claims["code_type"] == "dx") & (claims["code"].isin(rare_dx_codes))
    ][["member_id", "code"]].copy()
    rare_dx_claims["parent"] = rare_dx_claims["code"].map(_three_char_parent)

    parent_distinct = rare_dx_claims[["member_id", "parent"]].drop_duplicates()
    parent_counts = (
        parent_distinct.groupby("parent")["member_id"].count().reset_index(
            name="member_count"
        )
    )

    # Keep a parent token only if the rolled-up rares clear the floor, and only if
    # that 3-char string is not already a kept full code (avoid duplicate tokens).
    kept_parents = parent_counts[parent_counts["member_count"] >= min_count].copy()
    kept_parents = kept_parents[~kept_parents["parent"].isin(kept_full_codes)]
    kept_parents = kept_parents.rename(columns={"parent": "token"})
    kept_parents["code_type"] = "dx"
    kept_parents = kept_parents[["code_type", "token", "member_count"]]

    return pd.concat([kept_full, kept_parents], ignore_index=True)


def build_vocab(config: Config) -> pd.DataFrame:
    """Build and persist the token vocabulary for the configured LOB.

    Writes a parquet file to `config.vocab_path` with columns:
        token (str), token_id (int), code_type (str), member_count (int).
    The special tokens (PAD/MASK/CLS/UNK) occupy the first id slots and have
    code_type "special".

    Args:
        config: The run configuration.

    Returns:
        The vocabulary DataFrame that was written to disk.
    """
    claims = load_claims(config)
    counts = _member_counts_by_code(claims)

    min_count_by_type = {
        "dx": config.min_count_dx,
        "proc": config.min_count_proc,
        "rx": config.min_count_rx,
    }

    kept_per_type = []
    for code_type in CODE_TYPES:
        kept = _kept_tokens_for_type(
            counts=counts,
            claims=claims,
            code_type=code_type,
            min_count=min_count_by_type[code_type],
            rollup_rare_dx_to_3char=config.rollup_rare_dx_to_3char,
        )
        kept_per_type.append(kept)

    kept_tokens = pd.concat(kept_per_type, ignore_index=True)
    # Stable, readable ordering: by type then by descending prevalence.
    kept_tokens = kept_tokens.sort_values(
        ["code_type", "member_count"], ascending=[True, False]
    ).reset_index(drop=True)

    # Special tokens come first so PAD_TOKEN_ID/MASK_TOKEN_ID/etc. line up with config.
    special_rows = pd.DataFrame(
        {
            "token": list(SPECIAL_TOKENS),
            "code_type": ["special"] * len(SPECIAL_TOKENS),
            "member_count": [0] * len(SPECIAL_TOKENS),
        }
    )

    vocab = pd.concat([special_rows, kept_tokens], ignore_index=True)
    vocab["token_id"] = vocab.index.astype(int)
    vocab = vocab[["token", "token_id", "code_type", "member_count"]]

    vocab.to_parquet(config.vocab_path, index=False)
    print(
        f"[build_vocab] {len(vocab)} tokens "
        f"({len(special_rows)} special + {len(kept_tokens)} codes) "
        f"written to {config.vocab_path}"
    )
    return vocab


def build_token_lookup(vocab: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Build a {code_type: {token_string: token_id}} lookup from the vocab table.

    Args:
        vocab: The vocabulary DataFrame from `build_vocab`.

    Returns:
        Nested dict keyed first by code_type ("dx"/"proc"/"rx") then by token
        string, giving the integer token id. Special tokens are excluded.
    """
    lookup: dict[str, dict[str, int]] = {code_type: {} for code_type in CODE_TYPES}
    code_rows = vocab[vocab["code_type"].isin(CODE_TYPES)]
    for token, token_id, code_type in zip(
        code_rows["token"], code_rows["token_id"], code_rows["code_type"]
    ):
        lookup[code_type][token] = int(token_id)
    return lookup


def map_codes_to_tokens(
    code_type_series: pd.Series,
    code_series: pd.Series,
    token_lookup: dict[str, dict[str, int]],
    rollup_rare_dx_to_3char: bool,
    unk_token_id: int,
) -> pd.Series:
    """Map raw (code_type, code) pairs to token ids using the vocabulary rule.

    The rule, applied per row:
      * if the full code is a known token for its type -> that token id;
      * else if it is a diagnosis and its 3-char parent is a known token
        (and rollup is enabled) -> the parent token id;
      * else -> the <UNK> token id.

    Args:
        code_type_series: Series of "dx"/"proc"/"rx".
        code_series: Series of raw code strings (same index as code_type_series).
        token_lookup: Output of `build_token_lookup`.
        rollup_rare_dx_to_3char: Must match the value used when building the vocab.
        unk_token_id: The id to use when no token matches.

    Returns:
        Integer Series of token ids, aligned to the input index.
    """

    def map_one(code_type: str, code: str) -> int:
        type_tokens = token_lookup.get(code_type, {})
        if code in type_tokens:
            return type_tokens[code]
        if code_type == "dx" and rollup_rare_dx_to_3char:
            parent = _three_char_parent(code)
            if parent in type_tokens:
                return type_tokens[parent]
        return unk_token_id

    return pd.Series(
        [map_one(t, c) for t, c in zip(code_type_series, code_series)],
        index=code_type_series.index,
        dtype="int64",
    )
