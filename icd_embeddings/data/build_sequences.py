"""Turn validated claims into one ordered, typed code sequence per member.

Each member becomes a single row holding parallel lists:
    token_ids   - the code tokens, most-recent-first, truncated to max length
    type_ids    - dx/proc/rx tag aligned to token_ids
    recency_ids - which recency bucket each code fell in
plus member-level age, sex, and client id. The CLS token is NOT stored here; the
PyTorch dataset prepends it at training time.

We sort most-recent-first and truncate to the max length so that, for members
with very long histories, we keep their most recent (most predictive) codes.
"""

from __future__ import annotations

import pandas as pd

from icd_embeddings.config import (
    MAX_AGE,
    SEX_TO_ID,
    TYPE_TO_ID,
    UNKNOWN_AGE_ID,
    UNKNOWN_SEX_ID,
    UNK_TOKEN_ID,
    Config,
)
from icd_embeddings.data.build_vocab import build_token_lookup, map_codes_to_tokens
from icd_embeddings.data.load import load_claims


def _recency_bucket_id(days_ago: int, bucket_day_edges: tuple[int, ...]) -> int:
    """Map an age-in-days to a recency bucket index.

    Buckets are defined by ascending day edges measured backwards from the
    observation anchor. For edges (30, 90, 180): 0-30 days -> 0, 31-90 -> 1,
    91-180 -> 2, and anything older -> 3 (the final open-ended bucket).

    Args:
        days_ago: Whole days between the code's incurred date and observation_end.
        bucket_day_edges: Ascending day thresholds from the Config.

    Returns:
        Integer bucket id in [0, len(bucket_day_edges)].
    """
    for bucket_index, edge in enumerate(bucket_day_edges):
        if days_ago <= edge:
            return bucket_index
    return len(bucket_day_edges)


def _age_id_at(birth_date, observation_end) -> int:
    """Compute whole-years age at the observation anchor, with an unknown fallback.

    Args:
        birth_date: The member's birth date (may be NaT/None).
        observation_end: The anchor date (Config.observation_end as a Timestamp).

    Returns:
        Age clamped to [0, MAX_AGE], or UNKNOWN_AGE_ID when birth date is missing
        or the computed age is implausible (negative or absurdly large).
    """
    if pd.isna(birth_date):
        return UNKNOWN_AGE_ID
    age_years = int((observation_end - birth_date).days // 365)
    if age_years < 0 or age_years > MAX_AGE:
        # ASSUMPTION: ages outside [0, 100] are data errors, not real members.
        return UNKNOWN_AGE_ID
    return age_years


def _sex_id(raw_sex) -> int:
    """Map a raw sex value to its integer id, defaulting unknown/other to 0."""
    if pd.isna(raw_sex):
        return UNKNOWN_SEX_ID
    return SEX_TO_ID.get(str(raw_sex).strip().upper(), UNKNOWN_SEX_ID)


def _member_attributes(claims: pd.DataFrame, observation_end: pd.Timestamp) -> pd.DataFrame:
    """Collapse claim lines to one attribute row per member (age, sex, client).

    Client can change over time, so we take the client id from the member's most
    recent claim in the window. Age and sex are taken from the first non-null value.

    Args:
        claims: Validated claims with a recency-sorted helper not required.
        observation_end: Anchor date for the age calculation.

    Returns:
        DataFrame indexed by member_id with columns [client_id, age_id, sex_id].
    """
    # Most recent claim per member gives the client id we attribute them to.
    latest_rows = claims.sort_values("incurred_date").groupby("member_id").tail(1)
    latest_client = latest_rows.set_index("member_id")["client_id"]

    # First non-null birth date and sex per member.
    birth_by_member = (
        claims.dropna(subset=["member_birth_date"])
        .groupby("member_id")["member_birth_date"]
        .first()
    )
    sex_by_member = claims.groupby("member_id")["member_sex"].first()

    members = pd.DataFrame(index=latest_client.index)
    members["client_id"] = latest_client
    members["age_id"] = [
        _age_id_at(birth_by_member.get(member_id, pd.NaT), observation_end)
        for member_id in members.index
    ]
    members["sex_id"] = [
        _sex_id(sex_by_member.get(member_id)) for member_id in members.index
    ]
    return members


def build_sequences(config: Config, vocab: pd.DataFrame) -> pd.DataFrame:
    """Build and persist one sequence row per member for the configured LOB.

    Writes a parquet file to `config.sequences_path` with columns:
        member_id, client_id, age_id, sex_id,
        token_ids (list[int]), type_ids (list[int]), recency_ids (list[int]).

    Args:
        config: The run configuration.
        vocab: The vocabulary DataFrame from build_vocab.

    Returns:
        The member-sequence DataFrame that was written to disk.
    """
    claims = load_claims(config)
    observation_end = pd.Timestamp(config.observation_end)

    token_lookup = build_token_lookup(vocab)
    claims = claims.copy()
    claims["token_id"] = map_codes_to_tokens(
        code_type_series=claims["code_type"],
        code_series=claims["code"],
        token_lookup=token_lookup,
        rollup_rare_dx_to_3char=config.rollup_rare_dx_to_3char,
        unk_token_id=UNK_TOKEN_ID,
    )
    claims["type_id"] = claims["code_type"].map(TYPE_TO_ID).astype("int64")

    days_ago = (observation_end - claims["incurred_date"]).dt.days
    claims["recency_id"] = [
        _recency_bucket_id(int(d), config.recency_bucket_day_edges) for d in days_ago
    ]

    # Most-recent-first ordering so truncation keeps the latest codes.
    claims = claims.sort_values(["member_id", "incurred_date"], ascending=[True, False])

    member_attributes = _member_attributes(claims, observation_end)

    sequence_rows = []
    for member_id, member_claims in claims.groupby("member_id", sort=False):
        truncated = member_claims.head(config.max_sequence_length)
        attributes = member_attributes.loc[member_id]
        sequence_rows.append(
            {
                "member_id": member_id,
                "client_id": attributes["client_id"],
                "age_id": int(attributes["age_id"]),
                "sex_id": int(attributes["sex_id"]),
                "token_ids": truncated["token_id"].tolist(),
                "type_ids": truncated["type_id"].tolist(),
                "recency_ids": truncated["recency_id"].tolist(),
            }
        )

    sequences = pd.DataFrame(sequence_rows)
    sequences.to_parquet(config.sequences_path, index=False)
    print(
        f"[build_sequences] {len(sequences)} member sequences "
        f"written to {config.sequences_path}"
    )
    return sequences
