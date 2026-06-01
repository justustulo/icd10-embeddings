"""Load and validate the raw claims extract.

Both vocabulary building and sequence building start from the same filtered
claims table, so that logic lives here in one place. The loader renames your
data's columns (via Config.ColumnMap) to a fixed set of internal names so the
rest of the pipeline never refers to a raw column string.
"""

from __future__ import annotations

import pandas as pd

from icd_embeddings.config import CODE_TYPES, ColumnMap, Config


# Internal (canonical) column names used everywhere downstream of this loader.
INTERNAL_COLUMNS: tuple[str, ...] = (
    "member_id",
    "client_id",
    "line_of_business",
    "incurred_date",
    "code",
    "code_type",
    "member_birth_date",
    "member_sex",
)


def _read_claims_file(claims_path) -> pd.DataFrame:
    """Read the claims file as a DataFrame, picking the reader from the suffix.

    Args:
        claims_path: Path to a .parquet or .csv claims extract.

    Returns:
        The raw claims table, columns and dtypes as stored on disk.
    """
    suffix = claims_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(claims_path)
    if suffix == ".csv":
        return pd.read_csv(claims_path)
    raise ValueError(
        f"Unsupported claims file type '{suffix}' for {claims_path}; expected .parquet or .csv"
    )


def _rename_to_internal(claims_df: pd.DataFrame, columns: ColumnMap) -> pd.DataFrame:
    """Rename the user's columns to the fixed internal names, validating presence."""
    rename_map = {
        columns.member_id: "member_id",
        columns.client_id: "client_id",
        columns.line_of_business: "line_of_business",
        columns.incurred_date: "incurred_date",
        columns.code: "code",
        columns.code_type: "code_type",
        columns.member_birth_date: "member_birth_date",
        columns.member_sex: "member_sex",
    }

    missing = [source for source in rename_map if source not in claims_df.columns]
    if missing:
        raise ValueError(
            f"Claims data is missing expected column(s) {missing}. "
            f"Check the ColumnMap in your Config against the actual columns: "
            f"{list(claims_df.columns)}"
        )

    return claims_df.rename(columns=rename_map)[list(INTERNAL_COLUMNS)]


def load_claims(config: Config) -> pd.DataFrame:
    """Load claims for one LOB, restricted to the observation window, and validate.

    The returned frame uses internal column names (see INTERNAL_COLUMNS) and is
    guaranteed to have: valid datetime incurred_date with no nulls, code_type in
    {"dx","proc","rx"}, and only rows for the configured line of business that
    fall inside [observation_start, observation_end].

    Args:
        config: The run configuration (paths, LOB, window, column mapping).

    Returns:
        A filtered, validated claims DataFrame (one row per claim line).
    """
    if not config.claims_path.exists():
        raise FileNotFoundError(f"Claims file not found at {config.claims_path}")

    raw_claims = _read_claims_file(config.claims_path)
    claims = _rename_to_internal(raw_claims, config.columns)

    # Incurred date drives both windowing and recency. We require it to be parseable
    # and non-null -- a claim line with no service date cannot be placed in time.
    claims["incurred_date"] = pd.to_datetime(claims["incurred_date"], errors="coerce")

    # Birth date is optional but must be datetime for age calculation. Unparseable
    # values become NaT and fall through to UNKNOWN_AGE_ID in build_sequences.
    claims["member_birth_date"] = pd.to_datetime(claims["member_birth_date"], errors="coerce")
    n_bad_dates = int(claims["incurred_date"].isna().sum())
    if n_bad_dates > 0:
        # ASSUMPTION: rows with an unparseable/missing incurred date are unusable
        # and are dropped rather than guessed. Surface the count so it isn't silent.
        print(f"[load_claims] dropping {n_bad_dates} rows with missing/invalid incurred_date")
        claims = claims[claims["incurred_date"].notna()].copy()

    # Restrict to the requested line of business.
    claims = claims[claims["line_of_business"] == config.line_of_business].copy()
    if claims.empty:
        raise ValueError(
            f"No claims remain after filtering to line_of_business == "
            f"'{config.line_of_business}'. Check the LOB value and the data."
        )

    # Restrict to the observation window (inclusive on both ends).
    window_start = pd.Timestamp(config.observation_start)
    window_end = pd.Timestamp(config.observation_end)
    in_window = (claims["incurred_date"] >= window_start) & (
        claims["incurred_date"] <= window_end
    )
    claims = claims[in_window].copy()
    if claims.empty:
        raise ValueError(
            f"No claims fall within the observation window "
            f"[{config.observation_start}, {config.observation_end}]."
        )

    # Code type must be one of the three the model understands.
    claims["code_type"] = claims["code_type"].astype(str).str.lower()
    bad_types = sorted(set(claims["code_type"].unique()) - set(CODE_TYPES))
    if bad_types:
        raise ValueError(
            f"Found code_type value(s) {bad_types} not in {CODE_TYPES}. "
            f"Map every claim line to one of these three types before loading."
        )

    # Codes are compared as strings throughout (e.g. for the 3-char dx rollup).
    claims["code"] = claims["code"].astype(str).str.strip()

    return claims.reset_index(drop=True)
