"""Generate a small synthetic claims extract so the pipeline runs end-to-end.

This is a TEST FIXTURE, not real data. It builds members who each have one or
two "conditions"; every condition has associated diagnosis, procedure, and
pharmacy codes that therefore co-occur. That co-occurrence is what the embedding
model should learn -- so after training, codes from the same condition should be
near each other (a quick sanity check).

Run: python examples/make_synthetic_claims.py
Writes: examples/synthetic_claims.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Each condition ties together codes of all three types so they co-occur in the
# same members. (token, code_type) pairs; pharmacy codes are therapeutic classes.
CONDITION_CODES: dict[str, list[tuple[str, str]]] = {
    "diabetes": [
        ("E11.9", "dx"),
        ("E11.65", "dx"),
        ("E11.21", "dx"),
        ("83036", "proc"),  # hemoglobin A1c
        ("A10BA", "rx"),  # metformin (ATC class)
        ("A10AB", "rx"),  # fast-acting insulin (ATC class)
    ],
    "heart_failure": [
        ("I50.9", "dx"),
        ("I25.10", "dx"),
        ("I10", "dx"),
        ("93000", "proc"),  # electrocardiogram
        ("C03CA", "rx"),  # furosemide
        ("C09AA", "rx"),  # ACE inhibitor
    ],
    "ckd": [
        ("N18.3", "dx"),
        ("N18.4", "dx"),
        ("90999", "proc"),  # dialysis
        ("B03XA", "rx"),  # erythropoietin
    ],
    "copd": [
        ("J44.9", "dx"),
        ("J44.1", "dx"),
        ("94010", "proc"),  # spirometry
        ("R03AC", "rx"),  # short-acting beta agonist
        ("R03BB", "rx"),  # inhaled anticholinergic
    ],
}

# Low-signal codes any member can get, so the model sees noise too.
BACKGROUND_CODES: list[tuple[str, str]] = [
    ("Z00.00", "dx"),
    ("J06.9", "dx"),
    ("99213", "proc"),  # office visit
    ("N02BE", "rx"),  # acetaminophen
]

CLIENTS: list[str] = ["GROUP_A", "GROUP_B", "GROUP_C"]


def make_synthetic_claims(
    n_members: int = 2000,
    line_of_business: str = "Commercial",
    random_seed: int = 7,
) -> pd.DataFrame:
    """Build a synthetic long-format claims table.

    Args:
        n_members: Number of members to generate.
        line_of_business: LOB label written on every row.
        random_seed: Seed for reproducibility.

    Returns:
        A claims DataFrame with one row per claim line, matching the default
        ColumnMap (member_id, client_id, line_of_business, incurred_date, code,
        code_type, member_birth_date, member_sex).
    """
    rng = np.random.default_rng(random_seed)
    condition_names = list(CONDITION_CODES.keys())
    window_start = pd.Timestamp("2023-01-01")

    claim_rows = []
    for member_index in range(n_members):
        member_id = f"M{member_index:06d}"
        client_id = CLIENTS[member_index % len(CLIENTS)]
        sex = rng.choice(["M", "F"])
        age_years = int(rng.integers(20, 85))
        birth_date = window_start - pd.Timedelta(days=age_years * 365)

        # Each member gets one or two conditions (or zero -> background only).
        n_conditions = int(rng.integers(0, 3))
        member_conditions = list(
            rng.choice(condition_names, size=n_conditions, replace=False)
        ) if n_conditions > 0 else []

        # Collect the code pool this member draws from.
        member_code_pool: list[tuple[str, str]] = list(BACKGROUND_CODES)
        for condition in member_conditions:
            member_code_pool.extend(CONDITION_CODES[condition])

        # Emit several claim lines spread across the year.
        n_lines = int(rng.integers(4, 15))
        for _ in range(n_lines):
            code, code_type = member_code_pool[int(rng.integers(0, len(member_code_pool)))]
            day_offset = int(rng.integers(0, 365))
            incurred_date = window_start + pd.Timedelta(days=day_offset)
            claim_rows.append(
                {
                    "member_id": member_id,
                    "client_id": client_id,
                    "line_of_business": line_of_business,
                    "incurred_date": incurred_date,
                    "code": code,
                    "code_type": code_type,
                    "member_birth_date": birth_date,
                    "member_sex": sex,
                }
            )

    return pd.DataFrame(claim_rows)


def main() -> None:
    """Generate and write the synthetic claims parquet next to this script."""
    output_path = Path(__file__).resolve().parent / "synthetic_claims.parquet"
    claims = make_synthetic_claims()
    claims.to_parquet(output_path, index=False)
    print(f"wrote {len(claims)} synthetic claim lines to {output_path}")


if __name__ == "__main__":
    main()
