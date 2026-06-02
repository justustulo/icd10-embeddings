"""Central configuration for the ICD-10 embedding pipeline.

Everything that varies between runs lives here: where the claims data is, which
line of business (LOB) to train on, the observation/prediction windows, the
vocabulary frequency floors, and the model/training hyperparameters.

We build ONE model per LOB, so a typical workflow is to make one Config per LOB
(Commercial / MA / Medicaid) and run the pipeline once for each.

The column names below are a mapping layer: edit `ColumnMap` to match the
column names in your own claims extract. The rest of the code only refers to
these logical names, never to hard-coded column strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class ColumnMap:
    """Maps the logical fields the pipeline needs to the column names in your data.

    Change the right-hand-side default strings to match your claims extract.
    All of these refer to columns in a single long-format claims table where one
    row is one claim line.

    Attributes:
        member_id: Unique identifier for a covered individual.
        client_id: Employer group / plan sponsor the member belongs to.
        line_of_business: LOB label used to filter the data (e.g. "Commercial").
        incurred_date: Date of service (incurred date), NOT paid/processed date.
        code: The raw code string (e.g. "E11.9", "99213", an NDC or class code).
        code_type: Which kind of code this row is; must be one of {"dx", "proc", "rx"}.
        member_birth_date: Birth date, used to compute age at the observation anchor.
        member_sex: Member sex (kept as-is; encoded to an integer in the dataset).
    """

    member_id: str = "member_id"
    client_id: str = "client_id"
    line_of_business: str = "line_of_business"
    incurred_date: str = "incurred_date"
    code: str = "code"
    code_type: str = "code_type"
    member_birth_date: str = "member_birth_date"
    member_sex: str = "member_sex"


# The three code types the model understands. Pharmacy is expected to already be
# rolled up to a therapeutic class (see build_vocab.py) rather than raw NDC.
CODE_TYPES: tuple[str, ...] = ("dx", "proc", "rx")

# Each token carries a "type" embedding so the shared model knows whether it is a
# diagnosis, procedure, pharmacy, or special (CLS/PAD/MASK) token.
TYPE_TO_ID: dict[str, int] = {"dx": 0, "proc": 1, "rx": 2}
SPECIAL_TYPE_ID: int = 3
N_TYPE_IDS: int = 4

# Sex is encoded to a small integer. Unknown/other maps to 0 so missing data is
# always representable (claims sex fields are not always clean).
SEX_TO_ID: dict[str, int] = {"M": 1, "F": 2}
UNKNOWN_SEX_ID: int = 0
N_SEX_IDS: int = 3

# Age (in whole years at the observation anchor) is clamped to [0, MAX_AGE]; a
# dedicated id represents missing/implausible ages so they never crash the model.
MAX_AGE: int = 100
UNKNOWN_AGE_ID: int = 101
N_AGE_IDS: int = 102

# Special tokens that occupy the first id slots in the vocabulary. PAD fills
# short sequences, MASK is the masked-code training target, CLS is the pooled
# member-representation slot.
SPECIAL_TOKENS: tuple[str, ...] = ("<PAD>", "<MASK>", "<CLS>", "<UNK>")
PAD_TOKEN_ID: int = 0
MASK_TOKEN_ID: int = 1
CLS_TOKEN_ID: int = 2
UNK_TOKEN_ID: int = 3


@dataclass
class Config:
    """All settings for one end-to-end run (one LOB).

    Attributes:
        claims_path: Path to the long-format claims table (parquet or csv).
        output_dir: Directory where vocab, sequences, checkpoints and embeddings
            are written. Use a separate directory per LOB so runs don't collide.
        line_of_business: The LOB value to keep; rows with any other LOB are dropped.
        observation_start: First incurred date (inclusive) used to build a member's
            input sequence.
        observation_end: Last incurred date (inclusive) used to build the input
            sequence. Age is computed as of this date, and recency buckets are
            measured backwards from here.
        prediction_start: First incurred date of the downstream label window. Must
            be strictly after observation_end to avoid leakage. Only used by the
            Phase 3 downstream code; leave as None for embedding-only runs.
        prediction_end: Last incurred date of the label window.
        min_count_dx: Minimum number of distinct members a diagnosis must appear in
            to get its own token. Rarer diagnoses are rolled up to their 3-char
            ICD-10 parent (if enabled) or mapped to <UNK>.
        min_count_proc: Same idea for procedure codes (no parent rollup).
        min_count_rx: Same idea for pharmacy (therapeutic-class) codes.
        rollup_rare_dx_to_3char: If True, a rare full diagnosis (e.g. "E11.621")
            is replaced by its 3-character category ("E11") before counting; if the
            category still clears min_count_dx it gets a token, else <UNK>.
        recency_bucket_day_edges: Ascending day thresholds, measured backwards from
            observation_end, that define recency buckets. Example [30, 90, 180, 365]
            gives buckets: 0-30d, 31-90d, 91-180d, 181-365d, and >365d.
        max_sequence_length: Maximum number of code tokens per member (after the CLS
            token). Longer histories are truncated to the most recent tokens.
        embedding_dim: Size of each code/member vector.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads (must divide embedding_dim).
        feedforward_dim: Width of the transformer feed-forward sublayer.
        dropout: Dropout probability used throughout the encoder.
        mask_rate: Fraction of code tokens masked per sequence during pretraining.
        batch_size: Members per training batch.
        learning_rate: Adam learning rate.
        n_epochs: Number of passes over the training members.
        validation_fraction: Fraction of members held out to measure masked-code
            accuracy.
        early_stopping_patience: Stop training if validation loss does not improve
            for this many consecutive epochs. Set to n_epochs to disable early
            stopping. Requires validation_fraction > 0.
        device: "cuda" or "cpu". Set to "cuda" when a GPU is available.
        random_seed: Seed for reproducible vocab sampling, masking and splits.
    """

    # --- Data location ---
    claims_path: Path
    output_dir: Path
    line_of_business: str

    # --- Windows ---
    observation_start: date
    observation_end: date
    prediction_start: date | None = None
    prediction_end: date | None = None

    # --- Column mapping (edit ColumnMap defaults to match your extract) ---
    columns: ColumnMap = field(default_factory=ColumnMap)

    # --- Vocabulary ---
    min_count_dx: int = 50
    min_count_proc: int = 50
    min_count_rx: int = 50
    rollup_rare_dx_to_3char: bool = True

    # --- Sequence construction ---
    recency_bucket_day_edges: tuple[int, ...] = (30, 90, 180, 365, 730)
    max_sequence_length: int = 256

    # --- Model ---
    embedding_dim: int = 128
    n_layers: int = 4
    n_heads: int = 8
    feedforward_dim: int = 512
    dropout: float = 0.1

    # --- Training ---
    mask_rate: float = 0.15
    batch_size: int = 256
    learning_rate: float = 1e-3
    n_epochs: int = 10
    validation_fraction: float = 0.1
    early_stopping_patience: int = 5
    device: str = "cpu"
    random_seed: int = 12345

    # --- Derived file paths (filled in __post_init__) ---
    vocab_path: Path = field(init=False)
    sequences_path: Path = field(init=False)
    checkpoint_path: Path = field(init=False)
    code_vectors_path: Path = field(init=False)
    member_vectors_path: Path = field(init=False)

    def __post_init__(self) -> None:
        """Validate settings and compute the derived output file paths."""
        self.claims_path = Path(self.claims_path)
        self.output_dir = Path(self.output_dir)

        # Fail fast on window mistakes -- these are the most common source of
        # silent target leakage in claims modeling.
        if self.observation_start > self.observation_end:
            raise ValueError(
                f"observation_start ({self.observation_start}) must be on or before "
                f"observation_end ({self.observation_end})"
            )
        if self.prediction_start is not None:
            if self.prediction_start <= self.observation_end:
                raise ValueError(
                    f"prediction_start ({self.prediction_start}) must be strictly after "
                    f"observation_end ({self.observation_end}) to avoid label leakage"
                )
            if self.prediction_end is None or self.prediction_end < self.prediction_start:
                raise ValueError(
                    "prediction_end must be set and on or after prediction_start when "
                    "prediction_start is provided"
                )

        if self.embedding_dim % self.n_heads != 0:
            raise ValueError(
                f"embedding_dim ({self.embedding_dim}) must be divisible by "
                f"n_heads ({self.n_heads})"
            )
        if not 0.0 < self.mask_rate < 1.0:
            raise ValueError(f"mask_rate must be between 0 and 1, got {self.mask_rate}")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError(
                f"validation_fraction must be in [0, 1), got {self.validation_fraction}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_path = self.output_dir / "vocab.parquet"
        self.sequences_path = self.output_dir / "member_sequences.parquet"
        self.checkpoint_path = self.output_dir / "model.pt"
        self.code_vectors_path = self.output_dir / "code_vectors.parquet"
        self.member_vectors_path = self.output_dir / "member_vectors.parquet"

    @property
    def n_recency_buckets(self) -> int:
        """Number of recency buckets, including the final '>last edge' bucket."""
        return len(self.recency_bucket_day_edges) + 1
