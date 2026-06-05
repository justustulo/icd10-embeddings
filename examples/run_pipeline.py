"""End-to-end smoke test of the whole pipeline on synthetic data.

Runs every phase on the small synthetic claims file with a tiny, fast model, then
prints sanity checks: nearest neighbors of a diabetes code (should be other
diabetes-related codes), the client feature table shape, a few HCC suspects, and
the top fraud anomaly scores.

Run (from the project root):
    python examples/make_synthetic_claims.py
    python examples/run_pipeline.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

# Make the icd_embeddings package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icd_embeddings.config import Config  # noqa: E402
from icd_embeddings.data.build_sequences import build_sequences  # noqa: E402
from icd_embeddings.data.build_vocab import build_vocab  # noqa: E402
from icd_embeddings.downstream import fraud, premium, suspecting  # noqa: E402
from icd_embeddings.embeddings.extract import (  # noqa: E402
    extract_code_vectors,
    extract_member_vectors,
    nearest_neighbors,
)
from icd_embeddings.model.dataset import load_sequences  # noqa: E402
from icd_embeddings.model.pretrain import pretrain  # noqa: E402

# A toy code -> HCC mapping (illustrative; real runs plug in the LOB's framework).
TOY_DX_TO_HCC: dict[str, str] = {
    "E11.9": "HCC_DIABETES",
    "E11.65": "HCC_DIABETES",
    "E11.21": "HCC_DIABETES",
    "I50.9": "HCC_HEART_FAILURE",
    "I25.10": "HCC_HEART_FAILURE",
    "N18.3": "HCC_CKD",
    "N18.4": "HCC_CKD",
    "J44.9": "HCC_COPD",
    "J44.1": "HCC_COPD",
}


def build_smoke_config() -> Config:
    """Build a small, fast Config pointed at the synthetic data."""
    here = Path(__file__).resolve().parent
    return Config(
        claims_path=here / "synthetic_claims.parquet",
        output_dir=here / "output",
        line_of_business="Commercial",
        observation_start=date(2023, 1, 1),
        observation_end=date(2023, 12, 31),
        # Low floors because the synthetic data is small.
        min_count_dx=5,
        min_count_proc=5,
        min_count_rx=5,
        # Tiny model so the smoke test runs in seconds on CPU.
        embedding_dim=32,
        n_layers=2,
        n_heads=4,
        feedforward_dim=64,
        max_sequence_length=64,
        batch_size=64,
        n_epochs=5,
        device="cpu",
    )


def save_run_info(config) -> None:
    """Write a JSON record of this run's start time and full config to output_dir.

    Args:
        config: The run configuration to record.
    """
    run_info = {
        "run_started_at": datetime.now().isoformat(timespec="seconds"),
        "config": config.to_dict(),
    }
    with open(config.run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    print(f"[run_info] Written to {config.run_info_path}")


def main() -> None:
    config = build_smoke_config()
    save_run_info(config)

    print("\n=== Phase 0: vocab + sequences ===")
    vocab = build_vocab(config)
    build_sequences(config, vocab)

    print("\n=== Phase 1: pretrain ===")
    model = pretrain(config)

    print("\n=== Phase 2: extract embeddings ===")
    code_vectors = extract_code_vectors(config, model, vocab)
    member_vectors = extract_member_vectors(config, model)

    print("\n=== Sanity: nearest neighbors of diabetes code E11.9 ===")
    neighbors = nearest_neighbors(code_vectors, query_token="E11.9", query_code_type="dx", k=8)
    print(neighbors.to_string(index=False))

    print("\n=== Phase 3a: HCC suspecting (illustrative) ===")
    sequences = load_sequences(config)
    already_coded = suspecting.build_member_hcc_presence(
        member_vectors, sequences, vocab, TOY_DX_TO_HCC
    )
    # NOTE: in a real run, training labels come from your gold source. Here we
    # reuse currently-coded presence as the target just to exercise the interface.
    classifier, hcc_columns = suspecting.train_hcc_classifier(
        member_vectors, already_coded, n_epochs=20
    )
    suspects = suspecting.predict_suspects(
        classifier, hcc_columns, member_vectors, already_coded, probability_threshold=0.5
    )
    print(f"found {len(suspects)} suspect (member, HCC) pairs; showing up to 5:")
    print(suspects.head(5).to_string(index=False))

    print("\n=== Phase 3b: client premium features ===")
    client_features = premium.build_client_features(member_vectors)
    print(
        f"client feature table: {client_features.shape[0]} clients x "
        f"{client_features.shape[1]} columns"
    )

    print("\n=== Phase 3c: fraud anomaly scores (stub) ===")
    scores = fraud.anomaly_scores(member_vectors)
    print(scores.head(5).to_string(index=False))

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
