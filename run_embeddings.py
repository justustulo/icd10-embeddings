import json
from datetime import date, datetime

from icd_embeddings.config import Config
from icd_embeddings.data.build_vocab import build_vocab
from icd_embeddings.data.build_sequences import build_sequences
from icd_embeddings.model.pretrain import pretrain
from icd_embeddings.embeddings.extract import extract_code_vectors, extract_member_vectors


def save_run_info(config: Config) -> None:
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


config = Config(
    claims_path="data/embedding_data.csv",
    output_dir="output/ACA",
    line_of_business="ACA",
    observation_start=date(2015, 1, 1),
    observation_end=date(2017, 12, 31),
    # ACA HHS-HCC risk adjustment is binary (coded or not coded), so collapse
    # each code to its most-recent occurrence per member.
    unique_codes_per_member=True,
    device="cuda",
)

save_run_info(config)
vocab = build_vocab(config)
build_sequences(config, vocab)
model = pretrain(config)
extract_code_vectors(config, model, vocab)
extract_member_vectors(config, model)
