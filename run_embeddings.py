from datetime import date
from icd_embeddings.config import Config
from icd_embeddings.data.build_vocab import build_vocab
from icd_embeddings.data.build_sequences import build_sequences
from icd_embeddings.model.pretrain import pretrain
from icd_embeddings.embeddings.extract import extract_code_vectors, extract_member_vectors

config = Config(
    claims_path = "data/embedding_data.csv",
    output_dir = f"output/ACA",
    line_of_business="ACA",
    observation_start = date(2015, 1, 1),
    observation_end = date(2017, 12, 31),
    device="cuda"
)

vocab = build_vocab(config)
build_sequences(config, vocab)
model = pretrain(config)
extract_code_vectors(config, model, vocab)
extract_member_vectors(config,model)
