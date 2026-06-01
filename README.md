# ICD-10 (multi-code-type) Embeddings for Risk Modeling

Learn dense vector representations of medical codes from your own claims, and
reuse them across risk tasks: **HCC suspecting**, **client premium modeling**,
and (minor) **fraud**. The embeddings are the deliverable — they are *features*
you feed into your own downstream models, not a model that predicts cost itself.

One model is trained **per line of business** (Commercial / MA / Medicaid),
because the populations and HCC frameworks differ.

## How it works (one sentence)

A type-aware transformer (Med-BERT style) is trained purely on your claims by
masking out codes and learning to predict them; this teaches it which
diagnoses, procedures, and drugs co-occur, which is exactly the signal HCC
suspecting and fraud rely on.

## Install

```
pip install -r requirements.txt
```

(The `requirements.txt` pulls a CPU build of PyTorch by default. For GPU
training, install the CUDA build of torch for your machine and set
`device="cuda"` in the Config.)

## Expected input

One **long-format** claims table (parquet or csv), one row per claim line. The
default column names are in `ColumnMap` (`icd_embeddings/config.py`); edit them
to match your extract. Logical fields:

| field | meaning |
|---|---|
| `member_id` | covered individual |
| `client_id` | employer group / plan sponsor |
| `line_of_business` | LOB label used to filter (one model per LOB) |
| `incurred_date` | **date of service** (not paid/processed) |
| `code` | raw code string (e.g. `E11.9`, `99213`, a therapeutic class) |
| `code_type` | one of `dx`, `proc`, `rx` |
| `member_birth_date` | for age at the observation anchor |
| `member_sex` | `M` / `F` / other |

**Pharmacy must be rolled up to a therapeutic class (ATC/GPI) before loading —
do not feed raw NDCs.** Diagnoses and procedures are fed as-is; rare diagnoses
are rolled up to their 3-character ICD-10 parent automatically.

## Run it (per LOB)

```python
from datetime import date
from icd_embeddings.config import Config
from icd_embeddings.data.build_vocab import build_vocab
from icd_embeddings.data.build_sequences import build_sequences
from icd_embeddings.model.pretrain import pretrain
from icd_embeddings.embeddings.extract import extract_code_vectors, extract_member_vectors

config = Config(
    claims_path="data/claims_commercial.parquet",
    output_dir="output/commercial",
    line_of_business="Commercial",
    observation_start=date(2023, 1, 1),
    observation_end=date(2023, 12, 31),
    device="cuda",  # or "cpu"
)

vocab = build_vocab(config)            # Phase 0
build_sequences(config, vocab)         # Phase 0
model = pretrain(config)               # Phase 1
extract_code_vectors(config, model, vocab)   # Phase 2 -> code_vectors.parquet
extract_member_vectors(config, model)        # Phase 2 -> member_vectors.parquet
```

Re-run with a new `Config` (different `line_of_business` and `output_dir`) for
each LOB.

## Outputs

Everything lands in `config.output_dir`:

- `vocab.parquet` — the token vocabulary.
- `member_sequences.parquet` — one ordered, typed sequence per member.
- `model.pt` — the trained transformer.
- `code_vectors.parquet` — **one vector per code** (+ use `nearest_neighbors`).
- `member_vectors.parquet` — **one vector per member** (the main feature table).

## Downstream (Phase 3 — runs after embeddings exist)

- `downstream/suspecting.py` — build currently-coded HCC presence, train the
  example multi-label classifier on member vectors, and flag HCCs
  *predicted-present but not coded*. Pass in your LOB's `dx_code_to_hcc` mapping
  and **your own training labels** (confirmed HCCs). *Note:* if you train on
  currently-coded presence as a shortcut, you will get zero suspects by
  construction — real suspecting needs gold labels.
- `downstream/premium.py` — `build_client_features` aggregates member vectors to
  the client: per-dimension mean/std (+p10/p90), exposure-weighted if you pass
  member-months, plus demographic mix. Feed the result to your premium model.
- `downstream/fraud.py` — minimal embedding-outlier anomaly score (stub).

## Smoke test

No real data needed:

```
python examples/make_synthetic_claims.py
python examples/run_pipeline.py
```

The synthetic generator builds members with co-occurring condition codes, so
after training, `E11.9` (diabetes) should neighbor diabetes-related codes such
as the metformin therapeutic class — a quick check that cross-type co-occurrence
was learned.

## Important data caveats (read before using real claims)

- **IBNR / claim run-out:** the most recent months are incomplete. Either
  exclude the immature tail from your windows or completion-adjust it; never
  treat recent partial months as full. See the `# CHECK` notes in
  `data/build_sequences.py`.
- **Window leakage:** for downstream prediction tasks, keep the label window
  (`prediction_start`/`prediction_end`) strictly after `observation_end`. The
  Config validates this.
- **Member identity:** members move between LOBs and their `client_id` can
  change; we attribute each member to the client of their most recent claim in
  the window.
- **Dollars** (when you add a cost/premium target) stay floats; round only for
  display.
