"""HCC suspecting: turn member embeddings into suspect flags.

"Suspecting" means: a member's claims pattern looks like they have an HCC
condition that is NOT currently coded for them. We do this in three steps:

  1. Determine which HCCs each member ALREADY has coded (from their own dx
     tokens, via a pluggable code -> HCC mapping for the LOB).
  2. Train (or load) a multi-label classifier on the member embeddings that
     predicts the probability of each HCC. The training labels are supplied by
     YOU (e.g. confirmed HCCs from chart review or a later period) -- the
     embeddings are just the features.
  3. Flag, per member, the HCCs whose predicted probability is high but which
     are not already coded. Those are the suspects.

The HCC framework differs by LOB (HHS-HCC for Commercial, CMS-HCC for MA, a
Medicaid model for Medicaid), so the mapping is passed in rather than hard-coded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_member_hcc_presence(
    member_vectors: pd.DataFrame,
    sequences: pd.DataFrame,
    vocab: pd.DataFrame,
    dx_code_to_hcc: dict[str, str],
) -> pd.DataFrame:
    """Build a member x HCC binary table of currently-coded HCCs.

    A member "has" an HCC if any of their diagnosis tokens maps to it under the
    provided mapping. Procedure and pharmacy tokens are ignored here (HCCs are a
    diagnosis construct).

    Args:
        member_vectors: Output of extract_member_vectors (defines the member set
            and ordering).
        sequences: Member sequences (token_ids per member).
        vocab: Vocabulary table (token, token_id, code_type).
        dx_code_to_hcc: Mapping from a diagnosis token string to its HCC label,
            for this LOB's HCC framework.

    Returns:
        DataFrame indexed positionally like member_vectors, with a member_id
        column followed by one 0/1 column per HCC label (columns sorted).
    """
    # token_id -> (token string, code_type) for fast lookup.
    dx_token_to_hcc: dict[int, str] = {}
    dx_rows = vocab[vocab["code_type"] == "dx"]
    for token, token_id in zip(dx_rows["token"], dx_rows["token_id"]):
        if token in dx_code_to_hcc:
            dx_token_to_hcc[int(token_id)] = dx_code_to_hcc[token]

    all_hccs = sorted(set(dx_code_to_hcc.values()))
    tokens_by_member = sequences.set_index("member_id")["token_ids"]

    presence_rows = []
    for member_id in member_vectors["member_id"]:
        member_hccs = set()
        token_ids = tokens_by_member.get(member_id, [])
        for token_id in token_ids:
            hcc = dx_token_to_hcc.get(int(token_id))
            if hcc is not None:
                member_hccs.add(hcc)
        row = {hcc: int(hcc in member_hccs) for hcc in all_hccs}
        row["member_id"] = member_id
        presence_rows.append(row)

    presence = pd.DataFrame(presence_rows)
    ordered_columns = ["member_id"] + all_hccs
    return presence[ordered_columns]


class MultiLabelHCCClassifier(nn.Module):
    """A simple linear multi-label classifier over member embeddings.

    One shared linear layer maps a member vector to one logit per HCC. This is an
    illustrative, fully transparent head; swap in your own model when ready.
    """

    def __init__(self, embedding_dim: int, n_hccs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, n_hccs)

    def forward(self, member_vectors: torch.Tensor) -> torch.Tensor:
        """Return (batch, n_hccs) logits; apply sigmoid for probabilities."""
        return self.linear(member_vectors)


def _align_features_and_labels(
    member_vectors: pd.DataFrame, hcc_labels: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Join member vectors to their HCC label rows on member_id.

    Args:
        member_vectors: member_id + vector column.
        hcc_labels: member_id + one 0/1 column per HCC (the training target you supply).

    Returns:
        (feature_matrix, label_matrix, hcc_columns) as numpy arrays plus the
        ordered list of HCC column names.
    """
    hcc_columns = [column for column in hcc_labels.columns if column != "member_id"]
    merged = member_vectors[["member_id", "vector"]].merge(
        hcc_labels, on="member_id", how="inner"
    )
    if merged.empty:
        raise ValueError(
            "No members in common between member_vectors and hcc_labels; check member_id."
        )
    feature_matrix = np.asarray(merged["vector"].tolist(), dtype=np.float32)
    label_matrix = merged[hcc_columns].to_numpy(dtype=np.float32)
    return feature_matrix, label_matrix, hcc_columns


def train_hcc_classifier(
    member_vectors: pd.DataFrame,
    hcc_labels: pd.DataFrame,
    n_epochs: int = 30,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    random_seed: int = 12345,
) -> tuple[MultiLabelHCCClassifier, list[str]]:
    """Train the example multi-label HCC classifier on member embeddings.

    Args:
        member_vectors: Features (member_id + vector).
        hcc_labels: Training targets (member_id + 0/1 HCC columns). These are the
            "true" HCCs you want to detect -- supply them from your gold source.
        n_epochs: Training epochs.
        batch_size: Members per batch.
        learning_rate: Adam learning rate.
        device: "cuda" or "cpu".
        random_seed: Seed for reproducibility.

    Returns:
        (trained_classifier, hcc_columns) so predictions can be labeled later.
    """
    torch.manual_seed(random_seed)
    features, labels, hcc_columns = _align_features_and_labels(member_vectors, hcc_labels)

    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    classifier = MultiLabelHCCClassifier(
        embedding_dim=features.shape[1], n_hccs=labels.shape[1]
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    # BCEWithLogitsLoss handles the multi-label (independent yes/no per HCC) setup.
    loss_function = nn.BCEWithLogitsLoss()

    for epoch in range(1, n_epochs + 1):
        classifier.train()
        running_loss = 0.0
        for feature_batch, label_batch in loader:
            feature_batch = feature_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad()
            logits = classifier(feature_batch)
            loss = loss_function(logits, label_batch)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        if epoch % 10 == 0 or epoch == n_epochs:
            print(f"[suspecting] epoch {epoch:>3} | train BCE {running_loss / len(loader):.4f}")

    return classifier, hcc_columns


@torch.no_grad()
def predict_suspects(
    classifier: MultiLabelHCCClassifier,
    hcc_columns: list[str],
    member_vectors: pd.DataFrame,
    already_coded: pd.DataFrame,
    probability_threshold: float = 0.5,
    device: str = "cpu",
) -> pd.DataFrame:
    """Flag HCCs that are predicted-present but not currently coded, per member.

    Args:
        classifier: A trained MultiLabelHCCClassifier.
        hcc_columns: The HCC label order returned by train_hcc_classifier.
        member_vectors: Features to score (member_id + vector).
        already_coded: build_member_hcc_presence output (member_id + 0/1 HCC cols).
        probability_threshold: Minimum predicted probability to call a suspect.
        device: "cuda" or "cpu".

    Returns:
        Long DataFrame of suspects with columns member_id, hcc, predicted_probability,
        sorted by descending probability. A row appears only when the predicted
        probability clears the threshold AND the HCC is not already coded.
    """
    if not 0.0 <= probability_threshold <= 1.0:
        raise ValueError(
            f"probability_threshold must be in [0, 1], got {probability_threshold}"
        )

    features = np.asarray(member_vectors["vector"].tolist(), dtype=np.float32)
    classifier.eval()
    logits = classifier(torch.from_numpy(features).to(device))
    probabilities = torch.sigmoid(logits).cpu().numpy()

    # Align the already-coded table to the member order we just scored.
    coded_indexed = already_coded.set_index("member_id")
    member_ids = member_vectors["member_id"].tolist()

    suspect_rows = []
    for row_index, member_id in enumerate(member_ids):
        coded_for_member = coded_indexed.loc[member_id] if member_id in coded_indexed.index else None
        for hcc_index, hcc in enumerate(hcc_columns):
            probability = float(probabilities[row_index, hcc_index])
            if probability < probability_threshold:
                continue
            is_already_coded = (
                coded_for_member is not None
                and hcc in coded_for_member.index
                and int(coded_for_member[hcc]) == 1
            )
            if is_already_coded:
                continue
            suspect_rows.append(
                {
                    "member_id": member_id,
                    "hcc": hcc,
                    "predicted_probability": probability,
                }
            )

    suspects = pd.DataFrame(suspect_rows, columns=["member_id", "hcc", "predicted_probability"])
    return suspects.sort_values("predicted_probability", ascending=False).reset_index(drop=True)
