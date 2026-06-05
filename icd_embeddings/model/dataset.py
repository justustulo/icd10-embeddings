"""PyTorch dataset and masked-code collator for pretraining.

The dataset yields one member's parallel lists (token/type/recency ids plus
age and sex). The collator does three jobs per batch:
  1. prepend the CLS token (the member-representation slot),
  2. pad every sequence to the batch's longest length,
  3. apply masked-code masking and produce the prediction labels.

Masking uses whole-code masking: ~mask_rate fraction of the unique code types
in each sequence are selected, then every position carrying one of those codes
is replaced with <MASK>. This prevents the model from predicting a masked code
by copying from a sibling occurrence in a different recency bucket.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset

from icd_embeddings.config import (
    CLS_TOKEN_ID,
    MASK_TOKEN_ID,
    PAD_TOKEN_ID,
    SPECIAL_TOKENS,
    SPECIAL_TYPE_ID,
    TYPE_TO_ID,
    Config,
)

# Cross-entropy ignores this label, so non-predicted positions contribute no loss.
IGNORE_LABEL: int = -100


def load_sequences(config: Config) -> pd.DataFrame:
    """Read the member-sequence parquet written by build_sequences.

    Args:
        config: The run configuration (uses config.sequences_path).

    Returns:
        DataFrame with one row per member and list columns token_ids/type_ids/
        recency_ids plus age_id, sex_id, client_id, member_id.
    """
    if not config.sequences_path.exists():
        raise FileNotFoundError(
            f"Member sequences not found at {config.sequences_path}. "
            f"Run build_sequences first."
        )
    return pd.read_parquet(config.sequences_path)


class MemberSequenceDataset(Dataset):
    """Wraps the member-sequence DataFrame as a PyTorch Dataset.

    Each item is a dict of plain Python lists/ints; tensor creation, padding and
    masking all happen in the collator so masking is fresh every epoch.
    """

    def __init__(self, sequences: pd.DataFrame) -> None:
        """Store the sequence rows.

        Args:
            sequences: Output of build_sequences / load_sequences.
        """
        if len(sequences) == 0:
            raise ValueError("Cannot build a dataset from an empty sequences frame.")
        self._rows = sequences.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        row = self._rows.iloc[index]
        return {
            "token_ids": list(row["token_ids"]),
            "type_ids": list(row["type_ids"]),
            "recency_ids": list(row["recency_ids"]),
            "age_id": int(row["age_id"]),
            "sex_id": int(row["sex_id"]),
        }


class MaskedCodeCollator:
    """Builds padded, masked training batches from a list of dataset items.

    Attributes:
        vocab_size: Total number of tokens (used to bound random replacements).
        mask_rate: Fraction of code tokens chosen as prediction targets.
        special_recency_id: Recency id assigned to CLS/PAD (a dedicated extra slot).
        n_special_tokens: Number of special tokens; random replacements are drawn
            from the real-code id range [n_special_tokens, vocab_size).
        generator: Torch RNG for reproducible masking.
    """

    def __init__(self, config: Config, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.mask_rate = config.mask_rate
        # CLS/PAD sit in a recency slot just past the real buckets.
        self.special_recency_id = config.n_recency_buckets
        self.n_special_tokens = len(SPECIAL_TOKENS)
        self.generator = torch.Generator()
        self.generator.manual_seed(config.random_seed)
        # Integer type ids that are eligible for masking. Types not in this set
        # stay visible in context every training step and are never predicted.
        self.maskable_type_ids = frozenset(
            TYPE_TO_ID[t] for t in config.mask_code_types
        )

    def inference_batch(self, items: list[dict]) -> dict:
        """Build a padded batch WITHOUT masking (for embedding extraction).

        Same CLS-prepend and padding as training, but every code token is kept
        intact and no labels are produced. Use this as the collate_fn when you
        want member embeddings from the real, unmasked sequences.

        Args:
            items: A list of dataset items.

        Returns:
            Dict of batch tensors (token/type/recency ids, age/sex ids, mask).
        """
        return self._build_padded_batch(items)

    def __call__(self, items: list[dict]) -> dict:
        """Build a padded, masked training batch with prediction labels."""
        batch = self._build_padded_batch(items)
        masked_token_ids, labels = self._apply_masking(
            batch["token_ids"], batch["type_ids"], batch["attention_mask"]
        )
        batch["token_ids"] = masked_token_ids
        batch["labels"] = labels
        return batch

    def _build_padded_batch(self, items: list[dict]) -> dict:
        """Prepend CLS and pad all sequences to the batch's longest length."""
        batch_size = len(items)
        # +1 for the CLS token prepended to every sequence.
        max_len = max(len(item["token_ids"]) for item in items) + 1

        token_ids = torch.full((batch_size, max_len), PAD_TOKEN_ID, dtype=torch.long)
        type_ids = torch.full((batch_size, max_len), SPECIAL_TYPE_ID, dtype=torch.long)
        recency_ids = torch.full(
            (batch_size, max_len), self.special_recency_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        age_ids = torch.tensor([item["age_id"] for item in items], dtype=torch.long)
        sex_ids = torch.tensor([item["sex_id"] for item in items], dtype=torch.long)

        for row_index, item in enumerate(items):
            length = len(item["token_ids"])
            # Position 0 is CLS (already set to the special type/recency defaults).
            token_ids[row_index, 0] = CLS_TOKEN_ID
            attention_mask[row_index, 0] = 1

            if length > 0:
                slice_end = 1 + length
                token_ids[row_index, 1:slice_end] = torch.tensor(
                    item["token_ids"], dtype=torch.long
                )
                type_ids[row_index, 1:slice_end] = torch.tensor(
                    item["type_ids"], dtype=torch.long
                )
                recency_ids[row_index, 1:slice_end] = torch.tensor(
                    item["recency_ids"], dtype=torch.long
                )
                attention_mask[row_index, 1:slice_end] = 1

        return {
            "token_ids": token_ids,
            "type_ids": type_ids,
            "recency_ids": recency_ids,
            "age_ids": age_ids,
            "sex_ids": sex_ids,
            "attention_mask": attention_mask,
        }

    def _apply_masking(
        self,
        token_ids: torch.Tensor,
        type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose prediction targets using whole-code masking and replace with MASK.

        Selects ~mask_rate fraction of the unique codes among maskable positions,
        then replaces every position holding one of those codes with MASK_TOKEN_ID.
        Masking all occurrences together prevents the model from predicting a code
        by copying from a sibling occurrence in a different recency bucket.

        Only code types listed in self.maskable_type_ids are eligible for masking.
        Types excluded from that set stay visible in context every step, acting as
        permanent signals (e.g. procedure codes in an ACA suspecting model).

        Args:
            token_ids: (batch, seq) token ids with CLS at position 0 and PAD elsewhere.
            type_ids: (batch, seq) type ids aligned to token_ids.
            attention_mask: (batch, seq) 1 for real tokens, 0 for padding.

        Returns:
            (masked_token_ids, labels). labels hold the original id at predicted
            positions and IGNORE_LABEL everywhere else.
        """
        masked = token_ids.clone()
        labels = torch.full_like(token_ids, IGNORE_LABEL)

        # Build a boolean mask for positions whose code type can be masked.
        type_eligible = torch.zeros_like(attention_mask, dtype=torch.bool)
        for type_id in self.maskable_type_ids:
            type_eligible |= (type_ids == type_id)

        for b in range(token_ids.shape[0]):
            # Eligible = real code tokens only: exclude padding, CLS, and any
            # code types that are not in mask_code_types.
            eligible = attention_mask[b].bool() & type_eligible[b]
            eligible[0] = False  # never predict the CLS slot

            eligible_token_ids = token_ids[b][eligible]
            unique_codes = eligible_token_ids.unique()

            if len(unique_codes) == 0:
                continue

            # Select ~mask_rate fraction of unique codes to mask.
            code_probs = torch.rand(unique_codes.shape, generator=self.generator)
            selected_codes = unique_codes[code_probs < self.mask_rate]

            # Safeguard: guarantee at least one code is masked so the loss is never undefined.
            if len(selected_codes) == 0:
                selected_codes = unique_codes[:1]

            # Mark every position holding a selected code as a prediction target.
            selected_positions = eligible & torch.isin(token_ids[b], selected_codes)

            labels[b][selected_positions] = token_ids[b][selected_positions]
            masked[b][selected_positions] = MASK_TOKEN_ID

        return masked, labels
