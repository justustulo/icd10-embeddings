"""The shared masked-code transformer.

One model serves all three code types. Every token's input vector is the sum of
four learned embeddings:
    code     - what the code is (the reusable code embedding table)
    recency  - which time bucket the code fell in (coarse "when")
    position - order within the sequence (fine ordering inside a bucket)
    age/sex  - member-level demographics, broadcast across all positions

Token type ids (dx / proc / rx / special) are passed through the model for
masking decisions but are NOT added as a learned embedding — the code embedding
table alone captures per-type signal.

A standard transformer encoder then contextualizes the tokens. Two outputs are
reused downstream: the per-position logits over the vocabulary (for the masked-
code pretraining objective) and the CLS hidden state (the member embedding).
"""

from __future__ import annotations

import torch
from torch import nn

from icd_embeddings.config import (
    N_AGE_IDS,
    N_SEX_IDS,
    Config,
)


class MaskedCodeTransformer(nn.Module):
    """Type-aware transformer encoder trained by masked-code modeling.

    Attributes:
        embedding_dim: Size of every embedding and of the model's hidden state.
        code_embedding: The reusable (vocab_size, embedding_dim) code table.
    """

    def __init__(self, config: Config, vocab_size: int) -> None:
        """Build the embedding tables, encoder, and tied output head.

        Args:
            config: The run configuration (model hyperparameters).
            vocab_size: Number of tokens in the vocabulary (from the vocab table).
        """
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.vocab_size = vocab_size

        # Recency embedding has one extra slot for the CLS/PAD special bucket.
        n_recency_ids = config.n_recency_buckets + 1
        # Positions: CLS plus up to max_sequence_length code tokens.
        max_positions = config.max_sequence_length + 1

        self.code_embedding = nn.Embedding(
            vocab_size, config.embedding_dim, padding_idx=0
        )
        self.recency_embedding = nn.Embedding(n_recency_ids, config.embedding_dim)
        self.use_position_embedding = config.use_position_embedding
        if config.use_position_embedding:
            self.position_embedding = nn.Embedding(max_positions, config.embedding_dim)
        self.age_embedding = nn.Embedding(N_AGE_IDS, config.embedding_dim)
        self.sex_embedding = nn.Embedding(N_SEX_IDS, config.embedding_dim)

        self.embedding_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.n_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True,  # tensors are (batch, seq, dim)
            norm_first=True,  # pre-norm trains more stably
        )
        # enable_nested_tensor=False avoids a warning (and a no-op code path) when
        # using pre-norm layers; it does not change the maths.
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers, enable_nested_tensor=False
        )

        # Masked-code prediction head. Weights are tied to the code embedding table
        # (a standard trick that improves quality and reduces parameter count).
        self.mlm_head = nn.Linear(config.embedding_dim, vocab_size, bias=True)
        self.mlm_head.weight = self.code_embedding.weight

    def _embed_inputs(
        self,
        token_ids: torch.Tensor,
        recency_ids: torch.Tensor,
        age_ids: torch.Tensor,
        sex_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sum the four embedding sources into one (batch, seq, dim) tensor."""
        _, seq_len = token_ids.shape

        # Age and sex are member-level; broadcast them across all positions.
        age_vectors = self.age_embedding(age_ids).unsqueeze(1)  # (batch, 1, dim)
        sex_vectors = self.sex_embedding(sex_ids).unsqueeze(1)  # (batch, 1, dim)

        summed = (
            self.code_embedding(token_ids)
            + self.recency_embedding(recency_ids)
            + age_vectors
            + sex_vectors
        )
        if self.use_position_embedding:
            positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
            summed = summed + self.position_embedding(positions)  # (1, seq, dim)
        normalized = self.embedding_layer_norm(summed)
        return self.embedding_dropout(normalized)

    def encode(
        self,
        token_ids: torch.Tensor,
        type_ids: torch.Tensor,
        recency_ids: torch.Tensor,
        age_ids: torch.Tensor,
        sex_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the encoder and return contextualized hidden states.

        Args:
            token_ids: (batch, seq) code token ids.
            type_ids: (batch, seq) type ids.
            recency_ids: (batch, seq) recency bucket ids.
            age_ids: (batch,) member age ids.
            sex_ids: (batch,) member sex ids.
            attention_mask: (batch, seq) 1 for real tokens, 0 for padding.

        Returns:
            (batch, seq, dim) hidden states.
        """
        inputs = self._embed_inputs(token_ids, recency_ids, age_ids, sex_ids)
        # PyTorch's encoder expects True where a key should be IGNORED (i.e. padding).
        padding_mask = attention_mask == 0
        return self.encoder(inputs, src_key_padding_mask=padding_mask)

    def forward(
        self,
        token_ids: torch.Tensor,
        type_ids: torch.Tensor,
        recency_ids: torch.Tensor,
        age_ids: torch.Tensor,
        sex_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """Full forward pass producing masked-code logits and the member vector.

        Returns:
            Dict with:
              "logits": (batch, seq, vocab_size) masked-code prediction scores,
              "member_embedding": (batch, dim) the CLS hidden state.
        """
        hidden_states = self.encode(
            token_ids, type_ids, recency_ids, age_ids, sex_ids, attention_mask
        )
        logits = self.mlm_head(hidden_states)
        member_embedding = hidden_states[:, 0, :]  # CLS is always position 0
        return {"logits": logits, "member_embedding": member_embedding}
