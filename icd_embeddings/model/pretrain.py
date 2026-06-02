"""Pretrain the masked-code transformer for one LOB.

Trains by predicting masked code tokens (self-supervised on claims only), holds
out a fraction of members to measure masked-code accuracy each epoch, and saves
the trained weights to config.checkpoint_path.

Run order: build_vocab -> build_sequences -> pretrain.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from icd_embeddings.config import Config
from icd_embeddings.model.dataset import (
    IGNORE_LABEL,
    MaskedCodeCollator,
    MemberSequenceDataset,
    load_sequences,
)
from icd_embeddings.model.transformer import MaskedCodeTransformer


def _split_members(
    sequences: pd.DataFrame, validation_fraction: float, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split member rows into train and validation sets reproducibly.

    Args:
        sequences: One row per member.
        validation_fraction: Fraction of members to hold out for validation.
        random_seed: Seed for the shuffle.

    Returns:
        (train_sequences, validation_sequences). Validation may be empty if
        validation_fraction is 0.
    """
    shuffled = sequences.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    n_validation = int(len(shuffled) * validation_fraction)
    validation_sequences = shuffled.iloc[:n_validation].reset_index(drop=True)
    train_sequences = shuffled.iloc[n_validation:].reset_index(drop=True)
    return train_sequences, validation_sequences


def _move_batch_to_device(batch: dict, device: str) -> dict:
    """Move every tensor in a collated batch to the target device."""
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def _evaluate_validation(
    model: MaskedCodeTransformer,
    loader: DataLoader,
    loss_function: nn.CrossEntropyLoss,
    vocab_size: int,
    device: str,
) -> dict:
    """Compute validation loss and top-1/top-5 accuracy over masked positions.

    Args:
        model: The transformer (will be set to eval mode internally).
        loader: DataLoader yielding collated, masked batches.
        loss_function: The same cross-entropy loss used during training.
        vocab_size: Number of tokens in the vocabulary; used to flatten logits.
        device: "cuda" or "cpu".

    Returns:
        Dict with keys "loss" (float), "top1" (float in [0,1]), "top5" (float in
        [0,1]). Loss is float("inf") and accuracies are 0.0 if there were no
        masked positions.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_targets = 0
    top1_hits = 0
    top5_hits = 0

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        outputs = model(
            token_ids=batch["token_ids"],
            type_ids=batch["type_ids"],
            recency_ids=batch["recency_ids"],
            age_ids=batch["age_ids"],
            sex_ids=batch["sex_ids"],
            attention_mask=batch["attention_mask"],
        )
        labels = batch["labels"]

        logits_flat = outputs["logits"].view(-1, vocab_size)
        labels_flat = labels.view(-1)
        total_loss += float(loss_function(logits_flat, labels_flat).item())
        n_batches += 1

        target_positions = labels != IGNORE_LABEL
        if target_positions.sum() == 0:
            continue

        logits_at_targets = outputs["logits"][target_positions]
        true_tokens = labels[target_positions]

        top5_predictions = logits_at_targets.topk(5, dim=-1).indices
        top1_predictions = top5_predictions[:, 0]

        top1_hits += int((top1_predictions == true_tokens).sum().item())
        top5_hits += int((top5_predictions == true_tokens.unsqueeze(1)).any(dim=1).sum().item())
        total_targets += int(true_tokens.numel())

    if n_batches == 0 or total_targets == 0:
        return {"loss": float("inf"), "top1": 0.0, "top5": 0.0}
    return {
        "loss": total_loss / n_batches,
        "top1": top1_hits / total_targets,
        "top5": top5_hits / total_targets,
    }


def pretrain(config: Config) -> MaskedCodeTransformer:
    """Train the masked-code transformer and save it to config.checkpoint_path.

    Args:
        config: The run configuration. Expects vocab and sequences to already exist
            on disk (build_vocab and build_sequences run first).

    Returns:
        The trained model (also saved to disk).
    """
    torch.manual_seed(config.random_seed)

    vocab = pd.read_parquet(config.vocab_path)
    vocab_size = len(vocab)

    sequences = load_sequences(config)
    train_sequences, validation_sequences = _split_members(
        sequences, config.validation_fraction, config.random_seed
    )

    collator = MaskedCodeCollator(config=config, vocab_size=vocab_size)
    train_loader = DataLoader(
        MemberSequenceDataset(train_sequences),
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    has_validation = len(validation_sequences) > 0
    validation_loader = (
        DataLoader(
            MemberSequenceDataset(validation_sequences),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        if has_validation
        else None
    )

    model = MaskedCodeTransformer(config=config, vocab_size=vocab_size).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.n_epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = _move_batch_to_device(batch, config.device)
            optimizer.zero_grad()
            outputs = model(
                token_ids=batch["token_ids"],
                type_ids=batch["type_ids"],
                recency_ids=batch["recency_ids"],
                age_ids=batch["age_ids"],
                sex_ids=batch["sex_ids"],
                attention_mask=batch["attention_mask"],
            )
            # Flatten (batch, seq, vocab) and (batch, seq) for token-level loss.
            logits_flat = outputs["logits"].view(-1, vocab_size)
            labels_flat = batch["labels"].view(-1)
            loss = loss_function(logits_flat, labels_flat)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        average_train_loss = running_loss / max(n_batches, 1)

        if validation_loader is not None:
            val_metrics = _evaluate_validation(
                model, validation_loader, loss_function, vocab_size, config.device
            )
            print(
                f"[pretrain] epoch {epoch:>3} | train loss {average_train_loss:.4f} | "
                f"val loss {val_metrics['loss']:.4f} | "
                f"val top1 {val_metrics['top1']:.3f} | val top5 {val_metrics['top5']:.3f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                torch.save(
                    {"model_state": model.state_dict(), "vocab_size": vocab_size},
                    config.checkpoint_path,
                )
                print(f"[pretrain]          -> new best val loss {best_val_loss:.4f}, checkpoint saved")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    print(
                        f"[pretrain] early stopping: val loss did not improve for "
                        f"{config.early_stopping_patience} consecutive epochs"
                    )
                    break
        else:
            print(f"[pretrain] epoch {epoch:>3} | train loss {average_train_loss:.4f}")

    # When validation is disabled there is no best-checkpoint logic, so save at the end.
    if validation_loader is None:
        torch.save(
            {"model_state": model.state_dict(), "vocab_size": vocab_size},
            config.checkpoint_path,
        )
        print(f"[pretrain] saved model to {config.checkpoint_path}")
    else:
        print(f"[pretrain] best model (val loss {best_val_loss:.4f}) loaded from {config.checkpoint_path}")

    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_trained_model(config: Config) -> MaskedCodeTransformer:
    """Rebuild the model architecture and load saved weights from disk.

    Args:
        config: The run configuration (must match the one used for training).

    Returns:
        The model in eval mode on config.device.
    """
    if not config.checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {config.checkpoint_path}. Run pretrain first."
        )
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device)
    model = MaskedCodeTransformer(config=config, vocab_size=checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(config.device)
    model.eval()
    return model
