"""Streamlit app for interactive ICD code prediction visualization.

Enter a member's code history, hold out one or two codes, and see what the
trained transformer would predict at those masked positions.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from icd_embeddings.inference.predict import (
    ArchConfig,
    build_sequence_tensors,
    predict_chained,
    predict_single_masked,
    validate_codes_against_vocab,
)
from icd_embeddings.model.transformer import MaskedCodeTransformer

# ── Defaults ───────────────────────────────────────────────────────────────────

# Default architecture matches Config class defaults. These are overwritten
# automatically if the loaded checkpoint contains an "architecture" key.
_ARCH_DEFAULTS: dict = {
    "arch_embedding_dim": 128,
    "arch_n_layers": 4,
    "arch_n_heads": 8,
    "arch_feedforward_dim": 512,
    "arch_dropout": 0.1,
    "arch_max_sequence_length": 256,
    "arch_use_recency_bucketing": True,
    "arch_recency_edges_str": "30,90,180,365,730",
    "arch_rollup_rare_dx": True,
}

_DEFAULT_CODES = pd.DataFrame({
    "code":      ["E11.9", "I10",   "Z79.4"],
    "code_type": ["dx",    "dx",    "rx"],
    "days_ago":  [30,      90,      30],
})

# Color per code type, used consistently across bar charts.
_TYPE_COLORS = {"dx": "#1f77b4", "proc": "#ff7f0e", "rx": "#2ca02c"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_recency_edges(edges_str: str) -> tuple[int, ...]:
    """Parse a comma-separated string into an ascending tuple of ints.

    Args:
        edges_str: Comma-separated integers, e.g. "30,90,180,365,730".

    Returns:
        Tuple of ints in ascending order.

    Raises:
        ValueError: If the string can't be parsed or values are not ascending.
    """
    parts = [p.strip() for p in edges_str.split(",") if p.strip()]
    if not parts:
        raise ValueError("Recency bucket edges cannot be empty.")
    try:
        edges = tuple(int(p) for p in parts)
    except ValueError:
        raise ValueError(
            f"Recency bucket edges must be integers separated by commas. Got: {edges_str!r}"
        )
    if list(edges) != sorted(edges):
        raise ValueError(
            f"Recency bucket edges must be in ascending order. Got: {edges}"
        )
    return edges


def _render_bar_chart(predictions_df: pd.DataFrame, title: str) -> None:
    """Render a horizontal probability bar chart for top-K predictions.

    Color-codes predictions by code type (dx=blue, proc=orange, rx=green).

    Args:
        predictions_df: Output of predict_single_masked or one step of
            predict_chained. Expected columns: rank, token, code_type,
            probability, logit, member_count.
        title: Chart title. Pass "" for no title.
    """
    fig = px.bar(
        predictions_df,
        x="probability",
        y="token",
        color="code_type",
        color_discrete_map=_TYPE_COLORS,
        orientation="h",
        title=title,
        labels={
            "probability": "Probability",
            "token": "Predicted code",
            "code_type": "Type",
        },
        hover_data={
            "logit": ":.3f",
            "member_count": True,
            "rank": True,
        },
    )
    # Highest probability at top.
    fig.update_layout(yaxis=dict(autorange="reversed"), height=380)
    st.plotly_chart(fig, use_container_width=True)


def _show_hit_or_miss(true_code: str, predictions_df: pd.DataFrame, top_k: int) -> None:
    """Show a success or info message based on whether true_code is in the top-K.

    Args:
        true_code: The code that was held out.
        predictions_df: Top-K predictions DataFrame (must have "token" and "rank" columns).
        top_k: The K value, used in the message text.
    """
    if true_code in predictions_df["token"].values:
        rank = int(
            predictions_df.loc[predictions_df["token"] == true_code, "rank"].iloc[0]
        )
        st.success(f"**{true_code}** found at rank **{rank}** out of {top_k}.")
    else:
        st.info(f"**{true_code}** did not appear in the top-{top_k} predictions.")


# ── Model loading ──────────────────────────────────────────────────────────────

def _handle_load_model(checkpoint_path: str, vocab_path: str) -> None:
    """Read the checkpoint and vocab, build the model, store everything in session_state.

    If the checkpoint contains an "architecture" key (written by pretrain.py),
    the sidebar architecture settings are updated automatically. For older
    checkpoints, the current sidebar values are used as-is.

    Results are stored in:
        st.session_state["model"]        — MaskedCodeTransformer in eval mode
        st.session_state["vocab"]        — vocabulary DataFrame
        st.session_state["arch_config"]  — ArchConfig used to build the model
        st.session_state["model_loaded"] — True on success

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        vocab_path: Path to the vocab.parquet file.
    """
    with st.spinner("Loading model..."):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            st.error(f"Checkpoint not found: {checkpoint_path}")
            return
        except Exception as e:
            st.error(f"Failed to read checkpoint: {e}")
            return

        # If the checkpoint has architecture metadata, update session_state so
        # the sidebar widgets reflect the correct values on the next rerun.
        if "architecture" in checkpoint:
            arch_meta = checkpoint["architecture"]
            st.session_state["arch_embedding_dim"] = arch_meta["embedding_dim"]
            st.session_state["arch_n_layers"] = arch_meta["n_layers"]
            st.session_state["arch_n_heads"] = arch_meta["n_heads"]
            st.session_state["arch_feedforward_dim"] = arch_meta["feedforward_dim"]
            st.session_state["arch_dropout"] = float(arch_meta["dropout"])
            st.session_state["arch_max_sequence_length"] = arch_meta["max_sequence_length"]
            st.session_state["arch_use_recency_bucketing"] = arch_meta.get(
                "use_recency_bucketing", True
            )
            st.session_state["arch_recency_edges_str"] = ",".join(
                str(e) for e in arch_meta["recency_bucket_day_edges"]
            )
            st.session_state["arch_rollup_rare_dx"] = arch_meta.get(
                "rollup_rare_dx_to_3char", True
            )

        # Build ArchConfig from session_state (just updated above if checkpoint had metadata).
        try:
            use_recency = st.session_state["arch_use_recency_bucketing"]
            recency_edges = _parse_recency_edges(
                st.session_state["arch_recency_edges_str"]
            )
            arch_config = ArchConfig(
                embedding_dim=st.session_state["arch_embedding_dim"],
                n_layers=st.session_state["arch_n_layers"],
                n_heads=st.session_state["arch_n_heads"],
                feedforward_dim=st.session_state["arch_feedforward_dim"],
                dropout=st.session_state["arch_dropout"],
                max_sequence_length=st.session_state["arch_max_sequence_length"],
                use_recency_bucketing=use_recency,
                recency_bucket_day_edges=recency_edges,
                rollup_rare_dx_to_3char=st.session_state["arch_rollup_rare_dx"],
            )
        except ValueError as e:
            st.error(f"Architecture settings error: {e}")
            return

        try:
            model = MaskedCodeTransformer(
                config=arch_config, vocab_size=checkpoint["vocab_size"]
            )
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
        except RuntimeError as e:
            st.error(
                f"Architecture mismatch when loading weights: {e}. "
                "Check that the architecture settings match the checkpoint."
            )
            return
        except Exception as e:
            st.error(f"Failed to load model weights: {e}")
            return

        try:
            vocab = pd.read_parquet(vocab_path)
        except FileNotFoundError:
            st.error(f"Vocabulary not found: {vocab_path}")
            return
        except Exception as e:
            st.error(f"Failed to read vocabulary: {e}")
            return

        st.session_state["model"] = model
        st.session_state["vocab"] = vocab
        st.session_state["arch_config"] = arch_config
        st.session_state["model_loaded"] = True


# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    """Render all sidebar controls: paths, architecture, demographics, settings."""
    with st.sidebar:
        st.header("Model")
        checkpoint_path = st.text_input(
            "Checkpoint path (.pt)",
            key="checkpoint_path",
        )
        vocab_path = st.text_input(
            "Vocabulary path (.parquet)",
            key="vocab_path",
        )
        load_button = st.button("Load model", type="primary")

        if load_button:
            _handle_load_model(checkpoint_path, vocab_path)

        if st.session_state.get("model_loaded", False):
            st.success("Model loaded.")

        st.divider()
        st.header("Architecture")
        st.caption(
            "Must match the settings used during training. "
            "Auto-filled when the checkpoint contains architecture metadata."
        )

        with st.expander("Architecture settings", expanded=False):
            st.number_input(
                "Embedding dim",
                min_value=8,
                max_value=1024,
                step=8,
                key="arch_embedding_dim",
            )
            st.number_input(
                "Layers",
                min_value=1,
                max_value=24,
                step=1,
                key="arch_n_layers",
            )
            st.number_input(
                "Attention heads",
                min_value=1,
                max_value=32,
                step=1,
                key="arch_n_heads",
            )
            st.number_input(
                "Feedforward dim",
                min_value=16,
                max_value=4096,
                step=16,
                key="arch_feedforward_dim",
            )
            st.number_input(
                "Dropout",
                min_value=0.0,
                max_value=0.5,
                step=0.01,
                format="%.2f",
                key="arch_dropout",
            )
            st.number_input(
                "Max sequence length",
                min_value=1,
                max_value=1024,
                step=1,
                key="arch_max_sequence_length",
            )
            st.checkbox(
                "Use recency bucketing",
                key="arch_use_recency_bucketing",
                help="Uncheck if the model was trained with use_recency_bucketing=False",
            )
            st.text_input(
                "Recency bucket day edges (comma-separated)",
                key="arch_recency_edges_str",
                help="e.g. '30,90,180,365,730' creates 6 recency buckets",
                disabled=not st.session_state.get("arch_use_recency_bucketing", True),
            )
            st.checkbox(
                "Roll up rare dx to 3-char parent",
                key="arch_rollup_rare_dx",
            )

        st.divider()
        st.header("Member demographics")
        st.caption("Optional context for the prediction.")
        st.number_input(
            "Age (years, 101 = unknown)",
            min_value=0,
            max_value=101,
            step=1,
            key="member_age",
        )
        st.selectbox(
            "Sex",
            options=["unknown", "M", "F"],
            key="member_sex",
        )

        st.divider()
        st.header("Settings")
        st.slider(
            "Top-K predictions to show",
            min_value=3,
            max_value=20,
            value=10,
            key="top_k",
        )


# ── Prediction UI ──────────────────────────────────────────────────────────────

def _render_prediction_ui(
    model: MaskedCodeTransformer,
    vocab: pd.DataFrame,
    arch: ArchConfig,
    top_k: int,
) -> None:
    """Render the code entry table, mode selector, and prediction results.

    Args:
        model: Loaded model in eval mode.
        vocab: The vocabulary DataFrame.
        arch: Architecture / inference config.
        top_k: Number of top-K predictions to display.
    """
    st.subheader("Member code history")
    st.caption(
        "Enter the codes you want the model to reason over. "
        "Then hold out one or two to see what the model predicts at those positions."
    )

    edited_df = st.data_editor(
        _DEFAULT_CODES,
        num_rows="dynamic",
        column_config={
            "code": st.column_config.TextColumn(
                "Code",
                help="ICD-10 diagnosis, CPT/HCPCS procedure, or drug class",
                width="medium",
            ),
            "code_type": st.column_config.SelectboxColumn(
                "Type",
                options=["dx", "proc", "rx"],
                width="small",
            ),
            "days_ago": st.column_config.NumberColumn(
                "Days ago",
                min_value=0,
                step=1,
                width="small",
            ),
        },
        use_container_width=True,
        key="code_table",
    )

    # Drop rows with missing or blank code values.
    clean_df = edited_df.dropna(subset=["code"]).copy()
    clean_df = clean_df[clean_df["code"].str.strip() != ""].reset_index(drop=True)

    if len(clean_df) == 0:
        st.info("Add at least one code to the table above to begin.")
        return

    codes = clean_df["code"].str.strip().tolist()
    code_types = clean_df["code_type"].tolist()
    days_ago = clean_df["days_ago"].fillna(0).astype(int).tolist()

    # Surface a warning for codes the vocabulary doesn't recognize.
    unknown = validate_codes_against_vocab(
        codes, code_types, vocab, arch.rollup_rare_dx_to_3char
    )
    if unknown:
        st.warning(
            f"These codes are not in the vocabulary and will be treated as "
            f"<UNK>: {', '.join(unknown)}"
        )

    age = st.session_state.get("member_age", 101)
    sex = st.session_state.get("member_sex", "unknown")

    st.divider()
    mode = st.radio(
        "Prediction mode",
        options=["Single mask", "Chain prediction"],
        horizontal=True,
    )

    # Labels shown in the mask-selection dropdowns.
    code_labels = [
        f"{c} ({t}) — {d} days ago"
        for c, t, d in zip(codes, code_types, days_ago)
    ]

    if mode == "Single mask":
        _render_single_mask(
            model, vocab, arch, codes, code_types, days_ago,
            code_labels, age, sex, top_k,
        )
    else:
        _render_chain(
            model, vocab, arch, codes, code_types, days_ago,
            code_labels, age, sex, top_k,
        )


def _render_single_mask(
    model: MaskedCodeTransformer,
    vocab: pd.DataFrame,
    arch: ArchConfig,
    codes: list[str],
    code_types: list[str],
    days_ago: list[int],
    code_labels: list[str],
    age: int,
    sex: str,
    top_k: int,
) -> None:
    """Render the single-mask prediction UI: one dropdown, one chart, one table.

    Args:
        model: The trained model.
        vocab: The vocabulary DataFrame.
        arch: Architecture config.
        codes: Code strings from the entry table.
        code_types: Code type per code.
        days_ago: Days ago per code.
        code_labels: Human-readable dropdown labels.
        age: Member age id.
        sex: Member sex string.
        top_k: Number of predictions to show.
    """
    selected_label = st.selectbox("Code to hold out (mask)", options=code_labels)
    mask_index = code_labels.index(selected_label)
    held_out_code = codes[mask_index]

    if st.button("Predict", key="predict_single"):
        with st.spinner("Running model..."):
            try:
                tensors = build_sequence_tensors(
                    codes=codes,
                    code_types=code_types,
                    days_ago=days_ago,
                    vocab=vocab,
                    arch=arch,
                    age=age,
                    sex=sex,
                )
                results_df = predict_single_masked(
                    model=model,
                    tensors=tensors,
                    mask_position=mask_index,
                    vocab=vocab,
                    top_k=top_k,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        st.subheader(f"Top-{top_k} predictions for masked position: {held_out_code}")
        _show_hit_or_miss(held_out_code, results_df, top_k)
        _render_bar_chart(results_df, title="")
        display_cols = ["rank", "token", "code_type", "member_count", "probability", "logit"]
        st.dataframe(
            results_df[display_cols].style.format(
                {"probability": "{:.4f}", "logit": "{:.3f}"}
            ),
            use_container_width=True,
        )


def _render_chain(
    model: MaskedCodeTransformer,
    vocab: pd.DataFrame,
    arch: ArchConfig,
    codes: list[str],
    code_types: list[str],
    days_ago: list[int],
    code_labels: list[str],
    age: int,
    sex: str,
    top_k: int,
) -> None:
    """Render the chain prediction UI: two dropdowns, two side-by-side charts.

    Args:
        model: The trained model.
        vocab: The vocabulary DataFrame.
        arch: Architecture config.
        codes: Code strings from the entry table.
        code_types: Code type per code.
        days_ago: Days ago per code.
        code_labels: Human-readable dropdown labels.
        age: Member age id.
        sex: Member sex string.
        top_k: Number of predictions to show per step.
    """
    if len(codes) < 2:
        st.info("Chain prediction requires at least 2 codes in the table.")
        return

    col_left, col_right = st.columns(2)
    with col_left:
        first_label = st.selectbox(
            "First code to predict (original context)",
            options=code_labels,
            key="chain_first",
        )
    with col_right:
        # Exclude the first selection so the user can't pick the same code twice.
        remaining_labels = [lbl for lbl in code_labels if lbl != first_label]
        second_label = st.selectbox(
            "Second code to predict (after Step 1 is injected)",
            options=remaining_labels,
            key="chain_second",
        )

    first_index = code_labels.index(first_label)
    second_index = code_labels.index(second_label)
    first_code = codes[first_index]
    second_code = codes[second_index]

    st.caption(
        f"Step 1 predicts **{first_code}** with all other codes visible. "
        f"Step 2 injects the Step 1 top prediction as context, then predicts **{second_code}**."
    )

    if st.button("Run chain prediction", key="predict_chain"):
        with st.spinner("Running model..."):
            try:
                tensors = build_sequence_tensors(
                    codes=codes,
                    code_types=code_types,
                    days_ago=days_ago,
                    vocab=vocab,
                    arch=arch,
                    age=age,
                    sex=sex,
                )
                step1_df, step2_df = predict_chained(
                    model=model,
                    tensors=tensors,
                    first_mask_position=first_index,
                    second_mask_position=second_index,
                    vocab=vocab,
                    top_k=top_k,
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                return

        top1_predicted = step1_df["token"].iloc[0]
        display_cols = ["rank", "token", "code_type", "member_count", "probability", "logit"]

        result_left, result_right = st.columns(2)

        with result_left:
            st.subheader("Step 1")
            st.caption(f"Predicting **{first_code}** — original context")
            _show_hit_or_miss(first_code, step1_df, top_k)
            _render_bar_chart(step1_df, title="")
            st.dataframe(
                step1_df[display_cols].style.format(
                    {"probability": "{:.4f}", "logit": "{:.3f}"}
                ),
                use_container_width=True,
            )

        with result_right:
            st.subheader("Step 2")
            st.caption(
                f"Predicting **{second_code}** — context now includes "
                f"**{top1_predicted}** at the {first_code} position"
            )
            _show_hit_or_miss(second_code, step2_df, top_k)
            _render_bar_chart(step2_df, title="")
            st.dataframe(
                step2_df[display_cols].style.format(
                    {"probability": "{:.4f}", "logit": "{:.3f}"}
                ),
                use_container_width=True,
            )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the Streamlit app."""
    st.set_page_config(
        page_title="ICD Code Predictor",
        layout="wide",
    )
    st.title("ICD Code Prediction Demo")
    st.caption(
        "Hold out one or two codes from a member's history and see what the "
        "trained transformer predicts at those masked positions."
    )

    # Initialize session_state keys with defaults before any widgets render.
    # This ensures widgets that use key= have a value on first load.
    session_defaults = {
        **_ARCH_DEFAULTS,
        "checkpoint_path": "examples/output/model.pt",
        "vocab_path": "examples/output/vocab.parquet",
        "member_age": 101,
        "member_sex": "unknown",
        "top_k": 10,
        "model_loaded": False,
        "model": None,
        "vocab": None,
        "arch_config": None,
    }
    for key, val in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    _render_sidebar()

    if not st.session_state.get("model_loaded", False):
        st.info(
            "Load a trained model using the sidebar to begin. "
            "Point it at the checkpoint (.pt) and vocabulary (.parquet) files "
            "from your training run. If you have the example output, the default "
            "paths are already filled in."
        )
        return

    model = st.session_state["model"]
    vocab = st.session_state["vocab"]
    arch = st.session_state["arch_config"]
    top_k = st.session_state.get("top_k", 10)

    _render_prediction_ui(model, vocab, arch, top_k)


if __name__ == "__main__":
    main()
