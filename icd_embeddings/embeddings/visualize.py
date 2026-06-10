"""Core logic for visualizing the trained code embedding space.

Shared by the standalone visualize_embeddings.py script and the Streamlit app.
Given a code_vectors DataFrame (output of extract_code_vectors), this module
handles subsampling, dimensionality reduction (t-SNE or UMAP), and building
the Plotly scatter figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from icd_embeddings.config import CODE_TYPES

# Maps the first character of a dx code to a broad ICD-10 chapter group.
# Codes that share a chapter are given the same label so related clinical
# categories end up with the same color in the scatter plot.
_DX_CHAPTER_BY_FIRST_CHAR: dict[str, str] = {
    "A": "Infectious/Parasitic (A-B)",
    "B": "Infectious/Parasitic (A-B)",
    "C": "Neoplasms (C-D)",
    "D": "Neoplasms/Blood (C-D)",
    "E": "Endocrine/Metabolic (E)",
    "F": "Mental/Behavioral (F)",
    "G": "Nervous System (G)",
    "H": "Eye/Ear (H)",
    "I": "Circulatory (I)",
    "J": "Respiratory (J)",
    "K": "Digestive (K)",
    "L": "Skin (L)",
    "M": "Musculoskeletal (M)",
    "N": "Genitourinary (N)",
    "O": "Pregnancy/Childbirth (O-P)",
    "P": "Pregnancy/Childbirth (O-P)",
    "Q": "Congenital (Q)",
    "R": "Symptoms/Signs (R)",
    "S": "Injury/Poisoning (S-T)",
    "T": "Injury/Poisoning (S-T)",
    "U": "Special Purpose (U)",
    "V": "External Causes (V-Y)",
    "W": "External Causes (V-Y)",
    "X": "External Causes (V-Y)",
    "Y": "External Causes (V-Y)",
    "Z": "Factors/Health Status (Z)",
}


def _assign_color_label(token: str, code_type: str, color_by: str) -> str:
    """Return the color group label for one token.

    Args:
        token: The code string (e.g. "E11.9", "99213").
        code_type: "dx", "proc", or "rx".
        color_by: "type" (group by dx/proc/rx) or "chapter" (ICD-10 chapter for dx
            codes; proc and rx are labeled as their own groups).

    Returns:
        A string label used to assign a color in the scatter plot.
    """
    if color_by == "type":
        return code_type
    # color_by == "chapter"
    if code_type == "dx" and token:
        first = token[0].upper()
        return _DX_CHAPTER_BY_FIRST_CHAR.get(first, f"Other ({first})")
    if code_type == "proc":
        return "Procedure"
    if code_type == "rx":
        return "Pharmacy/Drug Class"
    return code_type


def _reduce_to_2d(
    matrix: np.ndarray,
    method: str,
    perplexity: int,
    n_iter: int,
    seed: int,
) -> np.ndarray:
    """Project high-dimensional code vectors down to 2 dimensions.

    Args:
        matrix: Float32 array of shape (n_tokens, embedding_dim).
        method: "tsne" or "umap".
        perplexity: t-SNE perplexity (ignored for UMAP). Should be smaller than
            n_tokens; values between 5 and 50 work well in practice.
        n_iter: Maximum t-SNE iterations (ignored for UMAP).
        seed: Random seed for reproducibility.

    Returns:
        Float32 array of shape (n_tokens, 2).

    Raises:
        ImportError: If method is "umap" and umap-learn is not installed.
        ValueError: If method is not "tsne" or "umap".
    """
    if method == "umap":
        try:
            import umap as umap_lib
        except ImportError:
            raise ImportError(
                "umap-learn is not installed. Install it with:\n"
                "    pip install umap-learn\n"
                "or switch to method='tsne'."
            )
        reducer = umap_lib.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(matrix).astype(np.float32)

    if method == "tsne":
        from sklearn.manifold import TSNE

        # ASSUMPTION: n_tokens > perplexity (t-SNE requires this). The caller
        # is responsible for ensuring a reasonable subsample size.
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=n_iter,
            random_state=seed,
            # PCA initialization is more stable and converges faster than
            # random init, which matters when n_tokens is large.
            init="pca",
        )
        return tsne.fit_transform(matrix).astype(np.float32)

    raise ValueError(f"method must be 'tsne' or 'umap', got {method!r}")


def build_scatter_df(
    code_vectors: pd.DataFrame,
    method: str = "tsne",
    color_by: str = "type",
    n_tokens: int = 2000,
    perplexity: int = 30,
    n_iter: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Subsample code vectors, reduce to 2-D, and return a plot-ready DataFrame.

    Special tokens (PAD, MASK, CLS, UNK) are always dropped before reduction.

    Args:
        code_vectors: DataFrame with columns token, code_type, member_count, vector.
            Produced by extract_code_vectors in icd_embeddings/embeddings/extract.py.
        method: "tsne" (default) or "umap".
        color_by: "type" (dx/proc/rx) or "chapter" (ICD-10 chapter for dx, separate
            groups for proc and rx).
        n_tokens: Number of tokens to sample before running the reduction. 0 means
            use all tokens (slow for large vocabularies). Sampling is random; the
            seed controls reproducibility.
        perplexity: t-SNE perplexity. Ignored for UMAP. Must be less than n_tokens.
        n_iter: Maximum t-SNE iterations. Ignored for UMAP.
        seed: Random seed for subsampling and the reduction algorithm.

    Returns:
        DataFrame with columns: token, code_type, member_count, color_label, x, y.

    Raises:
        ValueError: If no real-code tokens are found after filtering.
    """
    real_codes = code_vectors[code_vectors["code_type"].isin(CODE_TYPES)].copy()
    real_codes = real_codes.reset_index(drop=True)

    if len(real_codes) == 0:
        raise ValueError("No real-code tokens found in code_vectors (only special tokens present).")

    if n_tokens > 0 and len(real_codes) > n_tokens:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(real_codes), size=n_tokens, replace=False)
        real_codes = real_codes.iloc[sample_idx].reset_index(drop=True)

    # t-SNE perplexity must be less than n_tokens.
    effective_perplexity = min(perplexity, len(real_codes) - 1)
    if effective_perplexity != perplexity:
        import warnings
        warnings.warn(
            f"perplexity ({perplexity}) >= n_tokens ({len(real_codes)}); "
            f"clamping to {effective_perplexity}.",
            UserWarning,
            stacklevel=2,
        )

    matrix = np.asarray(real_codes["vector"].tolist(), dtype=np.float32)
    coords = _reduce_to_2d(matrix, method, effective_perplexity, n_iter, seed)

    result = real_codes[["token", "code_type", "member_count"]].copy()
    result["color_label"] = [
        _assign_color_label(tok, ctype, color_by)
        for tok, ctype in zip(result["token"], result["code_type"])
    ]
    result["x"] = coords[:, 0]
    result["y"] = coords[:, 1]
    return result


def make_scatter_figure(
    scatter_df: pd.DataFrame,
    method: str,
    color_by: str,
) -> go.Figure:
    """Build an interactive Plotly scatter figure from the 2-D projection.

    Each point is one code token. Hovering shows the token string, code type,
    and the number of distinct members that token appeared in during training.

    Args:
        scatter_df: Output of build_scatter_df. Expected columns: token,
            code_type, member_count, color_label, x, y.
        method: The reduction method used ("tsne" or "umap"), shown in axis labels.
        color_by: "type" or "chapter", shown in the chart title.

    Returns:
        A Plotly Figure ready to display or write to HTML.
    """
    method_label = method.upper()
    color_description = "code type" if color_by == "type" else "ICD-10 chapter"
    n_tokens = len(scatter_df)
    title = (
        f"Code Embedding Space — {method_label}, colored by {color_description} "
        f"({n_tokens:,} tokens)"
    )

    fig = px.scatter(
        scatter_df,
        x="x",
        y="y",
        color="color_label",
        hover_data={
            "token": True,
            "code_type": True,
            "member_count": True,
            "x": False,
            "y": False,
        },
        title=title,
        labels={
            "x": f"{method_label} 1",
            "y": f"{method_label} 2",
            "color_label": "Group",
            "member_count": "Members",
        },
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.update_layout(height=700, legend=dict(itemsizing="constant"))
    return fig
