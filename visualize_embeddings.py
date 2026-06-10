"""Generate a 2-D scatter plot of trained ICD code embeddings.

Reads the code_vectors.parquet file produced by the embedding pipeline and
projects the vectors down to 2 dimensions using t-SNE (default) or UMAP.
Writes an interactive Plotly HTML file you can open in any browser.

Usage
-----
Basic (t-SNE, colored by code type, 2000-token sample):
    python visualize_embeddings.py path/to/code_vectors.parquet

Color by ICD-10 chapter instead:
    python visualize_embeddings.py path/to/code_vectors.parquet --color-by chapter

Use UMAP instead of t-SNE:
    python visualize_embeddings.py path/to/code_vectors.parquet --method umap

Sample all tokens (slow for large vocabularies):
    python visualize_embeddings.py path/to/code_vectors.parquet --n-tokens 0

Write to a specific output path:
    python visualize_embeddings.py path/to/code_vectors.parquet --output my_scatter.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from icd_embeddings.embeddings.visualize import build_scatter_df, make_scatter_figure


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with fields: code_vectors, method, n_tokens, color_by,
        output, perplexity, n_iter, seed.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "code_vectors",
        type=Path,
        help="Path to code_vectors.parquet produced by the embedding pipeline.",
    )
    parser.add_argument(
        "--method",
        choices=["tsne", "umap"],
        default="tsne",
        help="Dimensionality reduction method. Default: tsne. "
             "UMAP requires: pip install umap-learn",
    )
    parser.add_argument(
        "--n-tokens",
        type=int,
        default=2000,
        metavar="N",
        help="Number of tokens to sample before reducing. "
             "0 = use all tokens (slow for large vocabularies). Default: 2000.",
    )
    parser.add_argument(
        "--color-by",
        choices=["type", "chapter"],
        default="type",
        help="Color scheme. 'type' groups by dx/proc/rx. "
             "'chapter' groups dx codes by ICD-10 chapter. Default: type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embedding_scatter.html"),
        help="Path to write the output HTML file. Default: embedding_scatter.html",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity. Ignored for UMAP. "
             "Should be smaller than n-tokens. Default: 30.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1000,
        metavar="N",
        help="Maximum t-SNE iterations. Ignored for UMAP. Default: 1000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling and the reduction algorithm. Default: 42.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the embedding visualization script."""
    args = _parse_args()

    if not args.code_vectors.exists():
        print(f"Error: code_vectors file not found: {args.code_vectors}", file=sys.stderr)
        sys.exit(1)

    print(f"[visualize] Loading code vectors from {args.code_vectors}...")
    code_vectors = pd.read_parquet(args.code_vectors)

    n_real = (code_vectors["code_type"].isin(["dx", "proc", "rx"])).sum()
    print(f"[visualize] {n_real:,} real-code tokens in vocabulary.")

    if args.n_tokens == 0:
        effective_n = n_real
    else:
        effective_n = min(args.n_tokens, n_real)
    print(
        f"[visualize] Using {effective_n:,} tokens "
        f"(method={args.method.upper()}, color-by={args.color_by})."
    )

    try:
        scatter_df = build_scatter_df(
            code_vectors=code_vectors,
            method=args.method,
            color_by=args.color_by,
            n_tokens=args.n_tokens,
            perplexity=args.perplexity,
            n_iter=args.n_iter,
            seed=args.seed,
        )
    except ImportError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    fig = make_scatter_figure(scatter_df, method=args.method, color_by=args.color_by)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output))
    print(f"[visualize] Scatter plot written to {args.output}")


if __name__ == "__main__":
    main()
