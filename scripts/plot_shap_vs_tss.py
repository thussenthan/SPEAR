#!/usr/bin/env python3
"""Utility for visualizing SHAP values vs TSS distance from cell-wise runs."""
ROLLING_WINDOW_SIZE = 51
ROLLING_MIN_PERIODS = 10


import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def _load_shap_table(model_dir: Path) -> pd.DataFrame:
    table_candidates = [
        model_dir / "shapley_values_mean.csv",
        model_dir / "shap_values_mean.csv",
        model_dir / "shap_importance_mean.csv",
        model_dir / "shap_importances_mean.csv",
    ]
    table_path = next((path for path in table_candidates if path.exists()), None)
    if table_path is None:
        raise FileNotFoundError(
            f"Could not locate SHAP table at any of: {', '.join(str(p) for p in table_candidates)}. "
            "Ensure the pipeline was run with enable_shap=true."
        )
    df = pd.read_csv(table_path)
    required = {"feature", "shap_mean_abs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"The SHAP importance table is missing required columns: {sorted(missing)}")
    # Normalize distance column names produced by different pipeline versions.
    if "distance_to_tss_kb" not in df.columns:
        if "signed_distance_to_tss_kb" in df.columns:
            df["distance_to_tss_kb"] = pd.to_numeric(df["signed_distance_to_tss_kb"], errors="coerce")
        elif "delta_to_tss_kb" in df.columns:
            df["distance_to_tss_kb"] = pd.to_numeric(df["delta_to_tss_kb"], errors="coerce")
        elif "distance_to_tss_abs_kb" in df.columns:
            df["distance_to_tss_kb"] = pd.to_numeric(df["distance_to_tss_abs_kb"], errors="coerce")
        elif "delta_to_tss_bp" in df.columns:
            df["distance_to_tss_kb"] = pd.to_numeric(df["delta_to_tss_bp"], errors="coerce") / 1000.0
        elif "distance_to_tss_bp" in df.columns:
            df["distance_to_tss_kb"] = pd.to_numeric(df["distance_to_tss_bp"], errors="coerce") / 1000.0
    return df


def _load_gene_summary(model_dir: Path) -> Optional[pd.DataFrame]:
    summary_candidates = [
        model_dir / "shapley_values_per_gene_summary.csv",
        model_dir / "shap_values_per_gene_summary.csv",
        model_dir / "shap_per_gene_summary.csv",
        model_dir / "shap_importance_per_gene_summary.csv",
    ]
    summary_path = next((path for path in summary_candidates if path.exists()), None)
    if summary_path is None:
        return None
    df = pd.read_csv(summary_path)
    if df.empty:
        return None
    return df


def _plot_top_features(ax: plt.Axes, table: pd.DataFrame, top_n: int) -> None:
    subset = table.sort_values("shap_mean_abs", ascending=False).head(top_n)
    sns.barplot(data=subset, x="shap_mean_abs", y="feature", ax=ax, color="#4C72B0")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} features by SHAP")


def _plot_distance_scatter(
    ax: plt.Axes,
    table: pd.DataFrame,
    show_scatter: bool = False,
    max_distance_kb: float = 10.0,
) -> None:
    """Plot SHAP values vs TSS distance in manuscript style (90th percentile line)."""
    if "distance_to_tss_kb" not in table.columns:
        ax.text(0.5, 0.5, "No TSS distance metadata", ha="center", va="center")
        ax.set_axis_off()
        return
    plot_df = table[["distance_to_tss_kb", "shap_mean_abs"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[np.abs(plot_df["distance_to_tss_kb"]) <= max_distance_kb]
    if plot_df.empty:
        ax.text(0.5, 0.5, "No finite points", ha="center", va="center")
        ax.set_axis_off()
        return
    plot_df = plot_df.sort_values("distance_to_tss_kb")
    
    # Calculate 90th percentile per distance bin (manuscript style)
    per_bin = (
        plot_df.groupby("distance_to_tss_kb", sort=True)["shap_mean_abs"]
        .quantile(0.9)
        .reset_index()
    )
    
    if show_scatter:
        sns.scatterplot(
            data=plot_df,
            x="distance_to_tss_kb",
            y="shap_mean_abs",
            s=30,
            alpha=0.45,
            edgecolor="none",
            color="#4daf4a",
            ax=ax,
        )
    
    ax.plot(
        per_bin["distance_to_tss_kb"],
        per_bin["shap_mean_abs"],
        color="#e41a1c",
        linewidth=1.5,
        label="90th percentile",
    )
    ax.legend(loc="upper right", frameon=True)
    ax.axvline(0.0, color="#999999", linestyle="--", linewidth=1)
    ax.set_xlim(-max_distance_kb, max_distance_kb)
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("SHAP value")
    ax.set_title("SHAP values vs TSS distance")


def _plot_bin_summary(ax: plt.Axes, table: pd.DataFrame) -> None:
    if "distance_to_tss_kb" not in table.columns:
        ax.text(0.5, 0.5, "No TSS distance metadata", ha="center", va="center")
        ax.set_axis_off()
        return
    df = table[["distance_to_tss_kb", "shap_mean_abs"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[np.abs(df["distance_to_tss_kb"]) <= 10.0]
    if df.empty:
        ax.text(0.5, 0.5, "No finite points", ha="center", va="center")
        ax.set_axis_off()
        return
    bins = np.arange(-10, 10.5, 0.5)
    df["bin"] = pd.cut(df["distance_to_tss_kb"], bins=bins, include_lowest=True)
    bin_summary = df.groupby("bin", observed=False)["shap_mean_abs"].agg(["mean", "count"]).reset_index()
    bin_summary["bin_center"] = bin_summary["bin"].apply(lambda x: x.mid)
    bin_summary = bin_summary[bin_summary["count"] > 0]
    if bin_summary.empty:
        ax.text(0.5, 0.5, "No binned data", ha="center", va="center")
        ax.set_axis_off()
        return
    sns.barplot(
        data=bin_summary,
        x="bin_center",
        y="mean",
        ax=ax,
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("SHAP by 500bp bins")
    ax.tick_params(axis="x", rotation=45)


def _plot_violin_with_nonzero(ax: plt.Axes, table: pd.DataFrame) -> None:
    if "distance_to_tss_kb" not in table.columns:
        ax.text(0.5, 0.5, "No TSS distance metadata", ha="center", va="center")
        ax.set_axis_off()
        return
    df = table[["distance_to_tss_kb", "shap_mean_abs"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[np.abs(df["distance_to_tss_kb"]) <= 10.0]
    if df.empty:
        ax.text(0.5, 0.5, "No finite points", ha="center", va="center")
        ax.set_axis_off()
        return
    bins = np.arange(-10, 10.5, 1.0)
    df["bin"] = pd.cut(df["distance_to_tss_kb"], bins=bins, include_lowest=True)
    df["bin_label"] = df["bin"].apply(lambda x: f"{x.left:.0f}")
    nonzero = df[df["shap_mean_abs"] > 0]
    if nonzero.empty:
        ax.text(0.5, 0.5, "No nonzero values", ha="center", va="center")
        ax.set_axis_off()
        return
    try:
        sns.violinplot(
            data=df,
            x="bin_label",
            y="shap_mean_abs",
            ax=ax,
            inner=None,
            color="#AAAAAA",
            alpha=0.5,
        )
        sns.stripplot(
            data=nonzero,
            x="bin_label",
            y="shap_mean_abs",
            ax=ax,
            size=2,
            alpha=0.6,
            color="#D62728",
        )
    except Exception:
        ax.text(0.5, 0.5, "Insufficient data for violin", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("|SHAP|")
    ax.set_title("Per-bin distribution (violin + nonzero dots)")
    ax.set_ylim(bottom=0)


def _plot_thresholded_scatter(ax: plt.Axes, table: pd.DataFrame, threshold: float = 0.0) -> None:
    if "distance_to_tss_kb" not in table.columns:
        ax.text(0.5, 0.5, "No TSS distance metadata", ha="center", va="center")
        ax.set_axis_off()
        return
    df = table[["distance_to_tss_kb", "shap_mean_abs"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["shap_mean_abs"] > threshold]
    if df.empty:
        ax.text(0.5, 0.5, f"No points above threshold {threshold}", ha="center", va="center")
        ax.set_axis_off()
        return
    sns.scatterplot(
        data=df,
        x="distance_to_tss_kb",
        y="shap_mean_abs",
        s=18,
        alpha=0.5,
        edgecolor="none",
        ax=ax,
    )
    sns.rugplot(data=df, x="distance_to_tss_kb", height=0.05, ax=ax, color="#444", alpha=0.6)
    ax.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-10, 10)
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("|SHAP|")
    ax.set_title(f"Scatter for |SHAP| > {threshold:g}")


def _plot_per_gene_panels(table: pd.DataFrame, output_path: Path, top_genes: int = 4) -> None:
    if "gene_name" not in table.columns or "distance_to_tss_kb" not in table.columns:
        return
    df = table[["gene_name", "distance_to_tss_kb", "shap_mean_abs"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return
    top = (
        df.groupby("gene_name")["shap_mean_abs"]
        .sum()
        .sort_values(ascending=False)
        .head(top_genes)
        .index.tolist()
    )
    subset = df[df["gene_name"].isin(top)]
    if subset.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, gene in enumerate(top):
        if i >= len(axes):
            break
        gene_df = subset[subset["gene_name"] == gene]
        gene_df = gene_df.sort_values("distance_to_tss_kb")
        ax = axes[i]
        sns.scatterplot(
            data=gene_df,
            x="distance_to_tss_kb",
            y="shap_mean_abs",
            s=20,
            alpha=0.5,
            ax=ax,
        )
        try:
            rolling_mean = gene_df["shap_mean_abs"].rolling(
                window=min(ROLLING_WINDOW_SIZE, len(gene_df)),
                min_periods=max(1, min(ROLLING_MIN_PERIODS, len(gene_df) // 2)),
            ).mean()
            sns.lineplot(
                x=gene_df["distance_to_tss_kb"],
                y=rolling_mean,
                color="#D62728",
                linewidth=1.5,
                ax=ax,
            )
        except Exception:
            pass
        ax.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Distance to TSS (kb)")
        ax.set_ylabel("|SHAP|")
        ax.set_title(gene)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-gene SHAP panels to {output_path}")


def create_summary_figure(
    model_dir: Path,
    output_path: Path,
    show_scatter: bool = False,
    max_distance_kb: float = 10.0,
) -> None:
    table = _load_shap_table(model_dir)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    _plot_top_features(axes[0, 0], table, top_n=20)
    _plot_distance_scatter(
        axes[0, 1],
        table,
        show_scatter=show_scatter,
        max_distance_kb=max_distance_kb,
    )
    _plot_bin_summary(axes[0, 2], table)
    _plot_violin_with_nonzero(axes[1, 0], table)
    _plot_thresholded_scatter(axes[1, 1], table, threshold=0.001)
    
    # Summary statistics in the last panel
    ax = axes[1, 2]
    if "distance_to_tss_kb" in table.columns:
        df = table[["distance_to_tss_kb", "shap_mean_abs"]].copy()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        stats_text = f"""SHAP Summary Statistics
        
Total features: {len(table):,}
With TSS distance: {len(df):,}
Nonzero |SHAP|: {(df['shap_mean_abs'] > 0).sum():,}

Mean |SHAP|: {df['shap_mean_abs'].mean():.6f}
Median |SHAP|: {df['shap_mean_abs'].median():.6f}
Max |SHAP|: {df['shap_mean_abs'].max():.6f}

Distance range: [{df['distance_to_tss_kb'].min():.1f}, {df['distance_to_tss_kb'].max():.1f}] kb
"""
    else:
        stats_text = "No TSS distance metadata available"
    
    ax.text(0.1, 0.5, stats_text, ha="left", va="center", fontsize=10, family="monospace")
    ax.set_axis_off()
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SHAP summary figure to {output_path}")


def create_manuscript_figure(model_dir: Path, output_path: Path, max_distance_kb: float = 10.0, 
                             show_scatter: bool = False, y_limits: Optional[tuple] = None) -> None:
    """Create a single manuscript-style figure: SHAP vs TSS distance."""
    table = _load_shap_table(model_dir)
    
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _plot_distance_scatter(
        ax,
        table,
        show_scatter=show_scatter,
        max_distance_kb=max_distance_kb,
    )
    if y_limits is not None:
        ax.set_ylim(y_limits)
    sns.despine(fig, left=True, bottom=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved manuscript figure to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SHAP values vs TSS distance")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to model directory containing shapley_values_mean.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for main figure (default: <model_dir>/shapley_values_vs_tss.png)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Output path for multi-panel summary figure (optional)",
    )
    parser.add_argument(
        "--per-gene-output",
        type=Path,
        help="Output path for per-gene panels (optional)",
    )
    parser.add_argument(
        "--top-genes",
        type=int,
        default=4,
        help="Number of top genes to plot in per-gene panels (default: 4)",
    )
    parser.add_argument(
        "--show-scatter",
        action="store_true",
        help="Show individual data points (scatter plot) in addition to 90th percentile line",
    )
    parser.add_argument(
        "--max-distance-kb",
        type=float,
        default=10.0,
        help="Absolute TSS distance limit in kb for plotting (default: 10.0)",
    )

    args = parser.parse_args()

    if not args.model_dir.exists():
        parser.error(f"Model directory does not exist: {args.model_dir}")

    # Create main manuscript-style figure
    output_path = args.output or (args.model_dir / "shapley_values_vs_tss.png")
    create_manuscript_figure(
        args.model_dir,
        output_path,
        max_distance_kb=args.max_distance_kb,
        show_scatter=args.show_scatter,
    )
    
    # Optionally create summary figure with multiple panels
    if args.summary_output:
        create_summary_figure(
            args.model_dir,
            args.summary_output,
            show_scatter=args.show_scatter,
            max_distance_kb=args.max_distance_kb,
        )

    if args.per_gene_output:
        table = _load_shap_table(args.model_dir)
        _plot_per_gene_panels(table, args.per_gene_output, top_genes=args.top_genes)


if __name__ == "__main__":
    main()
