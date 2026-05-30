#!/usr/bin/env python3
"""Utility for visualizing feature importance outputs from cell-wise runs."""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROLLING_WINDOW_SIZE = 51
ROLLING_MIN_PERIODS = 10

sns.set_style("whitegrid")


def _load_feature_table(model_dir: Path) -> pd.DataFrame:
    table_path = model_dir / "feature_importances_mean.csv"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Could not locate feature importance table at {table_path}. "
            "Ensure the pipeline was run with a model that records importances."
        )
    df = pd.read_csv(table_path)
    required = {"feature", "importance_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"The feature importance table is missing required columns: {sorted(missing)}"
        )
    return df


def _load_gene_summary(model_dir: Path) -> Optional[pd.DataFrame]:
    summary_path = model_dir / "feature_importance_per_gene_summary.csv"
    if not summary_path.exists():
        return None
    df = pd.read_csv(summary_path)
    if df.empty:
        return None
    return df


def _plot_top_features(ax: plt.Axes, table: pd.DataFrame, top_n: int) -> None:
    subset = table.sort_values("importance_mean", ascending=False).head(top_n)
    sns.barplot(data=subset, x="importance_mean", y="feature", ax=ax, color="#4C72B0")
    ax.set_xlabel("Mean importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Top {top_n} features")


def _plot_distance_scatter(ax: plt.Axes, table: pd.DataFrame) -> None:
    if "distance_to_tss_kb" not in table.columns:
        ax.text(0.5, 0.5, "No TSS distance metadata", ha="center", va="center")
        ax.set_axis_off()
        return
    plot_df = table[["distance_to_tss_kb", "importance_mean"]].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
    plot_df = plot_df[np.abs(plot_df["distance_to_tss_kb"]) <= 10.0]
    if plot_df.empty:
        ax.text(0.5, 0.5, "No finite points", ha="center", va="center")
        ax.set_axis_off()
        return
    plot_df = plot_df.sort_values("distance_to_tss_kb")
    pearson = plot_df["importance_mean"].corr(
        plot_df["distance_to_tss_kb"], method="pearson"
    )
    spearman = plot_df["importance_mean"].corr(
        plot_df["distance_to_tss_kb"], method="spearman"
    )
    sns.scatterplot(
        data=plot_df,
        x="distance_to_tss_kb",
        y="importance_mean",
        s=16,
        alpha=0.35,
        edgecolor="none",
        ax=ax,
    )
    sns.lineplot(
        x=plot_df["distance_to_tss_kb"],
        y=plot_df["importance_mean"]
        .rolling(window=ROLLING_WINDOW_SIZE, min_periods=ROLLING_MIN_PERIODS)
        .mean(),
        color="#D62728",
        linewidth=1.5,
        ax=ax,
        label="Rolling mean",
    )
    info = [
        f"Pearson={pearson:.3f}" if np.isfinite(pearson) else None,
        f"Spearman={spearman:.3f}" if np.isfinite(spearman) else None,
    ]
    info = [item for item in info if item]
    if info:
        ax.legend(loc="upper right", title="|".join(info))
    ax.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-10, 10)
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("Mean importance")
    ax.set_title("Importance vs TSS distance")


def _plot_bin_summary(ax: plt.Axes, table: pd.DataFrame) -> None:
    if "distance_to_tss_kb" not in table.columns:
        ax.text(0.5, 0.5, "No TSS distance metadata", ha="center", va="center")
        ax.set_axis_off()
        return
    df = table[["distance_to_tss_kb", "importance_mean"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        ax.text(0.5, 0.5, "No finite points", ha="center", va="center")
        ax.set_axis_off()
        return
    grouped = (
        df.groupby("distance_to_tss_kb")["importance_mean"]
        .agg(mean="mean", median="median", p90=lambda x: x.quantile(0.9))
        .reset_index()
        .sort_values("distance_to_tss_kb")
    )
    ax.plot(
        grouped["distance_to_tss_kb"], grouped["mean"], label="Mean", color="#1f77b4"
    )
    ax.plot(
        grouped["distance_to_tss_kb"], grouped["p90"], label="90th pct", color="#d62728"
    )
    ax.fill_between(
        grouped["distance_to_tss_kb"],
        grouped["median"],
        grouped["p90"],
        color="#d62728",
        alpha=0.15,
        label="Median–90th band",
    )
    ax.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-10, 10)
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("Importance")
    ax.set_title("Bin summary (mean/90th)")
    ax.legend()


def _plot_hexbin(ax: plt.Axes, table: pd.DataFrame) -> None:
    if "distance_to_tss_kb" not in table.columns:
        ax.text(0.5, 0.5, "No TSS distance metadata", ha="center", va="center")
        ax.set_axis_off()
        return
    df = table[["distance_to_tss_kb", "importance_mean"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        ax.text(0.5, 0.5, "No finite points", ha="center", va="center")
        ax.set_axis_off()
        return
    max_importance = df["importance_mean"].max()
    hb = ax.hexbin(
        df["distance_to_tss_kb"],
        df["importance_mean"],
        gridsize=35,
        extent=[-10, 10, 0, max_importance * 1.05 if max_importance > 0 else 1],
        cmap="mako",
        bins="log",
        mincnt=1,
    )
    ax.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-10, 10)
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("Importance")
    ax.set_title("Density (hexbin, log count)")
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")


def _plot_bin_violin(ax: plt.Axes, table: pd.DataFrame) -> None:
    if "distance_to_tss_kb" not in table.columns:
        ax.text(0.5, 0.5, "No TSS distance metadata", ha="center", va="center")
        ax.set_axis_off()
        return
    df = table[["distance_to_tss_kb", "importance_mean"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        ax.text(0.5, 0.5, "No finite points", ha="center", va="center")
        ax.set_axis_off()
        return
    df = df[np.abs(df["distance_to_tss_kb"]) <= 10]
    if df.empty:
        ax.text(0.5, 0.5, "No points within ±10 kb", ha="center", va="center")
        ax.set_axis_off()
        return
    sns.violinplot(
        data=df,
        x="distance_to_tss_kb",
        y="importance_mean",
        scale="width",
        inner=None,
        cut=0,
        ax=ax,
    )
    sns.stripplot(
        data=df[df["importance_mean"] > 0],
        x="distance_to_tss_kb",
        y="importance_mean",
        color="#d62728",
        alpha=0.5,
        size=2,
        ax=ax,
    )
    ax.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("Importance")
    ax.set_title("Per-bin distribution (violin + nonzero dots)")
    ax.set_ylim(bottom=0)


def _plot_thresholded_scatter(
    ax: plt.Axes, table: pd.DataFrame, threshold: float = 0.0
) -> None:
    if "distance_to_tss_kb" not in table.columns:
        ax.text(0.5, 0.5, "No TSS distance metadata", ha="center", va="center")
        ax.set_axis_off()
        return
    df = table[["distance_to_tss_kb", "importance_mean"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df[df["importance_mean"] > threshold]
    if df.empty:
        ax.text(
            0.5, 0.5, f"No points above threshold {threshold}", ha="center", va="center"
        )
        ax.set_axis_off()
        return
    sns.scatterplot(
        data=df,
        x="distance_to_tss_kb",
        y="importance_mean",
        s=18,
        alpha=0.5,
        edgecolor="none",
        ax=ax,
    )
    sns.rugplot(
        data=df, x="distance_to_tss_kb", height=0.05, ax=ax, color="#444", alpha=0.6
    )
    ax.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-10, 10)
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("Importance")
    ax.set_title(f"Scatter for importance > {threshold:g}")


def _plot_per_gene_panels(
    table: pd.DataFrame, output_path: Path, top_genes: int = 4
) -> None:
    if "gene_name" not in table.columns or "distance_to_tss_kb" not in table.columns:
        return
    df = table[["gene_name", "distance_to_tss_kb", "importance_mean"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return
    top = (
        df.groupby("gene_name")["importance_mean"]
        .sum()
        .sort_values(ascending=False)
        .head(top_genes)
        .index.tolist()
    )
    subset = df[df["gene_name"].isin(top)]
    if subset.empty:
        return
    rows = int(np.ceil(len(top) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4 * rows), squeeze=False)
    axes = axes.flatten()
    for ax, gene in zip(axes, top):
        gdf = subset[subset["gene_name"] == gene].sort_values("distance_to_tss_kb")
        sns.scatterplot(
            data=gdf,
            x="distance_to_tss_kb",
            y="importance_mean",
            s=20,
            alpha=0.6,
            edgecolor="none",
            ax=ax,
        )
        sns.lineplot(
            data=gdf,
            x="distance_to_tss_kb",
            y=gdf["importance_mean"].rolling(window=5, min_periods=1).mean(),
            color="#d62728",
            linewidth=1.4,
            ax=ax,
            label="Rolling mean (gene)",
        )
        ax.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
        ax.set_xlim(-10, 10)
        ax.set_title(gene)
        ax.set_xlabel("Distance to TSS (kb)")
        ax.set_ylabel("Importance")
    for ax in axes[len(top) :]:
        ax.axis("off")
    fig.suptitle("Per-gene panels (top by total importance)", y=0.99)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_gene_correlation(ax: plt.Axes, summary: pd.DataFrame) -> None:
    metrics = []
    if "spearman_distance_corr" in summary.columns:
        metrics.append(("Spearman", summary["spearman_distance_corr"]))
    if "pearson_distance_corr" in summary.columns:
        metrics.append(("Pearson", summary["pearson_distance_corr"]))
    if not metrics:
        ax.text(0.5, 0.5, "No per-gene correlations", ha="center", va="center")
        ax.set_axis_off()
        return
    melted = []
    for label, series in metrics:
        clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            continue
        melted.append(pd.DataFrame({"value": clean, "metric": label}))
    if not melted:
        ax.text(0.5, 0.5, "No finite per-gene metrics", ha="center", va="center")
        ax.set_axis_off()
        return
    data = pd.concat(melted, ignore_index=True)
    sns.boxplot(data=data, x="metric", y="value", ax=ax, color="#86BBD8")
    sns.stripplot(
        data=data, x="metric", y="value", ax=ax, color="#1F77B4", alpha=0.6, jitter=0.2
    )
    ax.set_ylabel("Correlation")
    ax.set_xlabel("Metric")
    ax.set_title("Per-gene TSS correlations")
    ax.axhline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feature importance diagnostics for a model run"
    )
    parser.add_argument("run_dir", type=Path, help="Path to a SPEAR run directory")
    parser.add_argument(
        "--model",
        default="mlp",
        help="Model subdirectory inside <run_dir>/models (default: mlp)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/figs/feature_importance_mlp.png"),
        help="Destination for the combined figure",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of features to show in the bar plot",
    )
    parser.add_argument(
        "--importance-threshold",
        type=float,
        default=0.0,
        help="Minimum importance for the thresholded scatter",
    )
    parser.add_argument(
        "--top-genes",
        type=int,
        default=4,
        help="Number of genes to show in the per-gene panels (if gene metadata is present)",
    )
    args = parser.parse_args()

    model_dir = args.run_dir / "models" / args.model
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist")

    feature_table = _load_feature_table(model_dir)
    gene_summary = _load_gene_summary(model_dir)

    # Main diagnostics grid
    panel_count = 3 if gene_summary is not None else 2
    fig, axes = plt.subplots(
        1, panel_count, figsize=(6 * panel_count, 5), squeeze=False
    )
    axes_flat = axes.flatten()
    _plot_top_features(axes_flat[0], feature_table, args.top_n)
    _plot_distance_scatter(axes_flat[1], feature_table)
    if gene_summary is not None:
        _plot_gene_correlation(axes_flat[2], gene_summary)
    fig.suptitle(f"Feature importance diagnostics | {args.model}", y=1.02)
    fig.tight_layout()
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure to {output_path}")

    # Additional TSS-focused plots
    extras = [
        ("feature_importance_bin_summary.png", _plot_bin_summary),
        ("feature_importance_bin_violin.png", _plot_bin_violin),
    ]
    for filename, fn in extras:
        fig, ax = plt.subplots(figsize=(7, 5))
        fn(ax, feature_table)
        out = model_dir / filename
        fig.tight_layout()
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"Saved {out}")

    # Thresholded scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_thresholded_scatter(ax, feature_table, threshold=args.importance_threshold)
    thresh_path = model_dir / "feature_importance_scatter_thresholded.png"
    fig.tight_layout()
    fig.savefig(thresh_path, dpi=300)
    plt.close(fig)
    print(f"Saved {thresh_path}")

    # Top bin per gene scatter (if metadata available)
    top_gene_path = model_dir / "feature_importance_top_bin_per_gene.png"
    _plot_top_bin_per_gene(feature_table, top_gene_path)


def _plot_top_bin_per_gene(table: pd.DataFrame, output_path: Path) -> None:
    if "gene_name" not in table.columns or "distance_to_tss_kb" not in table.columns:
        return
    df = table[["gene_name", "distance_to_tss_kb", "importance_mean"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return
    top_rows = df.loc[df.groupby("gene_name")["importance_mean"].idxmax()].copy()
    top_rows = top_rows.sort_values("distance_to_tss_kb")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=top_rows,
        x="distance_to_tss_kb",
        y="importance_mean",
        s=32,
        alpha=0.8,
        edgecolor="none",
        ax=ax,
    )
    sns.lineplot(
        x=top_rows["distance_to_tss_kb"],
        y=top_rows["importance_mean"].rolling(window=11, min_periods=3).mean(),
        color="#d62728",
        linewidth=1.6,
        ax=ax,
        label="Rolling mean (top bin per gene)",
    )
    ax.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
    ax.set_xlim(-10, 10)
    ax.set_xlabel("Distance to TSS (kb)")
    ax.set_ylabel("Importance (top bin per gene)")
    ax.set_title(f"Top bin per gene (n={len(top_rows)})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
