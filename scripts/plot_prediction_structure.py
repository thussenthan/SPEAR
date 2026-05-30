#!/usr/bin/env python
"""Plot cell-level structure from SPEAR raw predictions.

Inputs are the long-form ``predictions_raw.csv`` exported by SPEAR with columns:
``split``, ``cell_id``, ``gene``, ``y_true``, and ``y_pred``.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/spear_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/spear_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/spear_numba_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def _load_umap():
    try:
        import umap

        return umap.UMAP
    except Exception as exc:  # pragma: no cover - environment-dependent fallback
        raise RuntimeError(
            "umap-learn could not be imported. Try setting NUMBA_CACHE_DIR to a "
            "writable directory or install a working umap-learn/numba stack."
        ) from exc


def _select_genes(preds: pd.DataFrame, n_genes: int) -> list[str]:
    metrics = []
    for gene, group in preds.groupby("gene", sort=False):
        clean = (
            group[["y_true", "y_pred"]].apply(pd.to_numeric, errors="coerce").dropna()
        )
        if len(clean) < 10:
            continue
        corr = clean["y_true"].corr(clean["y_pred"], method="pearson")
        if pd.notna(corr):
            metrics.append((gene, float(corr), len(clean)))
    if not metrics:
        raise ValueError("No genes with valid y_true/y_pred values were found")
    ranked = pd.DataFrame(metrics, columns=["gene", "pearson", "n_cells"])
    ranked = ranked.sort_values(["pearson", "n_cells"], ascending=[False, False])
    return ranked.head(n_genes)["gene"].tolist()


def _pivot(preds: pd.DataFrame, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = preds[preds["gene"].isin(genes)].copy()
    true = sub.pivot_table(
        index="cell_id", columns="gene", values="y_true", aggfunc="mean"
    )
    pred = sub.pivot_table(
        index="cell_id", columns="gene", values="y_pred", aggfunc="mean"
    )
    cells = true.index.intersection(pred.index)
    common_genes = true.columns.intersection(pred.columns)
    true = true.loc[cells, common_genes].astype(float)
    pred = pred.loc[cells, common_genes].astype(float)
    keep_cells = ~(true.isna().any(axis=1) | pred.isna().any(axis=1))
    true = true.loc[keep_cells]
    pred = pred.loc[keep_cells]
    if true.empty or pred.empty:
        raise ValueError(
            "No complete cell x gene matrix remained after pivoting predictions"
        )
    return true, pred


def _plot_gene_scatters(
    preds: pd.DataFrame, genes: list[str], out: Path, split_label: str
) -> None:
    n_cols = 5
    n_rows = int(np.ceil(len(genes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.7))
    axes = np.asarray(axes).ravel()
    for ax, gene in zip(axes, genes):
        g = preds[preds["gene"] == gene].copy()
        g["y_true"] = pd.to_numeric(g["y_true"], errors="coerce")
        g["y_pred"] = pd.to_numeric(g["y_pred"], errors="coerce")
        g = g.dropna(subset=["y_true", "y_pred"])
        sns.scatterplot(
            data=g, x="y_true", y="y_pred", s=9, alpha=0.45, ax=ax, legend=False
        )
        if not g.empty:
            lo = float(min(g["y_true"].min(), g["y_pred"].min()))
            hi = float(max(g["y_true"].max(), g["y_pred"].max()))
            ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, alpha=0.55)
            corr = g["y_true"].corr(g["y_pred"], method="pearson")
            ax.set_title(f"{gene}\\nr={corr:.2f}", fontsize=8)
        ax.set_xlabel("True RNA", fontsize=7)
        ax.set_ylabel("Pred RNA", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[len(genes) :]:
        ax.axis("off")
    fig.suptitle(
        f"Per-gene true vs predicted RNA, top {len(genes)} genes ({split_label})",
        y=1.002,
        fontsize=14,
    )
    fig.tight_layout()
    safe_split = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in split_label
    )
    split_path = out / f"scatter_50_genes_true_vs_predicted_{safe_split}_only.png"
    fig.savefig(split_path, dpi=220, bbox_inches="tight")
    fig.savefig(
        out / "scatter_50_genes_true_vs_predicted.png", dpi=220, bbox_inches="tight"
    )
    plt.close(fig)


def _fit_embeddings(
    true: pd.DataFrame,
    pred: pd.DataFrame,
    n_clusters: int,
    seed: int,
    predicted_umap_mode: str,
):
    scaler = StandardScaler()
    x_true = scaler.fit_transform(true.to_numpy())
    x_pred = scaler.transform(pred.to_numpy())

    n_pcs = max(2, min(30, x_true.shape[0] - 1, x_true.shape[1]))
    pca = PCA(n_components=n_pcs, random_state=seed)
    true_pca = pca.fit_transform(x_true)
    pred_pca = pca.transform(x_pred)

    clusters = KMeans(n_clusters=n_clusters, random_state=seed, n_init=25).fit_predict(
        true_pca
    )
    UMAP = _load_umap()
    true_reducer = UMAP(
        n_neighbors=30, min_dist=0.25, metric="euclidean", random_state=seed
    )
    true_umap = true_reducer.fit_transform(true_pca)
    if predicted_umap_mode == "project":
        pred_umap = true_reducer.transform(pred_pca)
    else:
        pred_reducer = UMAP(
            n_neighbors=30, min_dist=0.25, metric="euclidean", random_state=seed
        )
        pred_umap = pred_reducer.fit_transform(pred_pca)
    pred_clusters = KMeans(
        n_clusters=n_clusters, random_state=seed, n_init=25
    ).fit_predict(pred_pca)
    return (
        x_true,
        x_pred,
        true_pca,
        pred_pca,
        true_umap,
        pred_umap,
        clusters,
        pred_clusters,
    )


def _cluster_palette(clusters: np.ndarray):
    unique = sorted(np.unique(clusters))
    palette = sns.color_palette("tab10", n_colors=max(len(unique), 10))
    color_map = {cluster: palette[i % len(palette)] for i, cluster in enumerate(unique)}
    colors = [color_map[int(c)] for c in clusters]
    return unique, color_map, colors


def _label_cluster_centroids(ax, embedding: np.ndarray, clusters: np.ndarray) -> None:
    for cluster in sorted(np.unique(clusters)):
        idx = clusters == cluster
        if not np.any(idx):
            continue
        x = float(np.median(embedding[idx, 0]))
        y = float(np.median(embedding[idx, 1]))
        ax.text(
            x,
            y,
            str(int(cluster)),
            ha="center",
            va="center",
            fontsize=16,
            color="black",
        )


def _plot_umaps(true_umap, pred_umap, clusters, out: Path) -> None:
    unique, color_map, colors = _cluster_palette(clusters)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    axes[0].scatter(
        true_umap[:, 0], true_umap[:, 1], c=colors, s=8, alpha=0.75, linewidths=0
    )
    axes[0].set_title("UMAP fit on real RNA")
    axes[1].scatter(
        pred_umap[:, 0], pred_umap[:, 1], c=colors, s=8, alpha=0.75, linewidths=0
    )
    axes[1].set_title("UMAP fit/projected from predicted RNA")
    _label_cluster_centroids(axes[0], true_umap, clusters)
    _label_cluster_centroids(axes[1], pred_umap, clusters)
    for ax in axes:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.grid(alpha=0.15)
    handles = [
        plt.Line2D(
            [0], [0], marker="o", linestyle="", color=color_map[i], label=f"cluster {i}"
        )
        for i in unique
    ]
    axes[1].legend(
        handles=handles,
        title="Real-RNA clusters",
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(
        out / "umap_real_vs_predicted_matched_clusters.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_composite_structure(
    true_umap: np.ndarray,
    pred_umap: np.ndarray,
    clusters: np.ndarray,
    dist_df: pd.DataFrame,
    out: Path,
) -> None:
    """Create one summary figure matching the real/predicted UMAP + distance layout."""
    unique, color_map, colors = _cluster_palette(clusters)
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.35], hspace=0.34, wspace=0.22)
    ax_real = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_dist = fig.add_subplot(gs[1, :])

    for ax, emb, title in [
        (ax_real, true_umap, "Real expression\nas input"),
        (ax_pred, pred_umap, "Predicted expression\nas input"),
    ]:
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=14, alpha=0.82, linewidths=0)
        _label_cluster_centroids(ax, emb, clusters)
        ax.set_title(title, fontsize=28, pad=8)
        ax.set_xlabel("UMAP-1", fontsize=22)
        ax.set_ylabel("UMAP-2", fontsize=22)
        ax.tick_params(labelsize=15, width=1.6)
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=color_map[c],
            label=str(int(c)),
            markersize=10,
        )
        for c in unique
    ]
    ax_real.legend(
        handles=handles,
        title="Cluster",
        frameon=False,
        loc="center right",
        bbox_to_anchor=(1.02, 0.35),
    )

    plot_df = dist_df.copy()
    if plot_df.empty:
        ax_dist.text(
            0.5, 0.5, "No pairwise distance data available", ha="center", va="center"
        )
    else:
        max_distance = float(plot_df["distance"].max())
        if max_distance > 0:
            plot_df["distance"] = plot_df["distance"] / max_distance
        order = []
        labels = []
        palette = {}
        x = 0
        centers = []
        for cluster in unique:
            cluster_positions = []
            for relation in ["intra", "inter"]:
                for source in ["real", "predicted"]:
                    key = f"{int(cluster)}_{relation}_{source}"
                    order.append(key)
                    labels.append("R" if source == "real" else "P")
                    palette[key] = color_map[cluster]
                    plot_df.loc[
                        (plot_df["cluster"] == cluster)
                        & (plot_df["relation"] == relation)
                        & (plot_df["source"] == source),
                        "category",
                    ] = key
                    cluster_positions.append(x)
                    x += 1
                x += 0.18
            centers.append((cluster, float(np.mean(cluster_positions))))
            x += 0.75

        sns.violinplot(
            data=plot_df.dropna(subset=["category"]),
            x="category",
            y="distance",
            order=order,
            palette=palette,
            inner="box",
            cut=0,
            linewidth=0,
            ax=ax_dist,
        )
        ax_dist.set_xticklabels(labels, fontsize=18)
        ax_dist.set_xlabel("")
        ax_dist.set_ylabel("Normalized pairwise distance", fontsize=18)
        ax_dist.set_title(
            "Pairwise distances between cells within and outside of the clusters",
            fontsize=24,
            pad=14,
        )
        ax_dist.tick_params(axis="y", labelsize=15, width=1.5)
        ax_dist.set_ylim(0, 1.03)
        ax_dist.grid(axis="y", alpha=0.18)
        for cluster, center in centers:
            ax_dist.text(
                center,
                -0.18,
                f"Cluster {int(cluster)}",
                ha="center",
                va="top",
                fontsize=18,
            )
        for cluster in unique:
            base = order.index(f"{int(cluster)}_intra_real")
            ax_dist.text(base + 0.5, -0.09, "Intra", ha="center", va="top", fontsize=16)
            ax_dist.text(
                base + 2.68, -0.09, "Inter", ha="center", va="top", fontsize=16
            )
        for spine in ax_dist.spines.values():
            spine.set_linewidth(1.8)

    fig.text(0.015, 0.965, "a)", fontsize=26, fontweight="bold")
    fig.text(0.015, 0.535, "b)", fontsize=26, fontweight="bold")
    fig.savefig(
        out / "prediction_structure_composite.png", dpi=260, bbox_inches="tight"
    )
    plt.close(fig)


def _sample_pairs(
    indices_a: np.ndarray,
    indices_b: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
):
    if len(indices_a) == 0 or len(indices_b) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if np.array_equal(indices_a, indices_b):
        pairs = np.array(np.triu_indices(len(indices_a), k=1)).T
        if len(pairs) == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        if len(pairs) > max_pairs:
            pairs = pairs[rng.choice(len(pairs), max_pairs, replace=False)]
        return indices_a[pairs[:, 0]], indices_a[pairs[:, 1]]
    left = rng.choice(
        indices_a, size=min(max_pairs, len(indices_a) * len(indices_b)), replace=True
    )
    right = rng.choice(indices_b, size=len(left), replace=True)
    return left, right


def _distance_plots(
    x_true: np.ndarray, x_pred: np.ndarray, clusters: np.ndarray, out: Path, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    pair_rows = []
    for cluster in sorted(np.unique(clusters)):
        inside = np.where(clusters == cluster)[0]
        outside = np.where(clusters != cluster)[0]
        for relation, a, b in [
            ("intra", inside, inside),
            ("inter", inside, outside),
        ]:
            i, j = _sample_pairs(a, b, max_pairs=8000, rng=rng)
            if len(i) == 0:
                continue
            real_d = np.linalg.norm(x_true[i] - x_true[j], axis=1)
            pred_d = np.linalg.norm(x_pred[i] - x_pred[j], axis=1)
            for source, vals in [("real", real_d), ("predicted", pred_d)]:
                sample = (
                    vals if len(vals) <= 2500 else rng.choice(vals, 2500, replace=False)
                )
                rows.extend(
                    {
                        "cluster": int(cluster),
                        "relation": relation,
                        "source": source,
                        "distance": float(v),
                    }
                    for v in sample
                )
            pair_sample = np.arange(len(real_d))
            if len(pair_sample) > 4000:
                pair_sample = rng.choice(pair_sample, 4000, replace=False)
            pair_rows.extend(
                {
                    "cluster": int(cluster),
                    "relation": relation,
                    "real_distance": float(real_d[k]),
                    "predicted_distance": float(pred_d[k]),
                }
                for k in pair_sample
            )
    dist_df = pd.DataFrame(rows)
    pair_df = pd.DataFrame(pair_rows)

    intra_df = dist_df[dist_df["relation"] == "intra"].copy()

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.violinplot(
        data=intra_df,
        x="cluster",
        y="distance",
        hue="source",
        split=True,
        inner="quartile",
        cut=0,
        ax=ax,
    )
    ax.set_title("Within-cluster pairwise distances: real vs predicted RNA")
    ax.set_xlabel("Real-RNA cluster")
    ax.set_ylabel("Euclidean distance in standardized selected-gene space")
    fig.tight_layout()
    fig.savefig(
        out / "pairwise_distances_intra_real_vs_predicted.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    plot_df = dist_df.copy()
    plot_df["cluster_relation"] = (
        "c" + plot_df["cluster"].astype(str) + "_" + plot_df["relation"]
    )
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.boxplot(
        data=plot_df,
        x="cluster_relation",
        y="distance",
        hue="source",
        ax=ax,
        showfliers=False,
    )
    ax.set_title("Intra- and inter-cluster distances by real-RNA cluster")
    ax.set_xlabel("Real-RNA cluster and pair type")
    ax.set_ylabel("Euclidean distance")
    ax.tick_params(axis="x", rotation=45)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(
        out / "pairwise_distance_distributions_by_cluster.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, relation in zip(axes, ["intra", "inter"]):
        sub = pair_df[pair_df["relation"] == relation]
        sns.scatterplot(
            data=sub,
            x="real_distance",
            y="predicted_distance",
            hue="cluster",
            s=8,
            alpha=0.25,
            palette="tab10",
            ax=ax,
            legend=(relation == "inter"),
        )
        ax.set_title(f"{relation} pair distances")
        ax.set_xlabel("Real RNA distance")
        ax.set_ylabel("Predicted RNA distance")
        if not sub.empty:
            r = sub["real_distance"].corr(sub["predicted_distance"])
            ax.text(0.03, 0.95, f"r={r:.2f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(
        out / "pairwise_distance_real_vs_predicted_scatter.png",
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    summary = (
        dist_df.groupby(["cluster", "relation", "source"])["distance"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    pair_summary = (
        pair_df.groupby(["cluster", "relation"])
        .apply(
            lambda g: pd.Series(
                {
                    "distance_corr": g["real_distance"].corr(g["predicted_distance"]),
                    "pairs": len(g),
                }
            )
        )
        .reset_index()
    )
    summary.to_csv(out / "pairwise_distance_summary.csv", index=False)
    pair_summary.to_csv(out / "pairwise_distance_correlation_summary.csv", index=False)
    return dist_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", required=True, type=Path, help="Path to predictions_raw.csv"
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output directory for figures/tables",
    )
    parser.add_argument(
        "--split", default="all", help="Split to plot, or 'all' for all splits"
    )
    parser.add_argument(
        "--scatter-split",
        default="test",
        help="Split to use for the 50-gene scatter plot, or 'same' to use --split. Default: test",
    )
    parser.add_argument("--n-scatter-genes", type=int, default=50)
    parser.add_argument("--n-umap-genes", type=int, default=1000)
    parser.add_argument("--n-clusters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--predicted-umap-mode",
        choices=["independent", "project"],
        default="independent",
        help=(
            "Use 'independent' to fit a separate UMAP on predicted RNA, matching the "
            "real-expression-vs-predicted-expression input comparison. Use 'project' "
            "to transform predicted RNA through the real-RNA UMAP."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(args.predictions)
    required = {"split", "cell_id", "gene", "y_true", "y_pred"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(
            f"Missing required columns from predictions CSV: {sorted(missing)}"
        )
    all_preds = preds.copy()
    if args.split != "all":
        preds = preds[preds["split"] == args.split].copy()
    preds["y_true"] = pd.to_numeric(preds["y_true"], errors="coerce")
    preds["y_pred"] = pd.to_numeric(preds["y_pred"], errors="coerce")
    preds = preds.dropna(subset=["cell_id", "gene", "y_true", "y_pred"])

    scatter_split = args.split if args.scatter_split == "same" else args.scatter_split
    scatter_preds = (
        all_preds
        if scatter_split == "all"
        else all_preds[all_preds["split"] == scatter_split].copy()
    )
    scatter_preds["y_true"] = pd.to_numeric(scatter_preds["y_true"], errors="coerce")
    scatter_preds["y_pred"] = pd.to_numeric(scatter_preds["y_pred"], errors="coerce")
    scatter_preds = scatter_preds.dropna(subset=["cell_id", "gene", "y_true", "y_pred"])
    if scatter_preds.empty:
        raise ValueError(f"No predictions remained for scatter split '{scatter_split}'")

    scatter_genes = _select_genes(scatter_preds, args.n_scatter_genes)
    umap_genes = _select_genes(preds, args.n_umap_genes)
    true, pred = _pivot(preds, umap_genes)

    (
        x_true,
        x_pred,
        true_pca,
        pred_pca,
        true_umap,
        pred_umap,
        clusters,
        pred_clusters,
    ) = _fit_embeddings(
        true, pred, args.n_clusters, args.seed, args.predicted_umap_mode
    )
    ari = adjusted_rand_score(clusters, pred_clusters)
    sil_real = (
        silhouette_score(true_pca, clusters) if len(set(clusters)) > 1 else np.nan
    )
    sil_pred = (
        silhouette_score(pred_pca, clusters) if len(set(clusters)) > 1 else np.nan
    )

    _plot_gene_scatters(
        scatter_preds[scatter_preds["gene"].isin(scatter_genes)],
        scatter_genes,
        args.out_dir,
        scatter_split,
    )
    _plot_umaps(true_umap, pred_umap, clusters, args.out_dir)
    dist_df = _distance_plots(x_true, x_pred, clusters, args.out_dir, args.seed)
    _plot_composite_structure(true_umap, pred_umap, clusters, dist_df, args.out_dir)

    pd.DataFrame({"gene": scatter_genes}).to_csv(
        args.out_dir / "scatter_genes.csv", index=False
    )
    pd.DataFrame({"gene": list(true.columns)}).to_csv(
        args.out_dir / "umap_genes.csv", index=False
    )
    pd.DataFrame(
        {
            "cell_id": true.index,
            "real_cluster": clusters,
            "predicted_cluster": pred_clusters,
            "real_umap1": true_umap[:, 0],
            "real_umap2": true_umap[:, 1],
            "predicted_umap1": pred_umap[:, 0],
            "predicted_umap2": pred_umap[:, 1],
        }
    ).to_csv(args.out_dir / "cell_umap_clusters.csv", index=False)
    pd.DataFrame(
        [
            {
                "n_cells": true.shape[0],
                "n_genes_umap": true.shape[1],
                "n_scatter_genes": len(scatter_genes),
                "n_clusters": args.n_clusters,
                "predicted_umap_mode": args.predicted_umap_mode,
                "adjusted_rand_real_vs_predicted_kmeans": ari,
                "silhouette_real_pca_real_clusters": sil_real,
                "silhouette_predicted_pca_real_clusters": sil_pred,
            }
        ]
    ).to_csv(args.out_dir / "structure_summary.csv", index=False)
    print(f"Wrote outputs to {args.out_dir}")
    print(
        f"n_cells={true.shape[0]} n_genes={true.shape[1]} ARI={ari:.3f} sil_real={sil_real:.3f} sil_pred={sil_pred:.3f}"
    )


if __name__ == "__main__":
    main()
