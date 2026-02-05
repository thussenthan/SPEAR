
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from .metrics import regression_metrics

sns.set_style("whitegrid")


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
    sample_size: Optional[int] = 200_000,
    annotate_r2: bool = True,
    annotation_metrics: Optional[Dict[str, float]] = None,
) -> None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred)) & (~np.isinf(y_true)) & (~np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return

    if sample_size is not None and y_true.size > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(y_true.size, size=sample_size, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    if min_val == max_val:
        max_val = min_val + 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, s=10, alpha=0.3, edgecolor="none")
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="crimson", linewidth=1.5)
    plt.xlabel("Actual expression")
    plt.ylabel("Predicted expression")
    plt.title(title)
    if annotate_r2:
        if annotation_metrics is None:
            metrics = regression_metrics(y_true, y_pred)
            mean_override = False
        else:
            metrics = annotation_metrics
            mean_override = True
        def _fmt(value: float) -> str:
            return "nan" if not np.isfinite(value) else f"{value:.3f}"

        lines = []
        r2_val = metrics.get("r2") if metrics is not None else None
        if r2_val is not None:
            lines.append(f"$R^2={_fmt(r2_val)}$")
        pearson_val = metrics.get("pearson") if metrics is not None else None
        if pearson_val is not None:
            if mean_override:
                label = "Mean per-gene Pearson"
            else:
                label = "Pearson"
            lines.append(f"{label}={_fmt(pearson_val)}")
        spearman_val = metrics.get("spearman") if metrics is not None else None
        if spearman_val is not None:
            if mean_override:
                label = "Mean per-gene Spearman"
            else:
                label = "Spearman"
            lines.append(f"{label}={_fmt(spearman_val)}")
        text = "\n".join(lines)
        if text:
            plt.text(
                min_val + 0.05 * (max_val - min_val),
                max_val - 0.12 * (max_val - min_val),
                text,
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_residual_histogram(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str) -> None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residuals = y_pred - y_true
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel("Residual (prediction - actual)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: Iterable[str],
    output_path: Path,
    title: str,
    top_n: int = 30,
) -> None:
    importances = np.asarray(importances, dtype=np.float64)
    names = np.asarray(list(feature_names))
    if importances.size == 0 or names.size != importances.size:
        return

    ranked = np.argsort(importances)[::-1]
    limit = int(min(max(top_n, 1), ranked.size))
    ranked = ranked[:limit]
    sorted_importances = importances[ranked]
    sorted_names = names[ranked]

    fig_height = max(4.0, 0.35 * limit)
    fig, ax = plt.subplots(figsize=(9, fig_height))
    bars = ax.barh(np.arange(limit), sorted_importances, color="#4C72B0")
    ax.set_yticks(np.arange(limit))
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()  # Largest importance at top for easier scanning
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    if np.any(sorted_importances > 0):
        ax.set_xlim(left=0)
    for bar, value in zip(bars, sorted_importances):
        if value <= 0:
            continue
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_per_gene_feature_panel(
    block_df: pd.DataFrame,
    gene_name: str,
    output_path: Path,
    top_n: int = 12,
) -> None:
    if block_df.empty:
        return
    sanitized = block_df.copy()
    sanitized = sanitized.sort_values("importance_mean", ascending=False)
    subset = sanitized.head(max(1, top_n)).copy()
    has_distance = "signed_distance_to_tss_kb" in sanitized.columns and sanitized["signed_distance_to_tss_kb"].notna().any()
    cols = 2 if has_distance else 1
    fig_height = max(4.0, subset.shape[0] * 0.35)
    fig, axes = plt.subplots(1, cols, figsize=(cols * 5.5, fig_height))
    if cols == 1:
        axes = [axes]

    bar_ax = axes[0]
    y_pos = np.arange(subset.shape[0])
    bar_ax.barh(y_pos, subset["importance_mean"], color="#4C72B0")
    bar_ax.set_yticks(y_pos)
    bar_ax.set_yticklabels(subset["feature"].astype(str))
    bar_ax.invert_yaxis()
    bar_ax.set_xlabel("Importance")
    bar_ax.set_title(f"{gene_name} | top features")
    for idx, value in enumerate(subset["importance_mean"]):
        if value <= 0:
            continue
        bar_ax.text(value, idx, f" {value:.3f}", va="center", ha="left", fontsize=8)

    if has_distance:
        scatter_ax = axes[1]
        scatter_data = sanitized.dropna(subset=["signed_distance_to_tss_kb", "importance_mean"])
        scatter_ax.scatter(
            scatter_data["signed_distance_to_tss_kb"],
            scatter_data["importance_mean"],
            s=20,
            alpha=0.6,
            edgecolor="none",
            color="#1f77b4",
        )
        scatter_ax.axvline(0.0, linestyle="--", color="#d62728", linewidth=1.0)
        scatter_ax.set_xlabel("Signed distance to TSS (kb)")
        scatter_ax.set_ylabel("Importance")
        scatter_ax.set_title(f"{gene_name} | distance profile")

    fig.suptitle(f"Feature importance panel | {gene_name}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_metric_table(metrics: Dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["metric,value"] + [f"{name},{value}" for name, value in metrics.items()]
    output_path.write_text("\n".join(lines) + "\n")


def plot_residual_barplot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gene_names: Sequence[str],
    output_path: Path,
    title: str,
    top_n: int = 30,
) -> None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
        return
    if y_true.size == 0:
        return

    residuals = y_pred - y_true
    mae = np.nanmean(np.abs(residuals), axis=0)
    mean_res = np.nanmean(residuals, axis=0)

    if mae.size == 0:
        return

    order = np.argsort(mae)[::-1]
    limit = min(top_n, mae.size)
    idx = order[:limit]

    selected_genes = np.asarray(gene_names)[idx]
    selected_mae = mae[idx]
    selected_mean = mean_res[idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(4.0, limit * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bar_colors = ["#d62728" if val >= 0 else "#1f77b4" for val in selected_mean]
    y_positions = np.arange(limit)
    ax.barh(y_positions, selected_mae, color=bar_colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(selected_genes)
    ax.invert_yaxis()  # put largest residuals at the top for readability
    ax.set_xlabel("Mean absolute residual")
    ax.set_ylabel("Gene")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_correlation_boxplot(
    values: Sequence[float],
    output_path: Path,
    title: str,
    metric_label: str,
    axes: plt.Axes | None = None,
) -> None:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    arr = arr[mask]
    if arr.size == 0:
        return

    if axes is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6.5, 4.0))
        ax = plt.gca()
        save_and_close = True
    else:
        ax = axes
        save_and_close = False

    sns.boxplot(y=arr, color="#4C72B0", orient="v", ax=ax)
    sns.stripplot(y=arr, orient="v", color="#1f77b4", alpha=0.7, size=4, jitter=0.15, ax=ax)
    ax.set_ylabel(metric_label)
    ax.set_xticks([])
    ax.set_title(title)
    if save_and_close:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


def plot_correlation_violin(
    values: Sequence[float],
    output_path: Path,
    title: str,
    metric_label: str,
    axes: plt.Axes | None = None,
) -> None:
    arr = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(arr)
    arr = arr[mask]
    if arr.size == 0:
        return

    if axes is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6.5, 4.0))
        ax = plt.gca()
        save_and_close = True
    else:
        ax = axes
        save_and_close = False

    sns.violinplot(y=arr, color="#A6CEE3", inner="quartile", cut=0, linewidth=1.1, ax=ax)
    sns.stripplot(y=arr, orient="v", color="#1f77b4", alpha=0.45, size=3.5, jitter=0.18, ax=ax)
    ax.set_ylabel(metric_label)
    ax.set_xticks([])
    ax.set_title(title)
    if save_and_close:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()


def plot_importance_distance_scatter(
    importances: Sequence[float],
    distances_kb: Sequence[float],
    output_path: Path,
    title: str,
    annotation: Optional[Dict[str, float]] = None,
    *,
    max_distance_kb: float | None = 10.0,
    rolling_window: int = 51,
) -> None:
    imp = np.asarray(importances, dtype=np.float64)
    dist = np.asarray(distances_kb, dtype=np.float64)
    mask = np.isfinite(imp) & np.isfinite(dist)
    imp = imp[mask]
    dist = dist[mask]
    if max_distance_kb is not None:
        span_mask = np.abs(dist) <= max_distance_kb
        imp = imp[span_mask]
        dist = dist[span_mask]
    if imp.size == 0:
        return

    order = np.argsort(dist)
    dist_sorted = dist[order]
    imp_sorted = imp[order]
    rolling = pd.Series(imp_sorted).rolling(window=rolling_window, min_periods=10).median()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=dist_sorted, y=imp_sorted, s=12, alpha=0.35, edgecolor="none")
    sns.lineplot(
        x=dist_sorted,
        y=rolling,
        color="#d62728",
        linewidth=1.4,
        label="Rolling median",
    )
    plt.axvline(0.0, linestyle="--", color="#444", linewidth=1.0, alpha=0.7)
    if max_distance_kb is not None:
        plt.xlim(-max_distance_kb, max_distance_kb)
    plt.xlabel("Distance to TSS (kb)")
    plt.ylabel("Feature importance")
    plt.title(title)
    if annotation:
        text = "\n".join(f"{k}={v:.3f}" for k, v in annotation.items() if np.isfinite(v))
        if text:
            plt.text(
                0.03,
                0.97,
                text,
                transform=plt.gca().transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_cumulative_importance_overlay(
    importances: Sequence[float],
    distances_kb: Sequence[float],
    output_path: Path,
    title: str,
) -> None:
    imp = np.asarray(importances, dtype=np.float64)
    dist = np.asarray(distances_kb, dtype=np.float64)
    mask = np.isfinite(imp) & np.isfinite(dist)
    imp = imp[mask]
    dist = np.abs(dist[mask])
    if imp.size == 0:
        return
    order = np.argsort(dist)
    dist = dist[order]
    imp = imp[order]
    imp = np.clip(imp, a_min=0.0, a_max=None)
    total = imp.sum()
    if total <= 0:
        cumulative = np.linspace(0.0, 1.0, imp.size)
    else:
        cumulative = np.cumsum(imp) / total

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(dist, cumulative, color="#1f77b4", linewidth=2.0)
    ax.fill_between(dist, cumulative, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("|Distance to TSS| (kb)")
    ax.set_ylabel("Cumulative importance fraction")
    ax.set_title(title)
    if dist.size:
        # Pick the 90th percentile index safely (clamp to valid range for small arrays)
        idx = max(0, min(len(dist) - 1, int(0.9 * len(dist)) - 1))
        ninety = float(dist[idx])
        ax.text(
            0.02,
            0.08,
            f"90% of ranked features within ~{ninety:.1f} kb",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
    ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_training_history_curves(
    history: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    include_train: bool = True,
    include_val: bool = True,
) -> None:
    if history.empty or "epoch" not in history.columns:
        return

    metric_clean = metric.lower()
    if metric_clean == "loss":
        train_col = "train_loss"
        val_col = "val_loss"
    else:
        train_col = f"train_{metric_clean}"
        val_col = f"val_{metric_clean}"

    curves = []
    labels = []
    if include_train and train_col in history:
        curves.append((history[train_col], "Train"))
    if include_val and val_col in history:
        curves.append((history[val_col], "Validation"))

    if not curves:
        return

    # Skip plots where all series are constant or all-NaN.
    def _is_constant_or_nan(series: pd.Series) -> bool:
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric.dropna()
        if finite.empty:
            return True
        return finite.nunique(dropna=True) <= 1

    if all(_is_constant_or_nan(series) for series, _ in curves):
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.5))
    for series, label in curves:
        plt.plot(history["epoch"], series, label=label, linewidth=2.0)
    plt.xlabel("Epoch")
    ylabel = "Loss" if metric_clean == "loss" else metric_clean.capitalize()
    plt.ylabel(ylabel)
    plt.title(title)
    if len(curves) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_training_history_series(
    history: pd.DataFrame,
    column: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    if history.empty or "epoch" not in history.columns or column not in history.columns:
        return

    series = pd.to_numeric(history[column], errors="coerce")
    finite = series.dropna()
    if finite.empty:
        return
    if finite.nunique(dropna=True) <= 1:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.5))
    plt.plot(history["epoch"], history[column], linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
