
import json
import logging
import math
import os
import re
import signal
import time
import traceback
from collections import OrderedDict, defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch

from .config import PipelineConfig, TrainingConfig
from .data import (
    GeneInfo,
    PeakIndexer,
    build_cellwise_dataset,
    build_cellwise_features_only,
    build_gene_dataset,
    filter_atac_by_genes,
    load_datasets,
    preprocess_modalities,
    parse_gtf,
    select_genes,
)
from .logging_utils import ResourceUsageTracker, get_logger
from .training import CellwiseModelResult, ModelResult, train_model_for_gene, train_multi_output_model, get_resource_summary
from . import predict
from .wandb_utils import (
    apply_sweep_overrides,
    infer_dataset_name,
    log_images_from_globs,
    log_metric_distribution_charts_from_csv,
    log_prediction_charts_from_csv,
    log_training_history_charts_from_csv,
    log_run_artifacts,
    log_tables_from_csv,
    maybe_init_wandb,
    wandb_finish,
    wandb_update_config,
    wandb_update_summary,
)
from .visualization import (
    plot_feature_importance,
    plot_correlation_boxplot,
    plot_correlation_violin,
    plot_predictions_vs_actual,
    plot_residual_barplot_by_split,
    plot_residual_histogram_by_split,
    plot_box_violin_half_split,
    plot_single_box_violin,
    plot_training_history_curves,
    plot_importance_distance_scatter,
    plot_per_gene_feature_panel,
    plot_cumulative_importance_overlay,
)

_LOG = get_logger(__name__)

_FEATURE_BIN_PATTERN = re.compile(r"bin_(-?\d+)_to_(-?\d+)", re.IGNORECASE)

# Regex to parse genomic interval notation: chr<name>:<start>-<end>
# Expected format examples: 'chr1:1000-2000', 'chrX:500000-600000', 'chrMT:100-200'
# Captures: (1) chromosome name, (2) start position, (3) end position
_FEATURE_INTERVAL_PATTERN = re.compile(r"^(chr[A-Za-z0-9_]+):(\d+)-(\d+)$", re.IGNORECASE)
_METRIC_ORDER = ("pearson", "r2", "spearman", "rmse", "mse", "mae")


def _compute_model_metric_summary(output_dir: Path) -> Dict[str, Dict[str, float | int]]:
    metrics = ("pearson", "r2", "spearman", "rmse", "mse", "mae")
    values_by_metric: Dict[str, List[float]] = {metric: [] for metric in metrics}

    metrics_per_gene_path = output_dir / "metrics_per_gene.csv"
    metrics_by_gene_path = output_dir / "metrics_by_gene.csv"
    metrics_aggregate_path = output_dir / "metrics_aggregate.csv"

    try:
        if metrics_per_gene_path.exists():
            df = pd.read_csv(metrics_per_gene_path)
            for metric in metrics:
                if metric not in df.columns:
                    continue
                series = pd.to_numeric(df[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not series.empty:
                    values_by_metric[metric].extend(float(v) for v in series.to_numpy())
        elif metrics_by_gene_path.exists():
            df = pd.read_csv(metrics_by_gene_path)
            for metric in metrics:
                cols = [f"train_{metric}", f"val_{metric}", f"test_{metric}"]
                for col in cols:
                    if col not in df.columns:
                        continue
                    series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                    if not series.empty:
                        values_by_metric[metric].extend(float(v) for v in series.to_numpy())
        elif metrics_aggregate_path.exists():
            df = pd.read_csv(metrics_aggregate_path)
            for metric in metrics:
                if metric not in df.columns:
                    continue
                series = pd.to_numeric(df[metric], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if not series.empty:
                    values_by_metric[metric].extend(float(v) for v in series.to_numpy())
    except Exception:
        _LOG.debug("Failed to compute model metric summary from %s", output_dir, exc_info=True)
        return {}

    summary: Dict[str, Dict[str, float | int]] = {}
    for metric, values in values_by_metric.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[metric] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr, ddof=0)),
            "count": int(arr.size),
        }
    return summary


def _flatten_numeric_metrics(payload: Dict[str, Any], *, prefix: str) -> Dict[str, float]:
    flattened: Dict[str, float] = {}

    def _visit(value: Any, key_prefix: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                _visit(v, f"{key_prefix}/{k}" if key_prefix else str(k))
            return
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            flattened[key_prefix] = float(value)

    _visit(payload, prefix)
    return flattened


def _compact_shap_metric_payload(shap_summary: Dict[str, Any]) -> Dict[str, float]:
    """Build short SHAP metric keys for W&B summary (e.g., ``shap.mse_mean``)."""

    payload: Dict[str, float] = {}
    num_features = shap_summary.get("num_features")
    if isinstance(num_features, (int, float, np.integer, np.floating)) and np.isfinite(float(num_features)):
        payload["shap.num_features"] = float(num_features)

    mean_abs_sum = shap_summary.get("shap_mean_abs_sum")
    if isinstance(mean_abs_sum, (int, float, np.integer, np.floating)) and np.isfinite(float(mean_abs_sum)):
        payload["shap.mean_abs_sum"] = float(mean_abs_sum)

    mean_sum = shap_summary.get("shap_mean_sum")
    if isinstance(mean_sum, (int, float, np.integer, np.floating)) and np.isfinite(float(mean_sum)):
        payload["shap.mean_sum"] = float(mean_sum)

    mean_abs_top1 = shap_summary.get("shap_mean_abs_top1")
    if isinstance(mean_abs_top1, (int, float, np.integer, np.floating)) and np.isfinite(float(mean_abs_top1)):
        payload["shap.mean_abs_top1"] = float(mean_abs_top1)

    mean_top1 = shap_summary.get("shap_mean_top1")
    if isinstance(mean_top1, (int, float, np.integer, np.floating)) and np.isfinite(float(mean_top1)):
        payload["shap.mean_top1"] = float(mean_top1)

    mean_signed_sum = shap_summary.get("shap_mean_signed_sum")
    if isinstance(mean_signed_sum, (int, float, np.integer, np.floating)) and np.isfinite(float(mean_signed_sum)):
        payload["shap.mean_signed_sum"] = float(mean_signed_sum)

    mean_signed_top_positive = shap_summary.get("shap_mean_signed_top_positive")
    if isinstance(mean_signed_top_positive, (int, float, np.integer, np.floating)) and np.isfinite(float(mean_signed_top_positive)):
        payload["shap.mean_signed_top_positive"] = float(mean_signed_top_positive)

    mean_signed_top_negative = shap_summary.get("shap_mean_signed_top_negative")
    if isinstance(mean_signed_top_negative, (int, float, np.integer, np.floating)) and np.isfinite(float(mean_signed_top_negative)):
        payload["shap.mean_signed_top_negative"] = float(mean_signed_top_negative)

    per_gene_count = shap_summary.get("per_gene_count")
    if isinstance(per_gene_count, (int, float, np.integer, np.floating)) and np.isfinite(float(per_gene_count)):
        payload["shap.per_gene_count"] = float(per_gene_count)

    per_gene_mean_abs_sum = shap_summary.get("per_gene_mean_abs_sum")
    if isinstance(per_gene_mean_abs_sum, (int, float, np.integer, np.floating)) and np.isfinite(float(per_gene_mean_abs_sum)):
        payload["shap.per_gene_mean_abs_sum"] = float(per_gene_mean_abs_sum)

    per_gene_mean_sum = shap_summary.get("per_gene_mean_signed_sum")
    if isinstance(per_gene_mean_sum, (int, float, np.integer, np.floating)) and np.isfinite(float(per_gene_mean_sum)):
        payload["shap.per_gene_mean_sum"] = float(per_gene_mean_sum)

    top10_weight_share = shap_summary.get("top10_weight_share")
    if isinstance(top10_weight_share, (int, float, np.integer, np.floating)) and np.isfinite(float(top10_weight_share)):
        payload["shap.top10_weight_share"] = float(top10_weight_share)

    tss_near_2kb_share = shap_summary.get("tss_near_2kb_share")
    if isinstance(tss_near_2kb_share, (int, float, np.integer, np.floating)) and np.isfinite(float(tss_near_2kb_share)):
        payload["shap.tss_near_2kb_share"] = float(tss_near_2kb_share)

    tss_corr = shap_summary.get("tss_correlation")
    if isinstance(tss_corr, dict):
        corr_pearson = tss_corr.get("pearson")
        if isinstance(corr_pearson, (int, float, np.integer, np.floating)) and np.isfinite(float(corr_pearson)):
            payload["shap.tss_corr_pearson"] = float(corr_pearson)
        corr_spearman = tss_corr.get("spearman")
        if isinstance(corr_spearman, (int, float, np.integer, np.floating)) and np.isfinite(float(corr_spearman)):
            payload["shap.tss_corr_spearman"] = float(corr_spearman)
        corr_count = tss_corr.get("count")
        if isinstance(corr_count, (int, float, np.integer, np.floating)) and np.isfinite(float(corr_count)):
            payload["shap.tss_corr_count"] = float(corr_count)

    model_metrics = shap_summary.get("model_metrics")
    if not isinstance(model_metrics, dict):
        return payload

    for metric_name, metric_stats in model_metrics.items():
        if not isinstance(metric_stats, dict):
            continue
        metric_token = str(metric_name).strip().lower()
        if not metric_token:
            continue
        for stat_name in ("mean", "std", "count"):
            value = metric_stats.get(stat_name)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                payload[f"shap.{metric_token}_{stat_name}"] = float(value)
    return payload


def _compact_feature_importance_metric_payload(fi_summary: Dict[str, Any]) -> Dict[str, float]:
    """Build short FI metric keys (e.g., ``fi.pearson_mean``)."""

    payload: Dict[str, float] = {}
    num_features = fi_summary.get("num_features")
    if isinstance(num_features, (int, float, np.integer, np.floating)) and np.isfinite(float(num_features)):
        payload["fi.num_features"] = float(num_features)

    importance_mean_sum = fi_summary.get("importance_mean_sum")
    if isinstance(importance_mean_sum, (int, float, np.integer, np.floating)) and np.isfinite(float(importance_mean_sum)):
        payload["fi.mean_sum"] = float(importance_mean_sum)

    importance_mean_abs_sum = fi_summary.get("importance_mean_abs_sum")
    if isinstance(importance_mean_abs_sum, (int, float, np.integer, np.floating)) and np.isfinite(float(importance_mean_abs_sum)):
        payload["fi.mean_abs_sum"] = float(importance_mean_abs_sum)

    importance_mean_top1 = fi_summary.get("importance_mean_top1")
    if isinstance(importance_mean_top1, (int, float, np.integer, np.floating)) and np.isfinite(float(importance_mean_top1)):
        payload["fi.mean_top1"] = float(importance_mean_top1)

    importance_mean_abs_top1 = fi_summary.get("importance_mean_abs_top1")
    if isinstance(importance_mean_abs_top1, (int, float, np.integer, np.floating)) and np.isfinite(float(importance_mean_abs_top1)):
        payload["fi.mean_abs_top1"] = float(importance_mean_abs_top1)

    importance_mean_signed_sum = fi_summary.get("importance_mean_signed_sum")
    if isinstance(importance_mean_signed_sum, (int, float, np.integer, np.floating)) and np.isfinite(float(importance_mean_signed_sum)):
        payload["fi.mean_signed_sum"] = float(importance_mean_signed_sum)

    importance_mean_signed_top_positive = fi_summary.get("importance_mean_signed_top_positive")
    if isinstance(importance_mean_signed_top_positive, (int, float, np.integer, np.floating)) and np.isfinite(float(importance_mean_signed_top_positive)):
        payload["fi.mean_signed_top_positive"] = float(importance_mean_signed_top_positive)

    importance_mean_signed_top_negative = fi_summary.get("importance_mean_signed_top_negative")
    if isinstance(importance_mean_signed_top_negative, (int, float, np.integer, np.floating)) and np.isfinite(float(importance_mean_signed_top_negative)):
        payload["fi.mean_signed_top_negative"] = float(importance_mean_signed_top_negative)

    per_gene_count = fi_summary.get("per_gene_count")
    if isinstance(per_gene_count, (int, float, np.integer, np.floating)) and np.isfinite(float(per_gene_count)):
        payload["fi.per_gene_count"] = float(per_gene_count)

    per_gene_mean_importance_sum = fi_summary.get("per_gene_mean_importance_sum")
    if isinstance(per_gene_mean_importance_sum, (int, float, np.integer, np.floating)) and np.isfinite(float(per_gene_mean_importance_sum)):
        payload["fi.per_gene_mean_sum"] = float(per_gene_mean_importance_sum)

    per_gene_mean_abs_sum = fi_summary.get("per_gene_mean_abs_sum")
    if isinstance(per_gene_mean_abs_sum, (int, float, np.integer, np.floating)) and np.isfinite(float(per_gene_mean_abs_sum)):
        payload["fi.per_gene_mean_abs_sum"] = float(per_gene_mean_abs_sum)

    top10_weight_share = fi_summary.get("top10_weight_share")
    if isinstance(top10_weight_share, (int, float, np.integer, np.floating)) and np.isfinite(float(top10_weight_share)):
        payload["fi.top10_weight_share"] = float(top10_weight_share)

    tss_near_2kb_share = fi_summary.get("tss_near_2kb_share")
    if isinstance(tss_near_2kb_share, (int, float, np.integer, np.floating)) and np.isfinite(float(tss_near_2kb_share)):
        payload["fi.tss_near_2kb_share"] = float(tss_near_2kb_share)

    tss_corr = fi_summary.get("tss_correlation")
    if isinstance(tss_corr, dict):
        corr_pearson = tss_corr.get("pearson")
        if isinstance(corr_pearson, (int, float, np.integer, np.floating)) and np.isfinite(float(corr_pearson)):
            payload["fi.tss_corr_pearson"] = float(corr_pearson)
        corr_spearman = tss_corr.get("spearman")
        if isinstance(corr_spearman, (int, float, np.integer, np.floating)) and np.isfinite(float(corr_spearman)):
            payload["fi.tss_corr_spearman"] = float(corr_spearman)
        corr_count = tss_corr.get("count")
        if isinstance(corr_count, (int, float, np.integer, np.floating)) and np.isfinite(float(corr_count)):
            payload["fi.tss_corr_count"] = float(corr_count)

    model_metrics = fi_summary.get("model_metrics")
    if not isinstance(model_metrics, dict):
        return payload

    for metric_name, metric_stats in model_metrics.items():
        if not isinstance(metric_stats, dict):
            continue
        metric_token = str(metric_name).strip().lower()
        if not metric_token:
            continue
        for stat_name in ("mean", "std", "count"):
            value = metric_stats.get(stat_name)
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                payload[f"fi.{metric_token}_{stat_name}"] = float(value)
    return payload


def _feature_name_metadata(feature_name: str) -> Dict[str, object]:
    """Parse common naming schemes to attach TSS-relative metadata."""

    if not feature_name:
        return {
            "feature_class": "unknown",
        }

    gene_name: Optional[str] = None
    if "|" in feature_name:
        gene_name, token = feature_name.split("|", 1)
    else:
        token = feature_name
    lowered = token.lower()
    meta: Dict[str, object] = {
        "feature_token": token,
        "feature_class": "unknown",
    }
    if gene_name:
        meta["gene_name"] = gene_name

    if "peak" in lowered:
        meta["feature_class"] = "atac_peak"

    interval_match = _FEATURE_INTERVAL_PATTERN.match(token)
    if interval_match:
        chrom, start_str, end_str = interval_match.groups()
        start = int(start_str)
        end = int(end_str)
        meta.update(
            {
                "feature_class": "atac_bin",
                "chrom": chrom,
                "genomic_start_bp": start,
                "genomic_end_bp": end,
                "genomic_center_bp": (start + end) / 2.0,
            }
        )
    else:
        # Only check bin pattern if interval pattern did not match
        bin_match = _FEATURE_BIN_PATTERN.search(token)
        if bin_match:
            start = int(bin_match.group(1))
            end = int(bin_match.group(2))
            center = (start + end) / 2.0
            meta.update(
                {
                    "feature_class": "atac_bin",
                    "relative_start_bp": start,
                    "relative_end_bp": end,
                    "relative_center_bp": center,
                    "delta_to_tss_bp": center,
                    "distance_to_tss_bp": abs(center),
                    "delta_to_tss_kb": center / 1_000.0,
                    # Preserve a signed distance for plotting/correlation; keep abs variant for convenience
                    "signed_distance_to_tss_kb": center / 1_000.0,
                    "distance_to_tss_abs_kb": abs(center) / 1_000.0,
                }
            )

    return meta


def _parse_genomic_interval(feature_name: str) -> Optional[Tuple[str, int, int]]:
    if not feature_name:
        return None
    token = feature_name.split("|", 1)[-1]
    match = _FEATURE_INTERVAL_PATTERN.match(token)
    if not match:
        return None
    chrom, start_str, end_str = match.groups()
    return chrom, int(start_str), int(end_str)


def _infer_signed_distance_from_blocks(
    feature_names: Sequence[str],
    gene_names: Optional[Sequence[str]],
    feature_block_indices: Optional[Sequence[Sequence[int]]],
    gene_infos: Optional[Sequence[GeneInfo]],
) -> Optional[np.ndarray]:
    if not feature_names or not gene_names or not feature_block_indices or not gene_infos:
        return None

    feature_count = len(feature_names)
    distance_lists: List[List[float]] = [[] for _ in range(feature_count)]
    gene_info_by_name = {g.gene_name: g for g in gene_infos}
    limit = min(len(gene_names), len(feature_block_indices))

    for idx in range(limit):
        gene_name = str(gene_names[idx])
        gene_info = gene_info_by_name.get(gene_name)
        if gene_info is None:
            continue
        block_indices = np.asarray(feature_block_indices[idx], dtype=np.int64)
        if block_indices.size == 0:
            continue
        valid_mask = (block_indices >= 0) & (block_indices < feature_count)
        block_indices = block_indices[valid_mask]
        if block_indices.size == 0:
            continue
        for feat_idx in np.unique(block_indices):
            parsed = _parse_genomic_interval(str(feature_names[int(feat_idx)]))
            if not parsed:
                continue
            chrom, start, end = parsed
            if chrom != gene_info.chrom:
                continue
            rel_start = start - gene_info.tss
            rel_end = end - gene_info.tss
            if gene_info.strand == "-":
                rel_start, rel_end = -rel_end, -rel_start
            center_kb = ((rel_start + rel_end) / 2.0) / 1_000.0
            distance_lists[int(feat_idx)].append(float(center_kb))

    inferred = np.full(feature_count, np.nan, dtype=np.float64)
    filled = 0
    for feat_idx, values in enumerate(distance_lists):
        if values:
            inferred[feat_idx] = float(np.mean(values))
            filled += 1
    if filled == 0:
        return None
    return inferred


def _write_placeholder_plot(output_path: Path, title: str, message: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _export_feature_importance_artifacts(
    output_dir: Path,
    model_name: str,
    importances: np.ndarray,
    feature_names: Sequence[str],
    *,
    feature_importance_mean_signed: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    export_per_gene_panels: bool = False,
    gene_names: Optional[Sequence[str]] = None,
    feature_block_slices: Optional[Sequence[Tuple[int, int]]] = None,
    feature_block_indices: Optional[Sequence[Sequence[int]]] = None,
    gene_infos: Optional[Sequence[GeneInfo]] = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fi = np.asarray(importances, dtype=np.float64)
    feature_count = int(fi.size if fi.ndim == 1 else fi.shape[-1])
    if feature_count == 0:
        _LOG.info("Feature importance export skipped | model=%s | reason=no features", model_name)
        return {}

    start_wall = time.perf_counter()
    start_ts = datetime.now(timezone.utc).isoformat()
    _LOG.info(
        "Feature importance export start | model=%s | features=%d | output_dir=%s | timestamp=%s",
        model_name,
        feature_count,
        output_dir,
        start_ts,
    )

    if fi.ndim == 1:
        fi_stack = fi[None, :]
    else:
        fi_stack = fi

    fi_mean = np.nanmean(fi_stack, axis=0)
    fi_std = np.nanstd(fi_stack, axis=0, ddof=0)
    fi_median = np.nanmedian(fi_stack, axis=0)

    raw_path = output_dir / "feature_importances_raw.npz"
    np.savez_compressed(
        raw_path,
        importances=fi_stack,
        feature_names=np.asarray(feature_names),
    )
    _LOG.info(
        "Saved raw feature importances (%s features, stack shape=%s) to %s",
        fi_stack.shape[-1],
        tuple(fi_stack.shape),
        raw_path,
    )

    plot_feature_importance(
        fi_mean,
        feature_names,
        output_dir / "feature_importance_mean.png",
        f"Feature importance | {model_name.upper()}",
    )

    metadata_records = [_feature_name_metadata(name) for name in feature_names]
    metadata_df = pd.DataFrame(metadata_records) if metadata_records else None

    aggregate_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": fi_mean,
            "importance_std": fi_std,
            "importance_median": fi_median,
        }
    )
    fi_signed: Optional[np.ndarray] = None
    if feature_importance_mean_signed is not None:
        signed_candidate = np.asarray(feature_importance_mean_signed, dtype=np.float64).reshape(-1)
        if signed_candidate.shape[0] == fi_mean.shape[0]:
            fi_signed = signed_candidate
            aggregate_df["importance_mean_signed"] = signed_candidate
    if metadata_df is not None and not metadata_df.empty:
        aggregate_df = pd.concat([aggregate_df, metadata_df], axis=1)
    if "signed_distance_to_tss_kb" not in aggregate_df.columns or not aggregate_df["signed_distance_to_tss_kb"].notna().any():
        inferred_dist = _infer_signed_distance_from_blocks(
            feature_names,
            gene_names,
            feature_block_indices,
            gene_infos,
        )
        if inferred_dist is not None:
            aggregate_df["signed_distance_to_tss_kb"] = inferred_dist

    aggregate_path = output_dir / "feature_importances_mean.csv"
    aggregate_df.to_csv(aggregate_path, index=False)
    _LOG.info(
        "Saved aggregate feature importance stats (%d rows) to %s",
        aggregate_df.shape[0],
        aggregate_path,
    )

    if "feature_class" in aggregate_df.columns:
        class_counts = aggregate_df["feature_class"].value_counts(dropna=False).head(5)
        if not class_counts.empty:
            breakdown = ", ".join(f"{str(cls)}:{int(cnt)}" for cls, cnt in class_counts.items())
            _LOG.info("Feature class breakdown | %s | %s", model_name, breakdown)

    top_feature_rows = aggregate_df.sort_values("importance_mean", ascending=False).head(5)
    if not top_feature_rows.empty:
        top_summary = ", ".join(
            f"{row.feature}={row.importance_mean:.4f}"
            for row in top_feature_rows.itertuples()
        )
        _LOG.info("Top feature importances | %s | %s", model_name, top_summary)

    fi_signed_plot_path: Optional[Path] = None
    if fi_signed is not None:
        signed_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_mean_signed": fi_signed,
            }
        )
        signed_df = signed_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["importance_mean_signed"])
        if not signed_df.empty:
            top_pos = signed_df.sort_values("importance_mean_signed", ascending=False).head(10)
            top_neg = signed_df.sort_values("importance_mean_signed", ascending=True).head(10)
            signed_top = pd.concat([top_neg, top_pos], axis=0).drop_duplicates(subset=["feature"], keep="first")
            if not signed_top.empty:
                fi_signed_plot_path = output_dir / "feature_importance_mean_signed.png"
                signed_top = signed_top.sort_values("importance_mean_signed", ascending=True)
                fig, ax = plt.subplots(figsize=(9.0, 6.0))
                bar_colors = ["#1f77b4" if v < 0 else "#d62728" for v in signed_top["importance_mean_signed"]]
                ax.barh(signed_top["feature"], signed_top["importance_mean_signed"], color=bar_colors)
                ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1)
                ax.set_xlabel("Mean signed feature importance")
                ax.set_ylabel("Feature")
                ax.set_title(f"Mean signed feature importance | {model_name.upper()}")
                plt.tight_layout()
                fig.savefig(fi_signed_plot_path, dpi=300)
                plt.close(fig)
                _LOG.info("Saved signed feature importance plot to %s", fi_signed_plot_path)

    per_gene_summary_path: Optional[Path] = None
    fi_per_gene_count: Optional[int] = None
    fi_per_gene_mean_importance_sum: Optional[float] = None
    fi_per_gene_mean_abs_sum: Optional[float] = None
    if feature_block_indices and gene_names:
        per_gene_records = []
        gene_block_indices: Dict[str, np.ndarray] = {}
        limit = min(len(feature_block_indices), len(gene_names))
        for idx in range(limit):
            indices = np.asarray(feature_block_indices[idx], dtype=np.int64)
            if indices.size == 0:
                continue
            valid_mask = (indices >= 0) & (indices < len(feature_names))
            invalid_mask = ~valid_mask
            if invalid_mask.any():
                invalid_indices = indices[invalid_mask]
                _LOG.warning(
                    "Out-of-bounds feature indices detected for gene block %d: %d invalid "
                    "indices (min=%s, max=%s) outside valid range [0, %d). These indices "
                    "will be ignored.",
                    idx,
                    invalid_indices.size,
                    int(invalid_indices.min()) if invalid_indices.size else "n/a",
                    int(invalid_indices.max()) if invalid_indices.size else "n/a",
                    len(feature_names),
                )
                continue
            indices = indices[valid_mask]
            if indices.size == 0:
                continue
            block = aggregate_df.iloc[indices].copy()
            if block.empty:
                continue
            gene_label = gene_names[idx]
            gene_block_indices[gene_label] = indices
            gene_info = gene_infos[idx] if gene_infos is not None and idx < len(gene_infos) else None
            record = {
                "gene": gene_label,
                "feature_count": int(block.shape[0]),
                "importance_mean_sum": float(block["importance_mean"].sum()),
                "importance_mean_abs_sum": float(np.abs(block["importance_mean"]).sum()),
                "importance_mean_avg": float(block["importance_mean"].mean()),
                "top_feature": str(block.loc[block["importance_mean"].idxmax(), "feature"]),
                "top_feature_importance": float(block["importance_mean"].max()),
            }
            if "importance_mean_signed" in block.columns:
                record["importance_mean_signed_sum"] = float(block["importance_mean_signed"].sum())
                record["importance_mean_signed_avg"] = float(block["importance_mean_signed"].mean())
            distances = None
            if "signed_distance_to_tss_kb" in block.columns and block["signed_distance_to_tss_kb"].notna().any():
                distances = pd.to_numeric(block["signed_distance_to_tss_kb"], errors="coerce")
            elif gene_info is not None:
                rel_centers = np.full(block.shape[0], np.nan, dtype=float)
                for row_idx, feature in enumerate(block["feature"].astype(str)):
                    parsed = _parse_genomic_interval(feature)
                    if not parsed:
                        continue
                    chrom, start, end = parsed
                    if chrom != gene_info.chrom:
                        continue
                    rel_start = start - gene_info.tss
                    rel_end = end - gene_info.tss
                    if gene_info.strand == "-":
                        rel_start, rel_end = -rel_end, -rel_start
                    rel_centers[row_idx] = (rel_start + rel_end) / 2.0
                distances = pd.Series(rel_centers / 1_000.0)
                block = block.copy()
                block["signed_distance_to_tss_kb"] = distances
            if distances is not None:
                mask = np.isfinite(distances) & np.isfinite(block["importance_mean"])
                if mask.any():
                    imp = block.loc[mask, "importance_mean"]
                    dist = distances[mask]
                    if imp.nunique() > 1 and dist.nunique() > 1:
                        record["pearson_distance_corr"] = float(imp.corr(dist, method="pearson"))
                        record["spearman_distance_corr"] = float(imp.corr(dist, method="spearman"))
                    top_idx = block.loc[mask, "importance_mean"].idxmax()
                    record["top_feature_distance_kb"] = float(distances.loc[top_idx])
            per_gene_records.append(record)
        if per_gene_records:
            per_gene_df = pd.DataFrame(per_gene_records)
            per_gene_summary_path = output_dir / "feature_importance_per_gene_summary.csv"
            per_gene_df.to_csv(per_gene_summary_path, index=False)
            fi_per_gene_count = int(per_gene_df.shape[0])
            fi_per_gene_mean_importance_sum = float(
                pd.to_numeric(per_gene_df["importance_mean_sum"], errors="coerce").mean(skipna=True)
            )
            if "importance_mean_abs_sum" in per_gene_df.columns:
                fi_per_gene_mean_abs_sum = float(
                    pd.to_numeric(per_gene_df["importance_mean_abs_sum"], errors="coerce").mean(skipna=True)
                )
            _LOG.info(
                "Saved per-gene feature importance summary (%d genes) to %s",
                per_gene_df.shape[0],
                per_gene_summary_path,
            )

            if export_per_gene_panels:
                panel_dir = output_dir / "per_gene_panels"
                panel_candidates = per_gene_df.sort_values("importance_mean_sum", ascending=False).head(12)
                generated = 0
                gene_info_map = {g.gene_name: g for g in gene_infos} if gene_infos is not None else {}
                for gene_value in panel_candidates["gene"]:
                    block_indices = gene_block_indices.get(gene_value)
                    if block_indices is None or block_indices.size == 0:
                        continue
                    block_slice = aggregate_df.iloc[block_indices].copy()
                    if block_slice.empty:
                        continue
                    gene_info = gene_info_map.get(gene_value)
                    if gene_info is not None:
                        # Compute relative centers only for features on the same chromosome
                        rel_centers = np.full(block_slice.shape[0], np.nan, dtype=float)
                        for row_idx, feature in enumerate(block_slice["feature"].astype(str)):
                            parsed = _parse_genomic_interval(feature)
                            if not parsed:
                                continue
                            chrom, start, end = parsed
                            if chrom != gene_info.chrom:
                                continue
                            rel_start = start - gene_info.tss
                            rel_end = end - gene_info.tss
                            if gene_info.strand == "-":
                                rel_start, rel_end = -rel_end, -rel_start
                            rel_centers[row_idx] = (rel_start + rel_end) / 2.0
                        block_slice["signed_distance_to_tss_kb"] = rel_centers / 1_000.0
                    safe_gene = re.sub(r"[^A-Za-z0-9._-]", "_", gene_value)
                    panel_path = panel_dir / f"{safe_gene}.png"
                    plot_per_gene_feature_panel(block_slice, gene_value, panel_path)
                    generated += 1
                if generated:
                    _LOG.info("Generated %d per-gene feature panels in %s", generated, panel_dir)
    elif feature_block_slices and gene_names:
        per_gene_records = []
        gene_block_ranges: Dict[str, Tuple[int, int]] = {}
        limit = min(len(feature_block_slices), len(gene_names))
        for idx in range(limit):
            start, end = feature_block_slices[idx]
            start = max(0, start)
            end = min(len(feature_names), end)
            if start >= end:
                continue
            block = aggregate_df.iloc[start:end].copy()
            if block.empty:
                continue
            gene_label = gene_names[idx]
            gene_block_ranges[gene_label] = (start, end)
            record = {
                "gene": gene_label,
                "feature_count": int(block.shape[0]),
                "importance_mean_sum": float(block["importance_mean"].sum()),
                "importance_mean_abs_sum": float(np.abs(block["importance_mean"]).sum()),
                "importance_mean_avg": float(block["importance_mean"].mean()),
                "top_feature": str(block.loc[block["importance_mean"].idxmax(), "feature"]),
                "top_feature_importance": float(block["importance_mean"].max()),
            }
            if "importance_mean_signed" in block.columns:
                record["importance_mean_signed_sum"] = float(block["importance_mean_signed"].sum())
                record["importance_mean_signed_avg"] = float(block["importance_mean_signed"].mean())
            if "signed_distance_to_tss_kb" in block.columns:
                distances = pd.to_numeric(block["signed_distance_to_tss_kb"], errors="coerce")
                mask = np.isfinite(distances) & np.isfinite(block["importance_mean"])
                if mask.any():
                    imp = block.loc[mask, "importance_mean"]
                    dist = distances[mask]
                    if imp.nunique() > 1 and dist.nunique() > 1:
                        record["pearson_distance_corr"] = float(imp.corr(dist, method="pearson"))
                        record["spearman_distance_corr"] = float(imp.corr(dist, method="spearman"))
                    top_idx = block.loc[mask, "importance_mean"].idxmax()
                    record["top_feature_distance_kb"] = float(distances.loc[top_idx])
            per_gene_records.append(record)
        if per_gene_records:
            per_gene_df = pd.DataFrame(per_gene_records)
            per_gene_summary_path = output_dir / "feature_importance_per_gene_summary.csv"
            per_gene_df.to_csv(per_gene_summary_path, index=False)
            fi_per_gene_count = int(per_gene_df.shape[0])
            fi_per_gene_mean_importance_sum = float(
                pd.to_numeric(per_gene_df["importance_mean_sum"], errors="coerce").mean(skipna=True)
            )
            if "importance_mean_abs_sum" in per_gene_df.columns:
                fi_per_gene_mean_abs_sum = float(
                    pd.to_numeric(per_gene_df["importance_mean_abs_sum"], errors="coerce").mean(skipna=True)
                )
            _LOG.info(
                "Saved per-gene feature importance summary (%d genes) to %s",
                per_gene_df.shape[0],
                per_gene_summary_path,
            )

            if export_per_gene_panels:
                panel_dir = output_dir / "per_gene_panels"
                panel_candidates = per_gene_df.sort_values("importance_mean_sum", ascending=False).head(12)
                generated = 0
                for gene_value in panel_candidates["gene"]:
                    block_range = gene_block_ranges.get(gene_value)
                    if not block_range:
                        continue
                    start, end = block_range
                    block_slice = aggregate_df.iloc[start:end].copy()
                    if block_slice.empty:
                        continue
                    safe_gene = re.sub(r"[^A-Za-z0-9._-]", "_", gene_value)
                    panel_path = panel_dir / f"{safe_gene}.png"
                    plot_per_gene_feature_panel(block_slice, gene_value, panel_path)
                    generated += 1
                if generated:
                    _LOG.info("Generated %d per-gene feature panels in %s", generated, panel_dir)

    fi_mean_finite = fi_mean[np.isfinite(fi_mean)]
    summary_payload: Dict[str, object] = {
        "method": method or "unknown",
        "num_features": int(fi_mean.size),
        "importance_mean_sum": float(fi_mean_finite.sum()) if fi_mean_finite.size else 0.0,
        "importance_mean_top1": float(fi_mean_finite.max()) if fi_mean_finite.size else 0.0,
        "raw_importances_file": raw_path.name,
        "aggregate_file": aggregate_path.name,
    }
    fi_abs = np.abs(fi_mean_finite)
    summary_payload["importance_mean_abs_sum"] = float(fi_abs.sum()) if fi_abs.size else 0.0
    summary_payload["importance_mean_abs_top1"] = float(fi_abs.max()) if fi_abs.size else 0.0
    if fi_signed is not None:
        fi_signed_finite = fi_signed[np.isfinite(fi_signed)]
        if fi_signed_finite.size:
            summary_payload["importance_mean_signed_sum"] = float(fi_signed_finite.sum())
            summary_payload["importance_mean_signed_top_positive"] = float(fi_signed_finite.max())
            summary_payload["importance_mean_signed_top_negative"] = float(fi_signed_finite.min())
    fi_weights = np.abs(fi_mean_finite)
    fi_weight_total = float(fi_weights.sum()) if fi_weights.size else 0.0
    if fi_weight_total > 0.0:
        top_k = min(10, int(fi_weights.size))
        if top_k > 0:
            top10_sum = float(np.sort(fi_weights)[-top_k:].sum())
            summary_payload["top10_weight_share"] = top10_sum / fi_weight_total
    if per_gene_summary_path is not None:
        summary_payload["per_gene_summary_file"] = per_gene_summary_path.name
    if fi_per_gene_count is not None:
        summary_payload["per_gene_count"] = fi_per_gene_count
    if fi_per_gene_mean_importance_sum is not None and np.isfinite(fi_per_gene_mean_importance_sum):
        summary_payload["per_gene_mean_importance_sum"] = fi_per_gene_mean_importance_sum
    if fi_per_gene_mean_abs_sum is not None and np.isfinite(fi_per_gene_mean_abs_sum):
        summary_payload["per_gene_mean_abs_sum"] = fi_per_gene_mean_abs_sum

    fi_scatter_path = output_dir / "feature_importance_vs_tss_distance.png"
    fi_overview_path = output_dir / "feature_importance_distance_overview.png"
    fi_distance_created = False
    if "signed_distance_to_tss_kb" in aggregate_df.columns:
        distances = pd.to_numeric(aggregate_df["signed_distance_to_tss_kb"], errors="coerce")
        mask = np.isfinite(distances) & np.isfinite(fi_mean)
        if mask.any():
            fi_weights_with_distance = np.abs(fi_mean[mask])
            fi_weights_with_distance_total = float(fi_weights_with_distance.sum())
            if fi_weights_with_distance_total > 0.0:
                near_mask = np.abs(distances[mask].to_numpy(dtype=float)) <= 2.0
                near_sum = float(fi_weights_with_distance[near_mask].sum())
                summary_payload["tss_near_2kb_share"] = near_sum / fi_weights_with_distance_total
        if mask.any():
            imp = pd.Series(fi_mean[mask])
            dist = distances[mask]
            if imp.nunique() > 1 and pd.Series(dist).nunique() > 1:
                pearson = float(imp.corr(dist, method="pearson"))
                spearman = float(imp.corr(dist, method="spearman"))
                corr_payload = {
                    "pearson": pearson,
                    "spearman": spearman,
                    "count": int(mask.sum()),
                    "method": method or "unknown",
                }
                summary_payload["tss_correlation"] = corr_payload
                plot_importance_distance_scatter(
                    fi_mean[mask],
                    dist,
                    fi_scatter_path,
                    f"FI vs TSS distance | {model_name.upper()}",
                    annotation={"Spearman": spearman, "Pearson": pearson},
                )
                _LOG.info(
                    "Saved FI vs TSS scatter and correlation stats (n=%d) to %s",
                    mask.sum(),
                    fi_scatter_path,
                )
                fi_distance_created = True

            plot_cumulative_importance_overlay(
                fi_mean[mask],
                dist,
                fi_overview_path,
                f"FI cumulative distance profile | {model_name.upper()}",
            )
            _LOG.info("Saved FI distance overlay to %s", fi_overview_path)
            fi_distance_created = True
    if not fi_distance_created:
        _write_placeholder_plot(
            fi_scatter_path,
            f"FI vs TSS distance | {model_name.upper()}",
            "Distance-to-TSS metadata unavailable for this run.",
        )
        _write_placeholder_plot(
            fi_overview_path,
            f"FI cumulative distance profile | {model_name.upper()}",
            "Distance-to-TSS metadata unavailable for this run.",
        )

    model_metric_summary = _compute_model_metric_summary(output_dir)
    if model_metric_summary:
        summary_payload["model_metrics"] = model_metric_summary
    if fi_signed_plot_path is not None:
        summary_payload["plot_signed_path"] = str(fi_signed_plot_path)

    summary_path = output_dir / "feature_importance_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))
    _LOG.info("Wrote feature importance manifest to %s", summary_path)

    duration = time.perf_counter() - start_wall
    end_ts = datetime.now(timezone.utc).isoformat()
    _LOG.info(
        "Feature importance export complete | model=%s | duration=%.2fs | timestamp=%s",
        model_name,
        duration,
        end_ts,
    )
    return summary_payload


def _export_shap_importance_artifacts(
    output_dir: Path,
    model_name: str,
    shap_importance: np.ndarray,
    feature_names: Sequence[str],
    *,
    shap_value_mean_signed: Optional[np.ndarray] = None,
    method: Optional[str] = None,
    gene_names: Optional[Sequence[str]] = None,
    feature_block_slices: Optional[Sequence[Tuple[int, int]]] = None,
    feature_block_indices: Optional[Sequence[Sequence[int]]] = None,
    gene_infos: Optional[Sequence[GeneInfo]] = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    shap_values = np.asarray(shap_importance, dtype=np.float64)
    if shap_values.size == 0 or shap_values.ndim != 1:
        _LOG.info("SHAP export skipped | model=%s | reason=no values", model_name)
        return {}

    plot_feature_importance(
        shap_values,
        feature_names,
        output_dir / "shapley_values_mean.png",
        f"SHAP mean | {model_name.upper()}",
        top_n=30,
    )

    # Parse feature metadata (same as feature importance export)
    metadata_records = [_feature_name_metadata(name) for name in feature_names]
    metadata_df = pd.DataFrame(metadata_records) if metadata_records else None

    shap_df = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_mean_abs": shap_values,
        }
    )
    signed_shap_values: Optional[np.ndarray] = None
    if shap_value_mean_signed is not None:
        signed_candidate = np.asarray(shap_value_mean_signed, dtype=np.float64).reshape(-1)
        if signed_candidate.shape[0] == shap_values.shape[0]:
            signed_shap_values = signed_candidate
            shap_df["shap_mean_signed"] = signed_candidate

    # Add metadata columns (TSS distance, gene name, etc.) if available.
    if metadata_df is not None and not metadata_df.empty:
        shap_df = pd.concat([shap_df, metadata_df], axis=1)
    if "signed_distance_to_tss_kb" not in shap_df.columns or not shap_df["signed_distance_to_tss_kb"].notna().any():
        inferred_dist = _infer_signed_distance_from_blocks(
            feature_names,
            gene_names,
            feature_block_indices,
            gene_infos,
        )
        if inferred_dist is not None:
            shap_df["signed_distance_to_tss_kb"] = inferred_dist
    shap_path = output_dir / "shapley_values_mean.csv"
    shap_df.to_csv(shap_path, index=False)
    _LOG.info("Saved SHAP mean importances (%d rows) to %s", shap_df.shape[0], shap_path)

    signed_plot_path: Optional[Path] = None
    if signed_shap_values is not None:
        signed_df = pd.DataFrame(
            {
                "feature": feature_names,
                "shap_mean_signed": signed_shap_values,
            }
        )
        signed_df = signed_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["shap_mean_signed"])
        if not signed_df.empty:
            top_pos = signed_df.sort_values("shap_mean_signed", ascending=False).head(10)
            top_neg = signed_df.sort_values("shap_mean_signed", ascending=True).head(10)
            signed_top = pd.concat([top_neg, top_pos], axis=0)
            signed_top = signed_top.drop_duplicates(subset=["feature"], keep="first")
            if not signed_top.empty:
                signed_plot_path = output_dir / "shapley_values_mean_signed.png"
                signed_top = signed_top.sort_values("shap_mean_signed", ascending=True)
                fig, ax = plt.subplots(figsize=(9.0, 6.0))
                bar_colors = ["#1f77b4" if v < 0 else "#d62728" for v in signed_top["shap_mean_signed"]]
                ax.barh(signed_top["feature"], signed_top["shap_mean_signed"], color=bar_colors)
                ax.axvline(0.0, color="#666666", linestyle="--", linewidth=1)
                ax.set_xlabel("Mean signed SHAP value")
                ax.set_ylabel("Feature")
                ax.set_title(f"Mean signed SHAP | {model_name.upper()}")
                plt.tight_layout()
                fig.savefig(signed_plot_path, dpi=300)
                plt.close(fig)
                _LOG.info("Saved signed SHAP mean plot to %s", signed_plot_path)

    shap_per_gene_summary_path: Optional[Path] = None
    shap_per_gene_count: Optional[int] = None
    shap_per_gene_mean_abs_sum: Optional[float] = None
    shap_per_gene_mean_signed_sum: Optional[float] = None
    if feature_block_indices and gene_names:
        per_gene_records: List[Dict[str, object]] = []
        limit = min(len(feature_block_indices), len(gene_names))
        for idx in range(limit):
            indices = np.asarray(feature_block_indices[idx], dtype=np.int64)
            if indices.size == 0:
                continue
            valid_mask = (indices >= 0) & (indices < len(feature_names))
            indices = indices[valid_mask]
            if indices.size == 0:
                continue
            block = shap_df.iloc[indices].copy()
            if block.empty:
                continue
            gene_label = gene_names[idx]
            record = {
                "gene": gene_label,
                "feature_count": int(block.shape[0]),
                "shap_mean_abs_sum": float(block["shap_mean_abs"].sum()),
                "shap_mean_abs_avg": float(block["shap_mean_abs"].mean()),
                "top_feature": str(block.loc[block["shap_mean_abs"].idxmax(), "feature"]),
                "top_feature_shap_mean_abs": float(block["shap_mean_abs"].max()),
            }
            if "shap_mean_signed" in block.columns:
                record["shap_mean_signed_sum"] = float(block["shap_mean_signed"].sum())
                record["shap_mean_signed_avg"] = float(block["shap_mean_signed"].mean())
            if "signed_distance_to_tss_kb" in block.columns:
                distances = pd.to_numeric(block["signed_distance_to_tss_kb"], errors="coerce")
                mask = np.isfinite(distances) & np.isfinite(block["shap_mean_abs"])
                if mask.any():
                    shap_vals = block.loc[mask, "shap_mean_abs"]
                    dist = distances[mask]
                    if shap_vals.nunique() > 1 and dist.nunique() > 1:
                        record["pearson_distance_corr"] = float(shap_vals.corr(dist, method="pearson"))
                        record["spearman_distance_corr"] = float(shap_vals.corr(dist, method="spearman"))
                    top_idx = block.loc[mask, "shap_mean_abs"].idxmax()
                    record["top_feature_distance_kb"] = float(distances.loc[top_idx])
            per_gene_records.append(record)

        if per_gene_records:
            per_gene_df = pd.DataFrame(per_gene_records)
            shap_per_gene_summary_path = output_dir / "shapley_values_per_gene_summary.csv"
            per_gene_df.to_csv(shap_per_gene_summary_path, index=False)
            shap_per_gene_count = int(per_gene_df.shape[0])
            shap_per_gene_mean_abs_sum = float(
                pd.to_numeric(per_gene_df["shap_mean_abs_sum"], errors="coerce").mean(skipna=True)
            )
            if "shap_mean_signed_sum" in per_gene_df.columns:
                shap_per_gene_mean_signed_sum = float(
                    pd.to_numeric(per_gene_df["shap_mean_signed_sum"], errors="coerce").mean(skipna=True)
                )
            _LOG.info(
                "Saved per-gene SHAP summary (%d genes) to %s",
                per_gene_df.shape[0],
                shap_per_gene_summary_path,
            )
    elif feature_block_slices and gene_names:
        per_gene_records = []
        limit = min(len(feature_block_slices), len(gene_names))
        for idx in range(limit):
            start, end = feature_block_slices[idx]
            start = max(0, start)
            end = min(len(feature_names), end)
            if start >= end:
                continue
            block = shap_df.iloc[start:end].copy()
            if block.empty:
                continue
            gene_label = gene_names[idx]
            record = {
                "gene": gene_label,
                "feature_count": int(block.shape[0]),
                "shap_mean_abs_sum": float(block["shap_mean_abs"].sum()),
                "shap_mean_abs_avg": float(block["shap_mean_abs"].mean()),
                "top_feature": str(block.loc[block["shap_mean_abs"].idxmax(), "feature"]),
                "top_feature_shap_mean_abs": float(block["shap_mean_abs"].max()),
            }
            if "shap_mean_signed" in block.columns:
                record["shap_mean_signed_sum"] = float(block["shap_mean_signed"].sum())
                record["shap_mean_signed_avg"] = float(block["shap_mean_signed"].mean())
            if "signed_distance_to_tss_kb" in block.columns:
                distances = pd.to_numeric(block["signed_distance_to_tss_kb"], errors="coerce")
                mask = np.isfinite(distances) & np.isfinite(block["shap_mean_abs"])
                if mask.any():
                    shap_vals = block.loc[mask, "shap_mean_abs"]
                    dist = distances[mask]
                    if shap_vals.nunique() > 1 and dist.nunique() > 1:
                        record["pearson_distance_corr"] = float(shap_vals.corr(dist, method="pearson"))
                        record["spearman_distance_corr"] = float(shap_vals.corr(dist, method="spearman"))
                    top_idx = block.loc[mask, "shap_mean_abs"].idxmax()
                    record["top_feature_distance_kb"] = float(distances.loc[top_idx])
            per_gene_records.append(record)
        if per_gene_records:
            per_gene_df = pd.DataFrame(per_gene_records)
            shap_per_gene_summary_path = output_dir / "shapley_values_per_gene_summary.csv"
            per_gene_df.to_csv(shap_per_gene_summary_path, index=False)
            shap_per_gene_count = int(per_gene_df.shape[0])
            shap_per_gene_mean_abs_sum = float(
                pd.to_numeric(per_gene_df["shap_mean_abs_sum"], errors="coerce").mean(skipna=True)
            )
            if "shap_mean_signed_sum" in per_gene_df.columns:
                shap_per_gene_mean_signed_sum = float(
                    pd.to_numeric(per_gene_df["shap_mean_signed_sum"], errors="coerce").mean(skipna=True)
                )
            _LOG.info(
                "Saved per-gene SHAP summary (%d genes) to %s",
                per_gene_df.shape[0],
                shap_per_gene_summary_path,
            )

    def _extract_distance_kb(table: pd.DataFrame) -> tuple[Optional[pd.Series], Optional[str], bool]:
        preferred_cols = [
            "signed_distance_to_tss_kb",
            "delta_to_tss_kb",
            "distance_to_tss_kb",
        ]
        for col in preferred_cols:
            if col in table.columns:
                return pd.to_numeric(table[col], errors="coerce"), col, True

        abs_cols = ["distance_to_tss_abs_kb"]
        for col in abs_cols:
            if col in table.columns:
                return pd.to_numeric(table[col], errors="coerce"), col, False

        bp_cols = ["delta_to_tss_bp", "distance_to_tss_bp", "relative_center_bp"]
        for col in bp_cols:
            if col in table.columns:
                return pd.to_numeric(table[col], errors="coerce") / 1_000.0, col, True

        return None, None, False

    def _plot_shap_vs_tss_distance(
        table: pd.DataFrame,
        distance_kb: pd.Series,
        output_path: Path,
        title: str,
        *,
        value_col: str = "shap_mean_abs",
        signed_values: bool = False,
        max_distance_kb: float,
        show_scatter: bool = False,
        y_limits: Optional[tuple[float, float]] = None,
    ) -> bool:
        plot_df = pd.DataFrame(
            {
                "distance_kb": distance_kb,
                "shap_value": pd.to_numeric(table[value_col], errors="coerce"),
            }
        )
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
        if plot_df.empty:
            return False
        plot_df = plot_df[plot_df["distance_kb"].abs() <= max_distance_kb].copy()
        if plot_df.empty:
            return False
        plot_df.sort_values("distance_kb", inplace=True)

        if signed_values:
            per_bin = (
                plot_df.groupby("distance_kb", sort=True)["shap_value"]
                .median()
                .reset_index()
            )
        else:
            per_bin = (
                plot_df.groupby("distance_kb", sort=True)["shap_value"]
                .quantile(0.9)
                .reset_index()
            )
        if per_bin.empty:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        if show_scatter:
            sns.scatterplot(
                data=plot_df,
                x="distance_kb",
                y="shap_value",
                s=30,
                alpha=0.45,
                edgecolor="none",
                color="#4daf4a",
                ax=ax,
            )
        ax.plot(
            per_bin["distance_kb"],
            per_bin["shap_value"],
            color="#e41a1c",
            linewidth=1.5,
            label="median" if signed_values else "90th percentile",
        )
        ax.legend(loc="upper right", frameon=True)
        ax.axvline(0.0, color="#999999", linestyle="--", linewidth=1)
        ax.set_xlim(-max_distance_kb, max_distance_kb)
        if y_limits is not None:
            ax.set_ylim(y_limits)
        ax.set_xlabel("Distance to TSS (kb)")
        ax.set_ylabel("SHAP value" if not signed_values else "Signed SHAP value")
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return True

    distance_kb, distance_col, distance_signed = _extract_distance_kb(shap_df)
    distance_plot_path = output_dir / "shapley_values_vs_tss.png"
    distance_plot_zoomed_path = output_dir / "shapley_values_vs_tss_zoomed.png"
    if distance_kb is not None and distance_col is not None:
        if not distance_signed:
            _LOG.info(
                "SHAP distance plot uses unsigned distances (%s); check feature metadata",
                distance_col,
            )
        created = _plot_shap_vs_tss_distance(
            shap_df,
            distance_kb,
            distance_plot_path,
            f"SHAP vs TSS distance | {model_name.upper()}",
            max_distance_kb=10.0,
            show_scatter=True,
        )
        if created:
            _LOG.info("Saved SHAP vs TSS plot to %s", distance_plot_path)
        else:
            distance_plot_path = None
        zoom_df = pd.DataFrame(
            {
                "distance_kb": distance_kb,
                "shap_value": pd.to_numeric(shap_df["shap_mean_abs"], errors="coerce"),
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()
        zoom_df = zoom_df[zoom_df["distance_kb"].abs() <= 5.0]
        if zoom_df.empty:
            zoom_y_limits = None
        else:
            zoom_max = (
                zoom_df.groupby("distance_kb", sort=True)["shap_value"]
                .quantile(0.9)
                .max()
            )
            zoom_y_limits = (0.0, float(zoom_max) * 1.15) if pd.notna(zoom_max) else None
        zoomed = _plot_shap_vs_tss_distance(
            shap_df,
            distance_kb,
            distance_plot_zoomed_path,
            f"SHAP vs TSS distance (zoomed) | {model_name.upper()}",
            max_distance_kb=5.0,
            y_limits=zoom_y_limits,
        )
        if zoomed:
            _LOG.info("Saved zoomed SHAP vs TSS plot to %s", distance_plot_zoomed_path)
        else:
            distance_plot_zoomed_path = None
    else:
        distance_plot_path = None
        distance_plot_zoomed_path = None

    if distance_plot_path is None:
        distance_plot_path = output_dir / "shapley_values_vs_tss.png"
        _write_placeholder_plot(
            distance_plot_path,
            f"SHAP vs TSS distance | {model_name.upper()}",
            "Distance-to-TSS metadata unavailable for this run.",
        )
    if distance_plot_zoomed_path is None:
        distance_plot_zoomed_path = output_dir / "shapley_values_vs_tss_zoomed.png"
        _write_placeholder_plot(
            distance_plot_zoomed_path,
            f"SHAP vs TSS distance (zoomed) | {model_name.upper()}",
            "Distance-to-TSS metadata unavailable for this run.",
        )

    shap_finite = shap_values[np.isfinite(shap_values)]
    summary = {
        "model": model_name,
        "method": method,
        "num_features": int(shap_values.size),
        "shap_mean_abs_sum": float(shap_finite.sum()) if shap_finite.size else 0.0,
        "shap_mean_abs_top1": float(shap_finite.max()) if shap_finite.size else 0.0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(shap_path),
        "plot_path": str(output_dir / "shapley_values_mean.png"),
    }
    if signed_shap_values is not None:
        signed_finite = signed_shap_values[np.isfinite(signed_shap_values)]
        if signed_finite.size:
            summary["shap_mean_sum"] = float(signed_finite.sum())
            summary["shap_mean_top1"] = float(signed_finite.max())
            summary["shap_mean_signed_sum"] = float(signed_finite.sum())
            summary["shap_mean_signed_top_positive"] = float(signed_finite.max())
            summary["shap_mean_signed_top_negative"] = float(signed_finite.min())
    if signed_plot_path is not None:
        summary["plot_signed_path"] = str(signed_plot_path)
    shap_weights = np.abs(shap_finite)
    shap_weight_total = float(shap_weights.sum()) if shap_weights.size else 0.0
    if shap_weight_total > 0.0:
        top_k = min(10, int(shap_weights.size))
        if top_k > 0:
            top10_sum = float(np.sort(shap_weights)[-top_k:].sum())
            summary["top10_weight_share"] = top10_sum / shap_weight_total
    if distance_plot_path is not None:
        summary["distance_plot_path"] = str(distance_plot_path)
    if distance_plot_zoomed_path is not None:
        summary["distance_plot_zoomed_path"] = str(distance_plot_zoomed_path)
    if shap_per_gene_summary_path is not None:
        summary["per_gene_summary_file"] = shap_per_gene_summary_path.name
    if shap_per_gene_count is not None:
        summary["per_gene_count"] = shap_per_gene_count
    if shap_per_gene_mean_abs_sum is not None and np.isfinite(shap_per_gene_mean_abs_sum):
        summary["per_gene_mean_abs_sum"] = shap_per_gene_mean_abs_sum
    if shap_per_gene_mean_signed_sum is not None and np.isfinite(shap_per_gene_mean_signed_sum):
        summary["per_gene_mean_signed_sum"] = shap_per_gene_mean_signed_sum

    if distance_kb is not None:
        distance_arr = np.asarray(distance_kb, dtype=np.float64)
        corr_mask = np.isfinite(distance_arr) & np.isfinite(shap_values)
        if corr_mask.any():
            shap_weights_with_distance = np.abs(shap_values[corr_mask])
            shap_weights_with_distance_total = float(shap_weights_with_distance.sum())
            if shap_weights_with_distance_total > 0.0:
                near_mask = np.abs(distance_arr[corr_mask]) <= 2.0
                near_sum = float(shap_weights_with_distance[near_mask].sum())
                summary["tss_near_2kb_share"] = near_sum / shap_weights_with_distance_total
        if corr_mask.any():
            corr_x = pd.Series(shap_values[corr_mask])
            corr_y = pd.Series(distance_arr[corr_mask])
            if corr_x.nunique() > 1 and corr_y.nunique() > 1:
                summary["tss_correlation"] = {
                    "pearson": float(corr_x.corr(corr_y, method="pearson")),
                    "spearman": float(corr_x.corr(corr_y, method="spearman")),
                    "count": int(corr_mask.sum()),
                    "method": method or "unknown",
                }

    model_metric_summary = _compute_model_metric_summary(output_dir)
    if model_metric_summary:
        summary["model_metrics"] = model_metric_summary
    summary_path = output_dir / "shapley_values_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _LOG.info("Wrote SHAP summary manifest to %s", summary_path)
    return summary

try:
    import torch.nn as _torch_nn
except ImportError:  # pragma: no cover - torch optional during some tests
    _torch_nn = None

def run_pipeline(config: PipelineConfig) -> Path:
    config.ensure_directories()

    wandb_run = maybe_init_wandb(config)
    run_status = "failed"
    run_dir: Optional[Path] = None
    wandb_finished = False
    prev_signal_handlers: Dict[int, Any] = {}

    def _safe_wandb_finish(status: str) -> None:
        nonlocal wandb_finished
        if wandb_finished:
            return
        wandb_finished = True
        wandb_finish(wandb_run, status=status, run_dir=run_dir)

    def _handle_termination(signum: int, frame: Any) -> None:
        nonlocal run_status
        _LOG.error("Received signal %s; marking run failed and finalizing W&B", signum)
        run_status = "failed"
        _safe_wandb_finish(run_status)
        raise SystemExit(128 + signum)

    candidate_signals = [
        getattr(signal, name)
        for name in ("SIGTERM", "SIGINT", "SIGQUIT", "SIGHUP", "SIGUSR1", "SIGUSR2")
        if hasattr(signal, name)
    ]
    for sig in candidate_signals:
        try:
            prev_signal_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handle_termination)
        except Exception:  # pragma: no cover - best-effort cleanup
            _LOG.debug("Failed to register signal handler for %s", sig, exc_info=True)
    try:
        apply_sweep_overrides(config, wandb_run)

        atac, rna = load_datasets(config.paths)
        atac, rna = preprocess_modalities(atac, rna, config.training)
    
        target_chromosomes = config.chromosomes
        if target_chromosomes and len(target_chromosomes) == 1:
            token = target_chromosomes[0].strip().lower()
            if token in {"all", "genome-wide", "genome"}:
                target_chromosomes = None
                _LOG.info("Chromosome filter explicitly set to all/genome-wide")
    
        genes_all = parse_gtf(
            config.paths.gtf_path,
            chromosomes=target_chromosomes,
            gene_names=config.genes,
        )
        max_genes_for_selection = config.max_genes
        if config.multi_output and not config.genes:
            max_genes_for_selection = None
        elif config.max_genes and not config.genes:
            max_genes_for_selection = None
    
        genes = select_genes(genes_all, requested_genes=config.genes, max_genes=max_genes_for_selection)
    
        candidate_count = len(genes)
        if (
            not config.multi_output
            and config.max_genes
            and not config.genes
            and candidate_count > config.max_genes
        ):
            rng = np.random.default_rng(config.training.random_state)
            sample_indices = np.asarray(rng.choice(len(genes), size=config.max_genes, replace=False))
            sample_indices.sort()
            genes = [genes[int(idx)] for idx in sample_indices]
            _LOG.info(
                "Randomly sampled %d genes (from %d candidates) for gene-wise processing",
                config.max_genes,
                candidate_count,
            )
    
        if config.genes:
            found_names = {gene.gene_name for gene in genes}
            found_ids = {gene.gene_id for gene in genes}
            missing = [name for name in config.genes if name not in found_names and name not in found_ids]
            if missing:
                raise RuntimeError(
                    "The following requested genes were not found in annotations: "
                    + ", ".join(missing[:10])
                    + (" ..." if len(missing) > 10 else "")
                )
    
        selected_gene_fractions: Dict[str, float] = {}
        manifest_mode = bool(config.genes)
        if config.multi_output:
            base_pool = genes if genes else genes_all
            expressed_candidates, fraction_map = _genes_expressed_above_fraction(
                base_pool,
                rna,
                min_expression=config.training.min_expression,
                min_fraction=config.training.min_expression_fraction,
            )
    
            if manifest_mode:
                missing = [
                    gene
                    for gene in genes
                    if fraction_map.get(gene.gene_name, 0.0) < config.training.min_expression_fraction
                ]
                if missing:
                    names = ", ".join(g.gene_name for g in missing[:10])
                    _LOG.error(
                        "Manifest supplied %d genes below expression fraction threshold: %s%s",
                        len(missing),
                        names,
                        " ..." if len(missing) > 10 else "",
                    )
                    raise RuntimeError(
                        "Gene manifest contains entries below the minimum expression fraction threshold"
                    )
                selected_gene_fractions = {
                    gene.gene_name: fraction_map.get(gene.gene_name, float("nan"))
                    for gene in genes
                }
                _LOG.info(
                    "Using %d genes from manifest with >=%.1f%% expressing cells",
                    len(genes),
                    config.training.min_expression_fraction * 100.0,
                )
            else:
                available_gene_count = len(expressed_candidates)
                if config.max_genes is None:
                    if available_gene_count == 0:
                        raise RuntimeError(
                            "No genes met the minimum expression fraction (>=%.2f of cells)"
                            % config.training.min_expression_fraction
                        )
                    genes = expressed_candidates
                    selected_gene_fractions = {
                        gene.gene_name: fraction_map.get(gene.gene_name, float("nan"))
                        for gene in genes
                    }
                    _LOG.info(
                        "Using all %d genes meeting the >=%.1f%% expression threshold",
                        len(genes),
                        config.training.min_expression_fraction * 100.0,
                    )
                else:
                    requested_gene_count = config.max_genes
                    if requested_gene_count <= 0:
                        raise RuntimeError("Configured max_genes must be >= 1 for multi-output mode")
    
                    if available_gene_count < requested_gene_count:
                        _LOG.warning(
                            "Only %d genes meet the expression threshold (requested %d); proceeding with available genes",
                            available_gene_count,
                            requested_gene_count,
                        )
    
                    selected_gene_count = min(requested_gene_count, available_gene_count)
                    if selected_gene_count == 0:
                        raise RuntimeError(
                            "No genes met the minimum expression fraction (>=%.2f of cells)"
                            % config.training.min_expression_fraction
                        )
    
                    genes = _choose_random_genes(
                        expressed_candidates,
                        selected_gene_count,
                        config.training.random_state,
                    )
                    selected_gene_fractions = {
                        gene.gene_name: fraction_map.get(gene.gene_name, float("nan"))
                        for gene in genes
                    }
                    _LOG.info(
                        "Selected %d genome-wide genes with >=%.1f%% expressing cells",
                        len(genes),
                        config.training.min_expression_fraction * 100.0,
                    )
                    _LOG.debug("Selected genes: %s", ", ".join(g.gene_name for g in genes))
    
        if config.multi_output and not genes:
            raise RuntimeError("No genes matched the provided filters for multi-output training")
    
        total_genes = len(genes)
        chunk_total = max(1, int(config.chunk_total))
        chunk_index = int(config.chunk_index)
        applied_chunking = False
        if chunk_index < 0:
            chunk_index = 0
        if chunk_index >= chunk_total:
            chunk_index = chunk_total - 1
        if chunk_total > 1 and total_genes:
            chunk_size = math.ceil(total_genes / chunk_total)
            start = chunk_index * chunk_size
            end = min(total_genes, start + chunk_size)
            _LOG.info(
                "Applying chunk selection: total_genes=%d | chunk_total=%d | chunk_index=%d | chunk_size=%d | start=%d | end=%d",
                total_genes,
                chunk_total,
                chunk_index,
                chunk_size,
                start,
                end,
            )
            genes = genes[start:end]
            applied_chunking = True
    
        if not genes:
            peak_indexer = PeakIndexer(atac, layer=config.training.atac_layer)
        else:
            atac = filter_atac_by_genes(atac, genes, config.training.window_bp)
            peak_indexer = PeakIndexer(atac, layer=config.training.atac_layer)
    
        if config.multi_output:
            if applied_chunking:
                _LOG.info(
                    "Multi-output chunk processed: index=%d/%d | genes_in_chunk=%d",
                    chunk_index,
                    chunk_total,
                    len(genes),
                )
            elif chunk_total > 1:
                _LOG.warning(
                    "Chunk parameters specified (chunk_total=%d, chunk_index=%d) but no genes selected; proceeding without chunking",
                    chunk_total,
                    chunk_index,
                )
            run_dir = _run_cellwise_pipeline(
                config,
                genes,
                atac,
                rna,
                peak_indexer,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                gene_expression_fraction=selected_gene_fractions,
                wandb_run=wandb_run,
            )
            run_status = "succeeded"
            return run_dir
    

        run_dir = _run_per_gene_pipeline(
            config,
            genes,
            atac,
            rna,
            peak_indexer,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            wandb_run=wandb_run,
        )
        run_status = "succeeded"
        return run_dir
    except BaseException as exc:
        run_status = "failed"
        _LOG.error(
            "Terminal pipeline error | type=%s | message=%s",
            type(exc).__name__,
            exc,
        )
        raise
    finally:
        for sig, handler in prev_signal_handlers.items():
            try:
                signal.signal(sig, handler)
            except Exception:  # pragma: no cover - best-effort cleanup
                _LOG.debug("Failed to restore signal handler for %s", sig, exc_info=True)
        _safe_wandb_finish(run_status)


def _run_per_gene_pipeline(
    config: PipelineConfig,
    genes: List[GeneInfo],
    atac: ad.AnnData,
    rna: ad.AnnData,
    peak_indexer: PeakIndexer,
    *,
    chunk_index: int = 0,
    chunk_total: int = 1,
    wandb_run: Optional[Any] = None,
) -> Path:
    """Execute per-gene training pipeline, processing each gene independently across all models."""
    
    base_dir = config.paths.output_dir
    run_dir = base_dir / config.run_name if config.run_name else base_dir
    
    if not genes:
        _LOG.warning(
            "No genes assigned to this chunk (chunk_index=%d, chunk_total=%d). Nothing to process.",
            chunk_index,
            chunk_total,
        )
        return run_dir
    
    _ensure_directory(run_dir)
    wandb_update_config(
        wandb_run,
        {
            "mode": "per_gene",
            "total_models": len(config.all_models()),
            "requested_genes": len(genes),
            "num_genes": len(genes),
            **(
                {"chunk_index": chunk_index, "chunk_total": chunk_total}
                if chunk_total > 1 or chunk_index > 0
                else {}
            ),
        },
    )
    
    summary_records: List[Dict[str, object]] = []
    model_store: Dict[str, Dict[str, object]] = defaultdict(lambda: {
        "predictions": [],
        "metrics": [],
        "feature_importances": [],
        "feature_importances_genes": [],
        "feature_names": None,
        "histories": [],
    })
    model_export_meta: Dict[str, Dict[str, Any]] = {
        name: {"successful_genes": [], "failures": []} for name in config.all_models()
    }
    model_config_snapshots: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    
    for gene in genes:
        _LOG.info("Processing gene %s", gene.gene_name)
        try:
            dataset = build_gene_dataset(
                gene,
                atac,
                rna,
                peak_indexer,
                config.training,
            )
        except ValueError as exc:
            _LOG.warning("Skipping gene %s: %s", gene.gene_name, exc)
            continue
    
        for model_name in config.all_models():
            _LOG.info("Training %s for gene %s", model_name, gene.gene_name)
            try:
                artifacts_dir = None
                if model_name == "catboost":
                    artifacts_dir = run_dir / "catboost_info" / gene.gene_name
                result = train_model_for_gene(
                    dataset,
                    model_name,
                    config.training,
                    artifacts_dir=artifacts_dir,
                    cache_dir=config.cache_dir,
                )
            except Exception as exc:
                _LOG.error(
                    "Model %s failed for gene %s: %s\n%s",
                    model_name,
                    gene.gene_name,
                    exc,
                    traceback.format_exc(),
                )
                failures.append(f"{model_name}|{gene.gene_name}: {exc}")
                model_export_meta.setdefault(model_name, {"successful_genes": [], "failures": []})
                model_export_meta[model_name]["failures"].append(
                    {"gene": gene.gene_name, "error": str(exc)}
                )
                continue
            preds_df = pd.DataFrame(result.predictions)
            preds_df["gene"] = gene.gene_name
    
            store = model_store[model_name]
            store["predictions"].append(preds_df)
            store["metrics"].append(
                {
                    "gene": gene.gene_name,
                    **{f"train_{k}": v for k, v in result.train_metrics.items()},
                    **{f"val_{k}": v for k, v in result.val_metrics.items()},
                    **{f"test_{k}": v for k, v in result.test_metrics.items()},
                }
            )
            feature_importances = _extract_feature_importance(result)
            if feature_importances is not None:
                store["feature_importances"].append(feature_importances)
                store["feature_importances_genes"].append(gene.gene_name)
                if store["feature_names"] is None:
                    store["feature_names"] = list(dataset.feature_names)
            if result.history:
                store["histories"].append((gene.gene_name, result.history))
    
            model_export_meta.setdefault(model_name, {"successful_genes": [], "failures": []})
            model_export_meta[model_name]["successful_genes"].append(gene.gene_name)
            if model_name not in model_config_snapshots and result.fitted_model is not None:
                model_config_snapshots[model_name] = _capture_model_configuration(result.fitted_model)
    
            summary_records.append(
                {
                    "gene": gene.gene_name,
                    "model": model_name,
                    **{f"cv_fold_{m.fold}_{k}": v for m in result.cv_metrics for k, v in m.metrics.items()},
                    **{f"train_{k}": v for k, v in result.train_metrics.items()},
                    **{f"val_{k}": v for k, v in result.val_metrics.items()},
                    **{f"test_{k}": v for k, v in result.test_metrics.items()},
                }
            )
    
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_df = _reorder_metric_columns(summary_df)
        summary_path = run_dir / "summary_metrics_per_gene.csv"
        summary_df.to_csv(summary_path, index=False)
        _LOG.info("Run summary metrics saved to %s", summary_path)
    
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_order = {name: idx for idx, name in enumerate(config.all_models(), start=1)}
    summary_aggregate_rows: List[Dict[str, Any]] = []
    for model_name, store in model_store.items():
        model_dir = models_dir / model_name
        model_dir.mkdir(exist_ok=True)
    
        predictions = store["predictions"]
        if predictions:
            preds_df = pd.concat(predictions, ignore_index=True)
            preds_path = model_dir / "predictions_raw.csv"
            preds_df.to_csv(preds_path, index=False)

            residuals_by_split: Dict[str, np.ndarray] = {}
            for split in ["train", "val", "test"]:
                subset = preds_df[preds_df["split"] == split]
                if subset.empty:
                    continue
                plot_predictions_vs_actual(
                    subset["y_true"].to_numpy(),
                    subset["y_pred"].to_numpy(),
                    model_dir / f"scatter_{split}.png",
                    f"{model_name.upper()} | {split}",
                )
                residuals_by_split[split] = (subset["y_pred"].to_numpy() - subset["y_true"].to_numpy()).astype(np.float64)

            if residuals_by_split:
                plot_residual_histogram_by_split(
                    residuals_by_split,
                    model_dir / "residuals.png",
                    f"Residuals | {model_name.upper()}",
                )
    
        metrics_records = store["metrics"]
        if metrics_records:
            metrics_df = pd.DataFrame(metrics_records)
            for metric in _METRIC_ORDER:
                split_cols = [f"{split}_{metric}" for split in ("train", "val", "test") if f"{split}_{metric}" in metrics_df.columns]
                if split_cols:
                    metrics_df[f"{metric}_mean"] = metrics_df[split_cols].mean(axis=1, skipna=True)
                    metrics_df[f"{metric}_std"] = metrics_df[split_cols].std(axis=1, ddof=0, skipna=True)
            metrics_df = _reorder_metric_columns(metrics_df)
            metrics_df.to_csv(model_dir / "metrics_by_gene.csv", index=False)

            metric_labels = {
                "pearson": "Pearson correlation coefficient",
                "spearman": "Spearman correlation coefficient",
                "r2": "R^2",
                "rmse": "RMSE",
                "mse": "MSE",
                "mae": "MAE",
            }
            for metric in _METRIC_ORDER:
                combined_records = []
                for split in ("train", "val", "test"):
                    col = f"{split}_{metric}"
                    if col not in metrics_df.columns:
                        continue
                    series = pd.to_numeric(metrics_df[col], errors="coerce")
                    series = series[np.isfinite(series)]
                    if series.empty:
                        continue
                    combined_records.extend(
                        {"split": split.title(), metric: val} for val in series.tolist()
                    )
                if not combined_records:
                    continue
                combined_df = pd.DataFrame(combined_records)
                values_by_group = {}
                for split in ("Train", "Val", "Test"):
                    vals = pd.to_numeric(
                        combined_df.loc[combined_df["split"] == split, metric],
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        values_by_group[split] = vals
                if values_by_group:
                    plot_box_violin_half_split(
                        values_by_group,
                        model_dir / f"by_split_{metric}.png",
                        f"{model_name.upper()} | {metric.upper()}",
                        metric_labels.get(metric, metric.upper()),
                        order=["Train", "Val", "Test"],
                    )

            if "train_pearson" in metrics_df.columns and "test_pearson" in metrics_df.columns:
                train_vals = pd.to_numeric(metrics_df["train_pearson"], errors="coerce")
                test_vals = pd.to_numeric(metrics_df["test_pearson"], errors="coerce")
                mask = np.isfinite(train_vals) & np.isfinite(test_vals)
                if mask.any():
                    gaps = (train_vals[mask] - test_vals[mask]).astype(float).tolist()
                    if gaps:
                        plot_single_box_violin(
                            gaps,
                            model_dir / "generalization_gap_pearson_train_test.png",
                            f"{model_name.upper()} | Train-Test Pearson Gap",
                            "Train - Test Pearson",
                        )

            metrics_mean = metrics_df.mean(numeric_only=True)
            metrics_mean.to_csv(model_dir / "metrics_summary.csv", header=["value"])
            if wandb_run is not None:
                summary_payload: Dict[str, Any] = {}
                for key, value in metrics_mean.items():
                    if "_" in key:
                        split, metric = key.split("_", 1)
                        summary_payload[f"{split}_{metric}"] = value
                    else:
                        summary_payload[key] = value
                if summary_payload:
                    wandb_update_summary(wandb_run, summary_payload)
            base_row: Dict[str, Any] = {
                "model_display": model_name,
                "dataset": infer_dataset_name(config),
                "model_id": model_name,
            }
            metrics = _METRIC_ORDER
            for split in ("train", "val", "test"):
                for metric in metrics:
                    col = f"{split}_{metric}"
                    if col in metrics_df.columns:
                        series = pd.to_numeric(metrics_df[col], errors="coerce")
                        base_row[f"{split}_{metric}_mean"] = float(series.mean())
                        base_row[f"{split}_{metric}_std"] = float(series.std(ddof=0))
            summary_aggregate_rows.append(base_row)
            if wandb_run is not None:
                summary_payload: Dict[str, Any] = {}
                for metric in ("pearson", "r2", "spearman"):
                    col = f"test_{metric}"
                    if col in metrics_df.columns:
                        try:
                            summary_payload[f"test_median_{metric}"] = float(
                                metrics_df[col].median(skipna=True)
                            )
                        except Exception:
                            _LOG.debug("Failed to compute median for %s", col, exc_info=True)
                if "test_r2" in metrics_df.columns and "gene" in metrics_df.columns:
                    try:
                        best_idx = metrics_df["test_r2"].astype(float).idxmax()
                        best_row = metrics_df.loc[best_idx]
                        summary_payload["best_gene"] = str(best_row["gene"])
                        if "test_r2" in best_row:
                            summary_payload["best_gene_r2"] = float(best_row["test_r2"])
                        if "test_pearson" in best_row:
                            summary_payload["best_gene_pearson"] = float(best_row["test_pearson"])
                        if "test_spearman" in best_row:
                            summary_payload["best_gene_spearman"] = float(best_row["test_spearman"])
                    except Exception:
                        _LOG.debug("Failed to compute best gene summary", exc_info=True)
                if summary_payload:
                    wandb_update_summary(wandb_run, summary_payload)
    
        feature_importances = store["feature_importances"]
        feature_importance_genes = store.get("feature_importances_genes", [])
        feature_names = store["feature_names"]
        if feature_importances and feature_names:
            try:
                fi_stack = np.vstack(feature_importances)
            except ValueError as exc:
                _LOG.warning(
                    "Skipping feature importance aggregation for %s due to shape mismatch: %s",
                    model_name,
                    exc,
                )
                fi_stack = None
            if fi_stack is not None:
                fi_mean = fi_stack.mean(axis=0)
                plot_feature_importance(
                    fi_mean,
                    feature_names,
                    model_dir / "feature_importance_mean.png",
                    f"Feature importance | {model_name.upper()}",
                )
    
                fi_std = fi_stack.std(axis=0, ddof=0)
                fi_median = np.median(fi_stack, axis=0)
    
                metadata_records = [_feature_name_metadata(name) for name in feature_names]
                metadata_df = pd.DataFrame(metadata_records) if metadata_records else None
    
                aggregate_df = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance_mean": fi_mean,
                        "importance_std": fi_std,
                        "importance_median": fi_median,
                    }
                )
                if metadata_df is not None and not metadata_df.empty:
                    aggregate_df = pd.concat([aggregate_df, metadata_df], axis=1)
                aggregate_df.to_csv(model_dir / "feature_importances_mean.csv", index=False)
    
                if feature_importance_genes:
                    long_records: List[Dict[str, object]] = []
                    metadata_lookup = (
                        {feature_names[idx]: metadata_records[idx] for idx in range(len(feature_names))}
                        if metadata_records
                        else {}
                    )
                    for gene_name, vector in zip(feature_importance_genes, feature_importances):
                        for idx, value in enumerate(vector):
                            entry: Dict[str, object] = {
                                "gene": gene_name,
                                "feature": feature_names[idx],
                                "importance": float(value),
                            }
                            if metadata_lookup:
                                entry.update(metadata_lookup.get(feature_names[idx], {}))
                            long_records.append(entry)
                    if long_records:
                        per_gene_df = pd.DataFrame(long_records)
                        per_gene_df.to_csv(
                            model_dir / "feature_importances_per_gene.csv",
                            index=False,
                        )

    if summary_aggregate_rows:
        summary_agg_df = pd.DataFrame(summary_aggregate_rows)
        summary_agg_df = _reorder_metric_columns(summary_agg_df)
        summary_path = run_dir / "summary_metrics.csv"
        summary_agg_df.to_csv(summary_path, index=False)
        _LOG.info("Aggregate summary metrics saved to %s", summary_path)
        histories = store["histories"]
        if histories:
            history_dir = model_dir / "histories"
            history_dir.mkdir(exist_ok=True)
            for gene_name, history_records in histories:
                history_df = _order_training_history_columns(pd.DataFrame(history_records))
                history_csv = history_dir / f"{gene_name}.csv"
                history_df.to_csv(history_csv, index=False)
                for metric in ("loss", "pearson", "r2", "spearman"):
                    plot_training_history_curves(
                        history_df,
                        metric,
                        history_dir / f"{gene_name}_{metric}.png",
                        title=f"{model_name.upper()} | {gene_name} | {metric.title()}",
                    )
    
    model_run_details: Dict[str, Any] = {}
    for name, meta in model_export_meta.items():
        successes = sorted(set(meta.get("successful_genes", [])))
        failure_records = meta.get("failures", [])
        if successes:
            status = "succeeded"
        elif failure_records:
            status = "failed"
        else:
            status = "skipped"
        entry: Dict[str, Any] = {"status": status}
        if successes:
            entry["successful_genes"] = successes
        if failure_records:
            entry["failures"] = failure_records
        if name in model_config_snapshots:
            entry["estimator"] = model_config_snapshots[name]
        model_run_details[name] = entry
    
    processed_genes = sorted({row["gene"] for row in summary_records}) if summary_records else []
    extra_context = {
        "mode": "per_gene",
        "requested_genes": sorted({gene.gene_name for gene in genes}) if genes else [],
        "processed_genes": processed_genes,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "total_models": len(config.all_models()),
    }
    _export_run_configuration(config, run_dir, model_run_details, extra_context)

    if failures:
        raise RuntimeError(
            "One or more gene-level model trainings failed: " + "; ".join(failures)
        )

    if wandb_run is not None and config.wandb.log_tables:
        table_max = config.wandb.table_max_rows
        log_tables_from_csv(
            wandb_run,
            "Tables/summary_metrics",
            run_dir / "summary_metrics.csv",
            max_rows=table_max,
        )
        log_tables_from_csv(
            wandb_run,
            "Tables/summary_metrics_per_gene",
            run_dir / "summary_metrics_per_gene.csv",
            max_rows=table_max,
        )
        for model_name in config.all_models():
            model_dir = run_dir / "models" / model_name
            log_tables_from_csv(
                wandb_run,
                "Tables/metrics_by_gene",
                model_dir / "metrics_by_gene.csv",
                max_rows=table_max,
            )
            log_tables_from_csv(
                wandb_run,
                "Tables/fi_per_gene_summary",
                model_dir / "feature_importance_per_gene_summary.csv",
                max_rows=table_max,
            )
            log_tables_from_csv(
                wandb_run,
                "Tables/shap_per_gene_summary",
                model_dir / "shapley_values_per_gene_summary.csv",
                max_rows=table_max,
            )
            if config.wandb.log_predictions_table:
                log_tables_from_csv(
                    wandb_run,
                    "Tables/predictions",
                    model_dir / "predictions_raw.csv",
                    max_rows=table_max,
                )

    if wandb_run is not None and config.wandb.log_media:
        for model_name in config.all_models():
            model_dir = run_dir / "models" / model_name
            chart_prefix = f"Per_Gene/{model_name.upper()}"
            log_prediction_charts_from_csv(
                wandb_run,
                model_dir / "predictions_raw.csv",
                prefix=chart_prefix,
            )
            log_metric_distribution_charts_from_csv(
                wandb_run,
                model_dir / "metrics_by_gene.csv",
                prefix=chart_prefix,
            )

        max_media = config.wandb.media_max_items
        log_images_from_globs(
            wandb_run,
            run_dir,
            patterns=["models/*/feature_importance_mean.png"],
            max_items=max_media,
            group_key="Plots/FI",
        )
        log_images_from_globs(
            wandb_run,
            run_dir,
            patterns=["models/*/feature_importance_vs_tss_distance.png"],
            max_items=max_media,
            group_key="Plots/FI/TSS_Distance",
        )
        log_images_from_globs(
            wandb_run,
            run_dir,
            patterns=["models/*/feature_importance_mean_signed.png"],
            max_items=max_media,
            group_key="Plots/FI/Signed",
        )
        log_images_from_globs(
            wandb_run,
            run_dir,
            patterns=["models/*/feature_importance_distance_overview.png"],
            max_items=max_media,
            group_key="Plots/FI/Cumulative_Overview",
        )
    if wandb_run is not None and config.wandb.log_artifacts:
        log_run_artifacts(wandb_run, run_dir)

    return run_dir

def _run_cellwise_pipeline(
    config: PipelineConfig,
    genes: List[GeneInfo],
    atac: ad.AnnData,
    rna: ad.AnnData,
    peak_indexer: PeakIndexer,
    *,
    chunk_index: int = 0,
    chunk_total: int = 1,
    gene_expression_fraction: Optional[Dict[str, float]] = None,
    wandb_run: Optional[Any] = None,
) -> Path:
    base_dir = config.paths.output_dir
    run_dir = base_dir / config.run_name if config.run_name else base_dir
    catboost_tmp_root = run_dir / "catboost_tmp"

    model_export_meta: Dict[str, Dict[str, Any]] = {
        name: {"status": "pending", "failures": []} for name in config.all_models()
    }
    model_config_snapshots: Dict[str, Dict[str, Any]] = {}
    overall_status = "failed"

    try:
        try:
            _LOG.info(
                "Constructing cell-wise dataset | genes=%d | chunk_index=%d | chunk_total=%d",
                len(genes),
                chunk_index,
                chunk_total,
            )
            dataset = build_cellwise_dataset(
                genes,
                atac,
                rna,
                peak_indexer,
                config.training,
            )
        except RuntimeError as exc:
            _LOG.error("Failed to construct cell-wise dataset: %s", exc)
            raise
        except ValueError as exc:
            _LOG.error("Failed to construct cell-wise dataset: %s", exc)
            raise

        _ensure_directory(run_dir)
        models_dir = _ensure_directory(run_dir / "models")

        _write_selected_genes(run_dir, dataset.genes, gene_expression_fraction)

        extra_context = {
            "mode": "multi_output",
            "gene_names": [gene.gene_name for gene in dataset.genes],
            "num_cells": dataset.num_cells(),
            "num_features": dataset.num_features(),
            "chunk_index": chunk_index,
            "chunk_total": chunk_total,
            "total_models": len(config.all_models()),
        }

        summary_records: List[Dict[str, object]] = []
        failures: List[str] = []

        def export_run_configuration_snapshot() -> None:
            model_run_details: Dict[str, Any] = {}
            for name, meta in model_export_meta.items():
                entry: Dict[str, Any] = {
                    "status": meta.get("status", "pending"),
                }
                failures_meta = meta.get("failures", [])
                if failures_meta:
                    entry["failures"] = failures_meta
                if name in model_config_snapshots:
                    entry["estimator"] = model_config_snapshots[name]
                model_run_details[name] = entry
            _export_run_configuration(config, run_dir, model_run_details, extra_context)

        export_run_configuration_snapshot()
        wandb_update_config(
            wandb_run,
            {
                "mode": "multi_output",
                "num_genes": dataset.num_genes(),
                "num_cells": dataset.num_cells(),
                "num_features": dataset.num_features(),
                **(
                    {"chunk_index": chunk_index, "chunk_total": chunk_total}
                    if chunk_total > 1 or chunk_index > 0
                    else {}
                ),
            },
        )

        for model_idx, model_name in enumerate(config.all_models(), start=1):
            model_dir = _ensure_directory(models_dir / model_name)

            _LOG.info(
                "Training %s for multi-output regression across %d genes",
                model_name,
                dataset.num_genes(),
            )

            tracker = ResourceUsageTracker(
                name=f"{model_name}_cellwise",
                output_dir=model_dir,
                interval_seconds=getattr(config.training, "resource_sample_seconds", 60.0),
            )
            artifacts_dir = None
            if model_name == "catboost":
                artifacts_dir = run_dir / "catboost_info"
                _ensure_directory(artifacts_dir)

            env_ctx = nullcontext()
            if model_name == "catboost":
                tmp_dir = _ensure_directory(catboost_tmp_root / f"{model_name}_tmp")
                env_ctx = _temporary_env_var("CATBOOST_TMPDIR", str(tmp_dir))

            try:
                with env_ctx:
                    with tracker:
                        result = train_multi_output_model(
                            dataset,
                            model_name,
                            config.training,
                            artifacts_dir=artifacts_dir,
                            cache_dir=config.cache_dir,
                        )
            except Exception as exc:  # pragma: no cover - defensive logging
                _LOG.error("Model %s failed in multi-output mode: %s", model_name, exc)
                failures.append(f"{model_name}: {exc}")
                model_export_meta[model_name]["status"] = "failed"
                model_export_meta[model_name]["failures"].append(str(exc))
            else:
                model_dir = _ensure_directory(model_dir)
                if not config.training.export_raw_predictions:
                    _LOG.info(
                        "Skipping raw predictions export | model=%s | reason=export_raw_predictions=False",
                        model_name,
                    )
                else:
                    preds_df = _cellwise_predictions_dataframe(result)
                    preds_df.to_csv(model_dir / "predictions_raw.csv", index=False)

                _write_cellwise_metrics(model_dir, result)
                _plot_cellwise_diagnostics(
                    model_dir,
                    result,
                    config=config,
                    wandb_run=wandb_run,
                )
                _persist_cellwise_model(model_dir, result, config.training)
                if result.history:
                    history_df = _order_training_history_columns(pd.DataFrame(result.history))
                    history_csv = model_dir / "training_history.csv"
                    history_df.to_csv(history_csv, index=False)
                    for metric in ("loss", "pearson", "r2", "spearman"):
                        plot_training_history_curves(
                            history_df,
                            metric,
                            model_dir / f"training_history_{metric}.png",
                            title=f"{model_name.upper()} | {metric.title()}",
                        )

                if getattr(result, "feature_importances", None) is not None and getattr(result, "feature_names", None) is not None:
                    fi_summary = _export_feature_importance_artifacts(
                        model_dir,
                        model_name,
                        np.asarray(result.feature_importances, dtype=np.float64),
                        list(result.feature_names),
                        feature_importance_mean_signed=(
                            np.asarray(result.feature_importance_mean_signed, dtype=np.float64)
                            if getattr(result, "feature_importance_mean_signed", None) is not None
                            else None
                        ),
                        method=getattr(result, "feature_importance_method", None),
                        export_per_gene_panels=config.training.enable_per_gene_panels,
                        gene_names=result.gene_names,
                        feature_block_slices=getattr(result, "feature_block_slices", None),
                        feature_block_indices=getattr(result, "feature_block_indices", None),
                        gene_infos=getattr(result, "gene_infos", None),
                    )
                    if wandb_run is not None and fi_summary:
                        fi_metric_payload = _compact_feature_importance_metric_payload(fi_summary)
                        if fi_metric_payload:
                            # Keep feature-importance aggregates in summary only (no time-series charts).
                            wandb_update_summary(wandb_run, fi_metric_payload)
                if getattr(result, "shap_importance_mean", None) is not None and getattr(result, "feature_names", None) is not None:
                    shap_summary = _export_shap_importance_artifacts(
                        model_dir,
                        model_name,
                        np.asarray(result.shap_importance_mean, dtype=np.float64),
                        list(result.feature_names),
                        shap_value_mean_signed=(
                            np.asarray(result.shap_value_mean_signed, dtype=np.float64)
                            if getattr(result, "shap_value_mean_signed", None) is not None
                            else None
                        ),
                        method=getattr(result, "shap_importance_method", None),
                        gene_names=result.gene_names,
                        feature_block_slices=getattr(result, "feature_block_slices", None),
                        feature_block_indices=getattr(result, "feature_block_indices", None),
                        gene_infos=getattr(result, "gene_infos", None),
                    )
                    if wandb_run is not None and shap_summary:
                        shap_metric_payload = _compact_shap_metric_payload(shap_summary)
                        if shap_metric_payload:
                            wandb_update_summary(wandb_run, shap_metric_payload)

                metric_payload = {
                    "model": model_name,
                    "dataset": infer_dataset_name(config),
                    "num_genes": dataset.num_genes(),
                }
                summary_payload: Dict[str, Any] = {}
                for split in ("train", "val", "test"):
                    metrics = result.aggregate_metrics.get(split, {})
                    for metric_name in ("pearson", "r2", "spearman", "rmse", "mse", "mae"):
                        key = f"{split}_{metric_name}"
                        value = metrics.get(metric_name)
                        metric_payload[key] = value
                        summary_payload[key] = value
                summary_records.append(metric_payload)
                if wandb_run is not None:
                    wandb_update_summary(wandb_run, summary_payload)

                # Verify all critical files were written before logging completion
                critical_files = [
                    model_dir / "metrics_aggregate.csv",
                ]
                if config.training.export_raw_predictions:
                    critical_files.append(model_dir / "predictions_raw.csv")
                all_files_exist = all(f.exists() for f in critical_files)
                if all_files_exist:
                    _LOG.info(
                        "✓ SUCCESS | Completed multi-output training | model=%s | genes=%d | all outputs saved & ready for analysis",
                        model_name,
                        dataset.num_genes(),
                    )
                else:
                    missing = [f.name for f in critical_files if not f.exists()]
                    _LOG.warning(
                        "Training completed but some outputs missing | model=%s | missing=%s",
                        model_name,
                        ", ".join(missing),
                    )

                model_export_meta[model_name]["status"] = "succeeded"
                if model_name not in model_config_snapshots and result.fitted_model is not None:
                    model_config_snapshots[model_name] = _capture_model_configuration(result.fitted_model)
            finally:
                export_run_configuration_snapshot()

        if failures:
            raise RuntimeError(
                "One or more models failed in multi-output mode: "
                + "; ".join(failures)
            )

        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_df = _reorder_metric_columns(summary_df)
            summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)

            export_run_configuration_snapshot()

            if wandb_run is not None and config.wandb.log_tables:
                table_max = config.wandb.table_max_rows
                log_tables_from_csv(
                    wandb_run,
                    "Tables/summary_metrics",
                    run_dir / "summary_metrics.csv",
                    max_rows=table_max,
                )
                for model_name in config.all_models():
                    model_dir = run_dir / "models" / model_name
                    log_tables_from_csv(
                        wandb_run,
                        "Tables/metrics_aggregate",
                        model_dir / "metrics_aggregate.csv",
                        max_rows=table_max,
                    )
                    log_tables_from_csv(
                        wandb_run,
                        "Tables/metrics_per_gene",
                        model_dir / "metrics_per_gene.csv",
                        max_rows=table_max,
                    )
                    log_tables_from_csv(
                        wandb_run,
                        "Tables/fi_per_gene_summary",
                        model_dir / "feature_importance_per_gene_summary.csv",
                        max_rows=table_max,
                    )
                    log_tables_from_csv(
                        wandb_run,
                        "Tables/shap_per_gene_summary",
                        model_dir / "shapley_values_per_gene_summary.csv",
                        max_rows=table_max,
                    )
                    if config.wandb.log_predictions_table:
                        log_tables_from_csv(
                            wandb_run,
                            "Tables/predictions",
                            model_dir / "predictions_raw.csv",
                            max_rows=table_max,
                        )
                    log_tables_from_csv(
                        wandb_run,
                        "Tables/training_history",
                        model_dir / "training_history.csv",
                        max_rows=table_max,
                    )
                    log_training_history_charts_from_csv(
                        wandb_run,
                        model_dir / "training_history.csv",
                        prefix=f"Training_History/{model_name.upper()}",
                    )

        if wandb_run is not None and config.wandb.log_media:
            for model_name in config.all_models():
                model_dir = run_dir / "models" / model_name
                chart_prefix = f"Multi_Output/{model_name.upper()}"
                log_prediction_charts_from_csv(
                    wandb_run,
                    model_dir / "predictions_raw.csv",
                    prefix=chart_prefix,
                )
                log_metric_distribution_charts_from_csv(
                    wandb_run,
                    model_dir / "metrics_per_gene.csv",
                    prefix=chart_prefix,
                )

            max_media = config.wandb.media_max_items
            log_images_from_globs(
                wandb_run,
                run_dir,
                patterns=["models/*/feature_importance_mean.png"],
                max_items=max_media,
                group_key="Plots/FI",
            )
            log_images_from_globs(
                wandb_run,
                run_dir,
                patterns=["models/*/feature_importance_vs_tss_distance.png"],
                max_items=max_media,
                group_key="Plots/FI/TSS_Distance",
            )
            log_images_from_globs(
                wandb_run,
                run_dir,
                patterns=["models/*/feature_importance_mean_signed.png"],
                max_items=max_media,
                group_key="Plots/FI/Signed",
            )
            log_images_from_globs(
                wandb_run,
                run_dir,
                patterns=["models/*/feature_importance_distance_overview.png"],
                max_items=max_media,
                group_key="Plots/FI/Cumulative_Overview",
            )
            log_images_from_globs(
                wandb_run,
                run_dir,
                patterns=["models/*/shapley_values_mean.png"],
                max_items=max_media,
                group_key="Plots/SHAP",
            )
            log_images_from_globs(
                wandb_run,
                run_dir,
                patterns=["models/*/shapley_values_mean_signed.png"],
                max_items=max_media,
                group_key="Plots/SHAP/Signed",
            )
            log_images_from_globs(
                wandb_run,
                run_dir,
                patterns=[
                    "models/*/shapley_values_vs_tss.png",
                    "models/*/shapley_values_vs_tss_zoomed.png",
                ],
                max_items=max_media,
                group_key="Plots/SHAP/TSS_Distance",
            )

        if wandb_run is not None and config.wandb.log_artifacts:
            log_run_artifacts(wandb_run, run_dir)

        overall_status = "succeeded"
        return run_dir
    finally:
        # Log resource usage summary from entire run
        try:
            resource_summary = get_resource_summary()
            if resource_summary["peak_rss_gib"] > 0 or resource_summary["peak_gpu_allocated_mb"] > 0:
                summary_parts = []
                if resource_summary["peak_rss_gib"] > 0:
                    summary_parts.append(f"peak_rss={resource_summary['peak_rss_gib']:.2f} GiB")
                if resource_summary["peak_cpu_pct"] > 0:
                    summary_parts.append(f"peak_cpu={resource_summary['peak_cpu_pct']:.1f}%")
                if resource_summary["peak_gpu_allocated_mb"] > 0:
                    summary_parts.append(f"peak_gpu_allocated={resource_summary['peak_gpu_allocated_mb']:.0f} MB")
                if resource_summary["peak_gpu_reserved_mb"] > 0:
                    summary_parts.append(f"peak_gpu_reserved={resource_summary['peak_gpu_reserved_mb']:.0f} MB")
                if resource_summary["peak_gpu_free_mb"] != float("inf") and resource_summary["peak_gpu_free_mb"] >= 0:
                    summary_parts.append(f"min_gpu_free={resource_summary['peak_gpu_free_mb']:.0f} MB")
                if resource_summary["max_gpu_devices"] > 0:
                    summary_parts.append(f"gpu_devices={resource_summary['max_gpu_devices']}")
                
                if summary_parts:
                    _LOG.info("═" * 80)
                    _LOG.info("RESOURCE USAGE SUMMARY (peak values across entire run)")
                    _LOG.info("═" * 80)
                    _LOG.info("Run resource peaks | %s", " | ".join(summary_parts))
                    _LOG.info("═" * 80)
                # W&B already logs system metrics; omit our resource summary from W&B to avoid redundancy.
                pass
        except Exception:  # pragma: no cover
            _LOG.debug("Failed to log resource summary", exc_info=True)
        
        model_status_snapshot = {
            name: meta.get("status", "pending") for name, meta in model_export_meta.items()
        }
        try:
            _update_run_status_overview(
                base_dir,
                run_dir,
                config.run_name,
                model_status_snapshot,
                overall_status,
            )
        except Exception:  # pragma: no cover - diagnostics only
            _LOG.warning("Failed to update run status overview", exc_info=True)


def _resolve_rna_index(name_to_idx: Dict[str, int], gene: GeneInfo) -> Optional[int]:
    idx = name_to_idx.get(gene.gene_name)
    if idx is not None:
        return idx
    return name_to_idx.get(gene.gene_id)


def _genes_expressed_above_fraction(
    genes: List[GeneInfo],
    rna: ad.AnnData,
    *,
    min_expression: float,
    min_fraction: float,
) -> Tuple[List[GeneInfo], Dict[str, float]]:
    total_cells = int(rna.n_obs)
    if total_cells == 0:
        return [], {}

    var_names = np.asarray(rna.var_names).astype(str)
    name_to_idx = {name: idx for idx, name in enumerate(var_names)}

    # Vectorized evaluation to avoid per-gene materialization of full columns
    lookup: List[Tuple[GeneInfo, int]] = []
    for gene in genes:
        idx = _resolve_rna_index(name_to_idx, gene)
        if idx is not None:
            lookup.append((gene, idx))

    if not lookup:
        return [], {}

    genes_present, indices = zip(*lookup)
    indices_arr = np.fromiter(indices, dtype=np.int64)

    _LOG.info(
        "Evaluating expression fractions | genes=%d | cells=%d | min_expression=%.3f | min_fraction=%.3f",
        len(indices_arr),
        total_cells,
        min_expression,
        min_fraction,
    )
    start = time.perf_counter()

    matrix = rna.X[:, indices_arr]
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
        if min_expression <= 0:
            counts = np.asarray(matrix.getnnz(axis=0)).ravel()
        else:
            mask = matrix.data >= min_expression
            counts = np.bincount(matrix.indices[mask], minlength=matrix.shape[1])
    else:
        arr = np.asarray(matrix)
        counts = (arr >= min_expression).sum(axis=0)

    fractions: Dict[str, float] = {}
    candidates: List[GeneInfo] = []

    for gene, count in zip(genes_present, counts):
        fraction = float(count / total_cells)
        fractions[gene.gene_name] = fraction
        if fraction >= min_fraction:
            candidates.append(gene)

    duration = time.perf_counter() - start
    _LOG.info(
        "Expression filtering complete | kept=%d/%d genes | duration=%.2fs",
        len(candidates),
        len(genes_present),
        duration,
    )

    return candidates, fractions


def _choose_random_genes(
    genes: List[GeneInfo],
    count: int,
    random_state: int,
) -> List[GeneInfo]:
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(genes), size=count, replace=False)
    sampled = [genes[int(i)] for i in indices]
    return sampled


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _order_training_history_columns(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df
    ordered_cols = []
    if "epoch" in history_df.columns:
        ordered_cols.append("epoch")
    for metric in ("loss",) + _METRIC_ORDER:
        for split in ("train", "val", "test"):
            metric_col = f"{split}_{metric}"
            if metric_col in history_df.columns:
                ordered_cols.append(metric_col)
    for col in (
        "effective_batch_size",
        "cpu_percent",
        "rss_gib",
        "thread_count",
        "gpu_alloc_mb",
        "gpu_reserved_mb",
        "gpu_peak_alloc_mb",
        "gpu_util_pct",
    ):
        if col in history_df.columns:
            ordered_cols.append(col)
    remaining = [c for c in history_df.columns if c not in ordered_cols]
    return history_df[ordered_cols + remaining]


def _reorder_metric_columns(
    df: pd.DataFrame,
    *,
    prefixes: Sequence[str] = ("importance", "train", "val", "test"),
    metric_order: Sequence[str] = _METRIC_ORDER,
    suffix_order: Sequence[str] = ("mean", "std"),
) -> pd.DataFrame:
    if df.empty:
        return df
    cols = list(df.columns)
    metric_cols = set()
    for metric in metric_order:
        metric_cols.add(metric)
        for suffix in suffix_order:
            metric_cols.add(f"{metric}_{suffix}")
        for prefix in prefixes:
            metric_cols.add(f"{prefix}_{metric}")
            for suffix in suffix_order:
                metric_cols.add(f"{prefix}_{metric}_{suffix}")

    if not any(col in metric_cols for col in cols):
        return df

    ordered: List[str] = []
    id_cols = [c for c in cols if c not in metric_cols]
    ordered.extend(id_cols)

    for metric in metric_order:
        if metric in cols:
            ordered.append(metric)

    for metric in metric_order:
        for prefix in prefixes:
            base = f"{prefix}_{metric}"
            if base in cols:
                ordered.append(base)
            for suffix in suffix_order:
                col = f"{prefix}_{metric}_{suffix}"
                if col in cols:
                    ordered.append(col)
            for col in cols:
                if col.startswith(f"{base}_") and col not in ordered:
                    ordered.append(col)

    for metric in metric_order:
        for suffix in suffix_order:
            col = f"{metric}_{suffix}"
            if col in cols and col not in ordered:
                ordered.append(col)
        for col in cols:
            if col.startswith(f"{metric}_") and col not in ordered:
                ordered.append(col)

    remaining = [c for c in cols if c not in ordered]
    return df[ordered + remaining]


@contextmanager
def _temporary_env_var(key: str, value: str) -> Iterator[None]:
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def _write_selected_genes(
    run_dir: Path,
    genes: List[GeneInfo],
    gene_expression_fraction: Optional[Dict[str, float]],
) -> None:
    if not genes:
        return
    out_path = run_dir / "selected_genes.csv"
    rows = ["gene_id,gene_name,chrom,expression_fraction"]
    expr_map = gene_expression_fraction or {}
    for gene in sorted(genes, key=lambda g: (g.gene_name.lower(), g.gene_id.lower())):
        frac = expr_map.get(gene.gene_name, float("nan"))
        rows.append(f"{gene.gene_id},{gene.gene_name},{gene.chrom},{frac}")
    out_path.write_text("\n".join(rows) + "\n")
    _LOG.info("Recorded selected gene list to %s", out_path)


def _cellwise_predictions_dataframe(result: CellwiseModelResult) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for split, payload in result.split_predictions.items():
        cell_ids = payload["cell_ids"]
        y_true = payload["y_true"]
        y_pred = payload["y_pred"]
        for cell_idx, cell_id in enumerate(cell_ids):
            for gene_idx, gene in enumerate(result.gene_names):
                rows.append(
                    {
                        "split": split,
                        "cell_id": cell_id,
                        "gene": gene,
                        "y_true": float(y_true[cell_idx, gene_idx]),
                        "y_pred": float(y_pred[cell_idx, gene_idx]),
                    }
                )
    return pd.DataFrame(rows)


def _write_cellwise_metrics(model_dir: Path, result: CellwiseModelResult) -> None:
    _ensure_directory(model_dir)
    agg_df = pd.DataFrame(result.aggregate_metrics).T
    agg_df.index.name = "split"
    agg_df = _reorder_metric_columns(agg_df)
    agg_df.to_csv(model_dir / "metrics_aggregate.csv")

    per_gene_rows: List[Dict[str, object]] = []
    for split, metrics_list in result.per_gene_metrics.items():
        for metrics in metrics_list:
            row = dict(metrics)
            row["split"] = split
            per_gene_rows.append(row)
    if per_gene_rows:
        per_gene_df = pd.DataFrame(per_gene_rows)
        if "gene" in per_gene_df.columns and "split" in per_gene_df.columns:
            per_gene_df = _reorder_metric_columns(per_gene_df, prefixes=("importance",))
        per_gene_df.to_csv(model_dir / "metrics_per_gene.csv", index=False)

    if result.cv_metrics:
        cv_df = pd.DataFrame(
            [{"fold": fm.fold, **fm.metrics} for fm in result.cv_metrics]
        )
        cv_df = _reorder_metric_columns(cv_df, prefixes=("importance",))
        cv_df.to_csv(model_dir / "metrics_cv.csv", index=False)


def _plot_cellwise_diagnostics(
    model_dir: Path,
    result: CellwiseModelResult,
    *,
    config: Optional[PipelineConfig] = None,
    wandb_run: Optional[Any] = None,
) -> None:
    _ensure_directory(model_dir)
    residuals_by_split: Dict[str, np.ndarray] = {}
    for split, payload in result.split_predictions.items():
        y_true_matrix = payload["y_true"]
        y_pred_matrix = payload["y_pred"]
        y_true = y_true_matrix.ravel()
        y_pred = y_pred_matrix.ravel()
        if y_true.size == 0:
            continue
        # Keep per-split residuals for combined plots.
        residuals_by_split[split] = (y_pred_matrix - y_true_matrix).astype(np.float64)
        plot_predictions_vs_actual(
            y_true,
            y_pred,
            model_dir / f"scatter_{split}.png",
            f"{result.model_name.upper()} | {split}",
            annotation_metrics=result.aggregate_metrics.get(split),
        )

    if residuals_by_split:
        plot_residual_histogram_by_split(
            residuals_by_split,
            model_dir / "residuals.png",
            f"Residuals | {result.model_name.upper()}",
        )
        plot_residual_barplot_by_split(
            residuals_by_split,
            result.gene_names,
            model_dir / "residuals_bar.png",
            f"Mean residuals by split | {result.model_name.upper()}",
        )

    def _collect_metric(per_gene: List[Dict[str, float]], key: str) -> List[float]:
        values: List[float] = []
        for entry in per_gene:
            val = entry.get(key)
            if val is None:
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(num):
                values.append(num)
        return values

    metric_labels = {
        "pearson": "Pearson correlation coefficient",
        "spearman": "Spearman correlation coefficient",
        "r2": "R^2",
        "rmse": "RMSE",
        "mse": "MSE",
        "mae": "MAE",
    }
    for metric in _METRIC_ORDER:
        values_by_group: Dict[str, List[float]] = {}
        for split in ("train", "val", "test"):
            per_gene = result.per_gene_metrics.get(split, [])
            split_values = _collect_metric(per_gene, metric)
            if not split_values:
                continue
            values_by_group[split.title()] = split_values
        if not values_by_group:
            continue
        plot_box_violin_half_split(
            values_by_group,
            model_dir / f"by_split_{metric}.png",
            f"{result.model_name.upper()} | {metric.upper()}",
            metric_labels.get(metric, metric.upper()),
            order=["Train", "Val", "Test"],
        )

    per_gene_train = result.per_gene_metrics.get("train", [])
    per_gene_test = result.per_gene_metrics.get("test", [])
    if per_gene_train and per_gene_test:
        train_map: Dict[str, float] = {}
        for entry in per_gene_train:
            gene = entry.get("gene")
            val = entry.get("pearson")
            if gene is None or val is None:
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            if math.isfinite(num):
                train_map[str(gene)] = num
        gaps: List[float] = []
        for entry in per_gene_test:
            gene = entry.get("gene")
            val = entry.get("pearson")
            if gene is None or val is None:
                continue
            try:
                test_val = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(test_val):
                continue
            train_val = train_map.get(str(gene))
            if train_val is None:
                continue
            gaps.append(train_val - test_val)
        if gaps:
            gap_mean = float(np.nanmean(gaps))
            if wandb_run is not None:
                wandb_update_summary(
                    wandb_run,
                    {
                        "generalization_gap_train_test_pearson_mean": gap_mean,
                    },
                )
            plot_single_box_violin(
                gaps,
                model_dir / "generalization_gap_pearson_train_test.png",
                f"{result.model_name.upper()} | Train-Test Pearson Gap",
                "Train - Test Pearson",
            )

def _persist_cellwise_model(
    model_dir: Path,
    result: CellwiseModelResult,
    training: TrainingConfig,
) -> None:
    """Persist fitted model and scalers for later inference."""

    model_dir.mkdir(parents=True, exist_ok=True)
    model = result.fitted_model
    if model is None:
        return

    meta = {
        "model_name": result.model_name,
        "gene_names": result.gene_names,
        "feature_names": result.feature_names,
        "feature_block_slices": result.feature_block_slices,
        "feature_block_indices": _serialize_value(result.feature_block_indices),
        "reshape": result.reshape,
        "training": _serialize_value(training),
    }
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    try:
        if _torch_nn is not None and isinstance(model, _torch_nn.Module):
            # Unwrap DataParallel to avoid 'module.' prefix in state dict keys
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            state = {
                "state_dict": model_to_save.state_dict(),
                "model_class": model_to_save.__class__.__name__,
                "reshape": result.reshape,
            }
            torch.save(state, model_dir / "model.pt")
        else:
            joblib.dump(model, model_dir / "model.pkl")
    except Exception as exc:  # pragma: no cover - defensive
        _LOG.warning("Failed to persist model artifact for %s: %s", result.model_name, exc)

    scalers = {
        "feature_scaler.pkl": result.feature_scaler,
        "target_scaler.pkl": result.target_scaler,
    }
    for name, scaler in scalers.items():
        if scaler is None:
            continue
        try:
            joblib.dump(scaler, model_dir / name)
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("Failed to persist %s for %s: %s", name, result.model_name, exc)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):  # numpy scalar
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _serialize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_value(item) for item in value]
    if is_dataclass(value):
        return {key: _serialize_value(val) for key, val in asdict(value).items()}
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive conversion
            pass
    if hasattr(value, "__dict__") and not isinstance(value, type):
        try:
            return {key: _serialize_value(val) for key, val in vars(value).items()}
        except Exception:  # pragma: no cover - defensive conversion
            pass
    return repr(value)


def _capture_model_configuration(model: object) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "type": model.__class__.__name__,
        "module": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
    }
    if _torch_nn is not None and isinstance(model, _torch_nn.Module):
        try:
            total_params = int(sum(param.numel() for param in model.parameters()))
            trainable_params = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
        except Exception:  # pragma: no cover - fallback when parameters unavailable
            total_params = trainable_params = 0
        summary.update(
            {
                "framework": "torch",
                "parameter_count": total_params,
                "trainable_parameter_count": trainable_params,
                "representation": repr(model),
            }
        )
        return summary
    if hasattr(model, "get_params"):
        try:
            params = model.get_params(deep=True)
        except Exception as exc:  # pragma: no cover - estimator without get_params support
            params = {"_error": f"get_params failed: {exc}"}
        summary.update(
            {
                "framework": "sklearn",
                "parameters": _serialize_value(params),
                "representation": repr(model),
            }
        )
        return summary
    summary["representation"] = repr(model)
    return summary


def _utc_timestamp() -> str:
    """Return a timezone-aware UTC timestamp with a trailing Z."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_dataset_manifest(config: PipelineConfig, run_dir: Path) -> None:
    if not config.wandb.log_dataset_manifest:
        return
    payload: Dict[str, Any] = {
        "created_utc": _utc_timestamp(),
        "base_dir": str(config.paths.base_dir),
        "datasets": {},
    }
    paths = {
        "atac_path": config.paths.atac_path,
        "rna_path": config.paths.rna_path,
        "gtf_path": config.paths.gtf_path,
    }
    for key, path in paths.items():
        entry: Dict[str, Any] = {"path": str(path)}
        try:
            resolved = path.expanduser().resolve()
            entry["resolved_path"] = str(resolved)
            if resolved.exists():
                stat = resolved.stat()
                entry["exists"] = True
                entry["size_bytes"] = stat.st_size
                entry["mtime_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(
                    microsecond=0
                ).isoformat().replace("+00:00", "Z")
            else:
                entry["exists"] = False
        except Exception as exc:  # pragma: no cover - best-effort manifest
            entry["exists"] = False
            entry["error"] = str(exc)
        payload["datasets"][key] = entry

    output_path = run_dir / "dataset_manifest.json"
    try:
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
    except Exception:  # pragma: no cover - best-effort manifest
        _LOG.debug("Failed to write dataset manifest to %s", output_path, exc_info=True)


def _export_run_configuration(
    config: PipelineConfig,
    run_dir: Path,
    model_details: Dict[str, Any],
    extra_context: Optional[Dict[str, Any]] = None,
) -> None:
    genes_payload: Optional[Sequence[str]] = config.genes
    if extra_context:
        for key in ("requested_genes", "gene_names"):
            candidates = extra_context.get(key)
            if isinstance(candidates, list) and candidates:
                genes_payload = candidates
                break
    payload: "OrderedDict[str, Any]" = OrderedDict()
    payload["model_configurations"] = _serialize_value(model_details)
    payload["pipeline_config"] = {
        "paths": _serialize_value(config.paths),
        "training": _serialize_value(config.training),
        "models": _serialize_value(config.models),
        "wandb": _serialize_value(config.wandb),
        "genes": _serialize_value(genes_payload),
        "chromosomes": _serialize_value(config.chromosomes),
        "max_genes": config.max_genes,
        "chunk_total": config.chunk_total,
        "chunk_index": config.chunk_index,
    }
    payload["run_name"] = config.run_name
    payload["timestamp_utc"] = _utc_timestamp()
    if extra_context:
        payload["run_context"] = _serialize_value(extra_context)

    output_path = run_dir / "run_configuration.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    _LOG.info("Exported run configuration snapshot to %s", output_path)
    _write_dataset_manifest(config, run_dir)


def _update_run_status_overview(
    base_dir: Path,
    run_dir: Path,
    run_name: Optional[str],
    model_statuses: Dict[str, str],
    overall_status: str,
) -> None:
    summary_path = base_dir / "run_status_overview.json"
    try:
        summary = json.loads(summary_path.read_text())
    except FileNotFoundError:
        summary = {}
    except json.JSONDecodeError:
        summary = {}

    summary.setdefault("succeeded", [])
    summary.setdefault("failed", [])

    identifier = run_name or run_dir.name
    run_path = str(run_dir.resolve())
    updated_at = _utc_timestamp()

    def _filtered(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            entry
            for entry in entries
            if entry.get("run_name") != identifier and entry.get("path") != run_path
        ]

    summary["succeeded"] = _filtered(summary["succeeded"])
    summary["failed"] = _filtered(summary["failed"])

    entry: Dict[str, Any] = {
        "run_name": identifier,
        "path": run_path,
        "model_statuses": model_statuses,
        "updated_at": updated_at,
    }
    target = "succeeded" if overall_status == "succeeded" else "failed"

    summary[target].append(entry)

    def _sort(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(
            entries,
            key=lambda item: item.get("updated_at", ""),
            reverse=True,
        )

    summary["succeeded"] = _sort(summary["succeeded"])
    summary["failed"] = _sort(summary["failed"])
    summary["generated_at"] = _utc_timestamp()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def _extract_feature_importance(result: ModelResult) -> np.ndarray | None:
    model = getattr(result, "fitted_model", None)
    if model is None:
        return None
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_, dtype=np.float64)
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        return np.abs(coef.ravel())
    return None
