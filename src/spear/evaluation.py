
import json
import logging
import math
import os
import re
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
    log_images_from_globs,
    log_run_artifacts,
    log_tables_from_csv,
    maybe_init_wandb,
    wandb_finish,
    wandb_log_metrics,
    wandb_update_summary,
)
from .visualization import (
    plot_feature_importance,
    plot_correlation_boxplot,
    plot_correlation_violin,
    plot_predictions_vs_actual,
    plot_residual_barplot,
    plot_residual_histogram,
    plot_training_history_curves,
    plot_training_history_series,
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


def _export_feature_importance_artifacts(
    output_dir: Path,
    model_name: str,
    importances: np.ndarray,
    feature_names: Sequence[str],
    *,
    method: Optional[str] = None,
    gene_names: Optional[Sequence[str]] = None,
    feature_block_slices: Optional[Sequence[Tuple[int, int]]] = None,
    feature_block_indices: Optional[Sequence[Sequence[int]]] = None,
    gene_infos: Optional[Sequence[GeneInfo]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fi = np.asarray(importances, dtype=np.float64)
    feature_count = int(fi.size if fi.ndim == 1 else fi.shape[-1])
    if feature_count == 0:
        _LOG.info("Feature importance export skipped | model=%s | reason=no features", model_name)
        return

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
    if metadata_df is not None and not metadata_df.empty:
        aggregate_df = pd.concat([aggregate_df, metadata_df], axis=1)

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

    per_gene_summary_path: Optional[Path] = None
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
                "importance_mean_avg": float(block["importance_mean"].mean()),
                "top_feature": str(block.loc[block["importance_mean"].idxmax(), "feature"]),
                "top_feature_importance": float(block["importance_mean"].max()),
            }
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
            _LOG.info(
                "Saved per-gene feature importance summary (%d genes) to %s",
                per_gene_df.shape[0],
                per_gene_summary_path,
            )

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
                "importance_mean_avg": float(block["importance_mean"].mean()),
                "top_feature": str(block.loc[block["importance_mean"].idxmax(), "feature"]),
                "top_feature_importance": float(block["importance_mean"].max()),
            }
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
            _LOG.info(
                "Saved per-gene feature importance summary (%d genes) to %s",
                per_gene_df.shape[0],
                per_gene_summary_path,
            )

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

    summary_payload: Dict[str, object] = {
        "method": method or "unknown",
        "num_features": int(fi_mean.size),
        "raw_importances_file": raw_path.name,
        "aggregate_file": aggregate_path.name,
    }
    if per_gene_summary_path is not None:
        summary_payload["per_gene_summary_file"] = per_gene_summary_path.name

    if "signed_distance_to_tss_kb" in aggregate_df.columns:
        distances = pd.to_numeric(aggregate_df["signed_distance_to_tss_kb"], errors="coerce")
        mask = np.isfinite(distances) & np.isfinite(fi_mean)
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
                scatter_path = output_dir / "feature_importance_vs_tss_distance.png"
                plot_importance_distance_scatter(
                    fi_mean[mask],
                    dist,
                    scatter_path,
                    f"FI vs TSS distance | {model_name.upper()}",
                    annotation={"Spearman": spearman, "Pearson": pearson},
                )
                corr_path = output_dir / "feature_importance_tss_correlation.json"
                corr_path.write_text(json.dumps(corr_payload, indent=2))
                _LOG.info(
                    "Saved FI vs TSS scatter and correlation stats (n=%d) to %s and %s",
                    mask.sum(),
                    scatter_path,
                    corr_path,
                )

            overlay_path = output_dir / "feature_importance_distance_overview.png"
            plot_cumulative_importance_overlay(
                fi_mean[mask],
                dist,
                overlay_path,
                f"FI cumulative distance profile | {model_name.upper()}",
            )
            _LOG.info("Saved FI distance overlay to %s", overlay_path)

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


def _export_shap_importance_artifacts(
    output_dir: Path,
    model_name: str,
    shap_importances: np.ndarray,
    feature_names: Sequence[str],
    *,
    method: Optional[str] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shap_values = np.asarray(shap_importances, dtype=np.float64)
    if shap_values.size == 0 or shap_values.ndim != 1:
        _LOG.info("SHAP export skipped | model=%s | reason=no values", model_name)
        return

    plot_feature_importance(
        shap_values,
        feature_names,
        output_dir / "shap_importance_mean.png",
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

    # Add metadata columns (TSS distance, gene name, etc.) if available
    if metadata_df is not None and not metadata_df.empty:
        shap_df = pd.concat([shap_df, metadata_df], axis=1)
    shap_path = output_dir / "shap_importances_mean.csv"
    shap_df.to_csv(shap_path, index=False)
    _LOG.info("Saved SHAP mean importances (%d rows) to %s", shap_df.shape[0], shap_path)

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
        max_distance_kb: float,
        show_scatter: bool = False,
        y_limits: Optional[tuple[float, float]] = None,
    ) -> bool:
        plot_df = pd.DataFrame(
            {
                "distance_kb": distance_kb,
                "shap_value": pd.to_numeric(table["shap_mean_abs"], errors="coerce"),
            }
        )
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna()
        if plot_df.empty:
            return False
        plot_df = plot_df[plot_df["distance_kb"].abs() <= max_distance_kb].copy()
        if plot_df.empty:
            return False
        plot_df.sort_values("distance_kb", inplace=True)

        per_bin = (
            plot_df.groupby("distance_kb", sort=True)["shap_value"]
            .quantile(0.9)
            .reset_index()
        )
        if per_bin.empty:
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
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
            label="90th percentile",
        )
        ax.legend(loc="upper right", frameon=True)
        ax.axvline(0.0, color="#999999", linestyle="--", linewidth=1)
        ax.set_xlim(-max_distance_kb, max_distance_kb)
        if y_limits is not None:
            ax.set_ylim(y_limits)
        ax.set_xlabel("Distance to TSS (kb)")
        ax.set_ylabel("Mean |SHAP|")
        ax.set_title(title)
        sns.despine(fig, left=True, bottom=True)
        plt.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return True

    distance_kb, distance_col, distance_signed = _extract_distance_kb(shap_df)
    distance_plot_path = output_dir / "shap_vs_tss.png"
    distance_plot_zoomed_path = output_dir / "shap_vs_tss_zoomed.png"
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

    summary = {
        "model": model_name,
        "method": method,
        "num_features": int(shap_values.size),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(shap_path),
        "plot_path": str(output_dir / "shap_importance_mean.png"),
    }
    if distance_plot_path is not None:
        summary["distance_plot_path"] = str(distance_plot_path)
    if distance_plot_zoomed_path is not None:
        summary["distance_plot_zoomed_path"] = str(distance_plot_zoomed_path)
    summary_path = output_dir / "shap_importance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _LOG.info("Wrote SHAP summary manifest to %s", summary_path)

try:
    import torch.nn as _torch_nn
except ImportError:  # pragma: no cover - torch optional during some tests
    _torch_nn = None

def run_pipeline(config: PipelineConfig) -> Path:
    config.ensure_directories()

    wandb_run = maybe_init_wandb(config)
    run_status = "failed"
    run_dir: Optional[Path] = None
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
    
    finally:
        wandb_finish(wandb_run, status=run_status, run_dir=run_dir)


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
    
    base_dir = config.paths.output_dir / "spear_results"
    run_dir = base_dir / config.run_name if config.run_name else base_dir
    
    if not genes:
        _LOG.warning(
            "No genes assigned to this chunk (chunk_index=%d, chunk_total=%d). Nothing to process.",
            chunk_index,
            chunk_total,
        )
        return run_dir
    
    _ensure_directory(run_dir)
    wandb_update_summary(
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
        summary_path = run_dir / "summary_metrics.csv"
        summary_df.to_csv(summary_path, index=False)
        _LOG.info("Run summary metrics saved to %s", summary_path)
    
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_order = {name: idx for idx, name in enumerate(config.all_models(), start=1)}
    for model_name, store in model_store.items():
        model_dir = models_dir / model_name
        model_dir.mkdir(exist_ok=True)
    
        predictions = store["predictions"]
        if predictions:
            preds_df = pd.concat(predictions, ignore_index=True)
            preds_path = model_dir / "predictions_raw.csv"
            preds_df.to_csv(preds_path, index=False)
    
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
                plot_residual_histogram(
                    subset["y_true"].to_numpy(),
                    subset["y_pred"].to_numpy(),
                    model_dir / f"residuals_{split}.png",
                    f"Residuals | {model_name.upper()} | {split}",
                )
    
        metrics_records = store["metrics"]
        if metrics_records:
            metrics_df = pd.DataFrame(metrics_records)
            metrics_df.to_csv(model_dir / "metrics_by_gene.csv", index=False)
            metrics_mean = metrics_df.mean(numeric_only=True)
            metrics_mean.to_csv(model_dir / "metrics_summary.csv", header=["value"])
            metrics_payload: Dict[str, Any] = {
            }
            for key, value in metrics_mean.items():
                if "_" in key:
                    split, metric = key.split("_", 1)
                    metrics_payload[f"{split}/{metric}"] = value
                else:
                    metrics_payload[key] = value
            wandb_log_metrics(
                wandb_run,
                metrics_payload,
            )
    
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
    
        histories = store["histories"]
        if histories:
            history_dir = model_dir / "histories"
            history_dir.mkdir(exist_ok=True)
            for gene_name, history_records in histories:
                history_df = pd.DataFrame(history_records)
                history_csv = history_dir / f"{gene_name}.csv"
                history_df.to_csv(history_csv, index=False)
                for metric in ("loss", "pearson", "spearman"):
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
            "summary_metrics",
            run_dir / "summary_metrics.csv",
            max_rows=table_max,
        )
        for model_name in config.all_models():
            model_dir = run_dir / "models" / model_name
            log_tables_from_csv(
                wandb_run,
                f"{model_name}_metrics_by_gene",
                model_dir / "metrics_by_gene.csv",
                max_rows=table_max,
            )
            if config.wandb.log_predictions_table:
                log_tables_from_csv(
                    wandb_run,
                    f"{model_name}_predictions",
                    model_dir / "predictions_raw.csv",
                    max_rows=table_max,
                )

    if wandb_run is not None and config.wandb.log_media:
        max_media = config.wandb.media_max_items
        log_images_from_globs(
            wandb_run,
            run_dir,
            patterns=[
                "models/*/scatter_*.png",
                "models/*/residuals_*.png",
                "models/*/feature_importance_mean.png",
                "models/*/histories/*.png",
            ],
            max_items=max_media,
            prefix="plots",
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
    base_dir = config.paths.output_dir / "spear_results"
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
        wandb_update_summary(
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
                _plot_cellwise_diagnostics(model_dir, result)
                _persist_cellwise_model(model_dir, result, config.training)
                if result.history:
                    history_df = pd.DataFrame(result.history)
                    history_csv = model_dir / "training_history.csv"
                    history_df.to_csv(history_csv, index=False)
                    for metric in ("loss", "pearson", "spearman"):
                        plot_training_history_curves(
                            history_df,
                            metric,
                            model_dir / f"training_history_{metric}.png",
                            title=f"{model_name.upper()} | {metric.title()}",
                        )
                    plot_training_history_series(
                        history_df,
                        "gpu_util_pct",
                        model_dir / "training_history_gpu_util.png",
                        title=f"{model_name.upper()} | GPU Utilization",
                        ylabel="GPU Utilization (%)",
                    )
                    plot_training_history_series(
                        history_df,
                        "gpu_peak_alloc_mb",
                        model_dir / "training_history_gpu_mem.png",
                        title=f"{model_name.upper()} | GPU Peak Alloc",
                        ylabel="GPU Peak Alloc (MB)",
                    )
                    plot_training_history_series(
                        history_df,
                        "gpu_alloc_mb",
                        model_dir / "training_history_gpu_alloc.png",
                        title=f"{model_name.upper()} | GPU Allocated",
                        ylabel="GPU Allocated (MB)",
                    )
                    plot_training_history_series(
                        history_df,
                        "gpu_reserved_mb",
                        model_dir / "training_history_gpu_reserved.png",
                        title=f"{model_name.upper()} | GPU Reserved",
                        ylabel="GPU Reserved (MB)",
                    )
                    plot_training_history_series(
                        history_df,
                        "cpu_percent",
                        model_dir / "training_history_cpu_util.png",
                        title=f"{model_name.upper()} | CPU Utilization",
                        ylabel="CPU Utilization (%)",
                    )
                    plot_training_history_series(
                        history_df,
                        "rss_gib",
                        model_dir / "training_history_rss_gib.png",
                        title=f"{model_name.upper()} | RSS Memory",
                        ylabel="RSS (GiB)",
                    )
                    plot_training_history_series(
                        history_df,
                        "thread_count",
                        model_dir / "training_history_threads.png",
                        title=f"{model_name.upper()} | Thread Count",
                        ylabel="Threads",
                    )

                if getattr(result, "feature_importances", None) is not None and getattr(result, "feature_names", None) is not None:
                    _export_feature_importance_artifacts(
                        model_dir,
                        model_name,
                        np.asarray(result.feature_importances, dtype=np.float64),
                        list(result.feature_names),
                        method=getattr(result, "feature_importance_method", None),
                        gene_names=result.gene_names,
                        feature_block_slices=getattr(result, "feature_block_slices", None),
                        feature_block_indices=getattr(result, "feature_block_indices", None),
                        gene_infos=getattr(result, "gene_infos", None),
                    )
                if getattr(result, "shap_importances_mean", None) is not None and getattr(result, "feature_names", None) is not None:
                    _export_shap_importance_artifacts(
                        model_dir,
                        model_name,
                        np.asarray(result.shap_importances_mean, dtype=np.float64),
                        list(result.feature_names),
                        method=getattr(result, "shap_importance_method", None),
                    )

                metric_payload = {"model": model_name, "num_genes": dataset.num_genes()}
                for split in ("train", "val", "test"):
                    metrics = result.aggregate_metrics.get(split, {})
                    for metric_name in ("r2", "rmse", "mae", "spearman", "pearson"):
                        key = f"{split}_{metric_name}"
                        metric_payload[key] = metrics.get(metric_name)
                summary_records.append(metric_payload)
                wandb_payload = {}
                for key, value in metric_payload.items():
                    if key == "model" or key == "num_genes":
                        continue
                    if "_" in key:
                        split, metric = key.split("_", 1)
                        wandb_payload[f"{split}/{metric}"] = value
                    else:
                        wandb_payload[key] = value
                wandb_log_metrics(wandb_run, wandb_payload)

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

        if summary_records:
            summary_df = pd.DataFrame(summary_records)
            summary_df.to_csv(run_dir / "summary_metrics.csv", index=False)

        export_run_configuration_snapshot()

        if failures:
            wandb_update_summary(wandb_run, {"status": "failed", "failure_count": len(failures)})
            raise RuntimeError(
                "One or more models failed in multi-output mode: "
                + "; ".join(failures)
            )

        if wandb_run is not None and config.wandb.log_tables:
            table_max = config.wandb.table_max_rows
            log_tables_from_csv(
                wandb_run,
                "summary_metrics",
                run_dir / "summary_metrics.csv",
                max_rows=table_max,
            )
            for model_name in config.all_models():
                model_dir = run_dir / "models" / model_name
                log_tables_from_csv(
                    wandb_run,
                    f"{model_name}_metrics_aggregate",
                    model_dir / "metrics_aggregate.csv",
                    max_rows=table_max,
                )
                log_tables_from_csv(
                    wandb_run,
                    f"{model_name}_metrics_per_gene",
                    model_dir / "metrics_per_gene.csv",
                    max_rows=table_max,
                )
                if config.wandb.log_predictions_table:
                    log_tables_from_csv(
                        wandb_run,
                        f"{model_name}_predictions",
                        model_dir / "predictions_raw.csv",
                        max_rows=table_max,
                    )
                log_tables_from_csv(
                    wandb_run,
                    f"{model_name}_training_history",
                    model_dir / "training_history.csv",
                    max_rows=table_max,
                )

        if wandb_run is not None and config.wandb.log_media:
            max_media = config.wandb.media_max_items
            log_images_from_globs(
                wandb_run,
                run_dir,
                patterns=[
                    "models/*/training_history_*.png",
                    "models/*/feature_importance_mean.png",
                    "models/*/shap_importance_mean.png",
                ],
                max_items=max_media,
                prefix="plots",
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
                if wandb_run is not None:
                    summary_payload = {
                        "peak_rss_gib": resource_summary.get("peak_rss_gib"),
                        "peak_cpu_pct": resource_summary.get("peak_cpu_pct"),
                        "peak_gpu_allocated_mb": resource_summary.get("peak_gpu_allocated_mb"),
                        "peak_gpu_reserved_mb": resource_summary.get("peak_gpu_reserved_mb"),
                        "min_gpu_free_mb": resource_summary.get("peak_gpu_free_mb"),
                    }
                    if resource_summary.get("max_gpu_devices", 0) > 1:
                        summary_payload["gpu_devices"] = resource_summary.get("max_gpu_devices")
                    wandb_update_summary(
                        wandb_run,
                        summary_payload,
                    )
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
    agg_df.to_csv(model_dir / "metrics_aggregate.csv")

    per_gene_rows: List[Dict[str, object]] = []
    for split, metrics_list in result.per_gene_metrics.items():
        for metrics in metrics_list:
            row = dict(metrics)
            row["split"] = split
            per_gene_rows.append(row)
    if per_gene_rows:
        per_gene_df = pd.DataFrame(per_gene_rows)
        per_gene_df.to_csv(model_dir / "metrics_per_gene.csv", index=False)

    if result.cv_metrics:
        cv_df = pd.DataFrame(
            [{"fold": fm.fold, **fm.metrics} for fm in result.cv_metrics]
        )
        cv_df.to_csv(model_dir / "metrics_cv.csv", index=False)


def _plot_cellwise_diagnostics(model_dir: Path, result: CellwiseModelResult) -> None:
    _ensure_directory(model_dir)
    for split, payload in result.split_predictions.items():
        y_true_matrix = payload["y_true"]
        y_pred_matrix = payload["y_pred"]
        y_true = y_true_matrix.ravel()
        y_pred = y_pred_matrix.ravel()
        if y_true.size == 0:
            continue
        plot_predictions_vs_actual(
            y_true,
            y_pred,
            model_dir / f"scatter_{split}.png",
            f"{result.model_name.upper()} | {split}",
            annotation_metrics=result.aggregate_metrics.get(split),
        )
        plot_residual_histogram(
            y_true,
            y_pred,
            model_dir / f"residuals_{split}.png",
            f"Residuals | {result.model_name.upper()} | {split}",
        )
        plot_residual_barplot(
            y_true_matrix,
            y_pred_matrix,
            result.gene_names,
            model_dir / f"residual_bar_{split}.png",
            f"Mean absolute residuals | {result.model_name.upper()} | {split}",
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

    spearman_by_split: Dict[str, List[float]] = {}
    for split in ("train", "val", "test"):
        per_gene = result.per_gene_metrics.get(split, [])
        split_values = _collect_metric(per_gene, "spearman")
        if not split_values:
            continue
        spearman_by_split[split] = split_values
        title = f"{result.model_name.upper()} | {split.title()} | Spearman"
        plot_correlation_boxplot(
            split_values,
            output_path=model_dir / f"spearman_boxplot_{split}.png",
            title=title,
            metric_label="Spearman correlation coefficient",
        )
        plot_correlation_violin(
            split_values,
            output_path=model_dir / f"spearman_violin_{split}.png",
            title=title,
            metric_label="Spearman correlation coefficient",
        )

    if spearman_by_split:
        combined_records = [
            {"split": split.title(), "spearman": val}
            for split, values in spearman_by_split.items()
            for val in values
        ]
        combined_df = pd.DataFrame(combined_records)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        sns.boxplot(
            data=combined_df,
            x="split",
            y="spearman",
            hue="split",
            palette="Set2",
            legend=False,
            ax=axes[0],
        )
        sns.stripplot(
            data=combined_df,
            x="split",
            y="spearman",
            color="#2c3e50",
            alpha=0.45,
            size=3.5,
            jitter=0.18,
            ax=axes[0],
        )
        axes[0].set_title(f"{result.model_name.upper()} | Spearman Boxplots")
        axes[0].set_xlabel("Data split")
        axes[0].set_ylabel("Spearman correlation coefficient")

        sns.violinplot(
            data=combined_df,
            x="split",
            y="spearman",
            hue="split",
            palette="Set2",
            inner="quartile",
            cut=0,
            legend=False,
            ax=axes[1],
        )
        sns.stripplot(
            data=combined_df,
            x="split",
            y="spearman",
            color="#2c3e50",
            alpha=0.4,
            size=3,
            jitter=0.16,
            ax=axes[1],
        )
        axes[1].set_title(f"{result.model_name.upper()} | Spearman Violins")
        axes[1].set_xlabel("Data split")
        axes[1].set_ylabel("")

        fig.tight_layout()
        fig.savefig(model_dir / "spearman_by_split.png", dpi=300)
        plt.close(fig)

    per_gene_val = result.per_gene_metrics.get("val", [])
    if per_gene_val:
        pearson_vals = _collect_metric(per_gene_val, "pearson")
        spearman_vals = spearman_by_split.get("val", [])
        if pearson_vals or spearman_vals:
            model_dir.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
            if spearman_vals:
                plot_correlation_boxplot(
                    spearman_vals,
                    output_path=model_dir / "correlation_boxplots_val.png",
                    title=f"{result.model_name.upper()} | Spearman",
                    metric_label="Spearman correlation coefficient",
                    axes=axes[0],
                )
            else:
                axes[0].axis("off")
            if pearson_vals:
                plot_correlation_boxplot(
                    pearson_vals,
                    output_path=model_dir / "correlation_boxplots_val.png",
                    title=f"{result.model_name.upper()} | Pearson",
                    metric_label="Pearson correlation coefficient",
                    axes=axes[1],
                )
            else:
                axes[1].axis("off")
            fig.suptitle(f"{result.model_name.upper()} | Validation Correlations", y=0.92)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(model_dir / "correlation_boxplots_val.png", dpi=300)
            plt.close(fig)


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
        "multi_output": config.multi_output,
    }
    payload["run_name"] = config.run_name
    payload["timestamp_utc"] = _utc_timestamp()
    if extra_context:
        payload["run_context"] = _serialize_value(extra_context)

    output_path = run_dir / "run_configuration.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    _LOG.info("Exported run configuration snapshot to %s", output_path)


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
