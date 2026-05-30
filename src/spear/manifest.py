"""Manifest generation utilities for SPEAR gene selection."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import TrainingConfig
from .data import GeneInfo, PeakIndexer, _gene_window_peak_indices, parse_gtf


def _expression_counts_and_fractions(
    rna: ad.AnnData,
    *,
    min_expression: float,
) -> tuple[dict[str, int], dict[str, float]]:
    """Return (counts, fractions) dicts — cells with expression >= min_expression per gene."""
    matrix = rna.X
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
        if min_expression <= 0.0:
            raw_counts = matrix.getnnz(axis=0)
        else:
            mask = matrix.copy()
            mask.data = (mask.data >= min_expression).astype(mask.data.dtype)
            raw_counts = np.asarray(mask.sum(axis=0)).ravel()
    else:
        raw_counts = np.asarray((matrix >= min_expression).sum(axis=0)).ravel()
    gene_names = np.asarray(rna.var_names).astype(str)
    n_cells = float(rna.n_obs)
    counts = {
        str(name): int(c) for name, c in zip(gene_names, raw_counts, strict=False)
    }
    fractions = {
        str(name): int(c) / n_cells
        for name, c in zip(gene_names, raw_counts, strict=False)
    }
    return counts, fractions


def compute_expression_fraction(
    rna: ad.AnnData,
    *,
    min_expression: float,
) -> dict[str, float]:
    _, fractions = _expression_counts_and_fractions(rna, min_expression=min_expression)
    return fractions


def gene_annotation_lookup(gtf_path: Path) -> dict[str, GeneInfo]:
    genes = parse_gtf(gtf_path)
    lookup: dict[str, GeneInfo] = {}
    for gene in genes:
        lookup.setdefault(gene.gene_name, gene)
        lookup.setdefault(gene.gene_id, gene)
    return lookup


def annotate_manifest(
    genes: list[str],
    *,
    gene_lookup: dict[str, GeneInfo],
    expression_fraction: dict[str, float],
    mean_expression: dict[str, float],
    variance_expression: dict[str, float],
    peak_counts: dict[str, int] | None = None,
    peak_window_stats: dict[str, dict[str, float]] | None = None,
    peak_window_bp: int | None = None,
) -> pd.DataFrame:
    rows = []
    for gene_name in genes:
        gene = gene_lookup.get(gene_name)
        if gene is None:
            continue
        row = {
            "gene_name": gene_name,
            "chrom": gene.chrom,
            "expression_fraction": expression_fraction.get(gene_name, 0.0),
            "mean_expression": mean_expression.get(gene_name, 0.0),
            "variance_expression": variance_expression.get(gene_name, 0.0),
        }
        if peak_counts is not None:
            row["peak_window_bp"] = int(peak_window_bp or 0)
            row["peak_count_in_window"] = int(peak_counts.get(gene_name, 0))
        if peak_window_stats is not None:
            stats = peak_window_stats.get(gene_name, {})
            row["peak_window_nonzero_fraction"] = float(
                stats.get("nonzero_fraction", 0.0)
            )
            row["peak_window_mean"] = float(stats.get("mean", 0.0))
            row["peak_window_variance"] = float(stats.get("variance", 0.0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gene_name").reset_index(drop=True)


def compute_peak_counts(
    genes: list[str],
    *,
    gene_lookup: dict[str, GeneInfo],
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
) -> dict[str, int]:
    peak_counts: dict[str, int] = {}
    for gene_name in genes:
        gene = gene_lookup.get(gene_name)
        if gene is None:
            continue
        peak_counts[gene_name] = int(
            _gene_window_peak_indices(gene, peak_indexer, training).size
        )
    return peak_counts


def filter_genes_by_peak_count(
    candidate_genes: list[str],
    *,
    peak_counts: dict[str, int],
    min_peaks_per_gene: int,
) -> list[str]:
    if min_peaks_per_gene <= 0:
        return list(candidate_genes)
    return [
        gene_name
        for gene_name in candidate_genes
        if int(peak_counts.get(gene_name, 0)) >= min_peaks_per_gene
    ]


def compute_peak_window_stats(
    genes: list[str],
    *,
    gene_lookup: dict[str, GeneInfo],
    peak_indexer: PeakIndexer,
    training: TrainingConfig,
    compute_variance: bool = True,
) -> dict[str, dict[str, float]]:
    """Compute simple sparsity/variance stats for each gene's peak window.

    Stats are computed over the full (cells × peaks-in-window) matrix:
    - nonzero_fraction: nnz / (n_cells * n_peaks)
    - mean: E[x] (optional; only when compute_variance=True)
    - variance: E[x^2] - E[x]^2 (optional; only when compute_variance=True)

    These are cheap to compute from a sparse peak matrix and are sufficient to
    proactively filter genes with all-zero or nearly-all-zero windows.
    """
    stats: dict[str, dict[str, float]] = {}
    n_cells = int(peak_indexer.n_cells)
    matrix = peak_indexer.matrix
    sparse = sp.issparse(matrix)
    peak_nnz: np.ndarray | None = None
    if sparse:
        # Precompute per-peak nnz so we can quickly detect all-zero windows
        # without slicing a submatrix for every gene.
        try:
            peak_nnz = np.asarray(matrix.getnnz(axis=0)).ravel()
        except Exception:
            peak_nnz = None

    for gene_name in genes:
        gene = gene_lookup.get(gene_name)
        if gene is None:
            continue
        idxs = _gene_window_peak_indices(gene, peak_indexer, training)
        n_peaks = int(idxs.size)
        if n_peaks == 0 or n_cells == 0:
            stats[gene_name] = {"nonzero_fraction": 0.0, "mean": 0.0, "variance": 0.0}
            continue
        if peak_nnz is not None:
            window_peak_nnz = peak_nnz[idxs]
            if int(np.any(window_peak_nnz > 0)) == 0:
                stats[gene_name] = {
                    "nonzero_fraction": 0.0,
                    "mean": 0.0,
                    "variance": 0.0,
                }
                continue
        sub = matrix[:, idxs]
        if sparse:
            sub = sub.tocsr()
            nnz = float(sub.nnz)
            if nnz == 0.0:
                stats[gene_name] = {
                    "nonzero_fraction": 0.0,
                    "mean": 0.0,
                    "variance": 0.0,
                }
                continue
            total = float(n_cells * n_peaks)
            if compute_variance:
                summed = float(sub.sum())
                sq_summed = float(sub.power(2).sum())
            else:
                summed = float("nan")
                sq_summed = float("nan")
        else:
            arr = np.asarray(sub, dtype=np.float32)
            nnz = float(np.count_nonzero(arr))
            if nnz == 0.0:
                stats[gene_name] = {
                    "nonzero_fraction": 0.0,
                    "mean": 0.0,
                    "variance": 0.0,
                }
                continue
            total = float(n_cells * n_peaks)
            if compute_variance:
                summed = float(arr.sum())
                sq_summed = float(np.square(arr).sum())
            else:
                summed = float("nan")
                sq_summed = float("nan")
        mean = (summed / total) if compute_variance else float("nan")
        if compute_variance:
            ex2 = sq_summed / total
            var = max(0.0, ex2 - (mean * mean))
        else:
            var = float("nan")
        stats[gene_name] = {
            "nonzero_fraction": nnz / total,
            "mean": mean,
            "variance": var,
        }
    return stats


def filter_genes_by_peak_window_quality(
    candidate_genes: list[str],
    *,
    peak_window_stats: dict[str, dict[str, float]],
    min_nonzero_fraction: float,
    min_variance: float,
) -> list[str]:
    # Always drop fully degenerate windows by default:
    # - nonzero_fraction == 0: entire window is zeros
    filtered: list[str] = []
    for gene_name in candidate_genes:
        stats = peak_window_stats.get(gene_name, {})
        nonzero_fraction = float(stats.get("nonzero_fraction", 0.0))
        if nonzero_fraction <= 0.0:
            continue
        if nonzero_fraction < min_nonzero_fraction:
            continue
        variance = float(stats.get("variance", float("nan")))
        if min_variance > 0.0:
            if not np.isfinite(variance) or variance < min_variance:
                continue
        filtered.append(gene_name)
    return filtered


def sample_random_genes(
    candidate_genes: list[str],
    *,
    gene_count: int,
    random_state: int,
) -> list[str]:
    rng = np.random.default_rng(random_state)
    if len(candidate_genes) < gene_count:
        raise RuntimeError(
            f"Need {gene_count} random genes but only found {len(candidate_genes)} candidates"
        )
    indices = rng.choice(len(candidate_genes), size=gene_count, replace=False)
    return sorted(candidate_genes[int(idx)] for idx in indices)


def top_hvg_genes(
    rna: ad.AnnData,
    *,
    gene_count: int,
    eligible_genes: set[str] | None = None,
) -> list[str]:
    matrix = rna.X
    if sp.issparse(matrix):
        matrix = matrix.tocsr().astype(np.float32)
        library_size = np.asarray(matrix.sum(axis=1)).ravel()
        scale = np.divide(
            1e4,
            np.maximum(library_size, 1e-12),
            out=np.zeros_like(library_size, dtype=np.float32),
            where=library_size > 0,
        )
        work = sp.diags(scale).dot(matrix).tocsr()
        work.data = np.log1p(work.data)
        mean = np.asarray(work.mean(axis=0)).ravel()
        sq_mean = np.asarray(work.power(2).mean(axis=0)).ravel()
        var = np.maximum(sq_mean - np.square(mean), 0.0)
    else:
        arr = np.asarray(matrix, dtype=np.float32)
        library_size = arr.sum(axis=1, keepdims=True)
        scale = np.divide(
            1e4,
            np.maximum(library_size, 1e-12),
            out=np.zeros_like(library_size, dtype=np.float32),
            where=library_size > 0,
        )
        work = np.log1p(arr * scale)
        mean = work.mean(axis=0)
        var = work.var(axis=0)

    dispersion = np.divide(
        var,
        np.maximum(mean, 1e-12),
        out=np.zeros_like(var, dtype=np.float32),
        where=mean > 0,
    )
    hvgs_df = pd.DataFrame(
        {
            "gene_name": np.asarray(rna.var_names).astype(str),
            "mean_expression_norm": mean,
            "variance_expression_norm": var,
            "dispersion_norm": dispersion,
        }
    ).sort_values(
        ["dispersion_norm", "variance_expression_norm", "gene_name"],
        ascending=[False, False, True],
        na_position="last",
    )
    hvgs = hvgs_df["gene_name"].astype(str).tolist()
    if eligible_genes is not None:
        hvgs = [gene for gene in hvgs if gene in eligible_genes]
    if len(hvgs) < gene_count:
        raise RuntimeError(f"Need {gene_count} HVGs but only found {len(hvgs)}")
    return sorted(hvgs[:gene_count])


def low_expression_or_noisy_genes(
    rna: ad.AnnData,
    *,
    gene_count: int,
    expression_fraction: dict[str, float],
    expression_counts: dict[str, int],
    gene_lookup: dict[str, GeneInfo],
    min_fraction: float,
    min_cells: int,
) -> list[str]:
    matrix = rna.X
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        sq_mean = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        var = np.maximum(sq_mean - np.square(mean), 0.0)
    else:
        arr = np.asarray(matrix)
        mean = arr.mean(axis=0)
        var = arr.var(axis=0)

    gene_names = np.asarray(rna.var_names).astype(str)
    df = pd.DataFrame(
        {"gene_name": gene_names, "mean_expression": mean, "variance_expression": var}
    )
    df["expression_fraction"] = df["gene_name"].map(expression_fraction).fillna(0.0)
    df["expression_count"] = (
        df["gene_name"].map(expression_counts).fillna(0).astype(int)
    )
    # Require both thresholds so low_noisy genes meet the same bar as all other manifests.
    df = df[
        (df["expression_fraction"] >= min_fraction)
        & (df["expression_count"] >= min_cells)
        & (df["gene_name"].isin(gene_lookup))
    ].copy()
    # Lowest mean expression first; tie-break on higher variance to bias toward noisier genes.
    df = df.sort_values(
        ["mean_expression", "expression_fraction", "variance_expression", "gene_name"],
        ascending=[True, True, False, True],
    )
    if df.shape[0] < gene_count:
        raise RuntimeError(
            f"Need {gene_count} low/noisy expressed genes but only found {df.shape[0]} "
            f"after requiring expression_fraction >= {min_fraction:.2f} and "
            f"expression_count >= {min_cells}"
        )
    return sorted(df["gene_name"].head(gene_count).tolist())


def dataset_specs(base_dir: Path) -> list[tuple[str, Path, Path]]:
    return [
        (
            "embryonic",
            base_dir / "embryonic" / "processed" / "per_sample",
            base_dir / "references" / "GCF_000001635.27_genomic.gtf",
        ),
        (
            "endothelial",
            base_dir / "endothelial" / "processed" / "per_sample",
            base_dir / "references" / "gencode.v44.annotation.gtf.gz",
        ),
    ]


def _build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Generate per-sample gene manifests (random, HVG, low_noisy) from preprocessed RNA data.",
    )
    parser.add_argument(
        "--base-dir",
        default=str(Path.cwd() / "data"),
        help="Base data directory containing <dataset>/processed/per_sample/ subdirs (default: <cwd>/data)",
    )
    parser.add_argument(
        "--gene-count",
        type=int,
        default=1000,
        help="Genes per manifest (default: 1000)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for gene sampling (default: 42)",
    )
    parser.add_argument(
        "--peak-window-bp",
        type=int,
        default=TrainingConfig().window_bp,
        help=(
            "Peak-count annotation window around TSS in base pairs "
            f"(default: {TrainingConfig().window_bp})"
        ),
    )
    parser.add_argument(
        "--min-peaks-per-gene",
        type=int,
        default=0,
        help=(
            "Require at least this many ATAC peaks within ±peak-window-bp for every gene in the manifest. "
            "Use this to pre-filter manifests for peak-only runs."
        ),
    )
    parser.add_argument(
        "--min-peak-window-nonzero-fraction",
        type=float,
        default=0.0,
        help=(
            "Require at least this fraction of nonzero entries in the (cells × peaks) peak window matrix. "
            "Use this to filter all/mostly-zero local windows (default: 0)."
        ),
    )
    parser.add_argument(
        "--min-peak-window-variance",
        type=float,
        default=0.0,
        help=(
            "Require at least this variance over the (cells × peaks) peak window matrix. "
            "Use this to filter constant/near-constant windows (default: 0)."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("embryonic", "endothelial", "all"),
        default=["all"],
        help="Dataset(s) to process (default: all)",
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        help="Optional sample-name filter; defaults to all processed samples in the selected dataset(s)",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=("random", "hvg", "low_noisy", "all"),
        default=["all"],
        help="Which manifest labels to generate (default: all)",
    )
    return parser


def generate_manifest_main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser(prog="spear generate-manifest")
    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir).expanduser().resolve()
    training = TrainingConfig()
    min_fraction = training.min_expression_fraction
    min_expression = training.min_expression
    min_cells = training.min_cells_per_gene
    training.window_bp = int(args.peak_window_bp)
    requested = set(args.datasets)
    if "all" in requested:
        requested = {"embryonic", "endothelial"}
    requested_samples = set(args.samples or [])
    requested_labels = set(args.labels or ["all"])
    if "all" in requested_labels:
        requested_labels = {"random", "hvg", "low_noisy"}

    for dataset_name, per_sample_dir, gtf_path in dataset_specs(base_dir):
        if dataset_name not in requested:
            continue
        if not per_sample_dir.exists():
            continue
        gene_lookup = gene_annotation_lookup(gtf_path)
        manifest_dir = base_dir / dataset_name / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        for sample_dir in sorted(per_sample_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_name = sample_dir.name
            if requested_samples and sample_name not in requested_samples:
                continue
            rna_path = sample_dir / f"{sample_name}_RNA_qc.h5ad"
            atac_path = sample_dir / f"{sample_name}_ATAC_qc.h5ad"
            if not rna_path.exists():
                continue
            if (
                args.min_peaks_per_gene > 0 or args.peak_window_bp > 0
            ) and not atac_path.exists():
                raise FileNotFoundError(
                    f"Missing ATAC file required for peak-count annotation: {atac_path}"
                )

            rna = ad.read_h5ad(rna_path)
            atac = ad.read_h5ad(atac_path)
            peak_indexer = PeakIndexer(atac)
            expr_counts, expr_fraction = _expression_counts_and_fractions(
                rna, min_expression=min_expression
            )
            # Per-sample manifests must stay feasible for small samples. When
            # the global absolute threshold exceeds the sample size, fall back
            # to the count implied by the expression-fraction rule.
            fraction_min_cells = max(1, math.ceil(min_fraction * int(rna.n_obs)))
            effective_min_cells = min(min_cells, fraction_min_cells)
            matrix = rna.X
            if sp.issparse(matrix):
                matrix = matrix.tocsr()
                mean_vals = np.asarray(matrix.mean(axis=0)).ravel()
                sq_mean_vals = np.asarray(matrix.power(2).mean(axis=0)).ravel()
                var_vals = np.maximum(sq_mean_vals - np.square(mean_vals), 0.0)
            else:
                arr = np.asarray(matrix)
                mean_vals = arr.mean(axis=0)
                var_vals = arr.var(axis=0)

            mean_map = {
                str(name): float(val)
                for name, val in zip(
                    np.asarray(rna.var_names).astype(str), mean_vals, strict=False
                )
            }
            var_map = {
                str(name): float(val)
                for name, val in zip(
                    np.asarray(rna.var_names).astype(str), var_vals, strict=False
                )
            }

            # A gene must pass BOTH thresholds to be eligible for any manifest.
            candidate_genes = [
                gene_name
                for gene_name in np.asarray(rna.var_names).astype(str)
                if expr_fraction.get(gene_name, 0.0) >= min_fraction
                and expr_counts.get(gene_name, 0) >= effective_min_cells
                and gene_name in gene_lookup
            ]
            peak_counts = compute_peak_counts(
                candidate_genes,
                gene_lookup=gene_lookup,
                peak_indexer=peak_indexer,
                training=training,
            )
            candidate_genes = filter_genes_by_peak_count(
                candidate_genes,
                peak_counts=peak_counts,
                min_peaks_per_gene=int(args.min_peaks_per_gene),
            )
            peak_window_stats = compute_peak_window_stats(
                candidate_genes,
                gene_lookup=gene_lookup,
                peak_indexer=peak_indexer,
                training=training,
                compute_variance=float(args.min_peak_window_variance) > 0.0,
            )
            candidate_genes = filter_genes_by_peak_window_quality(
                candidate_genes,
                peak_window_stats=peak_window_stats,
                min_nonzero_fraction=float(args.min_peak_window_nonzero_fraction),
                min_variance=float(args.min_peak_window_variance),
            )

            label_to_genes: dict[str, list[str]] = {}
            if "random" in requested_labels:
                label_to_genes["random"] = sample_random_genes(
                    candidate_genes,
                    gene_count=args.gene_count,
                    random_state=args.random_state,
                )
            if "hvg" in requested_labels:
                label_to_genes["hvg"] = top_hvg_genes(
                    rna,
                    gene_count=args.gene_count,
                    eligible_genes=set(candidate_genes),
                )
            if "low_noisy" in requested_labels:
                label_to_genes["low_noisy"] = low_expression_or_noisy_genes(
                    rna,
                    gene_count=args.gene_count,
                    expression_fraction=expr_fraction,
                    expression_counts=expr_counts,
                    gene_lookup=gene_lookup,
                    min_fraction=min_fraction,
                    min_cells=effective_min_cells,
                )

            for label, genes in label_to_genes.items():
                df = annotate_manifest(
                    genes,
                    gene_lookup=gene_lookup,
                    expression_fraction=expr_fraction,
                    mean_expression=mean_map,
                    variance_expression=var_map,
                    peak_counts=peak_counts,
                    peak_window_stats=peak_window_stats,
                    peak_window_bp=training.window_bp,
                )
                output_path = (
                    manifest_dir / f"{sample_name}_{label}_{args.gene_count}.csv"
                )
                df.to_csv(output_path, index=False)
                print(f"Wrote {output_path}")

    return 0
