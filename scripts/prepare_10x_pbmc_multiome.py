#!/usr/bin/env python3
"""Download and prepare 10x PBMC Multiome data for SPEAR.

This prepares the same public 10x PBMC Multiome dataset commonly used by
SCARlink/Signac examples: ``pbmc_granulocyte_sorted_10k``. The immediate SPEAR
model-zoo screen only needs the filtered feature-barcode H5 because it contains
paired RNA and ATAC peak counts for the same cell barcodes.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spear.data import PeakIndexer  # noqa: E402
from spear.manifest import (  # noqa: E402
    annotate_manifest,
    compute_expression_fraction,
    compute_peak_counts,
    compute_peak_window_stats,
    filter_genes_by_peak_count,
    filter_genes_by_peak_window_quality,
    gene_annotation_lookup,
    low_expression_or_noisy_genes,
    sample_random_genes,
    top_hvg_genes,
)
from spear.config import TrainingConfig  # noqa: E402

DATASET_URL = (
    "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/"
    "10k_PBMC_Multiome_nextgem_Chromium_X/"
    "10k_PBMC_Multiome_nextgem_Chromium_X_filtered_feature_bc_matrix.h5"
)


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"Found existing raw file: {output}")
        return
    print(f"Downloading {url}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request) as response, output.open("wb") as handle:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total and total.isdigit() else None
        transferred = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            transferred += len(chunk)
            if total_int:
                pct = 100.0 * transferred / total_int
                print(
                    f"\r  {transferred / 1e6:.1f}/{total_int / 1e6:.1f} MB ({pct:.1f}%)",
                    end="",
                )
        print()


def _feature_type_column(var: pd.DataFrame) -> str:
    for column in ("feature_types", "feature_type"):
        if column in var.columns:
            return column
    raise ValueError(
        f"Could not find feature type column in 10x H5 var columns: {list(var.columns)}"
    )


def _gene_symbol_series(var: pd.DataFrame) -> pd.Series:
    for column in ("gene_symbols", "gene_symbol", "name"):
        if column in var.columns:
            return var[column].astype(str)
    return pd.Series(var.index.astype(str), index=var.index)


def _gene_id_series(var: pd.DataFrame) -> pd.Series:
    for column in ("gene_ids", "gene_id", "id"):
        if column in var.columns:
            return var[column].astype(str)
    return pd.Series(var.index.astype(str), index=var.index)


def _parse_peak_name(name: str) -> tuple[str, int, int]:
    text = str(name)
    if ":" in text and "-" in text:
        chrom, rest = text.split(":", 1)
        start, end = rest.split("-", 1)
        return chrom, int(start), int(end)
    parts = text.replace(":", "-").split("-")
    if len(parts) >= 3:
        return parts[0], int(parts[1]), int(parts[2])
    raise ValueError(f"Could not parse ATAC peak coordinates from feature name: {name}")


def _make_unique(values: pd.Series) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for raw in values.astype(str):
        value = raw
        count = seen.get(value, 0)
        if count:
            value = f"{raw}-{count}"
        seen[raw] = count + 1
        result.append(value)
    return result


def _decode(values) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _read_10x_h5(raw_h5: Path) -> tuple[sp.csr_matrix, pd.Index, pd.DataFrame]:
    with h5py.File(raw_h5, "r") as handle:
        group = handle["matrix"]
        data = group["data"][:]
        indices = group["indices"][:]
        indptr = group["indptr"][:]
        shape = tuple(group["shape"][:])
        barcodes = pd.Index(_decode(group["barcodes"][:]), dtype=object)
        features = group["features"]
        feature_ids = _decode(features["id"][:])
        feature_names = _decode(features["name"][:])
        feature_types = _decode(features["feature_type"][:])
        genomes = (
            _decode(features["genome"][:])
            if "genome" in features
            else [""] * len(feature_ids)
        )

    # 10x stores a feature x barcode CSC matrix. AnnData expects observations x variables.
    feature_by_cell = sp.csc_matrix((data, indices, indptr), shape=shape)
    cell_by_feature = feature_by_cell.T.tocsr()
    var = pd.DataFrame(
        {
            "gene_ids": feature_ids,
            "gene_symbols": feature_names,
            "feature_types": feature_types,
            "genome": genomes,
        },
        index=pd.Index(feature_names, dtype=object),
    )
    return cell_by_feature, barcodes, var


def _split_modalities(raw_h5: Path, sample_name: str) -> tuple[ad.AnnData, ad.AnnData]:
    matrix, barcodes, var = _read_10x_h5(raw_h5)
    feature_col = _feature_type_column(var)
    feature_type = var[feature_col].astype(str).to_numpy()
    rna_mask = feature_type == "Gene Expression"
    atac_mask = feature_type == "Peaks"
    if int(rna_mask.sum()) == 0 or int(atac_mask.sum()) == 0:
        raise ValueError(
            f"Expected both Gene Expression and Peaks features; found RNA={int(rna_mask.sum())}, ATAC={int(atac_mask.sum())}"
        )

    rna = ad.AnnData(
        X=matrix[:, rna_mask].copy(),
        obs=pd.DataFrame(index=barcodes),
        var=var.loc[rna_mask].copy(),
    )
    atac = ad.AnnData(
        X=matrix[:, atac_mask].copy(),
        obs=pd.DataFrame(index=barcodes),
        var=var.loc[atac_mask].copy(),
    )

    rna_symbols = _make_unique(_gene_symbol_series(rna.var))
    rna_ids = _gene_id_series(rna.var).to_numpy()
    rna.var_names = pd.Index(rna_symbols, dtype=object)
    rna.var["gene_ids"] = rna_ids

    peak_names = pd.Index(atac.var_names.astype(str), dtype=object)
    coords = [_parse_peak_name(name) for name in peak_names]
    atac.var_names = peak_names
    atac.var["gene_ids"] = peak_names.astype(str)
    atac.var["chromosome"] = [chrom for chrom, _, _ in coords]
    atac.var["start"] = [start for _, start, _ in coords]
    atac.var["end"] = [end for _, _, end in coords]
    atac.var["peak"] = peak_names.astype(str)

    for obj, modality in ((rna, "rna"), (atac, "atac")):
        obj.obs["barcode"] = pd.Index(obj.obs_names).astype(str)
        obj.obs["sample"] = sample_name
        obj.obs["dataset"] = "pbmc"
        obj.obs["source"] = "10x_pbmc_granulocyte_sorted_10k"
        obj.obs["modality_preparation"] = modality

    return rna, atac


def _mean_var_maps(
    rna: ad.AnnData,
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    matrix = rna.X
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
        mean_vals = np.asarray(matrix.mean(axis=0)).ravel()
        sq_mean_vals = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        var_vals = np.maximum(sq_mean_vals - np.square(mean_vals), 0.0)
        expr_counts = np.asarray(matrix.getnnz(axis=0)).ravel()
    else:
        arr = np.asarray(matrix)
        mean_vals = arr.mean(axis=0)
        var_vals = arr.var(axis=0)
        expr_counts = (arr > 0).sum(axis=0)
    names = np.asarray(rna.var_names).astype(str)
    mean_map = {name: float(val) for name, val in zip(names, mean_vals, strict=False)}
    var_map = {name: float(val) for name, val in zip(names, var_vals, strict=False)}
    count_map = {name: int(val) for name, val in zip(names, expr_counts, strict=False)}
    return mean_map, var_map, count_map


def _write_manifests(
    *,
    rna: ad.AnnData,
    atac: ad.AnnData,
    gtf_path: Path,
    manifest_dir: Path,
    sample_name: str,
    gene_count: int,
    random_state: int,
    peak_window_bp: int,
    min_peaks_per_gene: int,
) -> None:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    training = TrainingConfig()
    training.window_bp = int(peak_window_bp)
    gene_lookup = gene_annotation_lookup(gtf_path)
    expr_fraction = compute_expression_fraction(
        rna, min_expression=training.min_expression
    )
    mean_map, var_map, expr_counts = _mean_var_maps(rna)
    min_cells = min(
        training.min_cells_per_gene,
        max(1, int(np.ceil(training.min_expression_fraction * rna.n_obs))),
    )
    candidate_genes = [
        gene
        for gene in np.asarray(rna.var_names).astype(str)
        if gene in gene_lookup
        and expr_fraction.get(gene, 0.0) >= training.min_expression_fraction
        and expr_counts.get(gene, 0) >= min_cells
    ]
    peak_indexer = PeakIndexer(atac)
    peak_counts = compute_peak_counts(
        candidate_genes,
        gene_lookup=gene_lookup,
        peak_indexer=peak_indexer,
        training=training,
    )
    candidate_genes = filter_genes_by_peak_count(
        candidate_genes,
        peak_counts=peak_counts,
        min_peaks_per_gene=min_peaks_per_gene,
    )
    peak_window_stats = compute_peak_window_stats(
        candidate_genes,
        gene_lookup=gene_lookup,
        peak_indexer=peak_indexer,
        training=training,
        compute_variance=False,
    )
    candidate_genes = filter_genes_by_peak_window_quality(
        candidate_genes,
        peak_window_stats=peak_window_stats,
        min_nonzero_fraction=0.0,
        min_variance=0.0,
    )
    print(
        f"PBMC manifest candidates after expression/annotation/peak filters: {len(candidate_genes)}"
    )

    label_to_genes = {
        "random": sample_random_genes(
            candidate_genes, gene_count=gene_count, random_state=random_state
        ),
        "hvg": top_hvg_genes(
            rna, gene_count=gene_count, eligible_genes=set(candidate_genes)
        ),
        "low_noisy": low_expression_or_noisy_genes(
            rna,
            gene_count=gene_count,
            expression_fraction=expr_fraction,
            expression_counts=expr_counts,
            gene_lookup=gene_lookup,
            min_fraction=training.min_expression_fraction,
            min_cells=min_cells,
        ),
    }
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
        output = manifest_dir / f"{sample_name}_{label}_{gene_count}.csv"
        df.to_csv(output, index=False)
        print(f"Wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("data/pbmc"))
    parser.add_argument("--sample-name", default="PBMC_10x")
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--gene-count", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--peak-window-bp", type=int, default=250_000)
    parser.add_argument("--min-peaks-per-gene", type=int, default=1)
    parser.add_argument(
        "--gtf-path",
        type=Path,
        default=Path("data/references/gencode.v44.annotation.gtf.gz"),
    )
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    raw_h5 = (
        base_dir / "raw" / "pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5"
    )
    if not args.skip_download:
        _download(args.url, raw_h5)
    if not raw_h5.exists():
        raise FileNotFoundError(f"Missing raw 10x H5: {raw_h5}")

    processed_dir = base_dir / "processed" / "per_sample" / args.sample_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    rna_path = processed_dir / f"{args.sample_name}_RNA_qc.h5ad"
    atac_path = processed_dir / f"{args.sample_name}_ATAC_qc.h5ad"

    if rna_path.exists() and atac_path.exists():
        print(f"Found existing processed files: {rna_path}, {atac_path}")
        rna = ad.read_h5ad(rna_path)
        atac = ad.read_h5ad(atac_path)
    else:
        rna, atac = _split_modalities(raw_h5, args.sample_name)
        rna.write_h5ad(rna_path)
        atac.write_h5ad(atac_path)
        print(f"Wrote {rna_path}")
        print(f"Wrote {atac_path}")

    _write_manifests(
        rna=rna,
        atac=atac,
        gtf_path=args.gtf_path.expanduser().resolve(),
        manifest_dir=base_dir / "manifests",
        sample_name=args.sample_name,
        gene_count=args.gene_count,
        random_state=args.random_state,
        peak_window_bp=args.peak_window_bp,
        min_peaks_per_gene=args.min_peaks_per_gene,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
