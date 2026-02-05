#!/usr/bin/env python3
"""Prepare and cache datasets for SPEAR training.

This script:
1. Loads embryonic and endothelial datasets from processed h5ad files
2. Subsets to specified genes (1000 genes from manifests)
3. Prepares cell-wise data and caches to disk
4. Ready for model training without recomputation
"""

import logging
from pathlib import Path
from typing import Optional
import json
import sys

repo_root = Path(__file__).resolve().parent.parent
src_root = repo_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import numpy as np
import pandas as pd
import scanpy as sc
from spear.config import TrainingConfig
from spear.training import prepare_cellwise_data
from spear.data import CellwiseDataset, GeneInfo

logging.basicConfig(level=logging.INFO)
_LOG = logging.getLogger(__name__)


def load_genes_from_manifest(manifest_path: Path) -> list[str]:
    """Load gene names from manifest CSV file.
    
    Parameters
    ----------
    manifest_path : Path
        Path to manifest CSV file (assumed to have 'gene_name' column).
    
    Returns
    -------
    list[str]
        List of gene names.
    """
    df = pd.read_csv(manifest_path)
    if "gene_name" in df.columns:
        return df["gene_name"].tolist()
    # If first column is gene names without header
    return df.iloc[:, 0].tolist()


def resolve_processed_path(data_dir: Path, filename: str) -> Path:
    processed_dir = data_dir / "processed"
    direct = processed_dir / filename
    if direct.exists():
        return direct
    stem = filename.replace(".h5ad", "")
    matches = sorted(processed_dir.glob(f"{stem}*.h5ad"))
    if not matches:
        raise FileNotFoundError(f"Missing processed file '{filename}' in {processed_dir}")
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
        _LOG.warning(
            "Multiple matches for %s in %s; using most recent: %s",
            filename,
            processed_dir,
            matches[0].name,
        )
    return matches[0]


def prepare_embryonic_dataset(
    data_dir: Path = Path("data/embryonic"),
    cache_dir: Optional[Path] = None,
) -> CellwiseDataset:
    """Prepare and cache embryonic dataset.
    
    Parameters
    ----------
    data_dir : Path
        Base directory containing embryonic data.
    cache_dir : Path, optional
        Directory to save cached preprocessed data.
    """
    _LOG.info("=" * 80)
    _LOG.info("PREPARING EMBRYONIC DATASET")
    _LOG.info("=" * 80)
    
    # Load manifest
    manifest_path = data_dir / "manifests" / "1000_random_genes.csv"
    _LOG.info("Loading gene manifest from %s", manifest_path)
    gene_names = load_genes_from_manifest(manifest_path)
    _LOG.info("Loaded %d genes from manifest", len(gene_names))
    
    # Load RNA data
    rna_path = resolve_processed_path(data_dir, "combined_RNA_qc.h5ad")
    _LOG.info("Loading RNA data from %s", rna_path)
    adata_rna = sc.read_h5ad(rna_path)
    _LOG.info("RNA data shape: %s", adata_rna.shape)
    
    # Subset to genes in manifest
    genes_in_data = [g for g in gene_names if g in adata_rna.var_names]
    missing_genes = set(gene_names) - set(genes_in_data)
    if missing_genes:
        _LOG.warning("Missing %d genes from manifest: %s", len(missing_genes), list(missing_genes)[:5])
    
    adata_rna = adata_rna[:, genes_in_data]
    _LOG.info("Subsetted RNA data to %d genes", len(genes_in_data))
    
    # Load ATAC data
    atac_path = resolve_processed_path(data_dir, "combined_ATAC_qc.h5ad")
    _LOG.info("Loading ATAC data from %s", atac_path)
    adata_atac = sc.read_h5ad(atac_path)
    _LOG.info("ATAC data shape: %s", adata_atac.shape)
    
    # Intersect cells
    common_cells = np.intersect1d(adata_rna.obs_names, adata_atac.obs_names)
    _LOG.info("Found %d common cells between RNA and ATAC", len(common_cells))
    
    adata_rna = adata_rna[common_cells, :]
    adata_atac = adata_atac[common_cells, :]
    
    # Create dataset object
    group_labels = adata_rna.obs["batch"].values if "batch" in adata_rna.obs else np.arange(len(adata_rna))
    dataset = CellwiseDataset(
        genes=[GeneInfo(gene_id=g, gene_name=g, chrom="unknown", tss=0, strand="+") for g in genes_in_data],
        X=adata_atac.X,  # Features from ATAC
        y=adata_rna.X,   # Targets from RNA (multi-output)
        cell_ids=adata_rna.obs_names.values,
        feature_names=adata_atac.var_names.tolist(),
        group_labels=group_labels,
    )
    
    # Prepare cache directory
    if cache_dir is None:
        cache_dir = Path("data/.spear_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard config for preprocessing
    config = TrainingConfig(
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        random_state=42,
        scaler="standard",
        target_scaler="standard",
        enable_smoothing=False,
        pseudobulk_group_size=1,
        force_dense_features=False,
    )
    

    # Prepare and cache
    _LOG.info("Preparing cell-wise data for embryonic dataset...")
    prepared = prepare_cellwise_data(dataset, config, cache_dir=cache_dir)
    _LOG.info("✓ Embryonic dataset prepared and cached")
    _LOG.info(
        "  Train: %d | Val: %d | Test: %d | Features: %d | Targets: %d",
        prepared.splits.X_train.shape[0],
        prepared.splits.X_val.shape[0],
        prepared.splits.X_test.shape[0],
        prepared.splits.X_train.shape[1],
        prepared.splits.y_train.shape[1],
    )
    
    return dataset


def prepare_endothelial_dataset(
    data_dir: Path = Path("data/endothelial"),
    cache_dir: Optional[Path] = None,
) -> CellwiseDataset:
    """Prepare and cache endothelial dataset using gene manifest.
    
    Parameters
    ----------
    data_dir : Path
        Base directory containing endothelial data.
    cache_dir : Path, optional
        Directory to save cached preprocessed data.

    Returns
    -------
    CellwiseDataset
        Prepared dataset object.
    """
    _LOG.info("=" * 80)
    _LOG.info("PREPARING ENDOTHELIAL DATASET")
    _LOG.info("=" * 80)
    
    # Load manifest
    manifest_path = data_dir / "manifests" / "1000_random_genes.csv"
    _LOG.info("Loading gene manifest from %s", manifest_path)
    gene_names = load_genes_from_manifest(manifest_path)
    _LOG.info("Loaded %d genes from manifest", len(gene_names))

    # Load RNA data
    rna_path = resolve_processed_path(data_dir, "combined_RNA_qc_<15%mito.h5ad")
    _LOG.info("Loading RNA data from %s", rna_path)
    adata_rna = sc.read_h5ad(rna_path)
    _LOG.info("RNA data shape: %s", adata_rna.shape)

    # Subset to genes in manifest
    genes_in_data = [g for g in gene_names if g in adata_rna.var_names]
    missing_genes = set(gene_names) - set(genes_in_data)
    if missing_genes:
        _LOG.warning("Missing %d genes from manifest: %s", len(missing_genes), list(missing_genes)[:5])

    adata_rna = adata_rna[:, genes_in_data]
    _LOG.info("Subsetted RNA data to %d genes", len(genes_in_data))
    
    # Load ATAC data
    atac_path = resolve_processed_path(data_dir, "combined_ATAC_qc_<15%mito.h5ad")
    _LOG.info("Loading ATAC data from %s", atac_path)
    adata_atac = sc.read_h5ad(atac_path)
    _LOG.info("ATAC data shape: %s", adata_atac.shape)
    
    # Intersect cells
    common_cells = np.intersect1d(adata_rna.obs_names, adata_atac.obs_names)
    _LOG.info("Found %d common cells between RNA and ATAC", len(common_cells))
    
    adata_rna = adata_rna[common_cells, :]
    adata_atac = adata_atac[common_cells, :]
    
    # Create dataset object
    group_labels = adata_rna.obs["batch"].values if "batch" in adata_rna.obs else np.arange(len(adata_rna))
    dataset = CellwiseDataset(
        genes=[GeneInfo(gene_id=g, gene_name=g, chrom="unknown", tss=0, strand="+") for g in genes_in_data],
        X=adata_atac.X,  # Features from ATAC
        y=adata_rna.X,   # Targets from RNA (multi-output)
        cell_ids=adata_rna.obs_names.values,
        feature_names=adata_atac.var_names.tolist(),
        group_labels=group_labels,
    )
    
    # Prepare cache directory
    if cache_dir is None:
        cache_dir = Path("data/.spear_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard config for preprocessing
    config = TrainingConfig(
        train_fraction=0.6,
        val_fraction=0.2,
        test_fraction=0.2,
        random_state=42,
        scaler="standard",
        target_scaler="standard",
        enable_smoothing=False,
        pseudobulk_group_size=1,
        force_dense_features=False,
    )
    
    # Initialize cache if needed
    if not hasattr(dataset, "prepared_cache"):
        dataset.prepared_cache = {}
    
    # Prepare and cache
    _LOG.info("Preparing cell-wise data for endothelial dataset...")
    prepared = prepare_cellwise_data(dataset, config, cache_dir=cache_dir)
    _LOG.info("✓ Endothelial dataset prepared and cached")
    _LOG.info(
        "  Train: %d | Val: %d | Test: %d | Features: %d | Targets: %d",
        prepared.splits.X_train.shape[0],
        prepared.splits.X_val.shape[0],
        prepared.splits.X_test.shape[0],
        prepared.splits.X_train.shape[1],
        prepared.splits.y_train.shape[1],
    )
    
    return dataset


def main():
    """Prepare both datasets."""
    # Repo root is the parent of scripts/ (…/spear)
    workspace_root = Path(__file__).parent.parent
    cache_dir = workspace_root / "data" / ".spear_cache"
    
    _LOG.info("Workspace root: %s", workspace_root)
    _LOG.info("Cache directory: %s", cache_dir)
    _LOG.info("")
    
    # Prepare embryonic dataset
    embryonic_dataset = prepare_embryonic_dataset(
        data_dir=workspace_root / "data" / "embryonic",
        cache_dir=cache_dir,
    )
    
    print()
    
    # Prepare endothelial dataset
    endothelial_dataset = prepare_endothelial_dataset(
        data_dir=workspace_root / "data" / "endothelial",
        cache_dir=cache_dir,
    )
    
    print()
    _LOG.info("=" * 80)
    _LOG.info("✓ ALL DATASETS PREPARED AND CACHED")
    _LOG.info("=" * 80)
    _LOG.info("Cache location: %s", cache_dir)
    _LOG.info("")
    _LOG.info("Next: Train models with:")
    _LOG.info("  python -m spear.cli train --dataset embryonic --model ridge")
    _LOG.info("  python -m spear.cli train --dataset endothelial --model ridge")


if __name__ == "__main__":
    main()
