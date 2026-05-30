#!/usr/bin/env python3
"""Generate a manifest of top genes by expression fraction for a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import anndata as ad
import numpy as np
import scipy.sparse as sp

from spear.config import TrainingConfig
from spear.data import GeneInfo, parse_gtf, select_genes

DATASET_DEFAULTS = {
    "embryonic": {
        "rna_path": Path("data/embryonic/processed/combined_RNA_qc.h5ad"),
        "manifest_dir": Path("data/embryonic/manifests"),
        "gtf_path": Path("data/references/GCF_000001635.27_genomic.gtf"),
    },
    "endothelial": {
        "rna_path": Path("data/endothelial/processed/combined_RNA_qc_<15%mito.h5ad"),
        "manifest_dir": Path("data/endothelial/manifests"),
        "gtf_path": Path("data/references/gencode.v44.annotation.gtf.gz"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=str(Path.cwd()),
        help="Project root (defaults to current working directory)",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_DEFAULTS.keys()),
        required=True,
        help="Dataset name to process",
    )
    parser.add_argument(
        "--gene-count",
        type=int,
        default=1000,
        help="Number of genes to include (default: 1000)",
    )
    parser.add_argument(
        "--min-expression",
        type=float,
        help="Expression threshold for counting expressing cells; defaults to TrainingConfig.min_expression",
    )
    parser.add_argument(
        "--rna-path",
        type=Path,
        help="Optional RNA AnnData path override",
    )
    parser.add_argument(
        "--gtf-path",
        type=Path,
        help="Optional GTF path override",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path override",
    )
    return parser.parse_args()


def load_rna_matrix(rna_path: Path) -> ad.AnnData:
    return ad.read_h5ad(rna_path.as_posix())


def compute_expression_fraction(
    rna: ad.AnnData,
    *,
    min_expression: float,
) -> dict[str, float]:
    matrix = rna.X
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
        if min_expression <= 0.0:
            counts = matrix.getnnz(axis=0)
        else:
            mask = matrix.copy()
            mask.data = (mask.data >= min_expression).astype(mask.data.dtype)
            counts = np.asarray(mask.sum(axis=0)).ravel()
    else:
        counts = np.asarray((matrix >= min_expression).sum(axis=0)).ravel()
    fractions = counts / float(rna.n_obs)
    gene_names = np.asarray(rna.var_names).astype(str)
    return {
        name: float(frac) for name, frac in zip(gene_names, fractions, strict=False)
    }


def rank_genes(
    genes: Iterable[GeneInfo],
    fractions: dict[str, float],
) -> List[tuple[GeneInfo, float]]:
    ranked: List[tuple[GeneInfo, float]] = []
    for gene in genes:
        frac = fractions.get(gene.gene_name)
        if frac is None:
            frac = fractions.get(gene.gene_id)
        if frac is None:
            continue
        ranked.append((gene, float(frac)))
    ranked.sort(
        key=lambda item: (-item[1], item[0].gene_name.lower(), item[0].gene_id.lower())
    )
    return ranked


def write_manifest(
    ranked: List[tuple[GeneInfo, float]],
    output_path: Path,
    *,
    limit: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["gene_name,chrom,expression_fraction"]
    for gene, frac in ranked[:limit]:
        lines.append(f"{gene.gene_name},{gene.chrom},{frac}")
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()

    defaults = DATASET_DEFAULTS[args.dataset]
    rna_path = args.rna_path or (base_dir / defaults["rna_path"])
    manifest_dir = base_dir / defaults["manifest_dir"]

    training = TrainingConfig()
    training.validate()

    min_expression = (
        args.min_expression
        if args.min_expression is not None
        else training.min_expression
    )

    rna = load_rna_matrix(rna_path)
    fractions = compute_expression_fraction(rna, min_expression=min_expression)

    gtf_path = args.gtf_path or (base_dir / defaults["gtf_path"])
    genes_all = parse_gtf(gtf_path)
    selected_pool = select_genes(genes_all, requested_genes=None, max_genes=None)

    ranked = rank_genes(selected_pool, fractions)
    if not ranked:
        raise SystemExit("No matching genes found between GTF and RNA matrix.")

    output_path = args.output or (
        manifest_dir / f"top_{args.gene_count}_expression_fraction.csv"
    )
    write_manifest(ranked, output_path, limit=args.gene_count)

    print(f"Wrote {min(args.gene_count, len(ranked))} genes to {output_path}")


if __name__ == "__main__":
    main()
