#!/usr/bin/env python3
"""Select a reproducible set of well-expressed genes for SPEAR modeling."""


import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import anndata as ad
import numpy as np
import scipy.sparse as sp

from spear.config import PathsConfig, TrainingConfig
from spear.data import GeneInfo, parse_gtf, select_genes

DEFAULT_GENE_COUNT = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default=str(Path.cwd()),
        help="Project root containing AnnData matrices and configuration (defaults to the current working directory)",
    )
    parser.add_argument(
        "--gene-count",
        type=int,
        default=DEFAULT_GENE_COUNT,
        help="Number of genes to sample (default: %(default)s)",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        help="Minimum fraction of cells expressing each gene; defaults to TrainingConfig.min_expression_fraction",
    )
    parser.add_argument(
        "--min-expression",
        type=float,
        help="Expression threshold used when counting expressing cells; defaults to TrainingConfig.min_expression",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        help="Random seed for gene sampling; defaults to TrainingConfig.random_state",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional output file path; defaults to data/manifests/selected_genes_100.csv"
        ),
    )
    parser.add_argument(
        "--rna-path",
        type=Path,
        help="Optional RNA AnnData path to override the default resolved via base-dir",
    )
    parser.add_argument(
        "--atac-path",
        type=Path,
        help="Optional ATAC AnnData path to override the default resolved via base-dir",
    )
    parser.add_argument(
        "--gtf-path",
        type=Path,
        help="Optional GTF path to override the default annotation in PathsConfig",
    )
    return parser.parse_args()


def load_rna_matrix(paths: PathsConfig) -> ad.AnnData:
    return ad.read_h5ad(paths.rna_path.as_posix())


def compute_expression_fraction(
    rna: ad.AnnData,
    *,
    min_expression: float,
) -> tuple[dict[str, float], np.ndarray]:
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
    mapping = {name: float(frac) for name, frac in zip(gene_names, fractions, strict=False)}
    return mapping, gene_names


def filter_candidates(
    genes: Sequence[GeneInfo],
    fractions: dict[str, float],
    *,
    min_fraction: float,
) -> List[GeneInfo]:
    return [
        gene
        for gene in genes
        if fractions.get(gene.gene_name, fractions.get(gene.gene_id, 0.0)) >= min_fraction
    ]


def sample_genes(
    genes: Sequence[GeneInfo],
    *,
    count: int,
    random_state: int,
) -> List[GeneInfo]:
    if count > len(genes):
        raise ValueError(f"Requested {count} genes, but only {len(genes)} candidates are available")
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(genes), size=count, replace=False)
    return [genes[int(idx)] for idx in indices]


def write_manifest(
    genes: Iterable[GeneInfo],
    fractions: dict[str, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gene_list = list(genes)
    lines = ["gene_name,chrom,expression_fraction"]
    for gene in sorted(gene_list, key=lambda g: g.gene_name):
        frac = fractions.get(gene.gene_name, fractions.get(gene.gene_id, 0.0))
        lines.append(f"{gene.gene_name},{gene.chrom},{frac}")
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    paths = PathsConfig.from_base(args.base_dir)
    training = TrainingConfig()
    training.validate()

    min_fraction = args.min_fraction if args.min_fraction is not None else training.min_expression_fraction
    min_expression = args.min_expression if args.min_expression is not None else training.min_expression
    random_state = args.random_state if args.random_state is not None else training.random_state

    # Set the random seed for reproducibility, using --random-state if provided or falling back to training config
    np.random.seed(random_state)

    if args.rna_path is not None:
        paths = PathsConfig(
            base_dir=paths.base_dir,
            atac_path=args.atac_path or paths.atac_path,
            rna_path=args.rna_path,
            gtf_path=paths.gtf_path,
            output_dir=paths.output_dir,
            logs_dir=paths.logs_dir,
            figures_dir=paths.figures_dir,
        )

    rna = load_rna_matrix(paths)
    fractions, _ = compute_expression_fraction(rna, min_expression=min_expression)

    gtf_path = args.gtf_path if args.gtf_path is not None else paths.gtf_path
    genes_all = parse_gtf(gtf_path)
    selected_pool = select_genes(genes_all, requested_genes=None, max_genes=None)
    candidates = filter_candidates(selected_pool, fractions, min_fraction=min_fraction)

    if len(candidates) < args.gene_count:
        raise RuntimeError(
            "Insufficient genes meeting expression fraction threshold: "
            f"needed {args.gene_count}, found {len(candidates)}"
        )

    sampled = sample_genes(candidates, count=args.gene_count, random_state=random_state)

    default_output = paths.base_dir / "data" / "embryonic" / "manifests" / "selected_genes_100.csv"
    output_path = args.output if args.output is not None else default_output
    write_manifest(sampled, fractions, output_path)

    print(
        f"Selected {len(sampled)} genes with expression fraction >= {min_fraction:.2f} "
        f"and saved manifest to {output_path}"
    )


if __name__ == "__main__":
    main()
